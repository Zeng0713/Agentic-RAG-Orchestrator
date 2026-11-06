"""Bounded, rotating, metadata-only logs for the desktop launcher."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import threading
import uuid


@dataclass(frozen=True)
class LogPolicy:
    max_bytes: int = 20 * 1024 * 1024
    max_files: int = 6  # Includes the active file, not six additional backups.
    days: int = 7

    def __post_init__(self):
        if self.max_bytes < 128 or self.max_files < 1 or self.days < 1:
            raise ValueError("日志大小、文件数量和保留时间配置无效")


class LogManager:
    CHANNELS = {"launcher", "backend", "frontend"}
    OWNED_NAME = re.compile(r"^(launcher|backend|frontend)(?:\.\d{8}-\d{6}-[0-9a-f]{6})?\.log$")

    def __init__(self, directory, policy=None, secrets=(), clock=None):
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("日志目录不能是符号链接")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory = self.directory.resolve()
        self.policy = policy or LogPolicy()
        self.secrets = sorted({s for s in secrets if s}, key=len, reverse=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.lock = threading.RLock()
        self.cleanup()

    def redact(self, text):
        text = str(text).replace("\r", " ").replace("\n", " ")
        for value in self.secrets:
            text = text.replace(value, "[REDACTED]")
        text = re.sub(r"\b(rediss?://)[^\s/@]+@", r"\1[REDACTED]@", text, flags=re.I)
        text = re.sub(r"\bBearer\s+\S+", "Bearer [REDACTED]", text, flags=re.I)
        text = re.sub(r"\bsk-[\w-]{8,}", "[REDACTED]", text)
        return text

    def _files(self, channel):
        return [p for p in self.directory.iterdir()
                if p.is_file() and not p.is_symlink() and self.OWNED_NAME.fullmatch(p.name)
                and p.name.startswith(channel + ".")]

    def _first_time(self, path):
        try:
            with path.open(encoding="utf-8") as stream:
                value = stream.readline(128).split(" ", 1)[0]
            return datetime.fromisoformat(value).astimezone(timezone.utc)
        except (OSError, ValueError):
            return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)

    def _unlink(self, path):
        # Cleanup never recurses or follows paths outside the named log directory.
        if path.parent == self.directory and not path.is_symlink() and self.OWNED_NAME.fullmatch(path.name):
            path.unlink(missing_ok=True)

    def cleanup(self):
        with self.lock:
            cutoff = self.clock() - timedelta(days=self.policy.days)
            for channel in self.CHANNELS:
                for path in self._files(channel):
                    if self._first_time(path) < cutoff:
                        self._unlink(path)
                active = self.directory / f"{channel}.log"
                archives = sorted((p for p in self._files(channel) if p != active),
                                  key=lambda p: (self._first_time(p), p.stat().st_mtime_ns), reverse=True)
                keep = self.policy.max_files - (1 if active.exists() else 0)
                for path in archives[keep:]:
                    self._unlink(path)

    def write(self, channel, message):
        if channel not in self.CHANNELS:
            raise ValueError("未知日志类型")
        with self.lock:
            now = self.clock()
            self.cleanup()
            path = self.directory / f"{channel}.log"
            if path.is_symlink():
                raise ValueError("日志文件不能是符号链接")
            prefix = now.isoformat(timespec="seconds") + " "
            budget = min(4096, self.policy.max_bytes) - len((prefix + "\n").encode("utf-8"))
            safe = self.redact(message).encode("utf-8")[:max(0, budget)].decode("utf-8", "ignore")
            line = (prefix + safe + "\n").encode("utf-8")
            if path.exists() and path.stat().st_size:
                if (self._first_time(path).date() != now.date()
                        or path.stat().st_size + len(line) > self.policy.max_bytes):
                    archive = self.directory / f"{channel}.{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}.log"
                    path.replace(archive)
            with path.open("ab") as stream:
                stream.write(line)
            self.cleanup()

    def service_line(self, channel, line):
        """Only record a closed set of safe metadata, never arbitrary stdout.

        Tracebacks, prompts, user text and unknown SDK diagnostics are intentionally
        not copied into persistent logs, even if their JSON formatting changes.
        """
        line = self.redact(line)
        access = re.search(r'"(GET|POST|DELETE|PUT|PATCH|OPTIONS|HEAD) ([^\s?]+)(?:\?[^\s]*)? HTTP/[\d.]+" (\d{3})', line)
        if access:
            method, path, status = access.groups()
            # Normalize IDs so access logs do not become a record of session IDs.
            if path != "/documents/upload":
                path = re.sub(r"/(sessions|documents)/[^/]+", r"/\1/:id", path)
            if path not in {"/health", "/ready", "/chat", "/chat/stream", "/documents",
                            "/documents/upload", "/documents/:id", "/sessions/:id",
                            "/docs", "/openapi.json", "/_stcore/health"}:
                path = "/:other"
            self.write(channel, f"http method={method} path={path} status={status}")
            return
        event = re.search(r"\b(chat_started|chat_finished|session_acquire|session_commit|session_commit_retry|rate_check|rate_limit_hit|session_lease_lost|model_invoke_failed|model_stream_failed|stream_failed|API 请求失败)\b", line)
        if event:
            fields = re.findall(r"\b(duration_ms|attempts|attempt|allowed|acquired|history_status|mode|error_type)=([A-Za-z0-9_.-]+)", line)
            self.write(channel, event.group(1) + " " + " ".join(f"{k}={v}" for k, v in fields))
            return
        if re.search(r"\b(ERROR|CRITICAL)\b", line):
            self.write(channel, "service_diagnostic severity=error details_omitted=true")
