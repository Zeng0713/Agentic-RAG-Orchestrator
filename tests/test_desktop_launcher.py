from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_launcher.logs import LogManager, LogPolicy
from desktop_launcher.runtime import Cancelled, LaunchConfig, LaunchError, OwnedProcess, RedisUnavailable, Runtime
import desktop_launcher.runtime as runtime_module


def config(tmp_path, **env):
    return LaunchConfig(tmp_path, {"DEEPSEEK_API_KEY": "offline-launcher-test",
                                  "SILICONFLOW_API_KEY": "offline-embedding-test", **env})


def test_log_size_file_count_and_secret_redaction(tmp_path):
    logs = LogManager(tmp_path / "logs", LogPolicy(max_bytes=256, max_files=6), secrets=["private-api-value"])
    for i in range(60):
        logs.write("launcher", f"line={i} private-api-value redis://user:password@host Bearer secret " + "x" * 90)
    files = list(logs.directory.glob("*.log"))
    assert len(files) == 6
    assert all(file.stat().st_size <= 256 for file in files)
    text = "".join(file.read_text(encoding="utf-8") for file in files)
    for secret in ("private-api-value", "password", "Bearer secret"):
        assert secret not in text
    assert "[REDACTED]" in text


def test_logs_expire_after_seven_days_and_rotate_on_new_day(tmp_path):
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    logs = LogManager(tmp_path / "logs", clock=lambda: now[0])
    logs.write("backend", "old event")
    unknown = logs.directory / "notes.log"
    unknown.write_text("do not delete", encoding="utf-8")
    now[0] += timedelta(days=1)
    logs.write("backend", "new day")
    assert len(list(logs.directory.glob("backend*.log"))) == 2
    now[0] += timedelta(days=8)
    logs.cleanup()
    assert not list(logs.directory.glob("backend*.log"))
    assert unknown.read_text() == "do not delete"


def test_service_logs_do_not_copy_prompts_tracebacks_or_query_strings(tmp_path):
    logs = LogManager(tmp_path / "logs")
    logs.service_line("backend", 'INFO: user - "POST /chat?secret=never HTTP/1.1" 200 OK')
    logs.service_line("backend", 'INFO {"question":"private conversation", "answer":"private answer"}')
    logs.service_line("backend", "ValueError: private conversation")
    logs.service_line("backend", "chat_finished request_id=private history_status=saved")
    text = (logs.directory / "backend.log").read_text()
    assert "path=/chat status=200" in text
    assert "history_status=saved" in text
    assert "never" not in text and "private" not in text


def test_unknown_access_paths_are_not_persisted(tmp_path):
    logs = LogManager(tmp_path / "logs")
    logs.service_line("backend", '"GET /private-question-content HTTP/1.1" 404 Not Found')
    logs.service_line("backend", '"POST /documents/upload HTTP/1.1" 200 OK')
    text = (logs.directory / "backend.log").read_text()
    assert "path=/:other status=404" in text
    assert "private-question-content" not in text
    assert "path=/documents/upload status=200" in text


@pytest.mark.parametrize("channel", ["backend", "frontend"])
def test_model_and_redis_credentials_only_pass_to_backend(tmp_path, monkeypatch, channel):
    runtime = Runtime(config(tmp_path, REDIS_URL="redis://user:private@host/0"))
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs["env"])
        raise RuntimeError("captured without starting a process")

    monkeypatch.setattr(runtime_module.subprocess, "Popen", capture)
    with pytest.raises(RuntimeError, match="captured"):
        runtime._spawn(channel, [])
    assert captured["RAG_API_URL"] == runtime.config.api_url
    for key in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "REDIS_URL"):
        assert (key in captured) is (channel == "backend")


def test_default_log_policy_matches_requested_limits(tmp_path):
    policy = config(tmp_path).log_policy()
    assert policy.max_bytes == 20 * 1024 * 1024
    assert policy.max_files == 6 and policy.days == 7


def test_config_errors_never_echo_secret_values(tmp_path):
    cfg = config(tmp_path, DEEPSEEK_API_KEY="your_key")
    with pytest.raises(LaunchError, match="DEEPSEEK_API_KEY"):
        cfg.validate()
    with pytest.raises(LaunchError, match="日志配置") as exc:
        config(tmp_path, LOG_MAX_FILES="private-config-value").log_policy()
    assert "private-config-value" not in str(exc.value)


def test_memory_mode_never_checks_docker_or_redis(tmp_path, monkeypatch):
    runtime = Runtime(config(tmp_path))

    def unexpected(*_args, **_kwargs):
        pytest.fail("Memory mode must not depend on Docker or Redis")

    monkeypatch.setattr(runtime, "_run_docker", unexpected)
    monkeypatch.setattr(runtime_module, "redis_ready", unexpected)
    runtime.prepare_redis()


def test_existing_external_redis_needs_no_docker(tmp_path, monkeypatch):
    runtime = Runtime(config(tmp_path, SESSION_BACKEND="redis", REDIS_URL="redis://external:6379/0"))
    monkeypatch.setattr(runtime_module, "redis_ready", lambda _url: True)
    monkeypatch.setattr(runtime, "_run_docker", lambda *_a, **_k: pytest.fail("Do not start Docker"))
    runtime.prepare_redis()


@pytest.mark.parametrize("approve", [False, True])
def test_redis_fallback_requires_explicit_choice_and_does_not_modify_env(tmp_path, monkeypatch, approve):
    env_file = tmp_path / ".env"
    env_file.write_text("SESSION_BACKEND=redis\n", encoding="utf-8")
    runtime = Runtime(config(tmp_path, SESSION_BACKEND="redis"))
    monkeypatch.setattr(runtime, "_existing", lambda *_args: False)

    def unavailable():
        raise RedisUnavailable("Redis not ready")

    monkeypatch.setattr(runtime, "prepare_redis", unavailable)
    spawned = []
    monkeypatch.setattr(runtime, "_spawn", lambda channel, args: spawned.append(channel))
    monkeypatch.setattr(runtime, "_wait", lambda *_args: None)
    if approve:
        runtime.start(lambda: True)
        assert runtime.config.env["SESSION_BACKEND"] == "memory"
        assert runtime.config.env["RATE_LIMIT_BACKEND"] == "memory"
        assert spawned == ["backend", "frontend"]
    else:
        with pytest.raises(RedisUnavailable):
            runtime.start(lambda: False)
        assert spawned == []
    assert env_file.read_text() == "SESSION_BACKEND=redis\n"


def test_foreign_port_is_not_killed_or_reused(tmp_path, monkeypatch):
    runtime = Runtime(config(tmp_path))
    monkeypatch.setattr(runtime_module, "listener_pid", lambda _port: 1234)
    monkeypatch.setattr(runtime_module, "is_project_service", lambda *_args: False)
    with pytest.raises(LaunchError, match="其他程序"):
        runtime._existing("backend", 8000)
    assert runtime.owned == []


def test_healthy_existing_services_are_reused_and_never_stopped(tmp_path, monkeypatch):
    runtime = Runtime(config(tmp_path))
    monkeypatch.setattr(runtime_module, "listener_pid", lambda _port: 1234)
    monkeypatch.setattr(runtime_module, "is_project_service", lambda *_args: True)
    monkeypatch.setattr(runtime_module.psutil, "Process", lambda _pid: SimpleNamespace(environ=lambda: {}))
    monkeypatch.setattr(runtime, "health", lambda _channel: True)
    monkeypatch.setattr(runtime, "_spawn", lambda *_args: pytest.fail("Must reuse services"))
    runtime.start()
    assert runtime.reused == ["backend", "frontend"]
    assert runtime.owned == []
    runtime.stop()


def test_startup_failure_rolls_back_only_owned_services(tmp_path, monkeypatch):
    runtime = Runtime(config(tmp_path))
    stopped = []
    monkeypatch.setattr(runtime, "_existing", lambda *_args: False)

    def spawn(channel, _args):
        record = SimpleNamespace(channel=channel, stop=lambda: stopped.append(channel))
        runtime.owned.append(record)
        return record

    def wait(channel, _record):
        if channel == "frontend":
            raise LaunchError("frontend failed")

    monkeypatch.setattr(runtime, "_spawn", spawn)
    monkeypatch.setattr(runtime, "_wait", wait)
    with pytest.raises(LaunchError, match="frontend failed"):
        runtime.start()
    assert stopped == ["frontend", "backend"]
    assert not runtime.owned


def test_cancel_before_start_never_spawns(tmp_path, monkeypatch):
    runtime = Runtime(config(tmp_path))
    runtime.cancel.set()
    monkeypatch.setattr(runtime, "_spawn", lambda *_args: pytest.fail("Must not spawn"))
    with pytest.raises(Cancelled):
        runtime.start()


def test_reused_pid_is_never_terminated(monkeypatch):
    calls = []
    process = SimpleNamespace(create_time=lambda: 200, terminate=lambda: calls.append("terminated"))
    monkeypatch.setattr(runtime_module.psutil, "Process", lambda _pid: process)
    OwnedProcess(SimpleNamespace(pid=123), 100, "backend").stop()
    assert not calls
