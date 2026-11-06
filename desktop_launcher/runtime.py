"""Process ownership and startup orchestration, independent of the desktop UI."""

from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
import webbrowser

from dotenv import dotenv_values
import psutil
import requests

from desktop_launcher.logs import LogManager, LogPolicy


class LaunchError(Exception):
    """Only intentionally user-safe messages may be shown by the GUI."""


class RedisUnavailable(LaunchError):
    pass


class Cancelled(LaunchError):
    def __init__(self):
        super().__init__("启动已取消")


@dataclass
class LaunchConfig:
    root: Path
    env: dict = field(default_factory=dict, repr=False)
    api_port: int = 8000
    ui_port: int = 8501

    @classmethod
    def load(cls, root):
        root = Path(root).resolve()
        values = {k: v for k, v in dotenv_values(root / ".env").items() if v is not None}
        values.update(os.environ)  # Same precedence as the application.
        try:
            return cls(root, values, int(values.get("LAUNCHER_API_PORT", 8000)),
                       int(values.get("LAUNCHER_UI_PORT", 8501)))
        except ValueError:
            raise LaunchError("启动器端口配置必须是整数") from None

    def validate(self):
        if sys.version_info[:2] != (3, 12):
            raise LaunchError("请使用 Python 3.12 创建项目虚拟环境，再运行安装依赖.bat")
        if not all(1024 <= p <= 65535 for p in (self.api_port, self.ui_port)) or self.api_port == self.ui_port:
            raise LaunchError("启动器端口需为不同的 1024–65535 范围整数")
        missing = [k for k in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY")
                   if not self.env.get(k) or self.env[k].startswith("your_")]
        if missing:
            raise LaunchError("请先在项目 .env 中填写：" + "、".join(missing))
        if any(self.env.get(k, "memory") not in {"memory", "redis"}
               for k in ("SESSION_BACKEND", "RATE_LIMIT_BACKEND")):
            raise LaunchError("SESSION_BACKEND 和 RATE_LIMIT_BACKEND 只能为 memory 或 redis")

    @property
    def python(self):
        for folder in (".venv", "venv"):
            candidate = self.root / folder / "Scripts" / "python.exe"
            if candidate.is_file():
                return str(candidate)
        # Useful for CI and Linux development; Windows bootstrap requires a venv.
        return sys.executable

    @property
    def uses_redis(self):
        return any(self.env.get(k, "memory") == "redis" for k in ("SESSION_BACKEND", "RATE_LIMIT_BACKEND"))

    @property
    def api_url(self):
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def ui_url(self):
        return f"http://127.0.0.1:{self.ui_port}"

    def log_policy(self):
        try:
            return LogPolicy(int(self.env.get("LOG_MAX_SIZE_MB", 20)) * 1024 * 1024,
                             int(self.env.get("LOG_MAX_FILES", 6)),
                             int(self.env.get("LOG_RETENTION_DAYS", 7)))
        except ValueError:
            raise LaunchError("日志配置无效，请检查大小、文件数和保留天数") from None

    def secrets(self):
        values = [self.env.get(k, "") for k in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "REDIS_URL")]
        try:
            password = urlsplit(self.env.get("REDIS_URL", "")).password
            if password:
                values.append(unquote(password))
        except ValueError:
            pass
        return values


@dataclass
class OwnedProcess:
    process: subprocess.Popen
    created: float
    channel: str

    def stop(self):
        """Do not act on a PID if the original process has exited and been reused."""
        try:
            parent = psutil.Process(self.process.pid)
            if abs(parent.create_time() - self.created) > 0.01:
                return
            family = parent.children(recursive=True) + [parent]
            identities = [(p, p.create_time()) for p in family]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        for process, created in reversed(identities):
            try:
                if abs(process.create_time() - created) < 0.01:
                    process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _gone, alive = psutil.wait_procs(family, timeout=3)
        for process in alive:
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def listener_pid(port):
    try:
        listeners = {c.pid for c in psutil.net_connections(kind="tcp")
                     if c.status == psutil.CONN_LISTEN and c.laddr.port == port and c.pid}
    except psutil.AccessDenied:
        raise LaunchError("无法确认端口进程归属，未结束任何进程。请检查系统权限。") from None
    if len(listeners) > 1:
        raise LaunchError(f"端口 {port} 有多个监听进程，请先检查占用")
    return next(iter(listeners), None)


def is_project_service(pid, root, channel):
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        cwd = Path(process.cwd()).resolve()
        expected = ("uvicorn", "api.main:app") if channel == "backend" else ("streamlit", "app.py")
        return cwd == Path(root).resolve() and all(token in command for token in expected)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def redis_ready(url):
    if not url:
        return False
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry
    client = None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname or parsed.query or parsed.fragment:
            return False
        client = redis.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=1,
                                      retry=Retry(NoBackoff(), 0))
        return bool(client.ping())
    except (redis.RedisError, ValueError):
        return False
    finally:
        if client:
            client.close()


class Runtime:
    def __init__(self, config, status=None, logs=None):
        self.config = config
        self.status = status or (lambda step, text: None)
        self.logs = logs or LogManager(config.root / "logs", config.log_policy(), config.secrets())
        self.cancel = threading.Event()
        self.owned = []
        self.reused = []
        self.browser = None
        self.pumps = []
        self.http = requests.Session()
        self.http.trust_env = False  # Loopback probes must not go through a VPN/proxy.
        self.lock = threading.RLock()

    def check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled()

    def report(self, step, text):
        self.logs.write("launcher", text)
        self.status(step, text)

    def health(self, channel):
        path = self.config.api_url + "/ready" if channel == "backend" else self.config.ui_url + "/_stcore/health"
        try:
            response = self.http.get(path, timeout=1)
            if channel == "backend":
                return response.ok and response.json().get("data", {}).get("status") == "ready"
            return response.ok and response.text.strip() == "ok"
        except (requests.RequestException, ValueError):
            return False

    def _existing(self, channel, port):
        pid = listener_pid(port)
        if pid is None:
            return False
        if not is_project_service(pid, self.config.root, channel):
            raise LaunchError(f"端口 {port} 被其他程序占用；请更换启动器端口或手动处理，未结束该程序。")
        if not self.health(channel):
            raise LaunchError(f"端口 {port} 上的项目服务未就绪。请先检查已有服务及 Redis 状态。")
        if channel == "frontend":
            try:
                env = psutil.Process(pid).environ()
                api = env.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                raise LaunchError("无法确认已有前端连接的后端地址，未接管该服务。") from None
            if api not in {self.config.api_url, self.config.api_url.replace("127.0.0.1", "localhost")}:
                raise LaunchError("已有前端连接了其他 API 地址，请先检查或关闭该前端。")
        self.reused.append(channel)
        self.logs.write(channel, "service_reused ownership=external log_capture=unchanged")
        return True

    def _run_docker(self, args, timeout=5):
        self.check_cancel()
        process = None
        try:
            process = subprocess.Popen(["docker", *args], cwd=self.config.root,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            until = time.monotonic() + timeout
            while process.poll() is None:
                self.check_cancel()
                if time.monotonic() >= until:
                    return False
                self.cancel.wait(0.2)
            return process.returncode == 0
        except OSError:
            return False
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

    def prepare_redis(self):
        if not self.config.uses_redis:
            self.report(1, "Memory 模式：无需 Docker 或 Redis")
            return
        url = self.config.env.get("REDIS_URL", "")
        if redis_ready(url):
            self.report(1, "已有 Redis 连接正常")
            return
        try:
            parsed = urlsplit(url)
            local_default = (parsed.scheme == "redis" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                             and (parsed.port or 6379) == 6379 and not parsed.username and not parsed.password
                             and parsed.path in {"", "/0"} and not parsed.query and not parsed.fragment)
        except ValueError:
            local_default = False
        if not local_default or not shutil.which("docker"):
            raise RedisUnavailable("未连接到配置的 Redis，且无法自动启动匹配的本地服务。")
        self.report(1, "正在检查 Docker Desktop…")
        if not self._run_docker(["info", "--format", "{{.ServerVersion}}"]):
            candidates = [Path(os.getenv("LOCALAPPDATA", "")) / "Programs/DockerDesktop/Docker Desktop.exe",
                          Path(os.getenv("ProgramFiles", "")) / "Docker/Docker/Docker Desktop.exe"]
            desktop = next((p for p in candidates if p.is_file()), None)
            if desktop is None:
                raise RedisUnavailable("Docker 服务未就绪，未找到 Docker Desktop。")
            subprocess.Popen([str(desktop)], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            until = time.monotonic() + 60
            while not self._run_docker(["info", "--format", "{{.ServerVersion}}"]):
                if time.monotonic() >= until:
                    raise RedisUnavailable("等待 Docker Desktop 就绪超时。")
                if self.cancel.wait(2):
                    raise Cancelled()
        self.report(1, "正在准备项目 Redis（首次可能需要下载镜像）…")
        if not self._run_docker(["compose", "-f", "compose.redis.yml", "up", "-d"], timeout=120):
            raise RedisUnavailable("Redis 容器启动失败，请检查 Docker 或选择 Memory。")
        until = time.monotonic() + 15
        while not redis_ready(url):
            self.check_cancel()
            if time.monotonic() >= until:
                raise RedisUnavailable("Redis 容器已请求启动，但连接检查未通过。")
            self.cancel.wait(0.5)
        self.report(1, "Redis 已就绪")

    def _pump(self, process, channel):
        try:
            for line in iter(process.stdout.readline, b""):
                self.logs.service_line(channel, line.decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            self.logs.write(channel, "log_capture_closed")
        finally:
            if process.stdout:
                process.stdout.close()

    def _spawn(self, channel, args):
        self.check_cancel()
        env = {**self.config.env, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "RAG_API_URL": self.config.api_url}
        if channel == "frontend":
            for key in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "REDIS_URL"):
                env.pop(key, None)
        process = subprocess.Popen([self.config.python, *args], cwd=self.config.root, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record = OwnedProcess(process, psutil.Process(process.pid).create_time(), channel)
        with self.lock:
            self.owned.append(record)
        self.logs.write(channel, f"service_started pid={process.pid}")
        pump = threading.Thread(target=self._pump, args=(process, channel), daemon=True)
        self.pumps.append(pump)
        pump.start()
        return record

    def _wait(self, channel, record, seconds=30):
        until = time.monotonic() + seconds
        while True:
            self.check_cancel()
            if record.process.poll() is not None:
                raise LaunchError(f"{'后端' if channel == 'backend' else '前端'}进程已退出，请检查依赖、配置和 logs 日志。")
            if self.health(channel):
                port = self.config.api_port if channel == "backend" else self.config.ui_port
                pid = listener_pid(port)
                try:
                    family = {record.process.pid, *(p.pid for p in psutil.Process(record.process.pid).children(recursive=True))}
                except psutil.NoSuchProcess:
                    family = set()
                if pid not in family:
                    raise LaunchError("启动期间端口归属发生变化，未接管其他程序。")
                return
            if time.monotonic() >= until:
                raise LaunchError(f"{'后端' if channel == 'backend' else '前端'}启动超时，请查看 logs 日志。")
            self.cancel.wait(0.25)

    def start(self, confirm_memory=None):
        self.check_cancel()
        self.reused = []
        self.report(0, "正在检查 Python、项目配置和端口…")
        self.config.validate()
        required = ("fastapi", "uvicorn", "streamlit", "requests", "redis", "psutil")
        if any(importlib.util.find_spec(name) is None for name in required):
            raise LaunchError("缺少项目依赖，请先运行安装依赖.bat")
        # Check both ports BEFORE creating any processes or containers.
        backend = self._existing("backend", self.config.api_port)
        frontend = self._existing("frontend", self.config.ui_port)
        try:
            if backend:
                self.report(1, "复用已有后端的状态配置，未更改其 Memory/Redis 模式")
            else:
                try:
                    self.prepare_redis()
                except RedisUnavailable:
                    if not confirm_memory or not confirm_memory():
                        raise
                    self.check_cancel()
                    self.config.env.update(SESSION_BACKEND="memory", RATE_LIMIT_BACKEND="memory")
                    self.report(1, "仅本次使用 Memory，未修改 .env；原 Redis 历史不迁移")
            if backend:
                self.report(2, "后端已运行，安全复用")
            else:
                self.report(2, "正在启动 FastAPI 后端…")
                record = self._spawn("backend", ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(self.config.api_port)])
                self._wait("backend", record)
            if frontend:
                self.report(3, "前端已运行，安全复用")
            else:
                self.report(3, "正在启动 Streamlit 前端…")
                record = self._spawn("frontend", ["-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1",
                                                    "--server.port", str(self.config.ui_port), "--server.headless", "true"])
                self._wait("frontend", record)
            self.check_cancel()
            self.report(4, "服务已就绪，可以打开工作台")
        except Exception:
            self.stop()
            raise

    def open_ui(self):
        self.check_cancel()
        if not self.health("frontend"):
            raise LaunchError("前端尚未就绪，请先启动服务")
        if self.browser and self.browser.process.poll() is None:
            if focus_process(self.browser.process.pid):
                return
            self.browser.stop()
            self.browser = None
        browser = find_app_browser()
        if browser:
            profile = self.config.root / ".runtime" / "browser-profile"
            if profile.parent.is_symlink() or profile.is_symlink():
                raise LaunchError("应用窗口运行目录不能是符号链接")
            profile.mkdir(parents=True, exist_ok=True)
            args = [str(browser), f"--app={self.config.ui_url}", f"--user-data-dir={profile}",
                    "--no-first-run", "--no-default-browser-check"]
            process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.browser = OwnedProcess(process, psutil.Process(process.pid).create_time(), "browser")
            self.report(4, "已打开独立工作台窗口")
        else:
            webbrowser.open(self.config.ui_url)
            self.report(4, "未找到应用模式浏览器，已使用默认浏览器打开工作台")

    def stop(self):
        self.cancel.set()
        with self.lock:
            owned, self.owned = self.owned, []
        if self.browser:
            self.browser.stop()
            self.browser = None
        for record in reversed(owned):
            record.stop()
            try:
                self.logs.write(record.channel, "service_stopped ownership=launcher")
            except OSError:
                pass
        for pump in self.pumps:
            pump.join(timeout=2)
        self.pumps = []
        try:
            self.logs.cleanup()
        except OSError:
            pass


def find_app_browser():
    candidates = []
    if os.name == "nt":
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for exe in ("msedge.exe", "chrome.exe"):
                try:
                    with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as key:
                        candidates.append(Path(winreg.QueryValue(key, None)))
                except OSError:
                    pass
    for base in (os.getenv("ProgramFiles", ""), os.getenv("ProgramFiles(x86)", ""), os.getenv("LOCALAPPDATA", "")):
        for relative in ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe"):
            candidates.append(Path(base) / relative)
    return next((path for path in candidates if path.is_file()), None)


def focus_process(pid):
    if os.name != "nt":
        return False
    import ctypes
    try:
        pids = {pid, *(p.pid for p in psutil.Process(pid).children(recursive=True))}
    except psutil.NoSuchProcess:
        return False
    user = ctypes.windll.user32
    user.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user.SetForegroundWindow.argtypes = [ctypes.c_void_p]
    user.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    found = []

    @callback_type
    def callback(hwnd, _data):
        owner = ctypes.c_ulong()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in pids and user.IsWindowVisible(hwnd):
            user.ShowWindow(hwnd, 9)
            user.SetForegroundWindow(hwnd)
            found.append(True)
            return False
        return True

    user.EnumWindows(callback, 0)
    return bool(found)
