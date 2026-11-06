"""Small native Windows controller. Slow startup work never blocks Tk's loop."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import psutil

from desktop_launcher.runtime import Cancelled, LaunchConfig, LaunchError, Runtime, focus_process


class SingleInstance:
    def __init__(self, root):
        self.root = Path(root)
        self.handle = None
        self.record = self.root / ".runtime" / "launcher.json"

    def acquire(self):
        if os.name != "nt":
            return True
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel.CreateMutexW.restype = ctypes.c_void_p
        name = "Local\\AgenticRAG-" + hashlib.sha256(str(self.root.resolve()).casefold().encode()).hexdigest()[:24]
        ctypes.set_last_error(0)
        self.handle = kernel.CreateMutexW(None, False, name)
        if not self.handle:
            raise LaunchError("无法创建启动器实例锁")
        if ctypes.get_last_error() == 183:
            self._focus_existing()
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle(self.handle)
            self.handle = None
            return False
        if self.record.parent.is_symlink():
            raise LaunchError("启动器运行目录不能是符号链接")
        self.record.parent.mkdir(exist_ok=True)
        self.record.write_text(json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()}), encoding="utf-8")
        return True

    def _focus_existing(self):
        try:
            record = json.loads(self.record.read_text(encoding="utf-8"))
            process = psutil.Process(record["pid"])
            if abs(process.create_time() - record["created"]) < 0.01:
                focus_process(record["pid"])
        except (OSError, ValueError, KeyError, psutil.Error):
            pass

    def close(self):
        if self.handle and os.name == "nt":
            try:
                record = json.loads(self.record.read_text(encoding="utf-8"))
                if record.get("pid") == os.getpid():
                    self.record.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
            kernel = ctypes.windll.kernel32
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle(self.handle)
            self.handle = None


class LauncherWindow:
    STEPS = ("检查运行环境", "准备状态服务", "启动后端 API", "启动网页界面", "打开知识工作台")

    def __init__(self, root, project):
        self.window, self.project = root, Path(project)
        self.events = queue.Queue()
        self.runtime = None
        self.working = False
        self.closing = False
        self.pending_confirm = None
        self.stop_requested = threading.Event()
        root.title("Agentic RAG · 本地启动器")
        root.geometry("640x510")
        root.minsize(600, 490)
        root.configure(bg="#111617")
        try:
            root.iconbitmap(str(self.project / "assets" / "app.ico"))
        except tk.TclError:
            pass
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("RAG.Horizontal.TProgressbar", troughcolor="#263032", background="#c6d6a0",
                        bordercolor="#263032", lightcolor="#c6d6a0", darkcolor="#c6d6a0")
        shell = tk.Frame(root, bg="#111617", padx=32, pady=28)
        shell.pack(fill="both", expand=True)
        tk.Label(shell, text="Agentic RAG Orchestrator", bg="#111617", fg="#edf0e8",
                 font=("Segoe UI", 20)).pack(anchor="w")
        tk.Label(shell, text="本地知识工作台 · 启动与服务管理", bg="#111617", fg="#a8b4b1",
                 font=("Microsoft YaHei UI", 10)).pack(anchor="w", pady=(8, 24))
        self.status = tk.StringVar(value="准备启动…")
        tk.Label(shell, textvariable=self.status, bg="#111617", fg="#c6d6a0", anchor="w",
                 justify="left", wraplength=565, font=("Microsoft YaHei UI", 11)).pack(fill="x")
        self.progress = ttk.Progressbar(shell, maximum=5, style="RAG.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(16, 16))
        self.steps = []
        for text in self.STEPS:
            label = tk.Label(shell, text="○  " + text, bg="#111617", fg="#a8b4b1", font=("Microsoft YaHei UI", 10))
            label.pack(anchor="w", pady=3)
            self.steps.append(label)
        self.detail = tk.StringVar(value="首次启动需已安装依赖，并在 .env 配置服务密钥。")
        tk.Label(shell, textvariable=self.detail, bg="#111617", fg="#a8b4b1", wraplength=565,
                 justify="left", font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=(20, 18))
        actions = tk.Frame(shell, bg="#111617")
        actions.pack(fill="x")
        self.open_button = self.button(actions, "打开工作台", self.open_ui, primary=True)
        self.logs_button = self.button(actions, "查看日志", self.open_logs)
        self.stop_button = self.button(actions, "停止服务", self.stop)
        self.start_button = self.button(actions, "重新启动", self.start)
        self.open_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.drain)
        root.after(250, self.start)
        root.after(60000, self.clean_logs)

    @staticmethod
    def button(parent, text, command, primary=False):
        button = tk.Button(parent, text=text, command=command, relief="flat", borderwidth=0,
                           bg="#c6d6a0" if primary else "#263032", fg="#111617" if primary else "#edf0e8",
                           activebackground="#d4dfb8" if primary else "#354144", disabledforeground="#75807a",
                           padx=12, pady=10, cursor="hand2", font=("Microsoft YaHei UI", 9))
        button.pack(side="left", padx=(0, 8))
        return button

    def start(self):
        if self.working or self.closing:
            return
        self.working = True
        self.stop_requested.clear()
        self.start_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress["value"] = 0
        for label, text in zip(self.steps, self.STEPS):
            label.configure(text="○  " + text, fg="#a8b4b1")
        threading.Thread(target=self.start_worker, daemon=True).start()

    def start_worker(self):
        try:
            config = LaunchConfig.load(self.project)
            self.runtime = Runtime(config, lambda step, text: self.events.put(("progress", step, text)))
            if self.stop_requested.is_set():
                raise Cancelled()
            self.runtime.start(self.confirm_memory)
            self.runtime.open_ui()
            self.events.put(("ready",))
        except Cancelled:
            if self.runtime:
                self.runtime.stop()
            self.events.put(("stopped",))
        except Exception as exc:
            if self.runtime:
                self.runtime.stop()
                try:
                    self.runtime.logs.write("launcher", f"startup_failed error_type={type(exc).__name__}")
                except OSError:
                    pass
            message = str(exc) if isinstance(exc, LaunchError) else "启动失败，请查看日志并确认依赖已安装。"
            self.events.put(("error", message))

    def confirm_memory(self):
        answer, signal = [], threading.Event()
        self.events.put(("confirm", answer, signal))
        while not signal.wait(0.1):
            if self.closing or (self.runtime and self.runtime.cancel.is_set()):
                return False
        return bool(answer and answer[0])

    def open_ui(self):
        if self.runtime and not self.working and not self.closing:
            try:
                self.runtime.open_ui()
            except LaunchError as exc:
                messagebox.showerror("打开工作台", str(exc), parent=self.window)

    def open_logs(self):
        directory = self.project / "logs"
        if not directory.is_dir():
            messagebox.showinfo("日志", "尚未生成日志，请先启动服务。", parent=self.window)
        elif os.name == "nt":
            os.startfile(directory)

    def stop(self):
        self.stop_requested.set()
        if self.runtime:
            self.runtime.cancel.set()
        if self.pending_confirm:
            self.pending_confirm.set()
        self.status.set("正在停止本次创建的服务…")
        self.stop_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        # Startup's worker performs its own rollback before enabling restart.
        if self.working:
            return
        self.working = True

        def worker():
            if self.runtime:
                self.runtime.stop()
            self.events.put(("stopped",))

        threading.Thread(target=worker, daemon=True).start()

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.stop()

    def clean_logs(self):
        if self.runtime:
            try:
                self.runtime.logs.cleanup()
            except OSError:
                pass
        if not self.closing:
            self.window.after(60000, self.clean_logs)

    def drain(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    step, text = event[1:]
                    self.status.set(text)
                    self.progress["value"] = step
                    for i, (label, title) in enumerate(zip(self.steps, self.STEPS)):
                        label.configure(text=("✓  " if i < step else "●  " if i == step else "○  ") + title,
                                        fg="#c6d6a0" if i <= step else "#a8b4b1")
                elif kind == "confirm":
                    answer, signal = event[1:]
                    self.pending_confirm = signal
                    choice = False if self.closing else messagebox.askyesno(
                        "Redis 未就绪", "无法连接配置的 Redis。是否仅本次改用 Memory？\n\n"
                        "不会修改 .env；原 Redis 历史不会迁移；Memory 历史在后端退出后清空。", parent=self.window)
                    answer.append(choice)
                    signal.set()
                    self.pending_confirm = None
                elif kind == "ready":
                    self.working = False
                    if self.closing or self.stop_requested.is_set():
                        self.stop()
                        continue
                    self.progress["value"] = 5
                    for label, title in zip(self.steps, self.STEPS):
                        label.configure(text="✓  " + title, fg="#c6d6a0")
                    self.open_button.configure(state="normal")
                    self.stop_button.configure(state="normal")
                    reused = "；复用的已有服务不会被本启动器停止或接管原日志" if self.runtime.reused else ""
                    self.detail.set("服务在后台运行。关闭启动器将退出本次创建的服务，Redis 和数据保留" + reused)
                elif kind in {"stopped", "error"}:
                    self.working = False
                    if self.closing:
                        self.window.destroy()
                        return
                    self.status.set("本次服务已停止，原有服务未受影响" if kind == "stopped" else event[1])
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.open_button.configure(state="disabled")
                    if kind == "error":
                        self.detail.set("修复配置后点“重新启动”；首次安装请双击安装依赖.bat。")
        except queue.Empty:
            pass
        self.window.after(100, self.drain)


def main(project):
    instance = SingleInstance(project)
    if not instance.acquire():
        return
    controller = None
    try:
        window = tk.Tk()
        controller = LauncherWindow(window, project)
        window.mainloop()
    finally:
        if controller and controller.runtime:
            controller.runtime.stop()
        instance.close()
