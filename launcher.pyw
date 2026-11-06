"""Double-click through 启动项目.vbs or the generated desktop shortcut."""

from pathlib import Path

if __name__ == "__main__":
    try:
        from desktop_launcher.gui import main
        main(Path(__file__).resolve().parent)
    except Exception as error:
        # pythonw has no console: even missing dependencies need a visible error.
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, "启动器无法加载。请先运行项目中的安装依赖.bat，再重试。\n"
            + f"错误类别：{type(error).__name__}", "Agentic RAG 启动器", 0x10,
        )
