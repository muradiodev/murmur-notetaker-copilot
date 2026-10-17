"""Murmur desktop app: runs the local server and shows it in its own native window.

This is the entry point for the packaged Windows build (PyInstaller). It starts the same FastAPI
server the browser version uses, bound to 127.0.0.1, then opens a pywebview window pointing at it.
Nothing leaves the machine except the AI summary/answer calls the user configured.
"""

import logging
import sys
import threading
import time
import traceback
import urllib.request

# Must run before anything imports ctranslate2 (CUDA libs, when present).
from app.config import DATA_DIR, FROZEN, HOST, PORT, add_nvidia_dll_dirs

add_nvidia_dll_dirs()

URL = f"http://{HOST}:{PORT}"


def _log_path():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return DATA_DIR / "murmur.log"
    except Exception:
        return None


def _setup_logging() -> None:
    path = _log_path()
    handlers = [logging.StreamHandler()] if not FROZEN else []
    if path:
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


log = logging.getLogger("murmur.desktop")


def _server() -> None:
    try:
        import uvicorn

        from app.server import app

        log.info("Starting server on %s", URL)
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning", use_colors=False)
    except Exception:
        log.error("Server failed to start:\n%s", traceback.format_exc())


def _wait_until_up(timeout: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/api/status", timeout=2):
                return True
        except Exception:
            time.sleep(0.3)
    return False


def _already_running() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/status", timeout=1):
            return True
    except Exception:
        return False


def main() -> None:
    _setup_logging()
    import webview

    # If a copy is already serving (e.g. the user started the browser version), just show it.
    if not _already_running():
        threading.Thread(target=_server, daemon=True, name="murmur-server").start()
        _wait_until_up()

    window = webview.create_window(
        "Murmur", URL, width=1280, height=820, min_size=(900, 600), confirm_close=True,
    )

    def _on_closing():
        # The server thread is a daemon; the process exits with the window.
        return True

    window.events.closing += _on_closing
    # GUI backend: EdgeChromium (WebView2) on Windows. gui=None lets pywebview pick what is installed.
    webview.start()


if __name__ == "__main__":
    # In the packaged build, double-clicking should not pop a console; PyInstaller is built with
    # --noconsole, so stdout/stderr are None. uvicorn's logging expects real streams (it calls
    # isatty()), so point them at the null device — a genuine file object — instead of a stub.
    if FROZEN:
        import os

        if sys.stdout is None or sys.stderr is None:
            devnull = open(os.devnull, "w")
            if sys.stdout is None:
                sys.stdout = devnull
            if sys.stderr is None:
                sys.stderr = devnull
    main()
