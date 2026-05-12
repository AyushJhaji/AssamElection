"""
server/launch.py  ─  V12 Server Launch Manager
================================================
Handles:
  • Finding a free port starting from the requested one
  • Starting Flask in a background daemon thread
  • Waiting until the port actually accepts connections
  • Opening the default browser once, with a short delay
  • Clean shutdown on Ctrl-C / SIGTERM

Usage from main.py:
    from server.launch import ServerManager
    mgr = ServerManager(port=8000, outputs_dir=OUT_DIR, web_dir=ROOT/"web")
    mgr.start()
    mgr.open_browser()
    mgr.print_banner()
    mgr.wait()          # blocks until Ctrl-C
"""

import logging
import signal
import socket
import threading
import time
import webbrowser
from pathlib import Path

log = logging.getLogger("launcher")

_ROOT = Path(__file__).resolve().parent.parent


# ── Port utilities ────────────────────────────────────────────────────────────

def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.15)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def _find_free_port(start: int = 8000, end: int = 8020,
                    host: str = "127.0.0.1") -> int:
    """Scan start..end and return first free port. Raises RuntimeError if none."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}–{end - 1}")


def _wait_ready(host: str, port: int,
                timeout: float = 12.0, interval: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_in_use(host, port):
            return True
        time.sleep(interval)
    return False


# ── Manager ───────────────────────────────────────────────────────────────────

class ServerManager:
    """
    Manages the Flask dev server running in a background daemon thread.

    Parameters
    ----------
    port         : preferred port (falls back to next free port if occupied)
    host         : bind address (default 127.0.0.1 — local only)
    no_ui        : if True, skip start / open_browser / wait entirely
    outputs_dir  : path to outputs/ directory served by Flask
    web_dir      : path to web/ directory containing index.html
    """

    def __init__(self,
                 port:        int  = 8000,
                 host:        str  = "127.0.0.1",
                 no_ui:       bool = False,
                 outputs_dir: Path | None = None,
                 web_dir:     Path | None = None):

        self.host   = host
        self.no_ui  = no_ui
        self._app   = None
        self._thread: threading.Thread | None = None

        self._outputs = Path(outputs_dir) if outputs_dir else _ROOT / "outputs"
        self._web     = Path(web_dir)     if web_dir     else _ROOT / "web"

        # Resolve port
        if _port_in_use(host, port):
            log.warning(f"  Port {port} already in use — scanning for next free port …")
            try:
                self.port = _find_free_port(port + 1, port + 20, host)
                log.info(f"  Found free port: {self.port}")
            except RuntimeError:
                log.error("  No free port available. Stop any existing server and retry.")
                sys.exit(1)
        else:
            self.port = port

        self.url = f"http://{host}:{self.port}"

    # ── start ─────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Import Flask app and spin it up in a daemon thread."""
        if self.no_ui:
            return

        # Suppress Flask/werkzeug noise
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        import sys as _sys
        _sys.path.insert(0, str(_ROOT))
        from server.app import create_app
        self._app = create_app(outputs_dir=self._outputs, web_dir=self._web)

        def _run():
            self._app.run(
                host=self.host,
                port=self.port,
                debug=False,        # must be False for threading
                use_reloader=False, # reloader forks → breaks daemon thread
            )

        self._thread = threading.Thread(target=_run, daemon=True, name="flask")
        self._thread.start()

        if _wait_ready(self.host, self.port, timeout=12):
            log.info(f"  Server ready → {self.url}")
        else:
            log.error("  Flask did not start within 12 s")
            sys.exit(1)

    # ── open_browser ──────────────────────────────────────────────────────────

    def open_browser(self, delay: float = 0.4) -> None:
        """Open the default browser after `delay` seconds (non-blocking)."""
        if self.no_ui:
            return

        def _open():
            time.sleep(delay)
            try:
                webbrowser.open(self.url)
                log.info(f"  Browser opened → {self.url}")
            except Exception as exc:
                log.warning(f"  Could not open browser automatically: {exc}")
                log.info(f"  Open manually: {self.url}")

        threading.Thread(target=_open, daemon=True).start()

    # ── print_banner ──────────────────────────────────────────────────────────

    def print_banner(self, status: str = "✅ PIPELINE COMPLETE") -> None:
        w = 65
        print()
        print("═" * w)
        print("  🗳  ASSAM 2026 ELECTION FORECAST — V12")
        print("═" * w)
        print(f"  Pipeline : {status}")
        print(f"  Dashboard: {self.url}")
        print(f"  Data API : {self.url}/api/summary")
        print()
        print("  The dashboard auto-refreshes when you re-run main.py")
        print("  Press Ctrl-C to stop the server")
        print("═" * w)
        print()

    # ── wait ──────────────────────────────────────────────────────────────────

    def wait(self) -> None:
        """Block main thread until Ctrl-C or SIGTERM."""
        if self.no_ui:
            return

        def _shutdown(sig, frame):
            print("\n\n  Server stopped. Goodbye.\n")
            sys.exit(0)

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while True:          # Flask thread is daemon — exits when main exits
            time.sleep(1)
