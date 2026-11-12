"""The lynx show: a local web page with the 3D lynx, fed by this process.

- EventHub: the state of the show, pushed to the page as Server-Sent Events.
- An HTTP server (standard library only) on 127.0.0.1: the page (web/), the
  model (assets/), the events (/events) and the camera preview (/camera.mjpg).
- Browser: opens Chrome in app mode and opens it again if the page goes away.
- Show: the facade that live_sentiment.py and lynx_demo.py use.

Nothing here reads the camera or stores images. The camera loop hands in
frames, and only the latest frame is kept in memory for the preview.
"""
import argparse
import atexit
import json
import os
import queue
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import lynx_lines
from lynx_voice import Voice

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
ASSETS_DIR = ROOT / "assets"
MODEL_FILE = ASSETS_DIR / "luchs-emerald-liegend.glb"  # Not in git, `make setup` fetches it
DEFAULT_PORT = 8765
PING_INTERVAL = 5  # Seconds between pings on an idle event stream. The page reconnects without them
CLIENT_QUEUE_SIZE = 256  # Events a page may fall behind before it is dropped
PREVIEW_WIDTH = 640  # Pixels
PREVIEW_FPS = 15
PREVIEW_JPEG_QUALITY = 70
BROWSER_RESTART_SECONDS = 20  # No page for this long: open the browser again
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".glb": "model/gltf-binary",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
}
CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def encode_event(event):
    """One Server-Sent Event."""
    return b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"


class Client:
    """One connected page."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=CLIENT_QUEUE_SIZE)
        self.alive = True


class EventHub:
    """Holds the state of the show and fans events out to the connected pages.

    A new page first gets a snapshot (the state and a speech that still plays),
    so a reload or a reconnect needs no help from the camera loop.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._clients = set()
        self._state = {"type": "state", "present": False, "phase": "idle", "mood": None}
        self._speech = None  # (speech_start event, clock time when the sound begins)
        self._frame = None
        self._frame_seq = 0
        self._frame_time = float("-inf")
        self._frame_cond = threading.Condition()
        self._viewers = 0
        self._last_client_time = clock()
        self.quit_requested = threading.Event()

    # --- events ---

    def publish(self, event):
        """Send an event to all pages. Never blocks: a page that is too slow is dropped."""
        with self._lock:
            if event.get("type") == "speech_start":
                self._speech = (event, self._clock() + event.get("delay", 0))
            elif event.get("type") == "speech_end":
                self._speech = None
            self._send(encode_event(event))

    def set_state(self, **fields):
        """Update the state. Sends an event only if something changed. Returns True if so."""
        with self._lock:
            state = {**self._state, **fields}
            if state == self._state:
                return False
            self._state = state
            self._send(encode_event(state))
            return True

    def _send(self, data):
        for client in list(self._clients):
            try:
                client.queue.put_nowait(data)
            except queue.Full:
                client.alive = False
                self._clients.discard(client)

    def snapshot(self):
        """The events a new page needs first."""
        events = [dict(self._state)]
        if self._speech is not None:
            event, start_time = self._speech
            # Negative delay: the sound started that many seconds ago
            events.append({**event, "delay": start_time - self._clock()})
        return events

    def subscribe(self):
        client = Client()
        with self._lock:
            for event in self.snapshot():
                client.queue.put_nowait(encode_event(event))
            self._clients.add(client)
        return client

    def unsubscribe(self, client):
        with self._lock:
            client.alive = False
            self._clients.discard(client)
            self._last_client_time = self._clock()

    def client_count(self):
        with self._lock:
            return len(self._clients)

    def seconds_without_client(self):
        with self._lock:
            return 0.0 if self._clients else self._clock() - self._last_client_time

    # --- camera preview ---

    def add_viewer(self, change):
        with self._frame_cond:
            self._viewers += change

    def frame_wanted(self):
        """True if a page shows the preview and the next frame is due. Saves the JPEG encoding otherwise."""
        return self._viewers > 0 and self._clock() - self._frame_time >= 1 / PREVIEW_FPS

    def set_frame(self, jpeg):
        with self._frame_cond:
            self._frame = jpeg
            self._frame_seq += 1
            self._frame_time = self._clock()
            self._frame_cond.notify_all()

    def wait_frame(self, last_seq, timeout):
        """The next frame after `last_seq` as (jpeg, seq), or (None, last_seq) after the timeout."""
        with self._frame_cond:
            if self._frame_cond.wait_for(lambda: self._frame_seq != last_seq, timeout):
                return self._frame, self._frame_seq
            return None, last_seq


class Handler(BaseHTTPRequestHandler):
    """Serves the page, the model, the event stream and the camera preview."""

    server_version = "LynxShow"

    def log_message(self, format, *args):
        pass  # The terminal belongs to the camera loop

    def _host_ok(self):
        # The camera preview must stay on this machine. A foreign Host header means
        # another site points its name at 127.0.0.1 (DNS rebinding).
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host in ("127.0.0.1", "localhost"):
            return True
        self.send_error(403)
        return False

    def do_GET(self):
        if not self._host_ok():
            return
        path = self.path.split("?", 1)[0]
        if path == "/events":
            self._events()
        elif path == "/camera.mjpg":
            self._camera()
        elif path.startswith("/assets/"):
            self._file(ASSETS_DIR, path[len("/assets/"):])
        else:
            self._file(WEB_DIR, path.lstrip("/") or "index.html")

    def do_POST(self):
        if not self._host_ok():
            return
        # The custom header blocks requests from other web pages (no CORS answer here)
        if self.path == "/quit" and self.headers.get("X-Lynx") == "1":
            self.server.hub.quit_requested.set()
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def _file(self, base, relative):
        target = (base / relative).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _stream_headers(self, content_type):
        self.connection.settimeout(10)  # A page that hangs must not hold its thread forever
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _events(self):
        hub = self.server.hub
        client = hub.subscribe()
        ping = encode_event({"type": "ping"})
        try:
            self._stream_headers("text/event-stream")
            while client.alive and not self.server.closing:
                try:
                    data = client.queue.get(timeout=PING_INTERVAL)
                except queue.Empty:
                    data = ping
                self.wfile.write(data)
                self.wfile.flush()
        except OSError:
            pass  # The page went away
        finally:
            hub.unsubscribe(client)

    def _camera(self):
        hub = self.server.hub
        hub.add_viewer(1)
        seq = 0
        try:
            self._stream_headers("multipart/x-mixed-replace; boundary=frame")
            while not self.server.closing:
                frame, seq = hub.wait_frame(seq, timeout=1)
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except OSError:
            pass
        finally:
            hub.add_viewer(-1)


class ShowServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, hub, port=DEFAULT_PORT):
        self.hub = hub
        self.closing = False
        try:
            super().__init__(("127.0.0.1", port), Handler)
        except OSError:
            super().__init__(("127.0.0.1", 0), Handler)  # Port in use: take a free one

    def handle_error(self, request, client_address):
        pass  # A page that closes its connection is normal. No traceback in the terminal

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/"

    def start(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def stop(self):
        self.closing = True
        self.shutdown()
        self.server_close()


class Browser:
    """Chrome in app mode with its own profile, so this process owns the window."""

    def __init__(self, url, fullscreen=True, extra_args=()):
        self._url = url
        self._fullscreen = fullscreen
        self._extra_args = list(extra_args)
        self._process = None
        self._opened_at = float("-inf")

    def command(self):
        chrome = next((path for path in CHROME_PATHS if os.path.exists(path)), None)
        if chrome is None:
            return ["open", self._url]  # Default browser. No kiosk mode and no restart
        profile = Path.home() / "Library" / "Caches" / "mood-greeter" / "chrome-profile"
        return [
            chrome,
            f"--app={self._url}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--noerrdialogs",
            "--disable-session-crashed-bubble",
            # Keep the animation running when the window is covered or loses focus
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--disable-features=CalculateNativeWinOcclusion,Translate",
            "--kiosk" if self._fullscreen else "--window-size=1280,800",
            *self._extra_args,
        ]

    def open(self):
        self.close()
        self._opened_at = time.monotonic()
        try:
            self._process = subprocess.Popen(self.command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            print(f"Cannot open the browser: {error}. Open {self._url} by hand.")

    def keep_open(self, hub):
        """Open the browser again if no page has been connected for a while (closed tab, crashed page)."""
        if (hub.seconds_without_client() > BROWSER_RESTART_SECONDS
                and time.monotonic() - self._opened_at > BROWSER_RESTART_SECONDS):
            print("No lynx page is connected. Opening the browser again.")
            self.open()

    def close(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None


def add_arguments(parser):
    parser.add_argument("--window", action="store_true", help="large window instead of full screen")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser, print the URL only")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port of the local page (default {DEFAULT_PORT})")


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    add_arguments(parser)
    return parser


class Show:
    """Everything the camera loop needs: speech, state, preview, quit."""

    def __init__(self, port=DEFAULT_PORT, fullscreen=True, browser=True):
        self.hub = EventHub()
        self.server = ShowServer(self.hub, port)
        self.server.start()
        self.voice = Voice(publish=self.hub.publish)
        self.voice.prepare(lynx_lines.all_lines())
        self.lines = lynx_lines.LinePicker()
        self.speak = self.voice.speak
        self.is_speaking = self.voice.is_speaking
        self.set_state = self.hub.set_state
        # Keep the display and the Mac awake while this process runs
        self._caffeinate = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])
        extra_args = os.environ.get("LYNX_CHROME_ARGS", "").split()
        self.browser = Browser(self.server.url, fullscreen, extra_args) if browser else None
        self._closed = False
        atexit.register(self.close)  # Also after an error in the camera loop
        if threading.current_thread() is threading.main_thread():
            # Ctrl+C ends the loop through quit_requested(), so the loop can clean up
            signal.signal(signal.SIGINT, lambda *_: self.hub.quit_requested.set())
        print(f"--- Lynx page: {self.server.url}  Quit: Ctrl+C here, or 'q' in the page ---")
        if not MODEL_FILE.is_file():
            print(f"The lynx model is missing: {MODEL_FILE}. Run 'make setup' to fetch it.")
        if self.browser:
            self.browser.open()

    def quit_requested(self):
        """True after 'q' in the page. Also the heartbeat: call it once per loop."""
        if self.browser:
            self.browser.keep_open(self.hub)
        return self.hub.quit_requested.is_set()

    def show_frame(self, frame):
        """Send a BGR camera frame to the preview, if a page shows it and a frame is due."""
        if not self.hub.frame_wanted():
            return
        import cv2  # Only the preview needs OpenCV. Tests of this module run without it

        height = frame.shape[0] * PREVIEW_WIDTH // frame.shape[1]
        small = cv2.resize(frame, (PREVIEW_WIDTH, height), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
        if ok:
            self.hub.set_frame(jpeg.tobytes())

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.browser:
            self.browser.close()
        self.voice.close()
        self.server.stop()
        self._caffeinate.terminate()
