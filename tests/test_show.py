import http.client
import json
import queue
import time

import pytest

import lynx_show
from lynx_show import Browser, EventHub, ShowServer


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def drain(client):
    events = []
    while True:
        try:
            events.append(json.loads(client.queue.get_nowait()[len(b"data: "):]))
        except queue.Empty:
            return events


SPEECH = {"type": "speech_start", "id": 1, "text": "hi", "mood": None,
          "envelope": [0, 50, 100], "step": 0.02, "duration": 2.0, "delay": 0.2}


# --- EventHub ---

def test_new_page_gets_the_state_first():
    hub = EventHub()
    hub.set_state(present=True, phase="greeting")
    assert drain(hub.subscribe()) == [{"type": "state", "present": True, "phase": "greeting", "mood": None}]


def test_state_is_sent_only_when_it_changes():
    hub = EventHub()
    client = hub.subscribe()
    drain(client)
    assert hub.set_state(present=True)
    assert not hub.set_state(present=True)  # The camera loop calls this on every frame
    assert hub.set_state(mood="happy")
    assert [e["mood"] for e in drain(client)] == [None, "happy"]


def test_events_reach_every_page():
    hub = EventHub()
    clients = [hub.subscribe(), hub.subscribe()]
    for client in clients:
        drain(client)
    hub.publish(SPEECH)
    assert [drain(client) for client in clients] == [[SPEECH], [SPEECH]]
    assert hub.client_count() == 2


def test_page_that_connects_during_speech_gets_the_running_speech():
    clock = FakeClock()
    hub = EventHub(clock)
    hub.publish(SPEECH)  # The sound begins 0.2 s later
    clock.now += 1.2
    state, speech = drain(hub.subscribe())
    assert speech["delay"] == pytest.approx(-1.0)  # The sound began 1 s ago: the jaw joins in sync
    assert speech["envelope"] == SPEECH["envelope"]


def test_page_that_connects_after_speech_gets_no_speech():
    hub = EventHub()
    hub.publish(SPEECH)
    hub.publish({"type": "speech_end", "id": 1})
    assert [e["type"] for e in drain(hub.subscribe())] == ["state"]


def test_slow_page_is_dropped_and_does_not_block_the_loop():
    hub = EventHub()
    slow = hub.subscribe()
    for i in range(lynx_show.CLIENT_QUEUE_SIZE + 5):
        hub.publish({"type": "ping", "n": i})
    assert not slow.alive
    assert hub.client_count() == 0


def test_seconds_without_client():
    clock = FakeClock()
    hub = EventHub(clock)
    clock.now += 30
    assert hub.seconds_without_client() == 30
    client = hub.subscribe()
    assert hub.seconds_without_client() == 0
    hub.unsubscribe(client)
    clock.now += 7
    assert hub.seconds_without_client() == 7


def test_frames_are_wanted_only_with_a_viewer_and_at_the_preview_rate():
    clock = FakeClock()
    hub = EventHub(clock)
    assert not hub.frame_wanted()  # No page shows the preview: no JPEG encoding
    hub.add_viewer(1)
    assert hub.frame_wanted()
    hub.set_frame(b"jpeg")
    assert not hub.frame_wanted()
    clock.now += 1 / lynx_show.PREVIEW_FPS + 0.001
    assert hub.frame_wanted()
    hub.add_viewer(-1)
    assert not hub.frame_wanted()


def test_wait_frame_returns_new_frames_only():
    hub = EventHub()
    assert hub.wait_frame(0, timeout=0.01) == (None, 0)
    hub.set_frame(b"one")
    assert hub.wait_frame(0, timeout=0.01) == (b"one", 1)
    assert hub.wait_frame(1, timeout=0.01) == (None, 1)


# --- HTTP server ---

@pytest.fixture
def server():
    server = ShowServer(EventHub(), port=0)
    server.start()
    yield server
    server.stop()


def request(server, method, path, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    connection.request(method, path, headers=headers or {})
    return connection.getresponse()


def test_serves_the_page_and_the_library(server):
    page = request(server, "GET", "/")
    assert page.status == 200
    assert b"lynx.js" in page.read()
    assert request(server, "GET", "/vendor/three/three.module.js").status == 200


@pytest.mark.skipif(not lynx_show.MODEL_FILE.is_file(), reason="the lynx model is not in git, `make setup` fetches it")
def test_serves_the_model(server):
    model = request(server, "GET", "/assets/luchs-emerald-liegend.glb")
    assert model.status == 200
    assert model.getheader("Content-Type") == "model/gltf-binary"
    assert model.read(4) == b"glTF"


def test_serves_other_assets_with_a_type(server):
    readme = request(server, "GET", "/assets/README.md")
    assert readme.status == 200
    assert readme.getheader("Content-Type") == "application/octet-stream"


def test_page_needs_no_other_server():
    """The party may have no internet: the page must load everything from here."""
    for path in (lynx_show.WEB_DIR / "index.html", lynx_show.WEB_DIR / "lynx.js"):
        text = path.read_text()
        assert "http://" not in text and "https://" not in text, path


def test_files_outside_web_and_assets_are_not_served(server):
    assert request(server, "GET", "/../lynx_show.py").status == 404
    assert request(server, "GET", "/assets/../lynx_show.py").status == 404
    assert request(server, "GET", "/missing.js").status == 404


def test_foreign_host_header_is_rejected(server):
    """The camera preview must not be readable through another site's name (DNS rebinding)."""
    assert request(server, "GET", "/camera.mjpg", {"Host": "evil.example"}).status == 403
    assert request(server, "GET", "/", {"Host": "evil.example"}).status == 403


def test_quit_needs_the_page_header(server):
    assert request(server, "POST", "/quit").status == 404
    assert not server.hub.quit_requested.is_set()
    assert request(server, "POST", "/quit", {"X-Lynx": "1"}).status == 204
    assert server.hub.quit_requested.is_set()


def test_event_stream_sends_snapshot_then_live_events(server):
    server.hub.set_state(present=True, phase="greeting")
    response = request(server, "GET", "/events")
    assert response.getheader("Content-Type") == "text/event-stream"
    assert json.loads(response.readline()[len(b"data: "):])["present"] is True
    response.readline()  # Blank line ends the event
    server.hub.publish(SPEECH)
    assert json.loads(response.readline()[len(b"data: "):]) == SPEECH
    response.close()


def test_closed_page_is_forgotten(server):
    response = request(server, "GET", "/events")
    response.readline()
    assert server.hub.client_count() == 1
    response.close()
    # The server notices on its next write
    for _ in range(50):
        server.hub.publish({"type": "ping"})
        if server.hub.client_count() == 0:
            break
        time.sleep(0.02)
    assert server.hub.client_count() == 0


def test_camera_stream_sends_frames(server):
    response = request(server, "GET", "/camera.mjpg")
    assert response.getheader("Content-Type").startswith("multipart/x-mixed-replace")
    for _ in range(100):  # The handler registers as viewer first
        if server.hub.frame_wanted():
            break
        time.sleep(0.01)
    server.hub.set_frame(b"\xff\xd8fake jpeg\xff\xd9")
    assert response.readline() == b"--frame\r\n"
    assert response.readline() == b"Content-Type: image/jpeg\r\n"
    assert response.readline() == b"Content-Length: 13\r\n"
    response.readline()
    assert response.read(13) == b"\xff\xd8fake jpeg\xff\xd9"
    response.close()


def test_port_in_use_falls_back_to_a_free_port(server):
    second = ShowServer(EventHub(), port=server.server_address[1])
    try:
        assert second.server_address[1] != server.server_address[1]
    finally:
        second.server_close()


# --- browser ---

def test_browser_command_full_screen_and_window():
    kiosk = Browser("http://127.0.0.1:8765/", fullscreen=True).command()
    window = Browser("http://127.0.0.1:8765/", fullscreen=False).command()
    if kiosk[0] == "open":  # No Chrome on this machine
        assert kiosk == ["open", "http://127.0.0.1:8765/"]
        return
    assert "--app=http://127.0.0.1:8765/" in kiosk
    assert "--kiosk" in kiosk and "--kiosk" not in window
    assert any(arg.startswith("--user-data-dir=") for arg in kiosk)


def test_browser_is_opened_again_when_no_page_is_connected(monkeypatch):
    clock = FakeClock()
    hub = EventHub(clock)
    browser = Browser("http://127.0.0.1:8765/")
    opened = []
    monkeypatch.setattr(browser, "open", lambda: opened.append(clock.now))
    browser.keep_open(hub)
    assert opened == []  # Just started: give the page time to connect
    clock.now += lynx_show.BROWSER_RESTART_SECONDS + 1
    browser.keep_open(hub)
    assert len(opened) == 1
    client = hub.subscribe()
    clock.now += 1000
    browser.keep_open(hub)
    assert len(opened) == 1  # A page is connected: leave the browser alone
    hub.unsubscribe(client)
