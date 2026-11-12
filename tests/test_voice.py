import subprocess
import threading
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import lynx_voice
from lynx_voice import ENVELOPE_STEP, SAMPLE_RATE, Voice, compute_envelope, spoken_text

WINDOW = int(SAMPLE_RATE * ENVELOPE_STEP)


def tone(seconds, amplitude=8000):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), np.int16)


# --- envelope ---

def test_envelope_of_silence_is_closed_mouth():
    assert compute_envelope(silence(0.5)) == [0] * 25


def test_envelope_of_nothing_is_empty():
    assert compute_envelope(np.array([], np.int16)) == []


def test_envelope_has_one_value_per_step():
    assert len(compute_envelope(tone(1.0))) == 50
    assert len(compute_envelope(tone(1.0)[:-1])) == 50  # A partial last window still counts
    assert len(compute_envelope(tone(1.0)[: WINDOW + 1])) == 2


def test_envelope_follows_the_sound():
    envelope = compute_envelope(np.concatenate([silence(0.2), tone(0.4), silence(0.2)]))
    assert envelope[:10] == [0] * 10  # Mouth closed before the sound
    assert all(value > 90 for value in envelope[11:29])  # Open during the sound
    assert envelope[30:] == [0] * 10  # Closed after it
    assert all(0 <= value <= 100 for value in envelope)


def test_envelope_is_relative_to_the_loud_parts():
    envelope = compute_envelope(np.concatenate([tone(0.4, 8000), tone(0.4, 2000)]))
    assert envelope[5] > 90
    assert 40 <= envelope[30] <= 60  # A quarter of the amplitude, lifted by the square root


def test_envelope_gate_removes_quiet_noise():
    envelope = compute_envelope(np.concatenate([tone(0.4, 8000), tone(0.4, 300)]))
    assert envelope[25:] == [0] * 15


def test_spoken_text_fixes_the_company_name_only():
    assert spoken_text("Welcome to the LYNQTECH party") == "Welcome to the Link tech party"
    assert spoken_text("Merry Christmas") == "Merry Christmas"


# --- speech queue ---

class FakeProcess:
    """Stands in for the player process. It runs until `finish` is set."""

    def __init__(self, command, finish, line="START 0.2\n"):
        self.command = command
        self._finish = finish
        self.stdout = mock.Mock(readline=mock.Mock(return_value=line))
        self.terminated = False

    def wait(self, timeout=None):
        self._finish.wait(5)
        return 0

    def poll(self):
        return 0 if self._finish.is_set() else None

    def terminate(self):
        self.terminated = True
        self._finish.set()

    kill = terminate


class Harness:
    """A Voice with fake `say` and a fake player."""

    def __init__(self, tmp_path, player=True, render_fails=False, player_line="START 0.2\n"):
        self.events = []
        self.renders = []
        self.processes = []
        self.finish = threading.Event()
        self.finish.set()  # Players end at once unless a test clears this
        self.render_fails = render_fails
        self.player_line = player_line
        player_path = tmp_path / "lynx_player"
        if player:
            player_path.write_text("")
        self._patches = [
            mock.patch.object(lynx_voice.subprocess, "run", side_effect=self._run),
            mock.patch.object(lynx_voice.subprocess, "Popen", side_effect=self._popen),
        ]
        for patch in self._patches:
            patch.start()
        self.voice = Voice(publish=self.events.append, player_path=player_path)

    def _run(self, command, **kwargs):
        if "-o" in command:  # say -o <file>: render
            if self.render_fails:
                raise subprocess.CalledProcessError(1, command)
            self.renders.append(command[-1])
            with wave.open(command[command.index("-o") + 1], "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(np.concatenate([tone(0.5), silence(0.5)]).tobytes())
        return mock.Mock(stdout="Samantha            en_US    # Hello!\n")

    def _popen(self, command, **kwargs):
        process = FakeProcess(command, self.finish, self.player_line)
        self.processes.append(process)
        return process

    def close(self):
        self.finish.set()
        self.voice.close()
        for patch in self._patches:
            patch.stop()


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(**kwargs):
        made.append(Harness(tmp_path, **kwargs))
        return made[-1]

    yield make
    for item in made:
        item.close()


def test_speak_sends_start_with_envelope_then_end(harness):
    h = harness()
    h.voice.speak("Welcome to the LYNQTECH Christmas party!", mood="happy")
    h.voice.wait()
    start, end = h.events
    assert start["type"] == "speech_start"
    assert start["text"] == "Welcome to the LYNQTECH Christmas party!"  # The page shows the real name
    assert start["mood"] == "happy"
    assert start["delay"] == 0.2  # From the player: the sound begins in 0.2 s
    assert start["duration"] == pytest.approx(1.0)
    assert start["step"] == ENVELOPE_STEP
    assert len(start["envelope"]) == 50
    assert start["envelope"][10] > 90 and start["envelope"][-1] == 0
    assert end == {"type": "speech_end", "id": start["id"]}
    assert h.renders == ["Welcome to the Link tech Christmas party!"]  # `say` gets the spoken form
    assert h.processes[0].command[0].endswith("lynx_player")


def test_say_gets_voice_rate_and_format(harness):
    h = harness()
    h.voice.speak("hello")
    h.voice.wait()
    command = lynx_voice.subprocess.run.call_args_list[-1].args[0]
    assert command[:5] == ["say", "-v", "Samantha", "-r", "185"]
    assert f"--data-format=LEI16@{SAMPLE_RATE}" in command


def test_lines_play_in_order(harness):
    h = harness()
    for text in ["one", "two", "three"]:
        h.voice.speak(text)
    h.voice.wait()
    assert [e["text"] for e in h.events if e["type"] == "speech_start"] == ["one", "two", "three"]
    assert [e["type"] for e in h.events] == ["speech_start", "speech_end"] * 3


def test_is_speaking_until_the_player_exits(harness):
    h = harness()
    h.finish.clear()
    assert not h.voice.is_speaking()
    h.voice.speak("slow")
    h.voice.speak("queued")  # Returns at once, although "slow" still plays
    assert h.voice.is_speaking()
    deadline = threading.Event()
    while not h.events and not deadline.wait(0.01):
        pass
    assert [e["type"] for e in h.events] == ["speech_start"]
    assert h.voice.is_speaking()  # The worker took the line, but it still plays
    h.finish.set()
    h.voice.wait()
    assert not h.voice.is_speaking()
    assert [e["type"] for e in h.events] == ["speech_start", "speech_end"] * 2


def test_a_line_is_rendered_once(harness):
    h = harness()
    h.voice.speak("same")
    h.voice.speak("same")
    h.voice.wait()
    assert h.renders == ["same"]


def test_prepare_renders_ahead(harness):
    h = harness()
    h.voice._prepare(["a", "b"])
    h.voice.speak("a")
    h.voice.wait()
    assert h.renders == ["a", "b"]


def test_without_player_afplay_is_used_with_a_guessed_delay(harness):
    h = harness(player=False)
    h.voice.speak("hello")
    h.voice.wait()
    assert h.processes[0].command[0] == "afplay"
    assert h.events[0]["delay"] == lynx_voice.AFPLAY_START_DELAY
    assert h.events[0]["envelope"]


def test_broken_player_falls_back_to_afplay(harness):
    h = harness(player_line="")
    h.voice.speak("hello")
    h.voice.wait()
    assert [Path(p.command[0]).name for p in h.processes] == ["lynx_player", "afplay"]
    assert [e["type"] for e in h.events] == ["speech_start", "speech_end"]


def test_render_failure_falls_back_to_plain_say(harness):
    h = harness(render_fails=True)
    h.voice.speak("Hello LYNQTECH")
    h.voice.wait()
    assert h.processes[0].command == ["say", "-v", "Samantha", "-r", "185", "Hello Link tech"]
    start, end = h.events
    assert start["envelope"] is None  # The page moves the jaw with noise
    assert end["type"] == "speech_end"


def test_worker_survives_a_failing_listener(harness):
    h = harness()
    calls = []

    def publish(event):
        calls.append(event)
        if len(calls) == 1:
            raise RuntimeError("listener broke")

    h.voice._publish = publish
    h.voice.speak("first")
    h.voice.speak("second")
    h.voice.wait()
    assert not h.voice.is_speaking()
    assert [e["text"] for e in calls if e["type"] == "speech_start"] == ["first", "second"]
    assert [e["type"] for e in calls] == ["speech_start", "speech_end"] * 2  # The broken line still ends
    assert len(h.processes) == 2


def test_close_stops_the_sound_and_the_worker(harness):
    h = harness()
    h.finish.clear()
    h.voice.speak("long line")
    while not h.processes:
        threading.Event().wait(0.01)
    h.voice.close()
    assert h.processes[0].terminated
    assert not h.voice._thread.is_alive()
    assert not h.voice._dir.exists()
