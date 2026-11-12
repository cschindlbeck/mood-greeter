"""Voice of the lynx: a sequential speech queue that also drives the lip sync.

`say` gives no amplitude. So each line is rendered to a WAV file (`say -o`),
a loudness envelope is computed from the samples, and a player plays the file.
A `speech_start` event carries the envelope and the time until the sound begins,
and the page moves the jaw along it. `speech_end` follows when the player exits.

Players, best first:
1. `.venv/bin/lynx_player` (built from lynx_player.swift by `make setup`). It
   reports the exact start of the sound.
2. `afplay` with a guessed start delay (AFPLAY_START_DELAY). Lip sync is less exact.
3. Plain `say`, if rendering fails. No envelope: the page moves the jaw with noise.

`is_speaking()` is true from `speak()` until the player has exited. The camera
loop uses it as the speech gate.
"""
import math
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

VOICE = "Samantha"  # Falls back to the system voice if it is not installed
RATE = 185  # Words per minute. The macOS default is about 175
SAMPLE_RATE = 22050
ENVELOPE_STEP = 0.02  # Seconds per envelope value
AFPLAY_START_DELAY = 0.25  # Guess. afplay does not report when the sound starts
PLAYER_PATH = Path(__file__).resolve().parent / ".venv" / "bin" / "lynx_player"

# `say` reads these words wrong. Text on screen keeps the original.
# "LYNQTECH" and "Linktech" come out as "link-tetch". Two words give "link tek".
PRONUNCIATION = {"LYNQTECH": "Link tech"}


def spoken_text(text):
    """The text as `say` gets it."""
    for word, sound in PRONUNCIATION.items():
        text = text.replace(word, sound)
    return text


def compute_envelope(samples, sample_rate=SAMPLE_RATE, step=ENVELOPE_STEP):
    """Loudness per `step` seconds as integers 0..100, for the jaw.

    RMS per window, relative to the loud parts of the clip (95th percentile).
    A gate removes breath noise and a square root lifts quiet syllables.
    """
    samples = np.asarray(samples, dtype=np.float64)
    window = max(1, int(round(sample_rate * step)))
    count = math.ceil(len(samples) / window)
    if count == 0:
        return []
    padded = np.zeros(count * window)
    padded[:len(samples)] = samples
    rms = np.sqrt((padded.reshape(count, window) ** 2).mean(axis=1))
    loud = np.percentile(rms, 95)
    if loud < 1:  # Silence (int16 scale)
        return [0] * count
    level = np.clip(rms / loud, 0, 1)
    level[level < 0.08] = 0
    return [int(round(v)) for v in np.sqrt(level) * 100]


def read_wav(path):
    """Mono int16 samples and the sample rate of a WAV file."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit samples: {path}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if wav.getnchannels() > 1:
            samples = samples.reshape(-1, wav.getnchannels()).mean(axis=1)
        return samples, wav.getframerate()


class Clip:
    """One rendered line: the audio file, its envelope and its length in seconds."""

    def __init__(self, path, envelope, duration):
        self.path = path
        self.envelope = envelope
        self.duration = duration


class Voice:
    """Speaks queued lines one after the other on a daemon thread."""

    def __init__(self, publish=None, voice=VOICE, rate=RATE, player_path=PLAYER_PATH):
        self._publish = publish or (lambda event: None)
        self._rate = rate
        self._voice = voice
        self._voice_checked = False
        self._player_path = Path(player_path)
        self._queue = queue.Queue()
        self._clips = {}
        self._render_lock = threading.Lock()
        self._process = None
        self._process_lock = threading.Lock()
        self._closed = False
        self._next_id = 0
        self._dir = Path(tempfile.mkdtemp(prefix="lynx-voice-"))
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # --- public API ---

    def speak(self, text, mood=None):
        """Queue text for speech. Non-blocking, lines play in order."""
        self._queue.put((text, mood))

    def is_speaking(self):
        """True while speech is queued or still playing."""
        return self._queue.unfinished_tasks > 0

    def wait(self):
        """Block until all queued speech has played."""
        self._queue.join()

    def prepare(self, texts):
        """Render lines in the background, so the first use starts without delay."""
        threading.Thread(target=self._prepare, args=(list(texts),), daemon=True).start()

    def close(self):
        """Stop the worker, stop the sound and delete the rendered files."""
        self._closed = True
        self._queue.put(None)
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
        self._thread.join(timeout=2)
        shutil.rmtree(self._dir, ignore_errors=True)

    # --- internals ---

    def _prepare(self, texts):
        for text in texts:
            if self._closed:
                return
            try:
                self._clip(text)
            except Exception:
                pass  # The worker tries again and falls back to plain `say`

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:  # Shutdown signal
                self._queue.task_done()
                break
            try:
                if not self._closed:
                    self._play(*item)
            except Exception as error:  # Speech must never kill the worker
                print(f"Speech failed: {error!r}")
            finally:
                self._queue.task_done()

    def _voice_args(self):
        if not self._voice_checked:
            self._voice_checked = True
            if self._voice:
                listed = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
                if not any(line.startswith(self._voice + " ") for line in listed.splitlines()):
                    print(f"Voice {self._voice!r} is not installed. Using the system voice.")
                    self._voice = None
        return ["-v", self._voice] if self._voice else []

    def _clip(self, text):
        """Render `text` once and keep the result."""
        with self._render_lock:
            clip = self._clips.get(text)
            if clip is None:
                path = self._dir / f"line-{len(self._clips)}.wav"
                subprocess.run(
                    ["say", *self._voice_args(), "-r", str(self._rate), "-o", str(path),
                     f"--data-format=LEI16@{SAMPLE_RATE}", spoken_text(text)],
                    check=True, stderr=subprocess.DEVNULL,  # A render that fails at shutdown is no news
                )
                samples, sample_rate = read_wav(path)
                clip = Clip(path, compute_envelope(samples, sample_rate), len(samples) / sample_rate)
                self._clips[text] = clip
            return clip

    def _run(self, command, **kwargs):
        with self._process_lock:
            if self._closed:
                return None
            self._process = subprocess.Popen(command, **kwargs)
            return self._process

    def _play(self, text, mood):
        self._next_id += 1
        event = {"type": "speech_start", "id": self._next_id, "text": text, "mood": mood,
                 "envelope": None, "step": ENVELOPE_STEP, "duration": None, "delay": 0}
        try:
            clip = self._clip(text)
        except Exception as error:
            print(f"Cannot render speech, using plain say: {error!r}")
            clip = None

        process = None
        try:
            if clip is None:
                process = self._run(["say", *self._voice_args(), "-r", str(self._rate), spoken_text(text)])
            else:
                event["envelope"] = clip.envelope
                event["duration"] = clip.duration
                process = self._start_player(clip, event)
            self._emit(event)
            if process is not None:
                try:
                    process.wait(timeout=(clip.duration if clip else 30) + 10)
                except subprocess.TimeoutExpired:
                    process.kill()
        finally:
            self._emit({"type": "speech_end", "id": event["id"]})

    def _emit(self, event):
        """Publish an event. A failing listener must not cut the speech or the speech gate."""
        try:
            self._publish(event)
        except Exception as error:
            print(f"Speech event failed: {error!r}")

    def _start_player(self, clip, event):
        """Start the sound and set event['delay'] to the seconds until it begins."""
        if self._player_path.is_file():
            process = self._run([str(self._player_path), str(clip.path)], stdout=subprocess.PIPE, text=True)
            if process is None:
                return None
            line = process.stdout.readline()  # "START <seconds>", or "" if the player failed
            if line.startswith("START "):
                event["delay"] = float(line.split()[1])
                return process
            process.wait()
        event["delay"] = AFPLAY_START_DELAY
        return self._run(["afplay", str(clip.path)])
