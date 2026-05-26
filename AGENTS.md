# AGENTS.md

Minimal instruction file for agents working in this repo.

## Fresh macOS Setup (from zero)

Do NOT use Homebrew's `python@3.11` — it has a known `libexpat` ABI crash with `ensurepip`.

```bash
# 1. Download Python 3.11.9 from python.org (macOS 64-bit universal2 installer)
#    https://www.python.org/downloads/release/python-3119/
# 2. Run the .pkg installer
# 3. Verify in a new terminal tab:
python3.11 --version   # should show Python 3.11.9

# 4. Create venv (use virtualenv if python3.11 -m venv fails on Framework Python):
python3.11 -m venv .venv
# fallback: pip3 install virtualenv && python3.11 -m virtualenv .venv

# 5. Activate and install:
source .venv/bin/activate
python3.11 -m pip install -r requirements.txt
```

- **Python 3.11 is required** (not 3.12/3.13) per `requirements.txt` — TensorFlow compatibility.
- `tensorflow-metal` enables GPU on Apple Silicon; without it, runs on CPU.
- `tf-keras` must be in `requirements.txt` alongside `tensorflow-macos`.

## Double-click Launch

After setup, double-click `Live Sentiment.command` in Finder to run the webcam.
It auto-changes to its own directory, activates `.venv`, and runs `live_sentiment.py`.
Press `q` on the webcam window to quit.

## Run (terminal)

```bash
python3.11 static_images.py     # analyze angryface.jpeg
python3.11 live_sentiment.py    # webcam face + sentiment + TTS
python3.11 speech_test.py       # standalone macOS TTS test
```

- `live_sentiment.py`: Press `q` to quit. Uses `subprocess.run(['say', ...])` for speech — **macOS only**. Face state machine: 3s analysis → lock sentiment → 5s cooldown → reset. Re-greets on face position change >150px.
- `speech_test.py` is an isolated test of the threaded `say` queue pattern used by `live_sentiment.py`.

## Architecture

- **Flat single-module layout.** No packages, no tests, no CI.
- `static_images.py` — Offline DeepFace analysis of one image.
- `live_sentiment.py` — Webcam loop: OpenCV Haar cascade for face detection, DeepFace `opencv` backend for emotion, threaded macOS `say` for TTS.
- No linting, typechecking, or formatting tooling.

## Key Quirks

- `live_sentiment.py` uses `face_roi` (cropped face) passed to `DeepFace.analyze` with `silent=True` to suppress TF warnings.
- `detector_backend='opencv'` is the fastest; `'retinaface'` is more accurate but slower.
- `enforce_detection=False` everywhere — prevents crashes on blurry/non-face frames.
