# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mood Greeter is a macOS-only Python app. It detects a face through the webcam, estimates the emotion with DeepFace, and greets the person with spoken text (macOS `say`). A static image mode also exists.

## Setup

Python 3.11 is required, not 3.12 or 3.13 (TensorFlow compatibility).

Do not use Homebrew's `python@3.11`. It crashes in `ensurepip` (libexpat ABI). On a fresh Mac:

1. Download Python 3.11.9 (macOS 64-bit universal2 installer) from https://www.python.org/downloads/release/python-3119/
2. Run the `.pkg` installer.
3. Verify in a new terminal tab: `python3.11 --version` must show `Python 3.11.9`.

```bash
python3.11 -m venv .venv
# fallback if venv fails on Framework Python:
# pip3 install virtualenv && python3.11 -m virtualenv .venv
source .venv/bin/activate
python3.11 -m pip install -r requirements.txt
```

## Running the Scripts

```bash
python3.11 static_images.py     # analyze angryface.jpeg
python3.11 live_sentiment.py    # webcam + sentiment + speech; press q to quit
python3.11 speech_test.py       # standalone test of the threaded `say` queue
```

`Live Sentiment.command` is a double-click launcher for Finder. It changes to its own directory, activates `.venv`, and runs `live_sentiment.py`. Run it only after setup.

There are no tests, linter, formatter, or CI.

## Architecture

Flat layout. Each script is standalone. There are no packages or shared modules.

- `live_sentiment.py` is one top-level loop with module-level state (no functions besides `speech_worker`/`speak`):
  - Haar cascade finds faces fast. The largest face is cropped and passed to `DeepFace.analyze` (emotion only).
  - Per-person state machine: greet and collect emotion samples for `ANALYSIS_DURATION` (3s), lock the most common emotion, speak it with a canned line, wait `COOLDOWN_DURATION` (5s), then reset.
  - A face center jump of more than `POSITION_CHANGE_THRESHOLD` (150px) counts as a new person. No face for `NO_FACE_THRESHOLD` (15) frames resets all state.
  - Speech runs on a daemon thread that reads a `queue.Queue` and calls `subprocess.run(['say', ...])`, so the video loop never blocks.
- `speech_worker`/`speak` are copy-pasted in `speech_test.py` and `live_sentiment.py`. Keep both in sync if you change one.
- `static_images.py` runs DeepFace once on one image, with optional Metal GPU on macOS.

## Key Dependencies

- `tensorflow-macos` + `tensorflow-metal`: backend with Apple GPU support (CPU without metal)
- `tf-keras`: must stay in `requirements.txt` next to TensorFlow, or DeepFace fails to load
- `deepface`: pre-trained emotion models
- `opencv-python`: capture, Haar cascade, display

## Configuration Notes

- `detector_backend` in DeepFace: `opencv` (fastest) or `retinaface` (most accurate, slower)
- `enforce_detection=False` prevents crashes on blurry or missing faces
- `silent=True` in `DeepFace.analyze` suppresses TF log noise
- Exceptions from `DeepFace.analyze` in the live loop are swallowed on purpose. A failed frame adds no sample.
