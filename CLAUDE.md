# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mood Greeter is a macOS-only Python app. It detects a face through the webcam, estimates the emotion with DeepFace, and greets the person with spoken text (macOS `say`). A static image mode also exists.

## Setup

Python 3.14. Homebrew's `python@3.14` works. `make setup` creates `.venv` and installs `requirements.txt`. It reruns when `requirements.txt` changes. `make clean` deletes `.venv`.

## Running the Scripts

```bash
make run       # live_sentiment.py (default target): webcam + sentiment + speech; press q to quit
make static    # static_images.py: analyze angryface.jpeg
make speech    # speech_test.py: standalone test of the threaded `say` queue
make test      # pytest: speech queue tests and the speech copy sync check
```

Each target runs `setup` first when needed.

`Live Sentiment.command` is a double-click launcher for Finder. It changes to its own directory, activates `.venv`, and runs `live_sentiment.py`. Run it only after setup.

`tests/` covers the speech queue only. `live_sentiment.py` runs at import (camera, DeepFace), so its loop has no tests. There is no linter, formatter, or CI.

## Architecture

Flat layout. Each script is standalone. There are no packages or shared modules.

- `live_sentiment.py` is one top-level loop with module-level state (no functions besides `speech_worker`/`speak`):
  - Haar cascade finds faces fast. The largest face is cropped and passed to `DeepFace.analyze` (emotion only).
  - Per-person state machine: greet and collect emotion samples for `ANALYSIS_DURATION` (3s), lock the most common emotion, speak it with a canned line, wait `COOLDOWN_DURATION` (5s), then reset.
  - A face center jump of more than `POSITION_CHANGE_THRESHOLD` (150px) counts as a new person. No face for `NO_FACE_THRESHOLD` (15) frames resets all state.
  - Speech runs on a daemon thread that reads a `queue.Queue` and calls `subprocess.run(['say', ...])`, so the video loop never blocks.
- `speech_worker`/`speak` are copy-pasted in `speech_test.py` and `live_sentiment.py`. Keep both in sync if you change one.
- `static_images.py` runs DeepFace once on one image.

## Key Dependencies

- `deepface` (>= 0.0.103): pre-trained emotion models, run on `onnxruntime` (CPU, about 6 ms per face). There is no TensorFlow in this project. Weights download to `~/.deepface/weights/` on first run.
- `werkzeug`: DeepFace imports it but does not declare it. It must stay in `requirements.txt`, or `from deepface import DeepFace` fails.
- `opencv-contrib-python-headless`: capture, Haar cascade, display. It is the same package DeepFace pulls in, so one package owns the `cv2` folder. On macOS it has the Cocoa window backend despite the name. OpenCV 5 moved `CascadeClassifier` to contrib, so keep `opencv-python` out of `requirements.txt`: it lacks the cascade and overwrites `cv2`. `make setup` asserts both the cascade and `GUI: COCOA`. OpenCV 5 wheels ship no cascade XML files (`cv2.data.haarcascades` is empty), so `live_sentiment.py` downloads the face cascade into `~/.deepface/weights/` with DeepFace's `weight_utils`, the same way DeepFace does.

## Configuration Notes

- `detector_backend` in DeepFace: `opencv` (fastest) or `retinaface` (most accurate, slower)
- `enforce_detection=False` prevents crashes on blurry or missing faces
- `silent=True` in `DeepFace.analyze` suppresses DeepFace progress logs
- Exceptions from `DeepFace.analyze` in the live loop are swallowed on purpose. A failed frame adds no sample.
