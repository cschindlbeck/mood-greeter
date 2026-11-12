# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mood Greeter is a macOS-only Python app. It detects a face through the webcam, estimates the emotion with DeepFace, and greets the person with spoken text. The output is the Christmas lynx show: a browser window with the 3D lynx mascot (Santa hat, mouth in sync with the voice) and a small camera preview. A static image mode also exists.

Privacy rules: no names, no face recognition, no stored images or emotion data. The page and the camera preview are served on `127.0.0.1` only. The app needs no internet at runtime.

## Setup

Python 3.14. Homebrew's `python@3.14` works. `make setup` creates `.venv`, installs `requirements.txt`, builds `.venv/bin/lynx_player` from `lynx_player.swift` with `swiftc` (Xcode command line tools), and fetches the lynx model. It reruns when `requirements.txt` or the Swift file changes.

The lynx model `assets/luchs-emerald-liegend.glb` is company property and this repo is public. The file is in `.gitignore`. Never commit it. `make setup` fetches it with `gh api` from the private repo named in the `Makefile` (`MODEL_REPO`), which needs `gh auth login` with access. `make clean` deletes `.venv`.

First run with internet: DeepFace and `live_sentiment.py` download the model weights and the face cascade into `~/.deepface/weights/`. After that the app runs offline.

## Running the Scripts

```bash
make run       # live_sentiment.py (default target): webcam + sentiment + lynx show in full screen
make demo      # lynx_demo.py: the lynx show with fake events and real speech, no camera
make static    # static_images.py: analyze angryface.jpeg
make speech    # speech_test.py: standalone test of the lynx voice
make test      # pytest: lines, voice (envelope, speech queue), show server
```

Each target runs `setup` first when needed. `make run` and `make demo` take `ARGS`: `--window` (large window, no full screen; `make demo` always uses it), `--no-browser`, `--port N`. `make demo` also takes `--moods happy,sad` and `--once`.

Quit: Ctrl+C in the terminal, or `q` in the lynx page.

`Live Sentiment.command` is a double-click launcher for Finder. It changes to its own directory and runs `live_sentiment.py` from `.venv`. Run it only after setup.

`live_sentiment.py` and `lynx_demo.py` run at import (camera, DeepFace, server), so they have no tests. The modules `lynx_lines.py`, `lynx_voice.py` and `lynx_show.py` have tests. The page (`web/lynx.js`) has no automated tests: check it with `make demo` and the debug URL parameters. There is no linter, formatter, or CI.

## Architecture

Flat layout, no packages. The scripts are `live_sentiment.py`, `lynx_demo.py`, `static_images.py`, `speech_test.py`. The shared modules are `lynx_lines.py`, `lynx_voice.py`, `lynx_show.py`.

- `live_sentiment.py` is one top-level loop with module-level state and no functions:
  - Haar cascade finds faces fast. The largest face is cropped and passed to `DeepFace.analyze` (emotion only).
  - Per-person state machine: greet, wait until the greeting ends, collect emotion samples for `ANALYSIS_DURATION` (1.5s), lock the most common emotion, speak one mood line, wait until the speech ends plus `COOLDOWN_DURATION` (5s), then reset.
  - Drawn on the frame (visible in the camera preview): a progress bar under the face during the analysis, and a tally with the share of each emotion next to the face (left side if there is no room on the right).
  - Speech gate: while `is_speaking()` is true (speech queued or playing), the loop takes no samples, starts no greeting, and holds the analysis and cooldown timers. So the queue holds at most one cycle of speech, and the spoken emotion matches the current face.
  - A face center jump of more than `POSITION_CHANGE_THRESHOLD` (150px) counts as a new person. No face for `NO_FACE_THRESHOLD` (15) frames resets all state.
  - Output: at the end of each iteration the loop calls `show.set_state(present, phase, mood)` and `show.show_frame(frame)`. There is no OpenCV window.
  - A failed camera read does not stop the loop. A black "NO CAMERA" frame goes through the loop as "no face", and the camera is opened again every `CAMERA_RETRY_SECONDS`.
- `lynx_lines.py`: the spoken lines as data. `WELCOME_LINES` (each starts with the welcome to the LYNQTECH Christmas party), `MOOD_LINES` per DeepFace emotion, `UNKNOWN_MOOD_LINES`. `LinePicker` picks a random variant, never the same twice in a row.
- `lynx_voice.py`: `Voice` is the speech queue on a daemon thread (`speak`, `is_speaking`). For lip sync it renders each line with `say -o` to a WAV file, computes a loudness envelope (`compute_envelope`, 50 values per second, 0..100), and plays the file. It publishes `speech_start` (text, mood, envelope, `delay` = seconds until the sound begins) and `speech_end`.
  - Player: `.venv/bin/lynx_player` prints `START <seconds>` with the exact start of the sound. Fallbacks: `afplay` with a guessed delay (its real start delay is not known and it exits up to 0.9s late), then plain `say` without envelope.
  - All lines are rendered in the background at startup (`prepare`), so speech starts without a render delay.
  - `PRONUNCIATION` maps written words to what `say` gets. "LYNQTECH" is spoken as "Link tech". "Linktech" as one word sounds like "link-tetch".
- `lynx_show.py`: the web part uses the standard library only.
  - `EventHub` holds the state and sends events to the pages (Server-Sent Events). A new page gets a snapshot first: the state and a speech that still plays, with a negative `delay`. So a reload joins in sync.
  - HTTP server on `127.0.0.1` (default port 8765): `/` serves `web/`, `/assets/` serves `assets/`, `/events` is the event stream (ping every 5s), `/camera.mjpg` is the preview (MJPEG, 640px, 15 fps, encoded only while a page shows it), `POST /quit` needs the header `X-Lynx: 1`. Requests with a foreign `Host` header get 403.
  - `Browser` starts Chrome in app mode with its own profile (`~/Library/Caches/mood-greeter/chrome-profile`), `--kiosk` for full screen. If no page is connected for 20s, it starts Chrome again. `LYNX_CHROME_ARGS` adds Chrome flags.
  - `Show` is the facade for the loop: `speak`, `is_speaking`, `lines`, `set_state`, `show_frame`, `quit_requested`, `close`. It also runs `caffeinate` so the display stays on.
- `web/`: the page. `index.html` (layout: lynx canvas, camera preview lower right, caption lower left), `lynx.js` (three.js scene), `vendor/` (three.js 0.186.1, see `web/vendor/README.md`).
  - The model stands in its rest pose. `lynx.js` applies the first frame of the clip `Liegen_Idle` (lying pose) and does not play the clip. All movement is procedural on the bones `head`, `neck01`, `jaw`, `ear_base*`, `ear_tip*`.
  - Mouth: the `jaw` bone also moves the nose and the mesh has no mouth cavity. At load time `prepareMouth` moves the skin weights above the lip line to the `head` bone, and a shader patch darkens the lip band while it stretches. Constants at the top of `lynx.js` (`JAW_MAX_ANGLE`, `LIP_LINE`, ...).
  - Lip sync: the jaw follows the envelope on the clock that starts at `speech_start` plus `delay`. `speech_end` or the end of the envelope closes the mouth.
  - Mood: head and ear pose per mood (`MOODS`), no facial expressions. During the analysis the head tilts (`ANALYSING`). Idle: slow look-around, ear flicks. The Santa hat is built in code and attached to the `head` bone. Snow falls behind the head only.
  - The page reconnects by itself: `EventSource` retries, a watchdog reconnects after 15s without a ping, the preview image reloads on error.
  - Debug URL parameters: `open=0..1`, `mood=<name>`, `present=1`, `still=1`, `noevents=1`, and the mouth constants `jaw`, `lip`, `band`, `mx`, `mw`. Keys: `q` quits, `[` and `]` shift the lip sync by 20 ms (stored in the browser profile).
- `lynx_demo.py`: the same steps as the live loop with fake events and a still image as preview. For tuning lines, poses and lip sync.
- `static_images.py` runs DeepFace once on one image.

## Key Dependencies

- `deepface` (>= 0.0.103): pre-trained emotion models, run on `onnxruntime` (CPU, about 6 ms per face). There is no TensorFlow in this project. Weights download to `~/.deepface/weights/` on first run.
- `numpy`: comes with OpenCV and DeepFace. `lynx_voice.py` uses it for the envelope.
- three.js 0.186.1 (MIT) is stored in `web/vendor/`. No CDN, the page must work offline.
- `werkzeug`: DeepFace imports it but does not declare it. It must stay in `requirements.txt`, or `from deepface import DeepFace` fails.
- `opencv-contrib-python-headless`: capture, Haar cascade, drawing on the frame, JPEG encoding for the preview. It is the same package DeepFace pulls in, so one package owns the `cv2` folder. On macOS it has the Cocoa window backend despite the name. OpenCV 5 moved `CascadeClassifier` to contrib, so keep `opencv-python` out of `requirements.txt`: it lacks the cascade and overwrites `cv2`. `make setup` asserts both the cascade and `GUI: COCOA` (the Cocoa check is a leftover from the OpenCV window, the lynx show does not need it). OpenCV 5 wheels ship no cascade XML files (`cv2.data.haarcascades` is empty), so `live_sentiment.py` downloads the face cascade into `~/.deepface/weights/` with DeepFace's `weight_utils`, the same way DeepFace does.

## Configuration Notes

- `detector_backend` in DeepFace: `opencv` (fastest) or `retinaface` (most accurate, slower)
- `enforce_detection=False` prevents crashes on blurry or missing faces
- `silent=True` in `DeepFace.analyze` suppresses DeepFace progress logs
- Exceptions from `DeepFace.analyze` in the live loop are swallowed on purpose. A failed frame adds no sample.
