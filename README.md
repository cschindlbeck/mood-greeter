# README

## Install

Requires Python 3.14 (`brew install python@3.14`).

```
make setup
```

This creates `.venv`, installs `requirements.txt`, and checks that OpenCV has the Haar cascade and window support. It also builds the audio player for the lip sync (needs `swiftc` from the Xcode command line tools) and fetches the lynx model.

The lynx model is LYNQTECH property and is not in this public repo. `make setup` fetches it with the GitHub CLI from a private repo. You need `gh auth login` with access to that repo. Without access, the page shows no lynx. `make clean` deletes `.venv`.

## Run the Scripts

### Static Image Sentiment Analysis

This script analyzes the sentiment of a face in a static image using DeepFace. By default, it uses the `angryface.jpeg` image for demonstration.

Run:
```
make static
```

Expected Output:
- Displays the dominant emotion (e.g., happy, sad, angry) and the confidence percentage.

### Live Sentiment Analysis with the Christmas Lynx

This script reads your mood through the webcam. A browser window in full screen shows the LYNQTECH lynx with a Santa hat. The lynx welcomes each person to the LYNQTECH Christmas party, reads the mood, and answers with a Christmas line. Its mouth moves with the voice. A small camera preview sits in the lower right corner.

Run:
```
make run
```

Or double-click `Live Sentiment.command` in Finder.

Instructions:
- Quit with Ctrl+C in the terminal, or press `q` in the lynx window.
- `make run ARGS=--window` opens a large window instead of full screen. Other options: `--no-browser`, `--port N`.
- While the app reads your mood, the camera preview shows a progress bar under your face and the share of each emotion next to it.
- When nobody is there, the lynx sits idle and does not speak.
- The window opens in Google Chrome. Without Chrome, the default browser opens the page (no full screen).
- If the window is closed, the app opens it again after 20 seconds.
- No images and no mood data are stored. The page is served on this Mac only (`127.0.0.1`).

> Note: Ensure your camera is accessible and functional. If the camera is missing, the lynx stays idle and the app retries.

Before the party:
- Run `make run` once with internet. The first run downloads the model weights. After that no internet is needed.
- Keep the Mac on power. The app keeps the display awake while it runs.
- Check the volume and the voice with `make speech`.

### Lynx Demo without a Camera

This script shows the lynx with fake events and real speech. Use it to tune the lines, the poses and the lip sync at a desk.

Run:
```
make demo
make demo ARGS="--moods happy,sad --once"
```

- Press `[` or `]` in the window to shift the lip sync by 20 ms. The page shows the value and keeps it.
- Change the spoken lines in `lynx_lines.py`. Change the voice and the speaking rate in `lynx_voice.py` (`VOICE`, `RATE`).
- Hold a pose for inspection with URL parameters, for example `http://127.0.0.1:8765/?mood=sad&open=0.6&still=1`.

## Dependencies

Ensure the following libraries are installed (also included in `requirements.txt`):

- `deepface`: Provides pre-trained models for face detection and sentiment analysis. Runs on `onnxruntime` (CPU), no TensorFlow needed.
- `werkzeug`: Imported by `deepface` but not declared as its dependency.
- `opencv-contrib-python-headless`: Used for the live video feed, face detection (Haar cascade), and the camera preview.

Not from `requirements.txt`:

- three.js 0.186.1 (MIT license) renders the lynx. It is stored in `web/vendor/`, see `web/vendor/README.md`.
- `lynx_player.swift` is a small audio player for the lip sync. `make setup` builds it with `swiftc` (Xcode command line tools). Without it the app uses `afplay` and the lip sync is less exact.
- macOS `say` speaks the lines. Google Chrome shows the lynx window.

Install all dependencies:
```
make setup
```

## Files Included

- **`static_images.py`**: Performs sentiment analysis on static images.
- **`live_sentiment.py`**: Analyzes facial sentiments in real-time using your camera and drives the lynx show.
- **`lynx_demo.py`**: The lynx show with fake events, no camera.
- **`lynx_lines.py`**: The spoken lines.
- **`lynx_voice.py`**: Speech queue and lip sync data.
- **`lynx_show.py`**: Local web server, events, camera preview, browser start.
- **`lynx_player.swift`**: Audio player that reports when the sound starts.
- **`web/`**: The lynx page (three.js).
- **`assets/`**: Place of the lynx model. The model is not in git, see `assets/README.md`.
- **`tests/`**: Tests for the lines, the voice and the show server (`make test`).
- **`requirements.txt`**: Contains all necessary dependencies.
- **`Makefile`**: Setup and run targets (`setup`, `run`, `demo`, `static`, `speech`, `test`, `clean`).
- **`angryface.jpeg`**: Example image for testing.

## Additional Notes

- Keep `opencv-contrib-python-headless` as the only OpenCV package. OpenCV 5 moved `CascadeClassifier` to the contrib build, so `opencv-python` breaks `live_sentiment.py` when it overwrites the `cv2` folder.
- For faster or more accurate detection, modify the `detector_backend` parameter in `static_images.py` or `live_sentiment.py`. Supported backends include `opencv` (fastest) and `retinaface` (most accurate).
