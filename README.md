# README

## Install

Requires Python 3.14 (`brew install python@3.14`).

```
make setup
```

This creates `.venv`, installs `requirements.txt`, and checks that OpenCV has the Haar cascade and window support. `make clean` deletes `.venv`.

## Run the Scripts

### Static Image Sentiment Analysis

This script analyzes the sentiment of a face in a static image using DeepFace. By default, it uses the `angryface.jpeg` image for demonstration.

Run:
```
make static
```

Expected Output:
- Displays the dominant emotion (e.g., happy, sad, angry) and the confidence percentage.

### Live Sentiment Analysis

This script performs live sentiment analysis using your webcam feed. Detected faces are processed for sentiment in real time.

Run:
```
make run
```

Or double-click `Live Sentiment.command` in Finder.

Instructions:
- Press `q` to quit.
- Detected emotions will display above faces as text.

> Note: Ensure your camera is accessible and functional.

## Dependencies

Ensure the following libraries are installed (also included in `requirements.txt`):

- `deepface`: Provides pre-trained models for face detection and sentiment analysis. Runs on `onnxruntime` (CPU), no TensorFlow needed.
- `werkzeug`: Imported by `deepface` but not declared as its dependency.
- `opencv-contrib-python-headless`: Used for the live video feed, face detection (Haar cascade), and the display window. On macOS this build has window support.

Install all dependencies:
```
make setup
```

## Files Included

- **`static_images.py`**: Performs sentiment analysis on static images.
- **`live_sentiment.py`**: Analyzes facial sentiments in real-time using your camera.
- **`requirements.txt`**: Contains all necessary dependencies.
- **`Makefile`**: Setup and run targets (`setup`, `run`, `static`, `speech`, `clean`).
- **`angryface.jpeg`**: Example image for testing.

## Additional Notes

- Keep `opencv-contrib-python-headless` as the only OpenCV package. OpenCV 5 moved `CascadeClassifier` to the contrib build, so `opencv-python` breaks `live_sentiment.py` when it overwrites the `cv2` folder.
- For faster or more accurate detection, modify the `detector_backend` parameter in `static_images.py` or `live_sentiment.py`. Supported backends include `opencv` (fastest) and `retinaface` (most accurate).
