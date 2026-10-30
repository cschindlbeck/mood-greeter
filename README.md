# README

## Install

Use a venv to isolate

```
python3.14 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
python3.14 -m pip install -r requirements.txt
```

## Run the Scripts

### Static Image Sentiment Analysis

This script analyzes the sentiment of a face in a static image using DeepFace. By default, it uses the `angryface.jpeg` image for demonstration.

Run:
```
python3.14 static_images.py
```

Expected Output:
- Displays the dominant emotion (e.g., happy, sad, angry) and the confidence percentage.

### Live Sentiment Analysis

This script performs live sentiment analysis using your webcam feed. Detected faces are processed for sentiment in real time.

Run:
```
python3.14 live_sentiment.py
```

Instructions:
- Press `q` to quit.
- Detected emotions will display above faces as text.

> Note: Ensure your camera is accessible and functional.

## Dependencies

Ensure the following libraries are installed (also included in `requirements.txt`):

- `deepface`: Provides pre-trained models for face detection and sentiment analysis. Runs on `onnxruntime` (CPU), no TensorFlow needed.
- `werkzeug`: Imported by `deepface` but not declared as its dependency.
- `opencv-python`: Used for live video feed processing.

Install all dependencies:
```
pip install -r requirements.txt
```

## Files Included

- **`static_images.py`**: Performs sentiment analysis on static images.
- **`live_sentiment.py`**: Analyzes facial sentiments in real-time using your camera.
- **`requirements.txt`**: Contains all necessary dependencies.
- **`angryface.jpeg`**: Example image for testing.

## Additional Notes

- `deepface` also installs `opencv-contrib-python-headless`, which has no window support and shares the `cv2` folder with `opencv-python`. If the live window does not open, check that `python -c "import cv2; print(cv2.getBuildInformation())" | grep "GUI:"` shows `COCOA`. If not, run `pip install --force-reinstall --no-deps opencv-python`.
- For faster or more accurate detection, modify the `detector_backend` parameter in `static_images.py` or `live_sentiment.py`. Supported backends include `opencv` (fastest) and `retinaface` (most accurate).
