# README

## Install

Use a venv to isolate

```
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
python3.11 -m pip install -r requirements.txt
```

## Run the Scripts

### Static Image Sentiment Analysis

This script analyzes the sentiment of a face in a static image using DeepFace. By default, it uses the `angryface.jpeg` image for demonstration.

Run:
```
python3.11 static_images.py
```

Expected Output:
- Displays the dominant emotion (e.g., happy, sad, angry) and the confidence percentage.
- Note: Install `tensorflow-metal` for GPU acceleration on Mac.

### Live Sentiment Analysis

This script performs live sentiment analysis using your webcam feed. Detected faces are processed for sentiment in real time.

Run:
```
python3.11 live_sentiment.py
```

Instructions:
- Press `q` to quit.
- Detected emotions will display above faces as text.

> Note: Ensure your camera is accessible and functional.

## Dependencies

Ensure the following libraries are installed (also included in `requirements.txt`):

- `tensorflow-macos` and `tensorflow-metal`: Used for leveraging GPU capabilities on macOS.
- `deepface`: Provides pre-trained models for face detection and sentiment analysis.
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

- This project uses TensorFlow and may require additional setup for GPU acceleration (especially on macOS with M1/M2 chips).
- For faster or more accurate detection, modify the `detector_backend` parameter in `static_images.py` or `live_sentiment.py`. Supported backends include `opencv` (fastest) and `retinaface` (most accurate).
