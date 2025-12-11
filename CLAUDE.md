# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mood Greeter is a Python-based facial sentiment analysis application using DeepFace and TensorFlow. It provides two modes: static image analysis and real-time webcam analysis.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install -r requirements.txt
```

## Running the Scripts

**Static image analysis:**
```bash
python3.11 static_images.py
```

**Live webcam sentiment analysis:**
```bash
python3.11 live_sentiment.py
```
Press `q` to quit the live feed.

## Architecture

- `static_images.py` - Analyzes sentiment from a single image file using DeepFace. Uses TensorFlow with optional Metal GPU acceleration on macOS.
- `live_sentiment.py` - Real-time webcam sentiment analysis. Uses OpenCV's Haar cascade for fast face detection, then DeepFace for emotion analysis on detected faces.

## Key Dependencies

- **TensorFlow** (`tensorflow-macos` + `tensorflow-metal`): ML backend with macOS GPU support
- **DeepFace**: Pre-trained facial analysis models
- **OpenCV**: Video capture and image processing

## Configuration Notes

- The `detector_backend` parameter in DeepFace controls detection speed vs accuracy: `opencv` (fastest) or `retinaface` (most accurate)
- `enforce_detection=False` prevents crashes on blurry/missing faces
