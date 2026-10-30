PYTHON ?= python3.14
VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.installed

.DEFAULT_GOAL := run
.PHONY: setup run static speech test clean

# live_sentiment.py needs cv2.CascadeClassifier and a window backend.
$(STAMP): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import cv2; gui = [l for l in cv2.getBuildInformation().splitlines() if 'GUI:' in l][0]; assert hasattr(cv2, 'CascadeClassifier'), 'cv2 has no CascadeClassifier'; assert 'COCOA' in gui, gui"
	touch $@

setup: $(STAMP)

run: $(STAMP)
	$(PY) live_sentiment.py

static: $(STAMP)
	$(PY) static_images.py

speech: $(STAMP)
	$(PY) speech_test.py

test: $(STAMP)
	$(PY) -m pip install -q -r requirements-dev.txt
	$(PY) -m pytest tests

clean:
	rm -rf $(VENV)
