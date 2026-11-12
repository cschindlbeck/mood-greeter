PYTHON ?= python3.14
VENV := .venv
PY := $(VENV)/bin/python
STAMP := $(VENV)/.installed

.DEFAULT_GOAL := run
.PHONY: setup run demo static speech test clean

PLAYER := $(VENV)/bin/lynx_player

# The lynx model is company property and this repo is public, so the file is not in git.
# `make setup` fetches it from the private repo. This needs `gh auth login` with access to it.
MODEL := assets/luchs-emerald-liegend.glb
MODEL_REPO ?= lynqtech/website
MODEL_REF ?= develop
MODEL_PATH ?= docs/Emerald_Lynx_Rig_Read/luchs-emerald-liegend.glb

# live_sentiment.py needs cv2.CascadeClassifier and a window backend.
$(STAMP): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import cv2; gui = [l for l in cv2.getBuildInformation().splitlines() if 'GUI:' in l][0]; assert hasattr(cv2, 'CascadeClassifier'), 'cv2 has no CascadeClassifier'; assert 'COCOA' in gui, gui"
	touch $@

# The player reports when the sound starts, for the lip sync. It needs swiftc (Xcode
# command line tools). Without it the build is skipped and lynx_voice.py uses afplay.
$(PLAYER): lynx_player.swift $(STAMP)
	swiftc -O lynx_player.swift -o $@ || echo "swiftc failed: no exact lip sync, afplay is used"

$(MODEL):
	gh api "repos/$(MODEL_REPO)/contents/$(MODEL_PATH)?ref=$(MODEL_REF)" -H "Accept: application/vnd.github.raw" > $@.tmp \
		&& mv $@.tmp $@ \
		|| { rm -f $@.tmp; echo "Cannot fetch the lynx model. Run 'gh auth login', or copy the file to $@ by hand."; exit 1; }

setup: $(STAMP) $(PLAYER) $(MODEL)

# ARGS: --window (no full screen), --no-browser, --port N
run: setup
	$(PY) live_sentiment.py $(ARGS)

# The lynx page with fake events and real speech, no camera. ARGS as above, plus --moods a,b and --once
demo: setup
	$(PY) lynx_demo.py --window $(ARGS)

static: $(STAMP)
	$(PY) static_images.py

speech: setup
	$(PY) speech_test.py

test: $(STAMP)
	$(PY) -m pip install -q -r requirements-dev.txt
	$(PY) -m pytest tests

clean:
	rm -rf $(VENV)
