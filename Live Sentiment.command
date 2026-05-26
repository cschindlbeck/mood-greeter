#!/bin/zsh
cd "$(dirname "${0:A}")"
source .venv/bin/activate
python3.11 live_sentiment.py
