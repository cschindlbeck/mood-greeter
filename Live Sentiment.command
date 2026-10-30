#!/bin/zsh
cd "$(dirname "${0:A}")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv found. Do the setup in AGENTS.md first."
  exit 1
fi
exec .venv/bin/python live_sentiment.py
