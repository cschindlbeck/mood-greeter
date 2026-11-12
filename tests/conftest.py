import sys
from pathlib import Path

# The scripts live in the repo root (flat layout, no package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
