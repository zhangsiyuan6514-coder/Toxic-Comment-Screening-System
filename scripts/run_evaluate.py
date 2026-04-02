"""Evaluate TextCNN checkpoint on a split. Run: python scripts/run_evaluate.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from training.evaluate import main

if __name__ == "__main__":
    main()
