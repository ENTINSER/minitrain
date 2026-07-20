"""Shared pytest fixtures and configuration for MiniTrain."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Avoid starting real GPU training when the API is exercised in tests.
os.environ.setdefault("TRAINING_DEMO_MODE", "true")
