"""Tests for the MiniTrain data cleaning pipeline."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _write_config(tmp_path: Path, source_path: Path) -> Path:
    """Create a temporary config that writes outputs into tmp_path."""
    cfg = {
        "model": {"base_model": "dummy"},
        "training": {"output_dir": str(tmp_path / "outputs")},
        "data": {
            "source_type": "file",
            "source_path": str(source_path),
            "output_dir": str(tmp_path / "processed_data"),
            "min_instruction_length": 10,
            "max_code_lines": 200,
            "quality_threshold": 0.5,
            "train_val_test_split": [0.8, 0.1, 0.1],
            "dedup_threshold": 0.8,
        },
        "evaluation": {"test_set_path": "evaluate/test_set/sample_test_cases.json"},
        "registry": {"mlflow_tracking_uri": "file:///tmp", "model_name": "dummy"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_pipeline_runs_without_error_type_filter(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, ROOT / "data" / "sample_input.json")
    result = subprocess.run(
        [sys.executable, "-m", "data.pipeline", "--config", str(cfg_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["counts"]["raw"] == 10
    assert report["counts"]["train"] > 0


def test_pipeline_filters_by_error_type(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, ROOT / "data" / "sample_input.json")
    result = subprocess.run(
        [sys.executable, "-m", "data.pipeline", "--config", str(cfg_path), "--error-type", "off_by_one"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["counts"]["raw"] == 4
    assert report["counts"]["train"] > 0
