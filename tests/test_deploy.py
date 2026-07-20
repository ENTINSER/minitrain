"""Tests for gray-release nginx config generation and A/B testing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_nginx_config_generator_creates_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "deploy_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy.nginx_config_generator",
            "--old-endpoint",
            "http://stable:8000",
            "--new-endpoint",
            "http://canary:8000",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config_file = out_dir / "gray_config.conf"
    plan_file = out_dir / "gray_plan.json"
    assert config_file.exists()
    assert plan_file.exists()
    config_text = config_file.read_text()
    assert "stable" in config_text
    assert "canary" in config_text
    plan = json.loads(plan_file.read_text())
    assert len(plan["stages"]) == 3


def test_ab_test_cli_detects_significant_difference() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy.ab_test",
            "--control-success",
            "30",
            "--control-total",
            "100",
            "--treatment-success",
            "50",
            "--treatment-total",
            "100",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "p_value" in data
    assert data["significant"] is True
    assert "treatment" in data["recommendation"]
