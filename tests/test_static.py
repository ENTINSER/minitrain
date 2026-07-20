"""Static validation tests that do not execute heavy model code."""
from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _compile(rel_path: str) -> None:
    py_compile.compile(str(ROOT / rel_path), doraise=True)


def test_train_module_compiles() -> None:
    _compile("train/train.py")


def test_evaluate_module_compiles() -> None:
    _compile("evaluate/benchmark.py")


def test_registry_module_compiles() -> None:
    _compile("registry/mlflow_integration.py")


def test_api_module_compiles() -> None:
    _compile("api/main.py")


def test_data_pipeline_module_compiles() -> None:
    _compile("data/pipeline.py")
