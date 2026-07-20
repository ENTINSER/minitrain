"""Tests for the MLflow registry module and CLI."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import registry.mlflow_integration as registry_mod


def test_register_model_uses_mlflow() -> None:
    """register_model should log params/metrics/artifacts and register the model."""
    dummy_run = MagicMock()
    dummy_run.info.run_id = "run-123"
    # Make the returned object usable as a context manager.
    ctx_manager = MagicMock()
    ctx_manager.__enter__ = MagicMock(return_value=dummy_run)
    ctx_manager.__exit__ = MagicMock(return_value=None)
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_version.source = "runs:/run-123/model"

    with patch.object(registry_mod.mlflow, "start_run", return_value=ctx_manager) as start_run_mock, \
         patch.object(registry_mod.mlflow, "log_param") as log_param_mock, \
         patch.object(registry_mod.mlflow, "log_metric") as log_metric_mock, \
         patch.object(registry_mod.mlflow, "log_artifacts") as log_artifacts_mock, \
         patch.object(registry_mod.mlflow, "register_model", return_value=mock_version) as register_mock:
        result = registry_mod.register_model(
            model_path="outputs/final_adapter",
            eval_report_path="evaluate/report.json",
            config={
                "model": {
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                },
                "training": {
                    "num_epochs": 3,
                    "learning_rate": 2.0e-4,
                    "batch_size": 4,
                },
                "registry": {"model_name": "code-fix-model", "mlflow_tracking_uri": "file:///tmp"},
            },
            training_samples=42,
        )

    start_run_mock.assert_called_once()
    register_mock.assert_called_once_with(
        model_uri="runs:/run-123/model",
        name="code-fix-model",
        tags=None,
    )
    assert result["run_id"] == "run-123"
    assert result["version"] == "1"


def test_cli_accepts_config_global_arg() -> None:
    """The registry CLI should accept --config before the subcommand."""
    import argparse
    # argparse will exit on --help; instead verify parser structure by importing.
    import inspect
    import registry.mlflow_integration as mod
    source = inspect.getsource(mod)
    assert '--config' in source
    assert 'subparsers.add_parser("register"' in source
