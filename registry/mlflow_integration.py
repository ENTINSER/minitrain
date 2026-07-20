"""MLflow integration for the MiniTrain model registry.

This module tracks, registers, and manages model versions in MLflow. It expects
a MiniTrain ``config.yaml`` with a ``registry`` section containing at least
``mlflow_tracking_uri`` and ``model_name``.

Environment variable overrides:
    MLFLOW_TRACKING_URI: overrides ``config.registry.mlflow_tracking_uri``.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import yaml
from mlflow.tracking import MlflowClient


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_file = path or CONFIG_PATH
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_tracking_uri(config: Dict[str, Any]) -> None:
    """Set the MLflow tracking URI from env or config."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or config["registry"]["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(uri)


def _get_client(config: Dict[str, Any]) -> MlflowClient:
    _set_tracking_uri(config)
    return MlflowClient()


def _load_eval_report(eval_report_path: str) -> Dict[str, Any]:
    if not eval_report_path or not Path(eval_report_path).exists():
        return {}
    with open(eval_report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def register_model(
    model_path: str,
    eval_report_path: str,
    config: Optional[Dict[str, Any]] = None,
    training_samples: Optional[int] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Log a model artifact, evaluation metrics, and register it in MLflow.

    Parameters
    ----------
    model_path:
        Path to the model directory or file to register.
    eval_report_path:
        Path to a JSON evaluation report with metrics to log.
    config:
        MiniTrain config dict. If ``None``, ``config.yaml`` is loaded.
    training_samples:
        Number of training samples used to produce the model.
    tags:
        Optional tags to attach to the registered model version.

    Returns
    -------
    A dict with ``run_id`` and ``version``.
    """
    if config is None:
        config = _load_config()

    _set_tracking_uri(config)
    model_name = config["registry"]["model_name"]
    eval_report = _load_eval_report(eval_report_path)

    with mlflow.start_run() as run:
        # Log model hyperparameters and lineage info.
        mlflow.log_param("base_model", config["model"]["base_model"])
        mlflow.log_param("lora_r", config["model"]["lora_r"])
        mlflow.log_param("lora_alpha", config["model"]["lora_alpha"])
        mlflow.log_param("lora_dropout", config["model"]["lora_dropout"])
        mlflow.log_param("num_epochs", config["training"]["num_epochs"])
        mlflow.log_param("learning_rate", config["training"]["learning_rate"])
        mlflow.log_param("batch_size", config["training"]["batch_size"])

        if training_samples is not None:
            mlflow.log_param("training_samples", training_samples)

        # Log evaluation metrics from the report.
        metrics = eval_report.get("metrics", eval_report)
        if isinstance(metrics, dict):
            # Flatten nested metric dicts for MLflow.
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

        # Log the model artifact under the ``model`` artifact path so it can be
        # registered via ``runs:/{run_id}/model``.
        model_artifact_path = Path(model_path)
        if model_artifact_path.is_dir():
            mlflow.log_artifacts(str(model_artifact_path), artifact_path="model")
        elif model_artifact_path.exists():
            mlflow.log_artifact(str(model_artifact_path), artifact_path="model")

        run_id = run.info.run_id

    # Register the model version from the run artifact.
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
        tags=tags,
    )

    return {
        "run_id": run_id,
        "version": model_version.version,
        "model_name": model_name,
        "source": model_version.source,
    }


def promote_model(version: str, stage: str = "Production", config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Promote a registered model version to a target stage.

    Common stages are ``Production``, ``Staging``, and ``Archived``.
    """
    if config is None:
        config = _load_config()

    client = _get_client(config)
    model_name = config["registry"]["model_name"]

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )

    return {"model_name": model_name, "version": version, "stage": stage}


def archive_model(version: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Archive a registered model version."""
    return promote_model(version, stage="Archived", config=config)


def list_registered_models(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """List all registered models known to the configured tracking server."""
    if config is None:
        config = _load_config()

    client = _get_client(config)
    models = client.search_registered_models()
    return [
        {
            "name": model.name,
            "latest_versions": [v.version for v in (model.latest_versions or [])],
        }
        for model in models
    ]


def list_model_versions(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """List versions of the configured MiniTrain model."""
    if config is None:
        config = _load_config()

    client = _get_client(config)
    model_name = config["registry"]["model_name"]
    versions = client.search_model_versions(f"name='{model_name}'")
    return [
        {
            "version": v.version,
            "stage": v.current_stage,
            "status": v.status,
            "run_id": v.run_id,
            "creation_timestamp": v.creation_timestamp,
        }
        for v in versions
    ]


def compare_versions(
    version_a: str,
    version_b: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare two model versions by logged metrics and current stage.

    Returns
    -------
    A dict containing each version's metadata and a diff of shared metrics.
    """
    if config is None:
        config = _load_config()

    client = _get_client(config)
    model_name = config["registry"]["model_name"]

    def _get_version(version: str):
        return client.get_model_version(name=model_name, version=version)

    va = _get_version(version_a)
    vb = _get_version(version_b)

    run_a = client.get_run(va.run_id)
    run_b = client.get_run(vb.run_id)

    metrics_a = dict(run_a.data.metrics)
    metrics_b = dict(run_b.data.metrics)
    shared_keys = set(metrics_a.keys()) & set(metrics_b.keys())

    diff = {
        key: {
            "version_a": metrics_a[key],
            "version_b": metrics_b[key],
            "delta": metrics_b[key] - metrics_a[key],
        }
        for key in shared_keys
    }

    return {
        "model_name": model_name,
        "version_a": {
            "version": va.version,
            "stage": va.current_stage,
            "metrics": metrics_a,
        },
        "version_b": {
            "version": vb.version,
            "stage": vb.current_stage,
            "metrics": metrics_b,
        },
        "metric_diff": diff,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniTrain MLflow registry CLI")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the global MiniTrain config YAML (used by orchestrators to pass the same config to every step).",
    )
    subparsers = parser.add_subparsers(dest="command")

    register_parser = subparsers.add_parser("register", help="Register a new model version")
    register_parser.add_argument("--model-path", required=True)
    register_parser.add_argument("--eval-report", required=True)
    register_parser.add_argument("--training-samples", type=int, default=None)

    promote_parser = subparsers.add_parser("promote", help="Promote a model version")
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument("--stage", default="Production")

    archive_parser = subparsers.add_parser("archive", help="Archive a model version")
    archive_parser.add_argument("--version", required=True)

    list_parser = subparsers.add_parser("list", help="List registered models")
    versions_parser = subparsers.add_parser("versions", help="List model versions")

    compare_parser = subparsers.add_parser("compare", help="Compare two model versions")
    compare_parser.add_argument("--version-a", required=True)
    compare_parser.add_argument("--version-b", required=True)

    args = parser.parse_args()

    if args.command == "register":
        result = register_model(
            model_path=args.model_path,
            eval_report_path=args.eval_report,
            training_samples=args.training_samples,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "promote":
        result = promote_model(args.version, args.stage)
        print(json.dumps(result, indent=2))
    elif args.command == "archive":
        result = archive_model(args.version)
        print(json.dumps(result, indent=2))
    elif args.command == "list":
        print(json.dumps(list_registered_models(), indent=2))
    elif args.command == "versions":
        print(json.dumps(list_model_versions(), indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_versions(args.version_a, args.version_b), indent=2))
    else:
        parser.print_help()
