"""MiniTrain API service.

Exposes REST endpoints for Agent Factory to trigger and monitor fine-tuning
jobs, list registered models, and promote model versions.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEMO_MODE = os.environ.get("TRAINING_DEMO_MODE", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure JSON logging for the API."""
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"levelname": "level", "asctime": "timestamp"},
    )
    log_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(log_handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


_configure_logging()
logger = logging.getLogger("minitrain.api")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

train_jobs_total = Counter(
    "train_jobs_total",
    "Total number of training jobs created",
    ["status"],
)
train_jobs_in_progress = Gauge(
    "train_jobs_in_progress",
    "Number of training jobs currently running",
)
train_job_duration_seconds = Histogram(
    "train_job_duration_seconds",
    "Time spent running a training job",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
CONFIG: dict[str, Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TrainJobRequest(BaseModel):
    """Request payload from Agent Factory to create a training job."""

    trigger_reason: str = Field(..., description="Why the job was triggered")
    error_type: str = Field(..., description="Error type to focus on")
    low_score_samples: int = Field(..., ge=0, description="Number of low-score samples")
    date_range: str = Field(..., description="Date range of the samples")
    current_model: str = Field(..., description="Currently deployed model name")
    current_accuracy: float = Field(..., ge=0.0, le=100.0)
    target_accuracy: float = Field(..., ge=0.0, le=100.0)


class AgentFactoryTriggerRequest(BaseModel):
    """Compatibility payload emitted by Agent Factory evaluation service."""

    error_type: str = Field(..., description="Error type to focus on")
    avg_score: float = Field(..., ge=0.0, le=100.0, description="Current average score")
    sample_count: int = Field(..., ge=0, description="Number of samples in the window")
    trigger_time: str = Field(..., description="ISO timestamp of the trigger")


class TrainJobResponse(BaseModel):
    """Summary of a training job."""

    job_id: str
    status: str
    created_at: str
    updated_at: str
    request: TrainJobRequest
    result: dict[str, Any] | None = None
    logs: list[str] = Field(default_factory=list)


class PromoteRequest(BaseModel):
    """Optional body for promoting a model version."""

    version: str | None = Field(None, description="Version to promote; latest if omitted")


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _run_subprocess(
    module: str,
    config_path: Path,
    subcommand: str | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a project module as a subprocess."""
    cmd = [sys.executable, "-m", module]
    if subcommand:
        cmd.append(subcommand)
    cmd.extend(["--config", str(config_path)])
    if extra_args:
        cmd.extend(extra_args)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    logger.info("Running subprocess", extra={"command": " ".join(cmd)})
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _append_log(job_id: str, message: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["logs"].append(f"[{_now()}] {message}")
        job["updated_at"] = _now()


def _set_status(job_id: str, status: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["status"] = status
        job["updated_at"] = _now()
        train_jobs_total.labels(status=status).inc()


def _run_training_pipeline(job_id: str, request: TrainJobRequest) -> None:
    """Background worker that runs the full training pipeline."""
    start_time = time.time()
    train_jobs_in_progress.inc()
    _set_status(job_id, "running")
    _append_log(job_id, "Training job started")

    try:
        if DEMO_MODE:
            logger.info("Running in TRAINING_DEMO_MODE", extra={"job_id": job_id})
            _append_log(job_id, "DEMO_MODE: skipping real training")
            for step in [
                "data.pipeline",
                "train.train",
                "evaluate.benchmark",
                "registry.mlflow_integration",
            ]:
                _append_log(job_id, f"DEMO_MODE: simulating {step}")
                time.sleep(2)
            _set_status(job_id, "completed")
            _append_log(job_id, "DEMO_MODE: training completed")
            return

        # Resolve artifact paths from the loaded config.
        training_cfg = CONFIG.get("training", {})
        eval_cfg = CONFIG.get("evaluation", {})
        output_dir = Path(training_cfg.get("output_dir", "./outputs"))
        adapter_path = output_dir / "final_adapter"
        eval_report_path = Path(eval_cfg.get("output", "./evaluate/report.json"))

        # Step 1: data pipeline
        _append_log(job_id, "Running data pipeline")
        result = _run_subprocess(
            "data.pipeline",
            CONFIG_PATH,
            extra_args=["--error-type", request.error_type],
        )
        _append_log(job_id, f"data.pipeline stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            _append_log(job_id, f"data.pipeline stderr: {result.stderr.strip()}")
            raise RuntimeError(f"data.pipeline failed with code {result.returncode}")

        # Step 2: training
        _append_log(job_id, "Running training")
        result = _run_subprocess(
            "train.train",
            CONFIG_PATH,
            extra_args=["--job-id", job_id],
        )
        _append_log(job_id, f"train.train stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            _append_log(job_id, f"train.train stderr: {result.stderr.strip()}")
            raise RuntimeError(f"train.train failed with code {result.returncode}")

        # Step 3: evaluation
        _append_log(job_id, "Running evaluation benchmark")
        result = _run_subprocess("evaluate.benchmark", CONFIG_PATH)
        _append_log(job_id, f"evaluate.benchmark stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            _append_log(job_id, f"evaluate.benchmark stderr: {result.stderr.strip()}")
            raise RuntimeError(f"evaluate.benchmark failed with code {result.returncode}")

        # Step 4: registry
        _append_log(job_id, "Registering model with MLflow")
        result = _run_subprocess(
            "registry.mlflow_integration",
            CONFIG_PATH,
            subcommand="register",
            extra_args=[
                "--model-path",
                str(adapter_path),
                "--eval-report",
                str(eval_report_path),
                "--training-samples",
                str(request.low_score_samples),
            ],
        )
        _append_log(job_id, f"registry.mlflow_integration stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            _append_log(job_id, f"registry.mlflow_integration stderr: {result.stderr.strip()}")
            raise RuntimeError(
                f"registry.mlflow_integration failed with code {result.returncode}"
            )

        _set_status(job_id, "completed")
        _append_log(job_id, "Training pipeline completed successfully")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training job failed", extra={"job_id": job_id})
        _set_status(job_id, "failed")
        _append_log(job_id, f"Training job failed: {exc}")
    finally:
        elapsed = time.time() - start_time
        train_job_duration_seconds.observe(elapsed)
        train_jobs_in_progress.dec()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load config and prime MLflow client."""
    global CONFIG
    logger.info("MiniTrain API starting", extra={"demo_mode": DEMO_MODE})
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                CONFIG = yaml.safe_load(fh) or {}
            logger.info("Loaded config.yaml", extra={"config": CONFIG})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load config.yaml", extra={"error": str(exc)})
    else:
        logger.warning("config.yaml not found", extra={"path": str(CONFIG_PATH)})
    yield
    logger.info("MiniTrain API shutting down")


app = FastAPI(
    title="MiniTrain API",
    description="Fine-tuning orchestration service for Agent Factory.",
    version="0.1.0",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "minitrain-api"}


@app.post("/train-jobs", response_model=TrainJobResponse, status_code=202)
async def create_train_job(
    request: TrainJobRequest,
    background_tasks: BackgroundTasks,
) -> TrainJobResponse:
    """Accept a training request from Agent Factory and start a background job."""
    job_id = str(uuid.uuid4())
    now = _now()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "request": request.model_dump(),
            "result": None,
            "logs": [f"[{now}] Job queued"],
        }
    train_jobs_total.labels(status="queued").inc()

    logger.info(
        "Training job created",
        extra={"job_id": job_id, "request": request.model_dump()},
    )

    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()

    return TrainJobResponse(**JOBS[job_id])


@app.post("/training-trigger", response_model=TrainJobResponse, status_code=202)
async def create_train_job_from_agent_factory(
    payload: AgentFactoryTriggerRequest,
    background_tasks: BackgroundTasks,
) -> TrainJobResponse:
    """Compatibility endpoint that accepts Agent Factory's native trigger payload.

    Agent Factory emits ``{error_type, avg_score, sample_count, trigger_time}``.
    This endpoint maps those fields onto the richer ``TrainJobRequest`` schema and
    starts the same background pipeline.
    """
    base_model = os.environ.get(
        "MINITRAIN_CURRENT_MODEL",
        CONFIG.get("model", {}).get("base_model", "unknown"),
    )
    target_accuracy = float(os.environ.get("MINITRAIN_TARGET_ACCURACY", "95.0"))

    request = TrainJobRequest(
        trigger_reason="Agent Factory automatic trigger",
        error_type=payload.error_type,
        low_score_samples=payload.sample_count,
        date_range=payload.trigger_time,
        current_model=base_model,
        current_accuracy=payload.avg_score,
        target_accuracy=target_accuracy,
    )
    logger.info(
        "Mapped Agent Factory trigger to TrainJobRequest",
        extra={"error_type": request.error_type, "current_accuracy": request.current_accuracy},
    )
    return await create_train_job(request, background_tasks)


@app.get("/train-jobs/{job_id}", response_model=TrainJobResponse)
async def get_train_job(job_id: str) -> TrainJobResponse:
    """Return the status and details of a training job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return TrainJobResponse(**job)


@app.get("/models")
async def list_models() -> list[dict[str, Any]]:
    """List MLflow registered model versions."""
    try:
        client = mlflow.tracking.MlflowClient()
        models = []
        for rm in client.search_registered_models():
            for version in client.search_model_versions(f"name='{rm.name}'"):
                models.append(
                    {
                        "name": rm.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "run_id": version.run_id,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                    }
                )
        return models
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list registered models")
        raise HTTPException(
            status_code=503,
            detail=f"MLflow unavailable: {exc}",
        ) from exc


@app.post("/models/{model_name}/promote")
async def promote_model(
    model_name: str,
    body: PromoteRequest | None = None,
) -> dict[str, str]:
    """Promote a model version to the Production stage."""
    try:
        client = mlflow.tracking.MlflowClient()
        if body and body.version:
            version = body.version
        else:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise HTTPException(
                    status_code=404,
                    detail=f"No versions found for model {model_name}",
                )
            version = max(v.version for v in versions)

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Promoted model to Production",
            extra={"model_name": model_name, "version": version},
        )
        return {
            "model_name": model_name,
            "version": str(version),
            "stage": "Production",
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to promote model")
        raise HTTPException(
            status_code=503,
            detail=f"MLflow promotion failed: {exc}",
        ) from exc
