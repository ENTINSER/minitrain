# MiniTrain

A small, opinionated LLM fine-tuning pipeline for code-fix models. MiniTrain is triggered by **Agent Factory** when monitored model performance degrades, runs data cleaning → LoRA fine-tuning → evaluation → model registration → gray release, and exposes a FastAPI orchestration service.

---

## 1. Project Overview & Goals

**MiniTrain** fine-tunes lightweight adapters (LoRA) on top of large causal LMs such as `Qwen/Qwen2.5-7B-Instruct` so that Agent Factory can self-heal from recurring error patterns.

Goals:

- **Automated retraining**: accept a trigger from Agent Factory and run the full pipeline unattended.
- **Cost efficiency**: use parameter-efficient fine-tuning (LoRA) and optional DeepSpeed ZeRO-3 on Linux GPU clusters.
- **Quality gate**: deduplicate, filter, and score training data before it reaches the model.
- **Traceability**: log every run, metric, and artifact to MLflow.
- **Safe rollout**: shift traffic gradually with nginx-backed gray releases and validate with A/B tests.
- **Developer friendly**: run end-to-end in Docker Compose or on a laptop with a CPU/MPS fallback.

---

## 2. Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Agent Factory                                   │
│  (detects accuracy drop / recurring error type)                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ POST /train-jobs
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MiniTrain API (api/main.py)                          │
│   • Accepts training job requests                                            │
│   • Orchestrates data → train → eval → registry pipeline                     │
│   • Exposes /health, /train-jobs, /models, /metrics                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ subprocess calls
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  data.pipeline ──► train.train ──► evaluate.benchmark ──► registry.mlflow   │
│                                                                              │
│  • Load raw data (file/PostgreSQL)   • LoRA fine-tuning       • Pass/fail   │
│  • MinHash dedup                     • DeepSpeed or PyTorch     • MLflow     │
│  • Quality/outlier filter            • Save final adapter       • register   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ registered model
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  deploy.nginx_config_generator          deploy.ab_test                       │
│  • Gray-release upstream weights        • Fisher / chi-squared test          │
│  • Stage plan (10% → 50% → 100%)        • Effect size & recommendation       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Model serving endpoints                              │
│            (existing stable endpoint + new model endpoint)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```text
minitrain/
├── api/                          # FastAPI orchestration service
│   ├── main.py                   # /train-jobs, /models, /health, /metrics
│   ├── Dockerfile
│   ├── requirements.txt
│   └── tests/test_api.py
├── data/                         # Data ingestion and cleaning
│   ├── pipeline.py               # Load, dedup, filter, split, export
│   ├── quality.py                # Quality-score heuristics
│   └── sample_input.json
├── deploy/                       # Gray release & A/B testing
│   ├── nginx_config_generator.py # Generate nginx upstream configs
│   ├── ab_test.py                # Statistical comparison helper
│   ├── gray_config.conf          # Generated nginx snippet
│   └── gray_plan.json            # Generated rollout plan
├── evaluate/                     # Benchmarking
│   ├── benchmark.py              # Run model on test cases, produce report
│   └── test_set/sample_test_cases.json
├── registry/                     # MLflow model registry
│   └── mlflow_integration.py     # Register, promote, compare versions
├── train/                        # Fine-tuning
│   ├── train.py                  # LoRA fine-tuning entrypoint
│   ├── lora_config.py            # LoRA config builder
│   └── deepspeed_config.json     # DeepSpeed ZeRO-3 config
├── config.yaml                   # Central configuration
├── docker-compose.yml            # MLflow + API services
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 4. Installation

### Requirements

- **Python 3.11+**
- **Linux** is recommended for DeepSpeed multi-GPU training.
- macOS and Windows work for development but fall back to single-device PyTorch.
- For CUDA training: a compatible NVIDIA driver + CUDA toolkit.

### Quick Install

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins `deepspeed; sys_platform == "linux"`, so DeepSpeed is installed only on Linux. On macOS it is skipped automatically.

---

## 5. Configuration

All pipeline stages read from `config.yaml`. Key sections:

| Section | Purpose | Key fields |
|---------|---------|------------|
| `model` | Base model & LoRA hyperparameters | `base_model`, `lora_r`, `lora_alpha`, `lora_dropout` |
| `training` | Fine-tuning schedule & hardware | `num_epochs`, `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `output_dir`, `deepspeed_config`, `fallback_to_single_gpu` |
| `data` | Data source & cleaning rules | `source_type`, `source_path`, `postgres_*`, `quality_threshold`, `train_val_test_split`, `dedup_threshold` |
| `evaluation` | Acceptance thresholds | `test_set_path`, `pass_threshold.accuracy`, `pass_threshold.code_quality`, `pass_threshold.perplexity_max` |
| `registry` | MLflow model registry | `mlflow_tracking_uri`, `model_name` |
| `deploy` | Gray-release schedule | `gray_stages` (list of `traffic_percent` + `duration`) |
| `api` | API bind address | `host`, `port` |
| `logging` | Log format & level | `level`, `format` |

PostgreSQL credentials can be overridden via environment variables:

```bash
export POSTGRES_HOST=db.example.com
export POSTGRES_PORT=5432
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=secret
export POSTGRES_DB=agent_factory
```

---

## 6. Step-by-Step Usage

### a. Data Pipeline

Load raw records, deduplicate with MinHash + LSH, filter by quality and code length, then split into `train/val/test` JSON files.

```bash
python -m data.pipeline --config config.yaml
```

Outputs:

- `processed_data/train.json`
- `processed_data/val.json`
- `processed_data/test.json`
- `processed_data/data_quality_report.json`

### b. Training

Fine-tune a LoRA adapter on top of the configured base model.

```bash
python -m train.train --config config.yaml
```

On Linux with DeepSpeed installed, `train/deepspeed_config.json` is used automatically. On macOS or when DeepSpeed is unavailable, training falls back to single-device PyTorch with `device_map="auto"`.

Outputs:

- `outputs/final_adapter/` — LoRA adapter + tokenizer
- `outputs/checkpoint-*/` — intermediate checkpoints

### c. Evaluation

Run the benchmark on the saved adapter.

```bash
python -m evaluate.benchmark \
  --model_path outputs/final_adapter \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --test_set evaluate/test_set/sample_test_cases.json
```

Produces `evaluate/report.json` with accuracy, code-quality, and perplexity metrics.

### d. Model Registry

Register the adapter with MLflow and attach the evaluation report.

```bash
python -m registry.mlflow_integration \
  --config config.yaml \
  --register \
  --model_path outputs/final_adapter \
  --eval_report_path outputs/eval_report.json
```

> **Note:** The current `registry.mlflow_integration` CLI uses subcommands and slightly different flag names:
>
> ```bash
> python -m registry.mlflow_integration register \
>   --model-path outputs/final_adapter \
>   --eval-report outputs/eval_report.json
> ```
>
> If you need the single-command form shown above, wrap it in a small shell alias or script.

Other useful registry commands:

```bash
# Promote version 3 to Production
python -m registry.mlflow_integration promote --version 3

# List versions
python -m registry.mlflow_integration versions

# Compare two versions
python -m registry.mlflow_integration compare --version-a 2 --version-b 3
```

### e. Gray Release

Generate an nginx upstream config that gradually shifts traffic to the new model.

```bash
python -m deploy.nginx_config_generator --config config.yaml
```

> **Note:** The generator needs the old and new endpoints. The current CLI requires:
>
> ```bash
> python -m deploy.nginx_config_generator \
>   --old-endpoint 127.0.0.1:8001 \
>   --new-endpoint 127.0.0.1:8002
> ```
>
> It reads the stage schedule from `config.deploy.gray_stages` and writes `deploy/gray_config.conf` and `deploy/gray_plan.json`.

Example generated plan:

```text
Stage 1: 10% new model for 2h   (old weight 90, new weight 10)
Stage 2: 50% new model for 6h   (old weight 50, new weight 50)
Stage 3: 100% new model steady  (old weight 0,  new weight 100)
```

### f. A/B Test

Compare success rates between the current (control) and new (treatment) models.

```bash
python -m deploy.ab_test 100 1000 120 1000
```

Arguments are interpreted as `control_success control_total treatment_success treatment_total`.

> **Note:** The current `ab_test.py` CLI uses named arguments:
>
> ```bash
> python -m deploy.ab_test \
>   --control-success 100 --control-total 1000 \
>   --treatment-success 120 --treatment-total 1000
> ```
>
> It automatically picks Fisher's exact test for low counts and chi-squared otherwise, returning p-value, effect size, and a recommendation.

### g. API Server

Start the orchestration API locally:

```bash
cd api && uvicorn main:app --host 0.0.0.0 --port 8001
```

For quick demos or CI, skip real GPU training and simulate the pipeline:

```bash
cd api && TRAINING_DEMO_MODE=true uvicorn main:app --host 0.0.0.0 --port 8001
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health |
| POST | `/train-jobs` | Create a training job |
| GET | `/train-jobs/{job_id}` | Get job status & logs |
| GET | `/models` | List registered MLflow models |
| POST | `/models/{model_name}/promote` | Promote a version to Production |
| GET | `/metrics` | Prometheus metrics |

---

## 7. Docker Compose Usage

The included `docker-compose.yml` spins up MLflow and the MiniTrain API.

```bash
docker compose up --build
```

Services:

- **MLflow** on `http://localhost:5000`
- **MiniTrain API** on `http://localhost:8001` (with `TRAINING_DEMO_MODE=true`)

You can then post a training job to `http://localhost:8001/train-jobs`.

---

## 8. Integration with Agent Factory

When Agent Factory detects degraded performance, it creates a training job by posting to the MiniTrain API.

### `POST /train-jobs` Payload

```json
{
  "trigger_reason": "Accuracy dropped below threshold on TypeError fixes",
  "error_type": "TypeError",
  "low_score_samples": 320,
  "date_range": "2026-07-01/2026-07-15",
  "current_model": "code-fix-model",
  "current_accuracy": 72.5,
  "target_accuracy": 85.0
}
```

### How It Triggers Retraining

1. Agent Factory collects low-score interactions for a specific error type.
2. It sends the payload to `POST /train-jobs`.
3. The API returns `202 Accepted` with a `job_id`.
4. A background thread runs:
   - `data.pipeline` (filtered to the requested `error_type`)
   - `train.train`
   - `evaluate.benchmark`
   - `registry.mlflow_integration`
5. Agent Factory polls `GET /train-jobs/{job_id}` until the status is `completed` or `failed`.
6. If evaluation passes the thresholds in `config.evaluation.pass_threshold`, Agent Factory can call `POST /models/{model_name}/promote` to move the new version to Production.

---

## 9. CI / GitHub Actions

A sample workflow is shown below. Save it as `.github/workflows/minitrain-ci.yml`.

```yaml
name: MiniTrain CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r api/requirements.txt

      - name: Lint
        run: |
          pip install pylint
          pylint data train evaluate registry deploy api/main.py

      - name: Unit tests
        run: |
          pip install pytest
          pytest api/tests

      - name: Data pipeline smoke test
        run: python -m data.pipeline --config config.yaml
```

Full GPU training is too slow for standard CI; run it on a self-hosted runner with CUDA or rely on `TRAINING_DEMO_MODE=true` for integration tests.

---

## 10. Notes on Mac Compatibility

MiniTrain can be developed and smoke-tested on macOS, but full DeepSpeed multi-GPU training is not available.

- `requirements.txt` uses `deepspeed; sys_platform == "linux"`, so `pip install` skips DeepSpeed on macOS.
- `train.py` detects missing DeepSpeed at runtime and automatically falls back to single-device PyTorch.
- On Apple Silicon, PyTorch uses the **MPS** backend if available; otherwise it runs on CPU.
- For local development, set `TRAINING_DEMO_MODE=true` when running the API to avoid long model downloads and training runs.

```bash
cd api
TRAINING_DEMO_MODE=true LOG_LEVEL=INFO uvicorn main:app --host 0.0.0.0 --port 8001
```

---

## License

This project is internal to Agent Factory. Update the license field once an open-source license is chosen.
