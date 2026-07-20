#!/usr/bin/env python3
"""MiniTrain LoRA fine-tuning entrypoint.

Supports DeepSpeed ZeRO-3 multi-GPU training on Linux and automatically falls
back to single-device PyTorch training on macOS / environments without DeepSpeed.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml

# Optional dependencies are imported lazily or guarded so that import errors are
# reported with actionable hints.
try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required. Install it with: pip install torch"
    ) from exc

try:
    from datasets import Dataset, load_dataset
    from peft import get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
        set_seed,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing training dependencies. Install them with:\n"
        "  pip install datasets peft transformers"
    ) from exc

from lora_config import get_lora_config

logger = logging.getLogger("minitrain")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM with LoRA for MiniTrain."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the global MiniTrain config YAML.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to a DeepSpeed config JSON. Overrides the value from config.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Optional job ID for tracking (used as MLflow run name if MLflow is enabled).",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    """Load the global config YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging for the training run."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def deepspeed_is_available() -> bool:
    """Return True if DeepSpeed can be imported, False otherwise."""
    try:
        import deepspeed  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


def load_json_dataset(path: Path) -> Dataset:
    """Load a JSON/JSONL dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return load_dataset("json", data_files=str(path), split="train")
    return load_dataset("json", data_files=str(path), split="train")


def build_prompt(example: dict[str, Any]) -> str:
    """Convert an instruction/response example into a training prompt.

    The function is tolerant of different field names used by preprocessing:
    ``instruction``/``input`` for the user content and ``response``/``output``
    for the assistant content.
    """
    instruction = example.get("instruction") or example.get("input") or ""
    response = example.get("response") or example.get("output") or ""

    if not instruction or not response:
        return ""

    # Use a simple chat-style template. Models such as Qwen2.5-Instruct expect
    # explicit roles; the tokenizer's chat template is preferred when present.
    return (
        f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{response.strip()}<|im_end|>"
    )


def tokenize_example(
    example: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> dict[str, Any]:
    """Tokenize a single example and attach labels for causal LM training."""
    prompt = build_prompt(example)
    if not prompt:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dataset:
    """Tokenize a Hugging Face dataset for supervised fine-tuning."""
    return dataset.map(
        lambda ex: tokenize_example(ex, tokenizer, max_seq_length),
        batched=False,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )


def log_config_to_mlflow(config: dict[str, Any]) -> None:
    """Log relevant hyper-parameters to the active MLflow run."""
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        logger.warning("MLflow not installed; skipping parameter logging.")
        return

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    params = {
        "base_model": model_cfg.get("base_model"),
        "lora_r": model_cfg.get("lora_r"),
        "lora_alpha": model_cfg.get("lora_alpha"),
        "lora_dropout": model_cfg.get("lora_dropout"),
        "num_epochs": training_cfg.get("num_epochs"),
        "learning_rate": training_cfg.get("learning_rate"),
        "batch_size": training_cfg.get("batch_size"),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
        "warmup_ratio": training_cfg.get("warmup_ratio"),
        "weight_decay": training_cfg.get("weight_decay"),
        "max_seq_length": training_cfg.get("max_seq_length"),
        "output_dir": training_cfg.get("output_dir"),
        "data_output_dir": data_cfg.get("output_dir"),
    }

    # Drop None values so the MLflow UI stays clean.
    params = {k: v for k, v in params.items() if v is not None}
    mlflow.log_params(params)


def main() -> int:
    """Run the LoRA fine-tuning pipeline."""
    args = parse_args()
    config = load_config(args.config)

    log_cfg = config.get("logging", {})
    setup_logging(log_cfg.get("level", "INFO"))

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    registry_cfg = config.get("registry", {})

    base_model_name = model_cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    output_dir = Path(training_cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)

    # ------------------------------------------------------------------
    # DeepSpeed detection and fallback
    # ------------------------------------------------------------------
    deepspeed_config_path = args.deepspeed or training_cfg.get("deepspeed_config")
    use_deepspeed = False
    if deepspeed_config_path:
        deepspeed_config_path = Path(deepspeed_config_path)
        if not deepspeed_config_path.is_absolute():
            deepspeed_config_path = Path(args.config).parent / deepspeed_config_path

        if not deepspeed_is_available():
            warnings.warn(
                "DeepSpeed is not available (common on macOS or when not installed). "
                "Falling back to single-device PyTorch training. "
                "The --deepspeed config path is retained for compatibility but will not be used.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "DeepSpeed not available; running on %s with PyTorch backend.",
                "MPS" if torch.backends.mps.is_available()
                else "CUDA" if torch.cuda.is_available()
                else "CPU",
            )
        else:
            use_deepspeed = True
            logger.info("DeepSpeed enabled with config: %s", deepspeed_config_path)

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    mlflow_run = None
    try:
        import mlflow
        tracking_uri = registry_cfg.get("mlflow_tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = registry_cfg.get("model_name", "minitrain")
        mlflow.set_experiment(experiment_name)
        run_name = args.job_id if args.job_id else None
        mlflow_run = mlflow.start_run(run_name=run_name)
        if run_name:
            logger.info("Using MLflow run name: %s", run_name)
        log_config_to_mlflow(config)
        logger.info("MLflow run started: %s", mlflow_run.info.run_id)
    except ImportError:
        logger.warning("MLflow not installed; metrics will not be logged externally.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not start MLflow run: %s", exc)

    try:
        # --------------------------------------------------------------
        # Tokenizer & model
        # --------------------------------------------------------------
        logger.info("Loading tokenizer and model: %s", base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map=None if use_deepspeed else "auto",
        )

        lora_config = get_lora_config(base_model_name, config)
        logger.info("LoRA config: r=%d alpha=%d dropout=%.3f targets=%s",
                    lora_config.r, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.target_modules)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # --------------------------------------------------------------
        # Datasets
        # --------------------------------------------------------------
        data_output_dir = Path(data_cfg.get("output_dir", "./processed_data"))
        train_path = data_output_dir / "train.json"
        val_path = data_output_dir / "val.json"

        logger.info("Loading datasets from %s", data_output_dir)
        train_dataset = prepare_dataset(
            load_json_dataset(train_path), tokenizer,
            training_cfg.get("max_seq_length", 1024),
        )
        eval_dataset = prepare_dataset(
            load_json_dataset(val_path), tokenizer,
            training_cfg.get("max_seq_length", 1024),
        )
        logger.info("Train examples: %d | Validation examples: %d",
                    len(train_dataset), len(eval_dataset))

        # --------------------------------------------------------------
        # Training arguments
        # --------------------------------------------------------------
        num_epochs = training_cfg.get("num_epochs", 3)
        batch_size = training_cfg.get("batch_size", 4)
        gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 4)
        max_steps = (len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=training_cfg.get("learning_rate", 2.0e-4),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            logging_steps=training_cfg.get("logging_steps", 10),
            save_steps=training_cfg.get("save_steps", 100),
            eval_steps=training_cfg.get("eval_steps", 100),
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
            deepspeed=str(deepspeed_config_path) if use_deepspeed else None,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------
        logger.info("Starting training for %d epochs (estimated %d steps)",
                    num_epochs, max_steps)
        train_result = trainer.train()
        logger.info("Training finished. Final loss: %.4f",
                    train_result.training_loss)

        # --------------------------------------------------------------
        # Save adapter and log metrics
        # --------------------------------------------------------------
        adapter_dir = output_dir / "final_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info("Adapter saved to %s", adapter_dir)

        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float))})
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to log training metrics to MLflow: %s", exc)

        return 0

    except Exception as exc:  # pragma: no cover
        logger.exception("Training failed: %s", exc)
        return 1

    finally:
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.end_run()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to end MLflow run: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
