#!/usr/bin/env python3
"""MiniTrain evaluation benchmark for code-fix models."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a code-fix model on the MiniTrain test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the global MiniTrain config YAML. When provided, model_path, base_model and test_set are inferred from it unless explicitly overridden.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the fine-tuned model or LoRA adapter. Required unless --config is provided.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path (used when model_path is a LoRA adapter).",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="evaluate/test_set/sample_test_cases.json",
        help="Path to the JSON file containing test cases.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluate/report.json",
        help="Path where the JSON evaluation report will be written.",
    )
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default=None,
        help="Optional baseline model path for comparison.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for each fix.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--test_timeout",
        type=int,
        default=10,
        help="Timeout in seconds for executing each test case.",
    )
    parser.add_argument(
        "--perplexity_max_samples",
        type=int,
        default=None,
        help="Limit number of test cases used for perplexity computation.",
    )
    args = parser.parse_args()

    if args.config:
        import yaml

        config_path = Path(args.config)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})

        output_dir = Path(training_cfg.get("output_dir", "./outputs"))
        inferred_model_path = str(output_dir / "final_adapter")
        inferred_base_model = model_cfg.get("base_model", args.base_model)
        inferred_test_set = eval_cfg.get("test_set_path", args.test_set)

        if args.model_path is None:
            args.model_path = inferred_model_path
        if args.base_model == parser.get_default("base_model"):
            args.base_model = inferred_base_model
        if args.test_set == parser.get_default("test_set"):
            args.test_set = inferred_test_set

    if not args.model_path:
        parser.error("--model_path is required unless --config is provided")

    return args


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_lora_adapter(model_path: str) -> bool:
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def load_model_and_tokenizer(
    model_path: str,
    base_model: str,
    device: torch.device,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and tokenizer, merging LoRA adapters when present."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora_adapter(model_path):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "peft is required to load a LoRA adapter; install it with: pip install peft"
            ) from exc

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(buggy_code: str, expected_behavior: str) -> str:
    return (
        "You are an expert Python programmer. Fix the bug in the following function.\n\n"
        f"Buggy code:\n```python\n{buggy_code}\n```\n\n"
        f"Expected behavior: {expected_behavior}\n\n"
        "Provide only the corrected Python code inside a single markdown code block.\n"
    )


def extract_code(response: str) -> str:
    """Extract the first Python code block from the model response."""
    pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response)
    if match:
        return match.group(1).strip()

    # Fallback: strip common explanation prefixes and return the whole text.
    lines = response.strip().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("def ", "class ", "import ")):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def generate_fix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_code(response)


def run_test_code(fixed_code: str, test_code: str, timeout: int) -> Dict[str, Any]:
    """Execute fixed_code followed by test_code in a subprocess sandbox."""
    script = f"{fixed_code}\n\n{test_code}\n"
    result = {"passed": False, "stdout": "", "stderr": "", "duration_sec": 0.0}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp_file:
        tmp_file.write(script)
        tmp_path = tmp_file.name

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result["duration_sec"] = time.time() - start
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["passed"] = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        result["duration_sec"] = time.time() - start
        result["stderr"] = f"Timeout after {timeout} seconds"
        result["stdout"] = exc.stdout or ""
    except Exception as exc:  # pylint: disable=broad-except
        result["stderr"] = str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


def compute_code_quality(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Run pylint on the generated code and return its score."""
    quality = {"score": 0.0, "tool": "fallback", "details": ""}

    pylint_path = None
    for candidate in ["pylint", "python3 -m pylint", "python -m pylint"]:
        try:
            subprocess.run(
                candidate.split() + ["--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            pylint_path = candidate.split()
            break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if pylint_path is None:
        quality["details"] = "pylint not available; using basic fallback score."
        quality["score"] = fallback_code_quality(code)
        return quality

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp_file:
        tmp_file.write(code)
        tmp_path = tmp_file.name

    try:
        proc = subprocess.run(
            pylint_path + ["--disable=all", "--enable=E,W,C", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        quality["tool"] = "pylint"
        quality["details"] = proc.stdout + proc.stderr

        # Pylint emits a final rating such as "Your code has been rated at 8.50/10"
        match = re.search(r"rated at ([\d\.]+)/10", proc.stdout)
        if match:
            quality["score"] = float(match.group(1)) * 10.0
        else:
            quality["score"] = fallback_code_quality(code)
            quality["tool"] = "fallback"
    except Exception as exc:  # pylint: disable=broad-except
        quality["details"] = f"pylint failed: {exc}"
        quality["score"] = fallback_code_quality(code)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return quality


def fallback_code_quality(code: str) -> float:
    """Simple heuristic quality score used when pylint is unavailable."""
    lines = code.splitlines()
    if not lines:
        return 0.0

    score = 50.0
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty:
        comment_ratio = sum(1 for ln in non_empty if ln.strip().startswith("#")) / len(non_empty)
        score += comment_ratio * 10.0

    long_lines = sum(1 for ln in lines if len(ln) > 100)
    score -= long_lines * 2.0

    if code.count("try:") == code.count("except"):
        score += 5.0

    return max(0.0, min(100.0, score))


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 1,
) -> float:
    """Compute average perplexity over a list of reference texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        if not text.strip():
            continue
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            n_tokens = inputs["input_ids"].numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


def evaluate_model(
    model_path: str,
    base_model: str,
    test_cases: List[Dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline for a single model."""
    model, tokenizer = load_model_and_tokenizer(model_path, base_model, device)

    per_case_results: List[Dict[str, Any]] = []
    perplexity_texts: List[str] = []

    for idx, case in enumerate(test_cases):
        case_id = case.get("id", f"case-{idx}")
        buggy_code = case["buggy_code"]
        expected_behavior = case["expected_behavior"]
        test_code = case["test_code"]

        prompt = build_prompt(buggy_code, expected_behavior)
        fixed_code = generate_fix(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )

        test_result = run_test_code(fixed_code, test_code, args.test_timeout)
        quality = compute_code_quality(fixed_code)

        case_result = {
            "id": case_id,
            "error_type": case.get("error_type"),
            "buggy_code": buggy_code,
            "expected_behavior": expected_behavior,
            "generated_fix": fixed_code,
            "test_code": test_code,
            "test_passed": test_result["passed"],
            "test_stdout": test_result["stdout"],
            "test_stderr": test_result["stderr"],
            "execution_time_sec": test_result["duration_sec"],
            "code_quality_score": quality["score"],
            "code_quality_tool": quality["tool"],
            "boundary_covered": test_result["passed"],
        }
        per_case_results.append(case_result)

        # Use the expected behavior as a reference for perplexity.
        reference = (
            f"Buggy code:\n{buggy_code}\n\n"
            f"Fixed code:\n{fixed_code}\n\n"
            f"Expected behavior: {expected_behavior}"
        )
        perplexity_texts.append(reference)

        if args.perplexity_max_samples and idx + 1 >= args.perplexity_max_samples:
            break

    total = len(per_case_results)
    passed = sum(1 for r in per_case_results if r["test_passed"])
    boundary_covered = sum(1 for r in per_case_results if r["boundary_covered"])
    avg_quality = (
        sum(r["code_quality_score"] for r in per_case_results) / total if total else 0.0
    )

    perplexity = compute_perplexity(
        model,
        tokenizer,
        perplexity_texts,
        device,
    )

    summary = {
        "model_path": model_path,
        "total_cases": total,
        "accuracy": passed / total if total else 0.0,
        "code_quality_avg": avg_quality,
        "boundary_pass_rate": boundary_covered / total if total else 0.0,
        "perplexity": perplexity,
    }

    return {
        "summary": summary,
        "cases": per_case_results,
    }


def evaluate_baseline(
    baseline_path: str,
    base_model: str,
    test_cases: List[Dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Evaluate the baseline model using the same pipeline."""
    return evaluate_model(baseline_path, base_model, test_cases, device, args)


def compare_results(
    target: Dict[str, Any], baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """Produce a comparison between target and baseline summaries."""
    metrics = ["accuracy", "code_quality_avg", "boundary_pass_rate"]
    comparison = {}
    for metric in metrics:
        target_val = target["summary"][metric]
        baseline_val = baseline["summary"][metric]
        comparison[metric] = {
            "target": target_val,
            "baseline": baseline_val,
            "delta": target_val - baseline_val,
            "improved": target_val > baseline_val,
        }

    comparison["perplexity"] = {
        "target": target["summary"]["perplexity"],
        "baseline": baseline["summary"]["perplexity"],
        "delta": target["summary"]["perplexity"] - baseline["summary"]["perplexity"],
        "improved": target["summary"]["perplexity"] < baseline["summary"]["perplexity"],
    }
    return comparison


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    if not os.path.isfile(args.test_set):
        raise FileNotFoundError(f"Test set not found: {args.test_set}")

    with open(args.test_set, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_cases = data.get("test_cases", [])
    if not test_cases:
        raise ValueError("No test cases found under key 'test_cases'.")

    print(f"Evaluating model: {args.model_path}")
    target_result = evaluate_model(
        args.model_path,
        args.base_model,
        test_cases,
        device,
        args,
    )

    report = {
        "target": target_result,
    }

    if args.baseline_model_path:
        print(f"Evaluating baseline: {args.baseline_model_path}")
        baseline_result = evaluate_baseline(
            args.baseline_model_path,
            args.base_model,
            test_cases,
            device,
            args,
        )
        report["baseline"] = baseline_result
        report["comparison"] = compare_results(target_result, baseline_result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report written to: {output_path}")
    print("Summary:")
    for key, value in target_result["summary"].items():
        print(f"  {key}: {value}")

    if report.get("comparison"):
        print("Comparison with baseline:")
        for metric, vals in report["comparison"].items():
            direction = "↑" if vals["improved"] else "↓"
            print(
                f"  {metric}: target={vals['target']:.4f}, "
                f"baseline={vals['baseline']:.4f}, delta={vals['delta']:+.4f} {direction}"
            )


if __name__ == "__main__":
    main()
