#!/usr/bin/env python3
"""Evaluate base vs fine-tuned MLX code-fix model on the MiniTrain test set."""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
TEST_SET_PATH = ROOT / "evaluate" / "test_set" / "sample_test_cases.json"
OUTPUT_PATH = ROOT / "evaluate" / "benchmark_result.json"
CHECKPOINT_DIR = ROOT / "checkpoints" / "lora-codefix"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

SYSTEM_PROMPT = "You are an expert Python programmer. Fix the bug in the provided function."


def load_test_cases() -> List[Dict[str, Any]]:
    data = json.loads(TEST_SET_PATH.read_text())
    return data.get("test_cases", data)


def build_messages(buggy_code: str, expected_behavior: str) -> List[Dict[str, str]]:
    user_content = (
        "Fix the following buggy Python code so that it satisfies the described "
        "expected behavior.\n\n"
        f"Expected behavior: {expected_behavior}\n\n"
        f"Buggy code:\n```python\n{buggy_code}\n```"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_code(response: str) -> str:
    pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    lines = response.strip().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("def ", "class ", "import ")):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def run_tests(fixed_code: str, test_code: str, timeout: int = 10) -> Dict[str, Any]:
    script = f"{fixed_code}\n\n{test_code}\n"
    result = {"passed": False, "stdout": "", "stderr": "", "duration_sec": 0.0}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(script)
        tmp_path = tmp.name
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
    except Exception as exc:  # noqa: BLE001
        result["stderr"] = str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return result


def compute_ppl_manual(model, tokenizer, jsonl_path: Path) -> float:
    """Compute perplexity on a chat-format JSONL file."""
    try:
        import math

        import mlx.core as mx
        import mlx.nn as nn

        all_tokens = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                toks = tokenizer.apply_chat_template(
                    data["messages"], return_dict=False
                )
                all_tokens.extend(toks)
        if len(all_tokens) < 2:
            return 0.0

        tokens = mx.array(all_tokens)
        seq_len = 512
        losses = []
        for start in range(0, len(tokens) - 1, seq_len):
            chunk = tokens[start : start + seq_len + 1]
            if len(chunk) < 2:
                break
            logits = model(chunk[:-1][None]).astype(mx.float32)
            loss = nn.losses.cross_entropy(logits, chunk[1:][None])
            mx.eval(loss)
            losses.append(loss.flatten())

        if not losses:
            return 0.0
        mean_loss = mx.concatenate(losses).mean().item()
        return round(math.exp(mean_loss), 3)
    except Exception as exc:  # noqa: BLE001
        print(f"    PPL computation skipped: {exc}")
        return 0.0


def fallback_quality_score(code: str) -> float:
    """Return a 0-100 quality score without external tools."""
    if not code:
        return 0.0
    syntax_ok = 0.0
    try:
        ast.parse(code)
        syntax_ok = 1.0
    except SyntaxError:
        pass
    lines = code.splitlines()
    max_len = max((len(l) for l in lines), default=0)
    length_ok = max(0.0, 1.0 - max(0, max_len - 100) / 100.0)
    has_func = 1.0 if any(l.strip().startswith(("def ", "class ")) for l in lines) else 0.5
    return (syntax_ok * 0.5 + length_ok * 0.3 + has_func * 0.2) * 100.0


def evaluate_model(
    model_name: str,
    adapter_path: str | None,
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    print(f"\nLoading model: {model_name}" + (f" adapter: {adapter_path}" if adapter_path else ""))
    model, tokenizer = load(model_name, adapter_path=adapter_path, lazy=False)

    results = []
    pass_count = 0
    quality_scores = []
    total_gen_time = 0.0

    for case in test_cases:
        messages = build_messages(case["buggy_code"], case["expected_behavior"])
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False
        )

        start = time.time()
        sampler = make_sampler(temp=0.0)
        response = generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=256,
            sampler=sampler,
            verbose=False,
        )
        gen_time = time.time() - start
        total_gen_time += gen_time

        fixed_code = extract_code(response)
        test_result = run_tests(fixed_code, case["test_code"])
        quality = fallback_quality_score(fixed_code)

        if test_result["passed"]:
            pass_count += 1
        quality_scores.append(quality)

        results.append(
            {
                "id": case.get("id"),
                "error_type": case.get("error_type"),
                "passed": test_result["passed"],
                "generated_code": fixed_code,
                "stdout": test_result["stdout"],
                "stderr": test_result["stderr"],
                "code_quality": round(quality, 2),
                "gen_time_sec": round(gen_time, 3),
            }
        )
        status = "PASS" if test_result["passed"] else "FAIL"
        print(f"  [{status}] {case.get('id')} ({case.get('error_type')}) Q={quality:.1f}")

    accuracy = pass_count / len(test_cases) if test_cases else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    ppl = compute_ppl_manual(model, tokenizer, ROOT / "data" / "mlx_lora" / "test.jsonl")
    return {
        "name": model_name if adapter_path is None else f"{model_name}-LoRA-codefix",
        "accuracy": round(accuracy * 100, 2),
        "code_quality": round(avg_quality, 2),
        "ppl": ppl,
        "avg_gen_time_sec": round(total_gen_time / len(test_cases), 3) if test_cases else 0.0,
        "per_case": results,
    }


def main() -> int:
    print("==> Loading test cases")
    test_cases = load_test_cases()
    print(f"    {len(test_cases)} cases")

    old_result = evaluate_model(BASE_MODEL, None, test_cases)
    new_result = evaluate_model(BASE_MODEL, str(CHECKPOINT_DIR), test_cases)

    report = {
        "old_model": {
            "name": old_result["name"],
            "accuracy": old_result["accuracy"],
            "code_quality": old_result["code_quality"],
            "ppl": old_result["ppl"],
        },
        "new_model": {
            "name": new_result["name"],
            "accuracy": new_result["accuracy"],
            "code_quality": new_result["code_quality"],
            "ppl": new_result["ppl"],
        },
        "improvement": {
            "accuracy_delta": round(new_result["accuracy"] - old_result["accuracy"], 2),
            "code_quality_delta": round(new_result["code_quality"] - old_result["code_quality"], 2),
            "ppl_delta": round(old_result["ppl"] - new_result["ppl"], 3),
        },
        "details": {
            "old": old_result["per_case"],
            "new": new_result["per_case"],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n==> Saved benchmark report to {OUTPUT_PATH}")

    print("\n===== Comparison Summary =====")
    print(f"Old model: {report['old_model']}")
    print(f"New model: {report['new_model']}")
    print(f"Improvement: {report['improvement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
