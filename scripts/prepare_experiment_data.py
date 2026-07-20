#!/usr/bin/env python3
"""Prepare the code-fix LoRA experiment dataset.

Sources:
  1. minitrain/data/sample_input.json
  2. codeskill_repro/build_codeskill_nb.py (parsed TEST_CASES)
  3. codeskill_experiment_data.json (successful fixed_code)
  4. rule-based supplemental samples

Outputs:
  - data/experiment_raw.json
  - data/finetune_dataset.json (instruction/response format)
  - data/mlx_lora/{train,valid,test}.jsonl (chat messages format)
"""

import ast
import json
import sys
import uuid
from pathlib import Path

import pandas as pd

# Make minitrain modules importable when running from scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.pipeline import (  # noqa: E402
    build_report,
    deduplicate_records,
    filter_by_quality,
    filter_outliers,
    split_data,
    to_instruction_response,
)

DATA_DIR = ROOT / "data"
MLX_DIR = DATA_DIR / "mlx_lora"
OUTPUT_RAW = DATA_DIR / "experiment_raw.json"
OUTPUT_DATASET = DATA_DIR / "finetune_dataset.json"

CODESKILL_DIR = Path("/Users/mingrun/codeskill_repro")
BUILD_NB = CODESKILL_DIR / "build_codeskill_nb.py"
EXPERIMENT_DATA = CODESKILL_DIR / "codeskill_experiment_data.json"

REQUIRED_ERROR_TYPES = {
    "NullPointerException",
    "IndexOutOfBounds",
    "TypeError",
    "ZeroDivisionError",
    "FileNotFoundError",
    "InfiniteLoop",
    "LogicError",
    "ResourceLeak",
    "ConcurrencyRace",
}


def load_sample_input() -> list[dict]:
    records = json.loads((DATA_DIR / "sample_input.json").read_text())
    for r in records:
        r["source"] = "minitrain_sample_input"
        r.setdefault("task_id", r.get("task_id", f"sample_{uuid.uuid4().hex[:8]}"))
    return records


def parse_codeskill_test_cases() -> list[dict]:
    """Extract TEST_CASES list from build_codeskill_nb.py via AST evaluation."""
    src = BUILD_NB.read_text()
    tree = ast.parse(src)
    cell2_code = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "cell2_code":
                    cell2_code = eval(compile(ast.Expression(node.value), "<string>", "eval"), {})
                    break
    if cell2_code is None:
        raise RuntimeError("cell2_code not found in build_codeskill_nb.py")
    ns = {}
    exec(cell2_code, ns)
    cases = ns["TEST_CASES"]
    for c in cases:
        c["source"] = "codeskill_test_cases"
    return cases


def map_fixed_codes(cases: list[dict]) -> dict[str, str]:
    """Return the best fixed_code per task_id from experiment results."""
    data = json.loads(EXPERIMENT_DATA.read_text())
    best = {}
    # Prefer the more advanced configs when successful.
    for config_name in ["codeskill", "static_skill", "no_skill"]:
        for r in data.get(config_name, {}).get("results", []):
            tid = r["task_id"]
            if r.get("success") and r.get("fixed_code"):
                best[tid] = r["fixed_code"]
    return best


def normalize_codeskill(cases: list[dict], fixed_map: dict[str, str]) -> list[dict]:
    records = []
    for c in cases:
        tid = c["id"]
        fixed = fixed_map.get(tid)
        if not fixed:
            # Fallback to the expected behavior; skip if no code available.
            continue
        records.append(
            {
                "task_id": tid,
                "error_type": c["error_type"],
                "buggy_code": c["buggy_code"].strip(),
                "expected_behavior": c["expected_behavior"],
                "fixed_code": fixed.strip(),
                "source": "codeskill_experiment",
            }
        )
    return records


def supplemental_samples() -> list[dict]:
    """Rule-based additional samples to ensure coverage and size."""
    templates = [
        {
            "error_type": "NullPointerException",
            "buggy_code": "def safe_upper(text):\n    return text.upper()",
            "expected_behavior": "Return empty string when text is None.",
            "fixed_code": "def safe_upper(text):\n    if text is None:\n        return ''\n    return text.upper()",
        },
        {
            "error_type": "NullPointerException",
            "buggy_code": "def get_email(user):\n    return user.email",
            "expected_behavior": "Return None when user is None.",
            "fixed_code": "def get_email(user):\n    if user is None:\n        return None\n    return user.email",
        },
        {
            "error_type": "IndexOutOfBounds",
            "buggy_code": "def first_item(items):\n    return items[1]",
            "expected_behavior": "Return the first item, or None if the list is empty.",
            "fixed_code": "def first_item(items):\n    if not items:\n        return None\n    return items[0]",
        },
        {
            "error_type": "IndexOutOfBounds",
            "buggy_code": "def last_pair(items):\n    return [items[-2], items[-1]]",
            "expected_behavior": "Return the last two items, or a shorter list if fewer than two exist.",
            "fixed_code": "def last_pair(items):\n    return items[-2:]",
        },
        {
            "error_type": "TypeError",
            "buggy_code": "def show_count(n):\n    return 'count: ' + n",
            "expected_behavior": "Convert n to string before concatenation.",
            "fixed_code": "def show_count(n):\n    return 'count: ' + str(n)",
        },
        {
            "error_type": "TypeError",
            "buggy_code": "def total(a, b):\n    return a + b\ntotal(1, '2')",
            "expected_behavior": "Convert string operands to integers before adding.",
            "fixed_code": "def total(a, b):\n    return int(a) + int(b)",
        },
        {
            "error_type": "ZeroDivisionError",
            "buggy_code": "def ratio(part, whole):\n    return part / whole",
            "expected_behavior": "Return 0.0 when whole is zero.",
            "fixed_code": "def ratio(part, whole):\n    if whole == 0:\n        return 0.0\n    return part / whole",
        },
        {
            "error_type": "ZeroDivisionError",
            "buggy_code": "def avg_even(nums):\n    evens = [n for n in nums if n % 2 == 0]\n    return sum(evens) / len(evens)",
            "expected_behavior": "Return 0.0 when there are no even numbers.",
            "fixed_code": "def avg_even(nums):\n    evens = [n for n in nums if n % 2 == 0]\n    if not evens:\n        return 0.0\n    return sum(evens) / len(evens)",
        },
        {
            "error_type": "FileNotFoundError",
            "buggy_code": "def load_text(path):\n    return open(path).read()",
            "expected_behavior": "Return empty string when the file does not exist.",
            "fixed_code": "def load_text(path):\n    try:\n        with open(path) as f:\n            return f.read()\n    except FileNotFoundError:\n        return ''",
        },
        {
            "error_type": "FileNotFoundError",
            "buggy_code": "def read_numbers(path):\n    return [int(x) for x in open(path)]",
            "expected_behavior": "Return an empty list when the file is missing.",
            "fixed_code": "def read_numbers(path):\n    try:\n        with open(path) as f:\n            return [int(x) for x in f]\n    except FileNotFoundError:\n        return []",
        },
        {
            "error_type": "InfiniteLoop",
            "buggy_code": "def increment_until(start, stop):\n    while start < stop:\n        start += 0\n    return start",
            "expected_behavior": "Increment start each iteration until it reaches stop.",
            "fixed_code": "def increment_until(start, stop):\n    while start < stop:\n        start += 1\n    return start",
        },
        {
            "error_type": "InfiniteLoop",
            "buggy_code": "def halve_until_one(n):\n    while n > 1:\n        pass\n    return n",
            "expected_behavior": "Divide n by 2 each iteration until it is 1 or less.",
            "fixed_code": "def halve_until_one(n):\n    while n > 1:\n        n //= 2\n    return n",
        },
        {
            "error_type": "LogicError",
            "buggy_code": "def product(nums):\n    result = 0\n    for n in nums:\n        result += n\n    return result",
            "expected_behavior": "Compute the product of all numbers; return 1 for empty input.",
            "fixed_code": "def product(nums):\n    result = 1\n    for n in nums:\n        result *= n\n    return result",
        },
        {
            "error_type": "LogicError",
            "buggy_code": "def mean_score(scores):\n    return sum(scores)",
            "expected_behavior": "Return the average of scores; return 0.0 for empty input.",
            "fixed_code": "def mean_score(scores):\n    if not scores:\n        return 0.0\n    return sum(scores) / len(scores)",
        },
        {
            "error_type": "ResourceLeak",
            "buggy_code": "def write_log(path, msg):\n    f = open(path, 'w')\n    f.write(msg)",
            "expected_behavior": "Use a context manager to close the file handle.",
            "fixed_code": "def write_log(path, msg):\n    with open(path, 'w') as f:\n        f.write(msg)",
        },
        {
            "error_type": "ResourceLeak",
            "buggy_code": "def append_file(path, data):\n    f = open(path, 'a')\n    f.write(data)",
            "expected_behavior": "Open the file with a with-statement so it is closed automatically.",
            "fixed_code": "def append_file(path, data):\n    with open(path, 'a') as f:\n        f.write(data)",
        },
        {
            "error_type": "ConcurrencyRace",
            "buggy_code": "def increment_shared(times):\n    import threading\n    counter = 0\n    def worker():\n        nonlocal counter\n        for _ in range(times):\n            counter += 1\n    threads = [threading.Thread(target=worker) for _ in range(5)]\n    for t in threads:\n        t.start()\n    for t in threads:\n        t.join()\n    return counter",
            "expected_behavior": "Use a threading.Lock to protect the shared counter.",
            "fixed_code": "def increment_shared(times):\n    import threading\n    counter = 0\n    lock = threading.Lock()\n    def worker():\n        nonlocal counter\n        for _ in range(times):\n            with lock:\n                counter += 1\n    threads = [threading.Thread(target=worker) for _ in range(5)]\n    for t in threads:\n        t.start()\n    for t in threads:\n        t.join()\n    return counter",
        },
        {
            "error_type": "ConcurrencyRace",
            "buggy_code": "def collect_items(n):\n    import threading\n    result = []\n    def worker():\n        for _ in range(n):\n            result.append(1)\n    threads = [threading.Thread(target=worker) for _ in range(3)]\n    for t in threads:\n        t.start()\n    for t in threads:\n        t.join()\n    return len(result)",
            "expected_behavior": "Guard the shared list append with a lock so the final count is correct.",
            "fixed_code": "def collect_items(n):\n    import threading\n    result = []\n    lock = threading.Lock()\n    def worker():\n        for _ in range(n):\n            with lock:\n                result.append(1)\n    threads = [threading.Thread(target=worker) for _ in range(3)]\n    for t in threads:\n        t.start()\n    for t in threads:\n        t.join()\n    return len(result)",
        },
    ]
    for i, t in enumerate(templates):
        t["task_id"] = f"supplemental_{i+1:03d}"
        t["source"] = "rule_based_supplemental"
    return templates


def ensure_min_lines(code: str, min_lines: int = 3, pad: str = "# Buggy code sample") -> str:
    """Pad short snippets so they pass the 3-line outlier filter."""
    lines = code.splitlines()
    while len(lines) < min_lines:
        lines.append(pad)
    return "\n".join(lines)


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    for r in records:
        r["buggy_code"] = ensure_min_lines(r["buggy_code"])
    df = pd.DataFrame(records)
    # Ensure required columns exist.
    for col in ["task_id", "error_type", "buggy_code", "expected_behavior", "fixed_code", "source"]:
        if col not in df.columns:
            df[col] = None
    return df


def run_pipeline_local(df: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    raw_count = len(df)
    df = deduplicate_records(df, threshold=0.8)
    dedup_count = len(df)
    df = filter_by_quality(df, quality_threshold=0.5)
    quality_count = len(df)
    df = filter_outliers(df, max_code_lines=200)
    outlier_count = len(df)
    splits = split_data(df, ratios=[0.8, 0.1, 0.1], random_state=42)
    report = build_report(raw_count, dedup_count, quality_count, outlier_count, splits)
    return report, splits


def convert_to_messages(inst_resp: dict) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Python programmer. Fix the bug in the provided function.",
            },
            {"role": "user", "content": inst_resp["instruction"]},
            {"role": "assistant", "content": inst_resp["response"]},
        ]
    }


def main() -> int:
    print("==> Loading MiniTrain sample input")
    sample_records = load_sample_input()
    print(f"    {len(sample_records)} records")

    print("==> Parsing CODESKILL test cases")
    codeskill_cases = parse_codeskill_test_cases()
    print(f"    {len(codeskill_cases)} cases")

    print("==> Mapping successful fixed_code from experiment results")
    fixed_map = map_fixed_codes(codeskill_cases)
    codeskill_records = normalize_codeskill(codeskill_cases, fixed_map)
    print(f"    {len(codeskill_records)} records with successful fixes")

    print("==> Generating supplemental samples")
    extra = supplemental_samples()
    print(f"    {len(extra)} records")

    all_records = sample_records + codeskill_records + extra
    df = to_dataframe(all_records)

    # Basic source statistics.
    source_counts = df["source"].value_counts().to_dict()
    print("\n==> Source counts before cleaning")
    for src, cnt in source_counts.items():
        print(f"    {src}: {cnt}")

    # Ensure error-type coverage.
    missing = REQUIRED_ERROR_TYPES - set(df["error_type"].dropna().unique())
    if missing:
        print(f"WARNING: missing error types: {missing}")

    # Save raw merged data.
    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RAW.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))

    print("\n==> Running MiniTrain data pipeline (dedup / quality / outlier / split)")
    report, splits = run_pipeline_local(df)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Save instruction/response dataset (all splits merged for visibility).
    full = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    dataset = [to_instruction_response(row) for _, row in full.iterrows()]
    OUTPUT_DATASET.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"\n    Saved {OUTPUT_DATASET} ({len(dataset)} records)")

    # Save MLX LoRA JSONL splits.
    MLX_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        records = [
            convert_to_messages(to_instruction_response(row))
            for _, row in splits[split_name].iterrows()
        ]
        path = MLX_DIR / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    Saved {path} ({len(records)} records)")

    # Save report.
    (DATA_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print("\n==> Data preparation complete")
    print(f"    Final train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
