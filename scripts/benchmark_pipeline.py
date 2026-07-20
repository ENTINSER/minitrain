#!/usr/bin/env python3
"""Benchmark the MiniTrain data pipeline on a scaled-up sample dataset.

Run:
    python scripts/benchmark_pipeline.py

Outputs a JSON report with wall-clock time and throughput (records/second).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.pipeline import run_pipeline


def _scale_dataset(factor: int) -> Path:
    """Create a temporary JSON dataset by replicating sample_input.json."""
    source = ROOT / "data" / "sample_input.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    scaled = []
    for i in range(factor):
        for rec in records:
            dup = dict(rec)
            dup["task_id"] = f"{rec['task_id']}-{i}"
            scaled.append(dup)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(scaled, tmp, ensure_ascii=False)
    tmp.close()
    return Path(tmp.name)


def _write_config(data_path: Path, output_dir: Path) -> Path:
    base_config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    base_config["data"]["source_type"] = "file"
    base_config["data"]["source_path"] = str(data_path)
    base_config["data"]["output_dir"] = str(output_dir / "processed_data")
    base_config["training"]["output_dir"] = str(output_dir / "outputs")
    cfg_path = output_dir / "benchmark_config.yaml"
    cfg_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")
    return cfg_path


def main() -> None:
    factor = 100  # 10 * 100 = 1000 records
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = _scale_dataset(factor)
        cfg_path = _write_config(data_path, tmp_path)

        start = time.perf_counter()
        report = run_pipeline(str(cfg_path))
        elapsed = time.perf_counter() - start

    raw_count = report["counts"]["raw"]
    throughput = raw_count / elapsed if elapsed > 0 else 0.0

    result = {
        "dataset_records": raw_count,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_records_per_second": round(throughput, 2),
        "report": report,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
