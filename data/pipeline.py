"""MiniTrain data processing pipeline.

This module loads raw training records, cleans them through deduplication,
quality filtering and outlier filtering, splits them into train/val/test sets,
and emits instruction/response JSON files plus a data-quality report.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow the module to be run both as `python -m data.pipeline` and as a script.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import pandas as pd
import yaml
from datasketch import MinHash, MinHashLSH
from sqlalchemy import create_engine

try:
    from minitrain.data.quality import compute_quality_score
except ImportError:  # pragma: no cover
    from quality import compute_quality_score


logger = logging.getLogger("minitrain.data.pipeline")


_TASK_DESCRIPTION = (
    "Fix the following buggy Python code so that it satisfies the "
    "described expected behavior."
)


def build_postgres_url(data_cfg: Dict[str, Any]) -> str:
    """Build a SQLAlchemy PostgreSQL URL from config and/or environment variables.

    Environment variables take precedence over the YAML config values.

    Args:
        data_cfg: The ``data`` section of the MiniTrain config.

    Returns:
        A SQLAlchemy-compatible PostgreSQL connection string.
    """
    host = os.environ.get("POSTGRES_HOST", data_cfg.get("postgres_host", "localhost"))
    port = int(
        os.environ.get("POSTGRES_PORT", data_cfg.get("postgres_port", 5432))
    )
    user = os.environ.get("POSTGRES_USER", data_cfg.get("postgres_user", "postgres"))
    password = os.environ.get(
        "POSTGRES_PASSWORD", data_cfg.get("postgres_password", "postgres")
    )
    db = os.environ.get("POSTGRES_DB", data_cfg.get("postgres_db", "postgres"))
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def load_data(data_cfg: Dict[str, Any], config_dir: Path) -> pd.DataFrame:
    """Load raw records from a file or a PostgreSQL database.

    Supported file formats are ``.json``, ``.jsonl`` and ``.csv``.

    Args:
        data_cfg: The ``data`` section of the MiniTrain config.
        config_dir: Directory containing the config file, used to resolve
            relative source paths.

    Returns:
        A :class:`pandas.DataFrame` with the raw records.

    Raises:
        ValueError: If the source type or file extension is unsupported, or if
            required columns are missing.
    """
    source_type = data_cfg.get("source_type", "file")

    if source_type == "file":
        source_path = Path(data_cfg["source_path"])
        if not source_path.is_absolute():
            source_path = config_dir / source_path
        suffix = source_path.suffix.lower()

        if suffix == ".json":
            df = pd.read_json(source_path)
        elif suffix == ".jsonl":
            df = pd.read_json(source_path, lines=True)
        elif suffix == ".csv":
            df = pd.read_csv(source_path)
        else:
            raise ValueError(f"Unsupported file extension: {suffix!r}")
    elif source_type == "postgresql":
        engine = create_engine(build_postgres_url(data_cfg))
        df = pd.read_sql(data_cfg["postgres_query"], engine)
    else:
        raise ValueError(f"Unsupported source_type: {source_type!r}")

    required_columns = {"buggy_code", "fixed_code", "expected_behavior"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def _get_shingles(text: str, n: int = 5) -> set:
    """Return the set of character n-grams (shingles) for a string.

    Args:
        text: Input text.
        n: Shingle size in characters.

    Returns:
        A set of shingles. Very short strings return a single shingle.
    """
    text = text.strip()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def deduplicate_records(
    df: pd.DataFrame,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> pd.DataFrame:
    """Remove near-duplicate records using MinHash + LSH on ``buggy_code``.

    MinHash is a locality-sensitive hashing technique that estimates Jaccard
    similarity between two sets using compact signatures. Here we treat each
    code snippet as a set of character n-grams (shingles), build a MinHash
    signature for each set, and insert the signatures into a Locality-Sensitive
    Hashing (LSH) index. LSH makes it efficient to find pairs whose estimated
    Jaccard similarity is at least ``threshold``. Records that match an earlier
    record are considered duplicates and are dropped; the first occurrence is
    retained.

    Args:
        df: Input dataframe containing a ``buggy_code`` column.
        threshold: MinHash similarity threshold above which records are
            considered duplicates.
        num_perm: Number of permutations (hash functions) used by MinHash.
        shingle_size: Character n-gram size.

    Returns:
        A dataframe with near-duplicate rows removed.
    """
    if df.empty:
        return df.copy()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicate_indices = set()

    for idx, row in df.iterrows():
        text = str(row["buggy_code"])
        shingles = _get_shingles(text, n=shingle_size)

        m = MinHash(num_perm=num_perm)
        for shingle in shingles:
            m.update(shingle.encode("utf-8"))

        # Query before insertion so the current record is not matched against
        # itself. Any earlier near-duplicate already in the LSH will be found.
        similar = lsh.query(m)
        if similar:
            duplicate_indices.add(idx)
        lsh.insert(idx, m)

    return df.drop(index=duplicate_indices).reset_index(drop=True)


def filter_by_quality(df: pd.DataFrame, quality_threshold: float) -> pd.DataFrame:
    """Drop records whose quality score is below the threshold.

    Args:
        df: Input dataframe.
        quality_threshold: Minimum acceptable quality score.

    Returns:
        Filtered dataframe with a new ``quality_score`` column.
    """
    df = df.copy()
    df["quality_score"] = df.apply(compute_quality_score, axis=1)
    return df[df["quality_score"] >= quality_threshold].reset_index(drop=True)


def filter_outliers(df: pd.DataFrame, max_code_lines: int) -> pd.DataFrame:
    """Drop records whose ``buggy_code`` line count is outside [3, max_code_lines].

    Args:
        df: Input dataframe.
        max_code_lines: Upper bound for acceptable line count.

    Returns:
        Filtered dataframe with a new ``line_count`` column.
    """
    df = df.copy()
    df["line_count"] = df["buggy_code"].astype(str).str.split("\n").str.len()
    mask = (df["line_count"] >= 3) & (df["line_count"] <= max_code_lines)
    return df[mask].reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    ratios: List[float],
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Shuffle and split the dataframe into train/val/test sets.

    Args:
        df: Input dataframe.
        ratios: A list of three floats summing to 1.0, e.g. ``[0.8, 0.1, 0.1]``.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with keys ``train``, ``val`` and ``test``.

    Raises:
        ValueError: If the ratios do not sum to 1.0.
    """
    ratios = [float(r) for r in ratios]
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df)

    if n < 3:
        # Not enough samples for a meaningful three-way split; keep them all
        # in train so downstream training can decide how to validate.
        return {
            "train": df.copy(),
            "val": df.iloc[0:0].reset_index(drop=True),
            "test": df.iloc[0:0].reset_index(drop=True),
        }

    # Ensure every split receives at least one record when possible, while
    # still honoring the requested ratios for larger datasets.
    train_size = min(max(1, int(n * ratios[0])), n - 2)
    val_size = max(1, int(n * ratios[1]))
    test_size = max(1, n - train_size - val_size)

    return {
        "train": df.iloc[:train_size].reset_index(drop=True),
        "val": df.iloc[train_size : train_size + val_size].reset_index(drop=True),
        "test": df.iloc[train_size + val_size :].reset_index(drop=True),
    }


def to_instruction_response(row: pd.Series) -> Dict[str, str]:
    """Convert a cleaned record into instruction/response format.

    Args:
        row: A pandas Series containing ``expected_behavior``, ``buggy_code``
            and ``fixed_code``.

    Returns:
        A dictionary with ``instruction`` and ``response`` keys.
    """
    instruction = (
        f"{_TASK_DESCRIPTION}\n\n"
        f"Expected behavior: {row['expected_behavior']}\n\n"
        f"Buggy code:\n```python\n{row['buggy_code']}\n```"
    )
    return {
        "instruction": instruction,
        "response": str(row["fixed_code"]),
    }


def build_report(
    raw_count: int,
    dedup_count: int,
    quality_count: int,
    outlier_count: int,
    splits: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Build the data-quality report.

    Args:
        raw_count: Number of records before any cleaning.
        dedup_count: Number of records after deduplication.
        quality_count: Number of records after quality filtering.
        outlier_count: Number of records after outlier filtering.
        splits: Dictionary of train/val/test dataframes.

    Returns:
        Report dictionary with counts and rates.
    """
    return {
        "counts": {
            "raw": raw_count,
            "after_dedup": dedup_count,
            "after_quality_filter": quality_count,
            "after_outlier_filter": outlier_count,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "rates": {
            "dedup_rate": (raw_count - dedup_count) / raw_count if raw_count else 0.0,
            "quality_filter_rate": (dedup_count - quality_count) / dedup_count
            if dedup_count
            else 0.0,
            "outlier_filter_rate": (quality_count - outlier_count) / quality_count
            if quality_count
            else 0.0,
            "overall_filter_rate": (raw_count - outlier_count) / raw_count
            if raw_count
            else 0.0,
        },
    }


def save_split(df: pd.DataFrame, output_dir: Path, split_name: str) -> None:
    """Save a split as a JSON file in instruction/response format.

    Args:
        df: Split dataframe.
        output_dir: Directory where the file will be written.
        split_name: Name of the split (``train``, ``val`` or ``test``).
    """
    records = df.apply(to_instruction_response, axis=1).tolist()
    out_path = output_dir / f"{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def run_pipeline(config_path: str, error_type: str | None = None) -> Dict[str, Any]:
    """Execute the full MiniTrain data pipeline.

    Args:
        config_path: Path to the MiniTrain YAML config file.
        error_type: Optional error type to filter on before cleaning.

    Returns:
        The data-quality report dictionary.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    config_dir = config_path.parent
    output_dir = Path(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data.
    df = load_data(data_cfg, config_dir)

    # 1b. Optional Agent Factory error-type filter.
    if error_type and "error_type" in df.columns:
        df = df[df["error_type"] == error_type]
        logger.info("Filtered data by error_type", extra={"error_type": error_type, "rows": len(df)})

    raw_count = len(df)

    # 2. Deduplicate with MinHash + LSH.
    dedup_threshold = data_cfg.get("dedup_threshold", 0.8)
    df = deduplicate_records(df, threshold=dedup_threshold)
    dedup_count = len(df)

    # 3. Quality filtering.
    quality_threshold = data_cfg.get("quality_threshold", 0.5)
    df = filter_by_quality(df, quality_threshold)
    quality_count = len(df)

    # 4. Outlier filtering by code line count.
    max_code_lines = data_cfg.get("max_code_lines", 200)
    df = filter_outliers(df, max_code_lines)
    outlier_count = len(df)

    # 5. Train/val/test split.
    split_ratios = data_cfg["train_val_test_split"]
    splits = split_data(df, split_ratios)

    # 6. Save instruction/response JSON files.
    for split_name in ("train", "val", "test"):
        save_split(splits[split_name], output_dir, split_name)

    # 7. Save data-quality report.
    report = build_report(
        raw_count=raw_count,
        dedup_count=dedup_count,
        quality_count=quality_count,
        outlier_count=outlier_count,
        splits=splits,
    )
    report_path = output_dir / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main(argv: List[str] = None) -> None:
    """CLI entrypoint for the MiniTrain data pipeline.

    Args:
        argv: Optional list of command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="MiniTrain data pipeline: clean, filter, split and export training data."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the MiniTrain YAML configuration file.",
    )
    parser.add_argument(
        "--error-type",
        default=None,
        help="Optional error_type to filter records before cleaning.",
    )
    args = parser.parse_args(argv)

    report = run_pipeline(args.config, error_type=args.error_type)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
