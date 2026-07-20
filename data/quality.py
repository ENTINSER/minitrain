"""Data quality scoring utilities for the MiniTrain pipeline."""

import ast
import re
from typing import Any, Dict

# Heuristic action verbs used to gauge whether an expected-behavior description
# is actionable and specific. This is intentionally lightweight so the pipeline
# does not require external NLP libraries.
_ACTION_VERBS = frozenset(
    [
        "return",
        "ensure",
        "handle",
        "convert",
        "iterate",
        "compare",
        "compute",
        "check",
        "validate",
        "parse",
        "fix",
        "correct",
        "output",
        "print",
        "raise",
        "use",
        "implement",
        "avoid",
        "prevent",
        "process",
        "generate",
        "calculate",
        "sum",
        "count",
        "find",
        "remove",
        "update",
    ]
)


def _score_instruction_completeness(expected_behavior: str, min_length: int = 10) -> float:
    """Score how complete the instruction is based on description length.

    The score rises linearly with the character length of ``expected_behavior``
    and saturates once ``min_length`` characters are reached.

    Args:
        expected_behavior: The natural-language expected behavior string.
        min_length: Target character length for a complete description.

    Returns:
        A score between 0 and 1.
    """
    if not expected_behavior:
        return 0.0
    length = len(expected_behavior.strip())
    return min(1.0, length / min_length)


def _score_code_parsability(buggy_code: str) -> float:
    """Score whether the buggy code can be parsed as valid Python syntax.

    Args:
        buggy_code: The buggy Python source snippet.

    Returns:
        1.0 if ``ast.parse`` succeeds, otherwise 0.0.
    """
    if not buggy_code:
        return 0.0
    try:
        ast.parse(buggy_code)
        return 1.0
    except SyntaxError:
        return 0.0


def _score_behavior_clarity(expected_behavior: str) -> float:
    """Score the clarity of the expected-behavior description.

    Clarity is approximated by word count and the presence of action verbs.
    Longer, verb-rich descriptions are assumed to be clearer instructions.

    Args:
        expected_behavior: The natural-language expected behavior string.

    Returns:
        A score between 0 and 1.
    """
    if not expected_behavior:
        return 0.0
    text = expected_behavior.strip().lower()
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    length_score = min(1.0, word_count / 10)
    has_action_verb = any(verb in words for verb in _ACTION_VERBS)
    action_score = 1.0 if has_action_verb else 0.5
    return length_score * action_score


def compute_quality_score(record: dict) -> float:
    """Compute an aggregate data quality score in the range [0, 1].

    The score averages three dimensions:
      * Instruction completeness (length of ``expected_behavior``).
      * Code parsability (whether ``buggy_code`` parses with ``ast``).
      * Expected behavior clarity (word count and action-verb heuristics).

    Args:
        record: A single data record containing at least ``expected_behavior``
            and ``buggy_code``.

    Returns:
        Aggregate quality score between 0 and 1.
    """
    expected_behavior = record.get("expected_behavior", "") or ""
    buggy_code = record.get("buggy_code", "") or ""

    completeness = _score_instruction_completeness(expected_behavior)
    parsability = _score_code_parsability(buggy_code)
    clarity = _score_behavior_clarity(expected_behavior)

    aggregate = (completeness + parsability + clarity) / 3.0
    return round(aggregate, 4)


def compute_quality_breakdown(record: dict) -> Dict[str, Any]:
    """Return a quality report with both the aggregate score and per-dimension breakdown.

    Args:
        record: A single data record containing at least ``expected_behavior``
            and ``buggy_code``.

    Returns:
        A dictionary with the structure::

            {
                "quality_score": float,
                "breakdown": {
                    "instruction_completeness": float,
                    "code_parsability": float,
                    "expected_behavior_clarity": float,
                },
            }
    """
    expected_behavior = record.get("expected_behavior", "") or ""
    buggy_code = record.get("buggy_code", "") or ""

    completeness = _score_instruction_completeness(expected_behavior)
    parsability = _score_code_parsability(buggy_code)
    clarity = _score_behavior_clarity(expected_behavior)

    aggregate = round((completeness + parsability + clarity) / 3.0, 4)
    return {
        "quality_score": aggregate,
        "breakdown": {
            "instruction_completeness": round(completeness, 4),
            "code_parsability": round(parsability, 4),
            "expected_behavior_clarity": round(clarity, 4),
        },
    }
