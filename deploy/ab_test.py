"""A/B test helper for comparing MiniTrain model variants.

The null hypothesis (H0) is that the new treatment model has the same success
rate as the control (baseline) model. The alternative hypothesis (H1) is that
the success rates differ. If the p-value is below ``alpha`` we reject H0 and
conclude the treatment performs significantly differently from control.

For low-frequency events where expected cell counts are below 5, this module
falls back to Fisher's exact test; otherwise it uses a chi-squared test of
independence on the 2x2 contingency table.
"""

import argparse
import json
import math
from typing import Any, Dict


def _cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def ab_test(
    control_success: int,
    control_total: int,
    treatment_success: int,
    treatment_total: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compare a control group against a treatment group.

    Parameters
    ----------
    control_success:
        Number of successful outcomes in the control group.
    control_total:
        Total number of observations in the control group.
    treatment_success:
        Number of successful outcomes in the treatment group.
    treatment_total:
        Total number of observations in the treatment group.
    alpha:
        Significance threshold for the p-value.

    Returns
    -------
    A dict with ``p_value``, ``effect_size``, ``significant``, and
    ``recommendation``.
    """
    if any(v < 0 for v in (control_success, control_total, treatment_success, treatment_total)):
        raise ValueError("All counts must be non-negative.")
    if control_success > control_total or treatment_success > treatment_total:
        raise ValueError("Success counts cannot exceed total counts.")

    control_failure = control_total - control_success
    treatment_failure = treatment_total - treatment_success

    contingency = [
        [control_success, control_failure],
        [treatment_success, treatment_failure],
    ]

    # Determine expected counts to choose between chi2 and Fisher's exact test.
    row_totals = [sum(row) for row in contingency]
    col_totals = [sum(col) for col in zip(*contingency)]
    grand_total = sum(row_totals)

    expected = [
        [row * col / grand_total for col in col_totals]
        for row in row_totals
    ]
    use_fisher = any(e < 5 for row in expected for e in row)

    try:
        from scipy.stats import chi2_contingency, fisher_exact
    except ImportError as exc:
        raise ImportError("scipy is required for A/B testing. Install it with `pip install scipy`.") from exc

    if use_fisher:
        # Fisher's exact test gives an exact p-value for small samples.
        odds_ratio, p_value = fisher_exact(contingency)
        test_name = "fisher_exact"
    else:
        _, p_value, _, _ = chi2_contingency(contingency)
        odds_ratio = None
        test_name = "chi2_contingency"

    p_control = control_success / control_total if control_total else 0.0
    p_treatment = treatment_success / treatment_total if treatment_total else 0.0
    effect_size = _cohens_h(p_treatment, p_control)

    significant = p_value < alpha
    if significant and p_treatment > p_control:
        recommendation = "treatment is significantly better; consider promoting it"
    elif significant and p_treatment < p_control:
        recommendation = "treatment is significantly worse; keep control"
    else:
        recommendation = "no significant difference; gather more data or keep control"

    return {
        "p_value": float(p_value),
        "alpha": float(alpha),
        "significant": bool(significant),
        "effect_size": float(effect_size),
        "test": test_name,
        "odds_ratio": None if odds_ratio is None else float(odds_ratio),
        "control_rate": p_control,
        "treatment_rate": p_treatment,
        "control": {"success": control_success, "total": control_total},
        "treatment": {"success": treatment_success, "total": treatment_total},
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test for MiniTrain model variants")
    parser.add_argument("--control-success", type=int, required=True)
    parser.add_argument("--control-total", type=int, required=True)
    parser.add_argument("--treatment-success", type=int, required=True)
    parser.add_argument("--treatment-total", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    result = ab_test(
        control_success=args.control_success,
        control_total=args.control_total,
        treatment_success=args.treatment_success,
        treatment_total=args.treatment_total,
        alpha=args.alpha,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
