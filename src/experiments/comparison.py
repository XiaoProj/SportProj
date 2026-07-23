"""Comparison and statistical-analysis helpers."""

from __future__ import annotations

from typing import Any


def before_after_difference(feature_table: Any, subject_column: str, phase_column: str, value_column: str) -> Any:
    """Create paired before-after differences when required labels exist."""

    required = {subject_column, phase_column, value_column}
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"Missing required columns for before-after analysis: {sorted(missing)}")
    pivot = feature_table.pivot_table(index=subject_column, columns=phase_column, values=value_column)
    if "before" not in pivot.columns or "after" not in pivot.columns:
        raise ValueError("phase_column must contain 'before' and 'after' labels for paired analysis.")
    pivot["after_minus_before"] = pivot["after"] - pivot["before"]
    return pivot


def paired_statistical_test(before_values: Any, after_values: Any) -> dict[str, Any]:
    """Run a paired t-test when scipy is available; otherwise return a clear skip reason."""

    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return {"status": "skipped", "reason": "scipy is not installed"}
    statistic, p_value = ttest_rel(before_values, after_values, nan_policy="omit")
    return {"status": "ok", "test": "paired_t_test", "statistic": float(statistic), "p_value": float(p_value)}
