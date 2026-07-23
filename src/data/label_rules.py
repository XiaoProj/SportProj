"""Candidate label-rule construction for the 5742821 table.

These rules are exploratory candidates. They are not ground-truth fatigue
labels and should not be used as final clinical labels without protocol review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelRuleSummary:
    """Summary of generated candidate label rules."""

    output_path: Path
    summary_path: Path
    rows: int
    columns: int
    rules: list[dict[str, Any]]
    explicit_ground_truth_fatigue_label: bool
    explicit_before_after_label: bool


def _numeric_columns() -> list[str]:
    return [
        "shot_1_accuracy",
        "shot_1_accuracy_1_20",
        "shot_1_accuracy_21_40",
        "shot_1_accuracy_41_60",
        "shot_2_accuracy",
        "shot_2_accuracy_1_20",
        "shot_2_accuracy_21_40",
        "shot_2_accuracy_41_60",
        "prontezza",
        "vas_mf_1",
        "vas_mf_2",
        "vas_mf_3",
        "vas_mot_1",
        "vas_mot_2",
        "vas_mot_3",
        "questions_accuracy",
    ]


def _value_counts_as_jsonable(series: Any) -> dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def build_label_rules(input_path: Path, output_path: Path, summary_path: Path) -> LabelRuleSummary:
    """Build candidate label-rule columns from the Phase 3 candidate table."""

    import pandas as pd

    data = pd.read_csv(input_path)
    for column in _numeric_columns():
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["condition_group_rule_candidate"] = data["condition_group"].map(
        {
            "mental_fatigue": 0,
            "mental_fatigue_plus_sleep_deprivation": 1,
        }
    )
    data["condition_group_rule_label"] = data["condition_group"].map(
        {
            "mental_fatigue": "mental_fatigue",
            "mental_fatigue_plus_sleep_deprivation": "mental_fatigue_plus_sleep_deprivation",
        }
    )
    data["condition_group_rule_note"] = (
        "candidate condition contrast only; not a strict high-vs-low fatigue ground-truth label"
    )

    data["vas_mf_mean"] = data[["vas_mf_1", "vas_mf_2", "vas_mf_3"]].mean(axis=1, skipna=True)
    data["vas_mot_mean"] = data[["vas_mot_1", "vas_mot_2", "vas_mot_3"]].mean(axis=1, skipna=True)

    median = data["vas_mf_mean"].median(skipna=True)
    lower_tertile = data["vas_mf_mean"].quantile(1.0 / 3.0)
    upper_tertile = data["vas_mf_mean"].quantile(2.0 / 3.0)
    data["vas_mf_median_threshold"] = median
    data["vas_mf_median_split_candidate"] = data["vas_mf_mean"].apply(
        lambda value: None if pd.isna(value) else int(value >= median)
    )
    data["vas_mf_median_split_label"] = data["vas_mf_median_split_candidate"].map(
        {0: "below_median", 1: "at_or_above_median"}
    )
    data["vas_mf_threshold_rule_note"] = (
        "exploratory median split only; not a clinical threshold or ground-truth fatigue label"
    )

    def tertile_label(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value <= lower_tertile:
            return "low_tertile"
        if value >= upper_tertile:
            return "high_tertile"
        return "middle_tertile"

    data["vas_mf_lower_tertile_threshold"] = lower_tertile
    data["vas_mf_upper_tertile_threshold"] = upper_tertile
    data["vas_mf_tertile_candidate"] = data["vas_mf_mean"].apply(tertile_label)
    data["vas_mf_tertile_rule_note"] = (
        "exploratory tertile grouping only; not a clinical threshold or ground-truth fatigue label"
    )

    data["shot_accuracy_change_shot2_minus_shot1"] = data["shot_2_accuracy"] - data["shot_1_accuracy"]
    data["performance_decline_candidate"] = data["shot_accuracy_change_shot2_minus_shot1"].apply(
        lambda value: None if pd.isna(value) else int(value < 0)
    )
    data["performance_decline_rule_note"] = (
        "candidate only; trial order and protocol interpretation must be confirmed"
    )

    data["explicit_ground_truth_fatigue_label"] = False
    data["explicit_before_after_label"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    rules = [
        {
            "rule": "condition_group_rule",
            "output_columns": ["condition_group_rule_candidate", "condition_group_rule_label"],
            "status": "usable_as_condition_contrast_candidate",
            "limitation": "Does not define strict high/low fatigue.",
        },
        {
            "rule": "vas_mf_continuous_score",
            "output_columns": ["vas_mf_mean"],
            "status": "usable_as_continuous_candidate_score",
            "limitation": "Scale direction and interpretation should be confirmed.",
        },
        {
            "rule": "vas_mf_threshold_candidate",
            "output_columns": ["vas_mf_median_split_candidate", "vas_mf_tertile_candidate"],
            "status": "exploratory_only",
            "limitation": "Median/tertile thresholds are data-derived and not clinical cut points.",
        },
        {
            "rule": "performance_decline_candidate",
            "output_columns": ["shot_accuracy_change_shot2_minus_shot1", "performance_decline_candidate"],
            "status": "exploratory_only",
            "limitation": "Trial order is not fully confirmed.",
        },
    ]
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "rules": rules,
        "class_counts": {
            "condition_group_rule_candidate": _value_counts_as_jsonable(data["condition_group_rule_candidate"]),
            "vas_mf_median_split_candidate": _value_counts_as_jsonable(data["vas_mf_median_split_candidate"]),
            "performance_decline_candidate": _value_counts_as_jsonable(data["performance_decline_candidate"]),
        },
        "vas_mf_thresholds": {
            "median": float(median),
            "lower_tertile": float(lower_tertile),
            "upper_tertile": float(upper_tertile),
        },
        "explicit_ground_truth_fatigue_label": False,
        "explicit_before_after_label": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return LabelRuleSummary(
        output_path=output_path,
        summary_path=summary_path,
        rows=int(data.shape[0]),
        columns=int(data.shape[1]),
        rules=rules,
        explicit_ground_truth_fatigue_label=False,
        explicit_before_after_label=False,
    )


def write_label_rule_doc(summary: LabelRuleSummary, doc_path: Path) -> None:
    """Write a Markdown comparison of candidate label rules."""

    lines = [
        "# Label Rule Comparison",
        "",
        "These rules are candidate labels for feasibility analysis only. They are not final ground-truth fatigue labels.",
        "",
        "## Outputs",
        "",
        f"- Candidate rule table: `{summary.output_path.as_posix()}`",
        f"- Summary JSON: `{summary.summary_path.as_posix()}`",
        f"- Rows: {summary.rows}",
        f"- Columns: {summary.columns}",
        "",
        "## Rule Comparison",
        "",
        "| Rule | Generated Columns | Feasibility | Main Limitation |",
        "| --- | --- | --- | --- |",
    ]
    for rule in summary.rules:
        lines.append(
            f"| {rule['rule']} | {', '.join(rule['output_columns'])} | {rule['status']} | {rule['limitation']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `condition_group_rule_candidate` distinguishes mental fatigue from mental fatigue plus sleep deprivation.",
            "- `vas_mf_mean` is a continuous mental-fatigue candidate score from `VAS MF 1/2/3`.",
            "- Median and tertile VAS MF splits are exploratory thresholds and not clinical cut points.",
            "- `shot_accuracy_change_shot2_minus_shot1` is a performance-change candidate; trial order must be confirmed before interpretation.",
            "",
            "## Label Status",
            "",
            "- Explicit ground-truth `fatigue_label`: not found.",
            "- Explicit `before_after_label`: not found.",
            "- Candidate labels should remain separate from raw ground-truth labels in future pipelines.",
            "",
        ]
    )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")
