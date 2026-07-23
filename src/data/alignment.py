"""Data-alignment feasibility checks for processed sample tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALIGNMENT_FIELDS = [
    "subject_id",
    "session_id",
    "dataset_name",
    "split",
    "scene_name",
    "condition_group",
    "trial_id",
    "before_after_label",
    "fatigue_label",
]

ALIGNMENT_TEMPLATE_COLUMNS = [
    "alignment_id",
    "pose_feature_table",
    "label_table",
    "pose_dataset_name",
    "label_dataset_name",
    "pose_subject_id",
    "label_subject_id",
    "pose_session_id",
    "label_condition_group",
    "match_evidence_file",
    "match_evidence_field",
    "alignment_status",
    "notes",
    "approved_by",
    "date_approved",
]


@dataclass(frozen=True)
class AlignmentSummary:
    """Summary of alignment feasibility."""

    doc_path: Path
    template_path: Path
    direct_alignment_possible: bool
    reason: str
    table_summaries: list[dict[str, Any]]
    shared_subject_values: dict[str, list[str]]


def _column_summary(frame: Any, table_name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "table_name": table_name,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "available_alignment_fields": [],
        "missing_alignment_fields": [],
        "unique_values": {},
    }
    for field in ALIGNMENT_FIELDS:
        if field in frame.columns:
            summary["available_alignment_fields"].append(field)
            values = frame[field].dropna().astype(str)
            values = values[values != ""]
            summary["unique_values"][field] = sorted(values.unique().tolist())[:30]
        else:
            summary["missing_alignment_fields"].append(field)
    return summary


def _write_empty_template(template_path: Path) -> None:
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_TEMPLATE_COLUMNS)
        writer.writeheader()


def check_alignment(
    humanm3_path: Path,
    epfl_path: Path,
    label_path: Path,
    template_path: Path,
    doc_path: Path,
) -> AlignmentSummary:
    """Check whether processed sample tables contain verifiable alignment fields."""

    import pandas as pd

    tables = {
        "humanm3_pose_features_sample": pd.read_csv(humanm3_path),
        "epfl_multiview_pose_features_sample": pd.read_csv(epfl_path),
        "fatigue_sleep_label_candidates": pd.read_csv(label_path),
    }
    summaries = [_column_summary(frame, name) for name, frame in tables.items()]

    label_subjects = set(tables["fatigue_sleep_label_candidates"]["subject_id"].dropna().astype(str))
    shared_subject_values = {}
    for name in ("humanm3_pose_features_sample", "epfl_multiview_pose_features_sample"):
        pose_subjects = set(tables[name]["subject_id"].dropna().astype(str))
        overlap = sorted(pose_subjects.intersection(label_subjects))
        shared_subject_values[name] = overlap

    direct_possible = False
    reason = (
        "No verified cross-dataset subject/session key links the pose feature tables to the 5742821 label table. "
        "Numeric subject_id overlap alone is insufficient evidence because dataset_name, session_id, scene_name, "
        "condition_group, and protocol provenance differ."
    )

    _write_empty_template(template_path)
    write_alignment_doc(doc_path, summaries, shared_subject_values, direct_possible, reason, template_path)
    return AlignmentSummary(
        doc_path=doc_path,
        template_path=template_path,
        direct_alignment_possible=direct_possible,
        reason=reason,
        table_summaries=summaries,
        shared_subject_values=shared_subject_values,
    )


def write_alignment_doc(
    doc_path: Path,
    summaries: list[dict[str, Any]],
    shared_subject_values: dict[str, list[str]],
    direct_possible: bool,
    reason: str,
    template_path: Path,
) -> None:
    """Write data-alignment feasibility notes."""

    lines = [
        "# Data Alignment Feasibility",
        "",
        "This analysis uses only already generated small-sample processed CSV files.",
        "No raw dataset files are read or modified.",
        "",
        f"- Direct supervised alignment possible: {'yes' if direct_possible else 'no'}",
        f"- Reason: {reason}",
        f"- Empty alignment template: `{template_path.as_posix()}`",
        "",
        "## Available Alignment Fields",
        "",
        "| Table | Rows | Available Fields | Missing Fields |",
        "| --- | ---: | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['table_name']} | {summary['rows']} | "
            f"{', '.join(summary['available_alignment_fields']) or 'none'} | "
            f"{', '.join(summary['missing_alignment_fields']) or 'none'} |"
        )
    lines.extend(["", "## Subject ID Overlap", ""])
    for table_name, overlap in shared_subject_values.items():
        overlap_text = ", ".join(overlap) if overlap else "none"
        lines.append(f"- `{table_name}` vs `fatigue_sleep_label_candidates`: {overlap_text}")
    lines.extend(
        [
            "",
            "Overlapping numeric IDs are not treated as verified matches. A future manual alignment file must cite evidence such as a protocol document, roster mapping, session log, or file-level metadata.",
            "",
            "## Required Evidence Before Merging",
            "",
            "- Shared subject identity across pose and fatigue/sleep data.",
            "- Shared session or experimental condition.",
            "- Trial/task mapping between movement samples and spreadsheet rows.",
            "- Approved label-construction rule for fatigue and sleep variables.",
            "- Clear before/after labels if before-after movement analysis is required.",
            "",
        ]
    )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")
