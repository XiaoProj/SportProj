"""Build small processed feature samples without full-data processing.

This script reads only selected JSON/XLSX samples and writes derived sample
tables outside dataset/. It does not extract frames, read MP4 content, train
models, or run full dataset processing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.data.label_candidates import LabelCandidateSummary, build_fatigue_sleep_label_candidates
from src.features.pose_table import (
    PoseTableSummary,
    build_epfl_multiview_pose_feature_sample,
    build_humanm3_pose_feature_sample,
)


DATASET_ROOT = PROJECT_ROOT / "dataset"
PROCESSED_ROOT = PROJECT_ROOT / "outputs" / "processed"
METADATA_ROOT = PROJECT_ROOT / "outputs" / "metadata"
LOG_PATH = PROJECT_ROOT / "logs" / "feature_sample_report.txt"
FEATURE_DESIGN_PATH = PROJECT_ROOT / "docs" / "feature_table_design.md"
LABEL_PLAN_PATH = PROJECT_ROOT / "docs" / "label_construction_plan.md"
PHASE3_REPORT_PATH = PROJECT_ROOT / "docs" / "feature_sample_report.md"
SUMMARY_JSON_PATH = METADATA_ROOT / "sample_summary.json"


def _safe_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _summary_to_dict(summary: PoseTableSummary | LabelCandidateSummary) -> dict[str, Any]:
    data = asdict(summary)
    data["output_path"] = _safe_rel(summary.output_path)
    return data


def write_feature_table_design(
    humanm3_summary: PoseTableSummary,
    epfl_summary: PoseTableSummary,
    label_summary: LabelCandidateSummary,
) -> None:
    """Update the processed feature table design document with validation."""

    ensure_not_inside_dataset(FEATURE_DESIGN_PATH, DATASET_ROOT)
    lines = [
        "# Processed Feature Table Design",
        "",
        "This document defines the standard feature table and records the small-sample validation.",
        "Derived sample data has been written only under `outputs/processed/`.",
        "",
        "## Generated Small-Sample Tables",
        "",
        "| Table | Rows | Columns | Purpose |",
        "| --- | ---: | ---: | --- |",
        f"| `{_safe_rel(humanm3_summary.output_path)}` | {humanm3_summary.rows} | {humanm3_summary.columns} | Human-M3 15-joint 3D pose features |",
        f"| `{_safe_rel(epfl_summary.output_path)}` | {epfl_summary.rows} | {epfl_summary.columns} | EPFL multiview 13-joint 3D pose features |",
        f"| `{_safe_rel(label_summary.output_path)}` | {label_summary.rows} | {label_summary.columns} | 5742821 fatigue/sleep/performance label candidates |",
        "",
        "## Required Metadata Columns",
        "",
        "| Column | Meaning | Notes |",
        "| --- | --- | --- |",
        "| `dataset_name` | Source dataset identifier | Required for all rows |",
        "| `subject_id` | Athlete/person identifier | Uses person/subject IDs when present |",
        "| `session_id` | Session or sequence identifier | EPFL sequence, Human-M3 split/scene, or spreadsheet condition |",
        "| `split` | train/test/validation | Uses official split when available |",
        "| `scene_name` | Scene or task setting | Human-M3 basketball1/basketball2 or EPFL SportCenter |",
        "| `camera_id` | Camera identifier | Empty for 3D pose-only rows |",
        "| `view_id` | View identifier | `3d_pose_calib` or `3d_triangulated` in sample tables |",
        "| `frame_id` | Frame index or pose JSON stem | Does not require reading image/video content |",
        "| `trial_id` | Trial or jump/shot ID | Empty until event segmentation is available |",
        "| `task_type` | Movement or experimental task | `basketball_pose` in pose sample tables |",
        "",
        "## Optional Label and Context Columns",
        "",
        "| Column | Current Status |",
        "| --- | --- |",
        "| `before_after_label` | Not found in probed data |",
        "| `fatigue_label` | No explicit binary label found; only candidates in 5742821 |",
        "| `sleep_condition` | Candidate from `Mental fatigue + Sleep deprivation` condition |",
        "| `soreness_score` | Not found |",
        "| `sRPE` | Not found |",
        "| `training_duration` | Not found |",
        "",
        "## Derived Pose Features",
        "",
        "- `left_knee_flexion_angle`, `right_knee_flexion_angle`: hip-knee-ankle angle.",
        "- `left_hip_flexion_angle`, `right_hip_flexion_angle`: pelvis-hip-knee for Human-M3; same-side shoulder-hip-knee proxy for EPFL multiview because pelvis is absent.",
        "- `left_ankle_angle`, `right_ankle_angle`: currently blank because neither sample schema includes foot/toe joints.",
        "- `left_right_knee_angle_difference`: absolute left-right knee flexion difference.",
        "- `left_right_ankle_angle_difference`: blank until distal foot/toe joints are available.",
        "- `left_knee_valgus_proxy`, `right_knee_valgus_proxy`: geometric proxy from hip-knee-ankle in coordinate plane, not a clinical valgus measure.",
        "- `pose_valid_joint_count`, `pose_expected_joint_count`, `pose_missing_joint_ratio`: pose completeness checks.",
        "",
        "## Features Requiring Event Segmentation",
        "",
        "- takeoff-to-landing duration",
        "- lowest center-of-mass timing",
        "- landing-phase knee angle fluctuation",
        "- session/trial aggregation features",
        "",
        "## Training Readiness",
        "",
        "Formal model training should not start yet. The current sample tables validate feature computation, but there is no confirmed subject/session alignment between pose datasets and fatigue/sleep labels.",
        "",
    ]
    FEATURE_DESIGN_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_label_plan(label_summary: LabelCandidateSummary) -> None:
    """Write label construction feasibility notes."""

    ensure_not_inside_dataset(LABEL_PLAN_PATH, DATASET_ROOT)
    lines = [
        "# Label Construction Plan",
        "",
        "Generated during Phase 3 from `dataset/5742821/Data.xlsx` using read-only workbook access.",
        "The `.AW5` file remains an unknown proprietary format and is not parsed.",
        "",
        "## Generated Candidate Table",
        "",
        f"- Path: `{_safe_rel(label_summary.output_path)}`",
        f"- Rows: {label_summary.rows}",
        f"- Columns: {label_summary.columns}",
        "",
        "## Fields Available From Data.xlsx",
        "",
        "- `Subject`",
        "- condition group `Mental fatigue`",
        "- condition group `Mental fatigue + Sleep deprivation`",
        "- `Shot 1 (accuracy)` and windowed Shot 1 accuracy fields",
        "- `Shot 2 (accuracy)` and windowed Shot 2 accuracy fields",
        "- `Prontezza`",
        "- `VAS MF 1`, `VAS MF 2`, `VAS MF 3`",
        "- `VAS Mot 1`, `VAS Mot 2`, `VAS Mot 3`",
        "- `Questions (accuracy)`",
        "",
        "## Candidate Variables",
        "",
        "- Fatigue variables: condition group, VAS MF fields, VAS Mot fields.",
        "- Sleep variable: sleep-deprivation condition candidate from the `Mental fatigue + Sleep deprivation` column group.",
        "- Performance variables: Shot 1/2 accuracy fields and Questions accuracy.",
        "",
        "## Feasibility",
        "",
        "- `sleep_condition` can be represented as a candidate condition based on the explicit condition group.",
        "- `fatigue_label` is not explicit. A binary label should not be treated as ground truth until the study protocol confirms how conditions map to labels.",
        "- `before_after_label` was not found.",
        "- There is no confirmed row-level alignment between the spreadsheet subjects and pose datasets.",
        "",
        "## Candidate Rules Requiring Approval",
        "",
        "- Compare `Mental fatigue` versus `Mental fatigue + Sleep deprivation` as condition groups.",
        "- Use VAS MF thresholds only after confirming scale direction and clinically meaningful cut points.",
        "- Use performance decline from Shot 1 to Shot 2 only after confirming trial order and task design.",
        "",
    ]
    LABEL_PLAN_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_phase3_report(
    humanm3_summary: PoseTableSummary,
    epfl_summary: PoseTableSummary,
    label_summary: LabelCandidateSummary,
) -> None:
    """Write Phase 3 feature sample report to docs and logs."""

    ensure_not_inside_dataset(PHASE3_REPORT_PATH, DATASET_ROOT)
    ensure_not_inside_dataset(LOG_PATH, DATASET_ROOT)
    lines = [
        "# Phase 3 Feature Sample Report",
        "",
        "This phase generated small processed samples only. No video frames were extracted, no MP4 content was read, and no model training was run.",
        "",
        "## Outputs",
        "",
        "| Output | Rows | Columns |",
        "| --- | ---: | ---: |",
        f"| `{_safe_rel(humanm3_summary.output_path)}` | {humanm3_summary.rows} | {humanm3_summary.columns} |",
        f"| `{_safe_rel(epfl_summary.output_path)}` | {epfl_summary.rows} | {epfl_summary.columns} |",
        f"| `{_safe_rel(label_summary.output_path)}` | {label_summary.rows} | {label_summary.columns} |",
        "",
        "## Human-M3 Pose Features",
        "",
        "- Source: small samples from basketball1 and basketball2 train/test `pose_calib/*.json`.",
        "- Joint schema: 15 joints from the Human-M3 README.",
        "- Successfully computed: " + ", ".join(humanm3_summary.features_with_values),
        "- All missing by design: " + ", ".join(humanm3_summary.features_all_missing),
        "- Ankle angles are missing because no distal foot/toe joint exists in the 15-joint schema.",
        "",
        "## EPFL Multiview Pose Features",
        "",
        "- Source: `pose_subject7.json` and `pose_subject12.json`; both are test subjects per README.",
        "- Joint schema: 13 joints from `vis.py`.",
        "- Successfully computed: " + ", ".join(epfl_summary.features_with_values),
        "- All missing by design: " + ", ".join(epfl_summary.features_all_missing),
        "- Hip flexion uses a same-side shoulder proxy because pelvis is absent.",
        "- Ankle angles are missing because no distal foot/toe joint exists.",
        "",
        "## 5742821 Label Candidates",
        "",
        "- Fatigue/sleep fields: " + ", ".join(label_summary.fatigue_sleep_fields),
        "- Performance fields: " + ", ".join(label_summary.performance_fields),
        "- Explicit fatigue label found: no",
        "- Explicit before/after label found: no",
        "",
        "## Training Readiness",
        "",
        "Do not start formal model training yet. The next step is data alignment and label-construction approval: map pose subjects/sessions to fatigue/sleep labels or define a validated label construction rule.",
        "",
    ]
    text = "\n".join(lines)
    PHASE3_REPORT_PATH.write_text(text, encoding="utf-8")
    LOG_PATH.write_text(text.replace("# Phase 3 Feature Sample Report", "Phase 3 Feature Sample Report"), encoding="utf-8")


def write_summary_json(
    humanm3_summary: PoseTableSummary,
    epfl_summary: PoseTableSummary,
    label_summary: LabelCandidateSummary,
) -> None:
    """Write machine-readable Phase 3 metadata."""

    ensure_not_inside_dataset(SUMMARY_JSON_PATH, DATASET_ROOT)
    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "small_sample_feature_table_generation",
        "safety": {
            "dataset_writes": False,
            "frame_extraction": False,
            "model_training": False,
            "full_dataset_processing": False,
        },
        "tables": {
            "humanm3_pose_features": _summary_to_dict(humanm3_summary),
            "epfl_multiview_pose_features": _summary_to_dict(epfl_summary),
            "fatigue_sleep_label_candidates": _summary_to_dict(label_summary),
        },
        "training_readiness": {
            "can_train_formal_model": False,
            "reason": "No confirmed subject/session alignment between pose features and fatigue/sleep labels.",
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for path in (PROCESSED_ROOT, METADATA_ROOT, LOG_PATH.parent, FEATURE_DESIGN_PATH.parent):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)

    humanm3_summary = build_humanm3_pose_feature_sample(
        DATASET_ROOT / "humanm3",
        PROCESSED_ROOT / "humanm3_pose_features_sample.csv",
        PROJECT_ROOT,
    )
    epfl_summary = build_epfl_multiview_pose_feature_sample(
        DATASET_ROOT / "sportcenter_multiview_dataset",
        PROCESSED_ROOT / "epfl_multiview_pose_features_sample.csv",
        PROJECT_ROOT,
    )
    label_summary = build_fatigue_sleep_label_candidates(
        DATASET_ROOT / "5742821" / "Data.xlsx",
        PROCESSED_ROOT / "fatigue_sleep_label_candidates.csv",
    )

    write_feature_table_design(humanm3_summary, epfl_summary, label_summary)
    write_label_plan(label_summary)
    write_phase3_report(humanm3_summary, epfl_summary, label_summary)
    write_summary_json(humanm3_summary, epfl_summary, label_summary)

    print("Small-sample feature generation complete.")
    print(f"Human-M3: {humanm3_summary.rows} rows x {humanm3_summary.columns} columns")
    print(f"EPFL multiview: {epfl_summary.rows} rows x {epfl_summary.columns} columns")
    print(f"Label candidates: {label_summary.rows} rows x {label_summary.columns} columns")
    print(f"Wrote: {humanm3_summary.output_path}")
    print(f"Wrote: {epfl_summary.output_path}")
    print(f"Wrote: {label_summary.output_path}")
    print(f"Wrote: {PHASE3_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
