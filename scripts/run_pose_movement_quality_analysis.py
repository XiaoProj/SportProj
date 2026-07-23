"""Run Route A pose movement-quality exploratory analysis.

Inputs are the Phase 3 small-sample pose feature tables under outputs/processed.
This script does not read dataset/, MP4 files, frames, or fatigue labels.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset


DATASET_ROOT = PROJECT_ROOT / "dataset"
HUMANM3_PATH = PROJECT_ROOT / "outputs" / "processed" / "humanm3_pose_features_sample.csv"
EPFL_PATH = PROJECT_ROOT / "outputs" / "processed" / "epfl_multiview_pose_features_sample.csv"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "pose_movement_quality"
LOG_PATH = PROJECT_ROOT / "logs" / "pose_movement_quality_analysis.txt"

FEATURES = [
    "left_knee_flexion_angle",
    "right_knee_flexion_angle",
    "left_hip_flexion_angle",
    "right_hip_flexion_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "left_right_knee_angle_difference",
    "left_right_ankle_angle_difference",
    "left_knee_valgus_proxy",
    "right_knee_valgus_proxy",
    "pose_valid_joint_count",
    "pose_expected_joint_count",
    "pose_missing_joint_ratio",
]

ANGLE_FEATURES = [
    "left_knee_flexion_angle",
    "right_knee_flexion_angle",
    "left_hip_flexion_angle",
    "right_hip_flexion_angle",
]


def _load_pose_tables():
    import pandas as pd

    humanm3 = pd.read_csv(HUMANM3_PATH)
    epfl = pd.read_csv(EPFL_PATH)
    humanm3["pose_dataset_source"] = "Human-M3"
    epfl["pose_dataset_source"] = "EPFL multiview"
    combined = pd.concat([humanm3, epfl], ignore_index=True, sort=False)
    for feature in FEATURES:
        if feature in combined.columns:
            combined[feature] = pd.to_numeric(combined[feature], errors="coerce")
    return combined


def _feature_summary(frame):
    rows = []
    for dataset_name, subset in frame.groupby("pose_dataset_source"):
        for feature in FEATURES:
            values = subset[feature] if feature in subset.columns else None
            if values is None:
                continue
            values = values.dropna()
            rows.append(
                {
                    "pose_dataset_source": dataset_name,
                    "feature": feature,
                    "n_total": int(len(subset)),
                    "n_non_missing": int(values.shape[0]),
                    "mean": values.mean() if not values.empty else None,
                    "std": values.std() if values.shape[0] > 1 else None,
                    "min": values.min() if not values.empty else None,
                    "q1": values.quantile(0.25) if not values.empty else None,
                    "median": values.median() if not values.empty else None,
                    "q3": values.quantile(0.75) if not values.empty else None,
                    "max": values.max() if not values.empty else None,
                }
            )
    return rows


def _missingness_summary(frame):
    rows = []
    for dataset_name, subset in frame.groupby("pose_dataset_source"):
        for feature in FEATURES:
            if feature not in subset.columns:
                continue
            values = subset[feature]
            rows.append(
                {
                    "pose_dataset_source": dataset_name,
                    "feature": feature,
                    "n_total": int(len(values)),
                    "n_missing": int(values.isna().sum()),
                    "n_non_missing": int(values.notna().sum()),
                    "missing_rate": float(values.isna().mean()),
                    "analysis_type": "pose_movement_quality_missingness",
                }
            )
    return rows


def _dataset_comparison(frame):
    import math

    rows = []
    for feature in ANGLE_FEATURES + [
        "left_right_knee_angle_difference",
        "left_knee_valgus_proxy",
        "right_knee_valgus_proxy",
    ]:
        if feature not in frame.columns:
            continue
        human = frame.loc[frame["pose_dataset_source"] == "Human-M3", feature].dropna()
        epfl = frame.loc[frame["pose_dataset_source"] == "EPFL multiview", feature].dropna()
        pooled = None
        cohen_d = None
        if len(human) > 1 and len(epfl) > 1:
            pooled_var = ((len(human) - 1) * human.var() + (len(epfl) - 1) * epfl.var()) / (len(human) + len(epfl) - 2)
            pooled = math.sqrt(pooled_var) if pooled_var >= 0 else None
            cohen_d = (epfl.mean() - human.mean()) / pooled if pooled and pooled > 0 else None
        rows.append(
            {
                "feature": feature,
                "humanm3_n": int(len(human)),
                "epfl_multiview_n": int(len(epfl)),
                "humanm3_mean": human.mean() if not human.empty else None,
                "epfl_multiview_mean": epfl.mean() if not epfl.empty else None,
                "epfl_minus_humanm3_mean": (epfl.mean() - human.mean()) if not human.empty and not epfl.empty else None,
                "pooled_sd": pooled,
                "cohen_d_exploratory": cohen_d,
                "interpretation": "dataset_source_comparison_only_not_fatigue_analysis",
            }
        )
    return rows


def main() -> int:
    import pandas as pd

    for path in (RESULT_ROOT, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)

    frame = _load_pose_tables()
    feature_summary = pd.DataFrame(_feature_summary(frame))
    missingness = pd.DataFrame(_missingness_summary(frame))
    comparison = pd.DataFrame(_dataset_comparison(frame))

    output_feature = RESULT_ROOT / "pose_movement_quality_feature_summary.csv"
    output_missing = RESULT_ROOT / "pose_movement_quality_missingness_summary.csv"
    output_comparison = RESULT_ROOT / "pose_movement_quality_dataset_comparison.csv"
    for path in (output_feature, output_missing, output_comparison):
        ensure_not_inside_dataset(path, DATASET_ROOT)

    feature_summary.to_csv(output_feature, index=False)
    missingness.to_csv(output_missing, index=False)
    comparison.to_csv(output_comparison, index=False)

    LOG_PATH.write_text(
        "\n".join(
            [
                "Route A pose movement-quality analysis complete.",
                "No fatigue labels were used.",
                "No pose-fatigue merge was performed.",
                f"Rows analyzed: {len(frame)}",
                f"Feature summary: {output_feature}",
                f"Missingness summary: {output_missing}",
                f"Dataset comparison: {output_comparison}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("Pose movement-quality analysis complete.")
    print(f"Wrote: {output_feature}")
    print(f"Wrote: {output_missing}")
    print(f"Wrote: {output_comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
