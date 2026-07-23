"""Run Phase 4 smoke tests using processed sample tables only.

Under Route A, tabular smoke tests are exploratory and pose checks remain
movement-quality summaries. The script never merges pose rows with 5742821
fatigue/sleep candidate labels.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.models.smoke_test import run_tabular_smoke_tests, summarize_pose_features, write_modeling_readiness_doc


DATASET_ROOT = PROJECT_ROOT / "dataset"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
HUMANM3_PATH = PROJECT_ROOT / "outputs" / "processed" / "humanm3_pose_features_sample.csv"
EPFL_PATH = PROJECT_ROOT / "outputs" / "processed" / "epfl_multiview_pose_features_sample.csv"
TABULAR_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance_phase4_smoke_test_results.csv"
POSE_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "results" / "pose_movement_quality_phase4_feature_summary.csv"
DOC_PATH = PROJECT_ROOT / "docs" / "phase4_modeling_readiness.md"
LOG_PATH = PROJECT_ROOT / "logs" / "phase4_smoke_tests.txt"


def main() -> int:
    for path in (TABULAR_RESULTS_PATH, POSE_SUMMARY_PATH, DOC_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)

    tabular_results = run_tabular_smoke_tests(LABEL_RULES_PATH, TABULAR_RESULTS_PATH)
    pose_summary = summarize_pose_features(HUMANM3_PATH, EPFL_PATH, POSE_SUMMARY_PATH)
    write_modeling_readiness_doc(DOC_PATH, tabular_results, POSE_SUMMARY_PATH)
    LOG_PATH.write_text(
        "\n".join(
            [
                "Phase 4 smoke tests complete.",
                "Route A active: tabular exploratory smoke tests and pose movement-quality summaries only.",
                "These are diagnostic smoke tests only, not final results.",
                "Pose rows were not merged with 5742821 candidate labels.",
                f"Tabular results: {TABULAR_RESULTS_PATH}",
                f"Pose summary rows: {len(pose_summary)}",
                f"Pose summary: {POSE_SUMMARY_PATH}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("Phase 4 smoke tests complete under Route A.")
    print("No pose-fatigue merge was performed.")
    print(f"Wrote: {TABULAR_RESULTS_PATH}")
    print(f"Wrote: {POSE_SUMMARY_PATH}")
    print(f"Wrote: {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
