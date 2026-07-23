"""Run full/large-sample Route A pose movement-quality processing."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.features.full_pose_processing import (
    process_epfl_multiview_full,
    process_epfl_single_metadata,
    process_humanm3_full,
    write_pose_analysis_outputs,
)


DATASET_ROOT = PROJECT_ROOT / "dataset"
PROCESSED_ROOT = PROJECT_ROOT / "outputs" / "processed"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "pose_movement_quality"
LOG_PATH = PROJECT_ROOT / "logs" / "full_pose_movement_quality_processing.txt"


def run() -> dict:
    for path in (PROCESSED_ROOT, RESULT_ROOT, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    human_path = PROCESSED_ROOT / "pose_movement_quality_humanm3_features_full.csv"
    epfl_path = PROCESSED_ROOT / "pose_movement_quality_epfl_multiview_features_full.csv"
    single_path = PROCESSED_ROOT / "pose_movement_quality_epfl_single_metadata_summary.csv"
    combined_path = PROCESSED_ROOT / "pose_movement_quality_features_analysis_table_full.csv"

    human = process_humanm3_full(
        dataset_root=DATASET_ROOT / "humanm3",
        output_path=human_path,
        project_root=PROJECT_ROOT,
    )
    epfl = process_epfl_multiview_full(
        dataset_root=DATASET_ROOT / "sportcenter_multiview_dataset",
        output_path=epfl_path,
        project_root=PROJECT_ROOT,
    )
    single = process_epfl_single_metadata(
        dataset_root=DATASET_ROOT / "sportcenter_camerapose_dataset",
        output_path=single_path,
        project_root=PROJECT_ROOT,
    )
    result_paths = write_pose_analysis_outputs(
        human_path=human_path,
        epfl_path=epfl_path,
        combined_path=combined_path,
        result_root=RESULT_ROOT,
        audit_rows=[human, epfl, single],
    )
    summary = {"humanm3": human, "epfl_multiview": epfl, "epfl_single": single, "result_paths": result_paths}
    LOG_PATH.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run()
    print("Full/large-sample pose movement-quality processing complete.")
    print(json.dumps({k: v for k, v in summary.items() if k != "result_paths"}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
