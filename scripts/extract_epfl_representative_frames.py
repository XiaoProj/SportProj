"""Extract EPFL multiview video metadata and start/middle/near-end frames."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.video.representative_frames import extract_epfl_multiview_representative_frames


DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "pose_movement_quality"
FRAME_ROOT = PROJECT_ROOT / "outputs" / "extracted" / "video_frames" / "epfl_multiview"
FIGURE_BASE = PROJECT_ROOT / "outputs" / "figures" / "pose_movement_quality" / "pose_movement_quality_epfl_multiview_camera_examples"
LOG_PATH = PROJECT_ROOT / "logs" / "epfl_multiview_representative_frames.txt"


def run() -> dict:
    for path in (RESULT_ROOT, FRAME_ROOT, FIGURE_BASE, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    summary = extract_epfl_multiview_representative_frames(
        multicam_root=DATASET_ROOT / "sportcenter_multiview_dataset" / "multicam",
        frame_output_root=FRAME_ROOT,
        metadata_csv=RESULT_ROOT / "epfl_multiview_video_metadata.csv",
        audit_csv=RESULT_ROOT / "epfl_multiview_representative_frames_audit.csv",
        figure_base=FIGURE_BASE,
        project_root=PROJECT_ROOT,
    )
    LOG_PATH.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run()
    print("EPFL video metadata/representative frames step complete.")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
