"""Check whether processed sample tables can be safely aligned."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.alignment import check_alignment
from src.data.inventory import ensure_not_inside_dataset


DATASET_ROOT = PROJECT_ROOT / "dataset"
HUMANM3_PATH = PROJECT_ROOT / "outputs" / "processed" / "humanm3_pose_features_sample.csv"
EPFL_PATH = PROJECT_ROOT / "outputs" / "processed" / "epfl_multiview_pose_features_sample.csv"
LABEL_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_candidates.csv"
TEMPLATE_PATH = PROJECT_ROOT / "outputs" / "processed" / "alignment_template.csv"
DOC_PATH = PROJECT_ROOT / "docs" / "data_alignment_feasibility.md"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "metadata" / "alignment_feasibility_summary.json"
LOG_PATH = PROJECT_ROOT / "logs" / "data_alignment_feasibility.txt"


def main() -> int:
    for path in (TEMPLATE_PATH, DOC_PATH, SUMMARY_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = check_alignment(HUMANM3_PATH, EPFL_PATH, LABEL_PATH, TEMPLATE_PATH, DOC_PATH)
    data = asdict(summary)
    data["doc_path"] = str(DOC_PATH)
    data["template_path"] = str(TEMPLATE_PATH)
    SUMMARY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG_PATH.write_text(
        "\n".join(
            [
                "Data alignment feasibility check complete.",
                f"Direct alignment possible: {summary.direct_alignment_possible}",
                f"Reason: {summary.reason}",
                f"Template: {TEMPLATE_PATH}",
                f"Doc: {DOC_PATH}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("Data alignment feasibility check complete.")
    print(f"Direct alignment possible: {summary.direct_alignment_possible}")
    print(f"Wrote: {TEMPLATE_PATH}")
    print(f"Wrote: {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
