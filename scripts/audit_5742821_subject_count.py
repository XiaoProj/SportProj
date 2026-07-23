"""Audit parsed 5742821 subject counts from processed label-rule table."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset


DATASET_ROOT = PROJECT_ROOT / "dataset"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance" / "audit_5742821_subject_count.csv"


def run() -> dict:
    import pandas as pd

    ensure_not_inside_dataset(OUTPUT_PATH, DATASET_ROOT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(LABEL_RULES_PATH)
    rows = [
        {
            "source": "outputs/processed/fatigue_sleep_label_rules.csv",
            "rows": len(data),
            "unique_subjects": data["subject_id"].nunique() if "subject_id" in data else None,
            "condition_groups": ",".join(sorted(data["condition_group"].dropna().astype(str).unique())) if "condition_group" in data else "",
            "zenodo_expected_subjects_note": "Zenodo protocol reports 19 participants; parsed processed table contains the count shown here and should be reconciled before final inferential claims.",
        }
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    return {"output_path": OUTPUT_PATH.as_posix(), **rows[0]}


def main() -> int:
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
