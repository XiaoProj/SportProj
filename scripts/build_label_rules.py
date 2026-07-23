"""Build exploratory candidate label rules for 5742821.

The output labels are candidates only, not ground-truth fatigue labels.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.data.label_rules import build_label_rules, write_label_rule_doc


DATASET_ROOT = PROJECT_ROOT / "dataset"
INPUT_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_candidates.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "metadata" / "label_rule_summary.json"
DOC_PATH = PROJECT_ROOT / "docs" / "label_rule_comparison.md"
LOG_PATH = PROJECT_ROOT / "logs" / "label_rule_comparison.txt"


def main() -> int:
    for path in (OUTPUT_PATH, SUMMARY_PATH, DOC_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_label_rules(INPUT_PATH, OUTPUT_PATH, SUMMARY_PATH)
    write_label_rule_doc(summary, DOC_PATH)
    LOG_PATH.write_text(
        "\n".join(
            [
                "Label rule construction complete.",
                "Candidate labels are not ground-truth labels.",
                f"Rows: {summary.rows}",
                f"Columns: {summary.columns}",
                f"Output: {OUTPUT_PATH}",
                f"Summary: {SUMMARY_PATH}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("Label rule construction complete.")
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {SUMMARY_PATH}")
    print(f"Wrote: {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
