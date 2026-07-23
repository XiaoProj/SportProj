"""Generate read-only dataset inventory reports.

Outputs:
  - logs/dataset_inventory.txt
  - docs/dataset_summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import build_inventory, write_markdown_summary, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only dataset inventory scanner.")
    parser.add_argument("--dataset-root", default="dataset", help="Raw dataset root. Must be read-only.")
    parser.add_argument("--log-path", default="logs/dataset_inventory.txt", help="Plain-text report path.")
    parser.add_argument("--summary-path", default="docs/dataset_summary.md", help="Markdown summary path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    log_path = (PROJECT_ROOT / args.log_path).resolve()
    summary_path = (PROJECT_ROOT / args.summary_path).resolve()

    inventory = build_inventory(dataset_root)
    write_text_report(inventory, log_path, dataset_root)
    write_markdown_summary(inventory, summary_path, dataset_root)

    print(f"Read-only scan complete: {inventory.total_files} files across {inventory.dataset_count} datasets")
    print(f"Wrote: {log_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
