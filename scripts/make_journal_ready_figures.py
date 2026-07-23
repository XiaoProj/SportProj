"""Generate Phase 7.5 journal-ready figures in a new output directory.

Existing Phase 7 figures are copied to `outputs/figures/archive_phase7/`
before new figures are written. Raw dataset files are not read.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.runtime import add_conda_dll_directories
from src.visualization.journal_figures import make_all_phase7_figures, phase7_figure_notes


DATASET_ROOT = PROJECT_ROOT / "dataset"
OLD_POSE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "pose_movement_quality"
OLD_TABULAR_ROOT = PROJECT_ROOT / "outputs" / "figures" / "tabular_fatigue_sleep_performance"
ARCHIVE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "archive_phase7"
POSE_READY_ROOT = PROJECT_ROOT / "outputs" / "figures" / "journal_ready" / "pose_movement_quality"
TABULAR_READY_ROOT = PROJECT_ROOT / "outputs" / "figures" / "journal_ready" / "tabular_fatigue_sleep_performance"
HUMANM3_PATH = PROJECT_ROOT / "outputs" / "processed" / "pose_movement_quality_humanm3_features_full.csv"
EPFL_PATH = PROJECT_ROOT / "outputs" / "processed" / "pose_movement_quality_epfl_multiview_features_full.csv"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
TABULAR_RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance"
FIGURE_NOTES_PATH = PROJECT_ROOT / "docs" / "figure_notes.md"
LOG_PATH = PROJECT_ROOT / "logs" / "phase7_5_journal_ready_figures.txt"


def archive_phase7_figures() -> list[Path]:
    """Copy existing Phase 7 PNG/PDF figures to archive without deleting originals."""

    copied: list[Path] = []
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for source_root in (OLD_POSE_ROOT, OLD_TABULAR_ROOT):
        if not source_root.exists():
            continue
        target_root = ARCHIVE_ROOT / source_root.name
        target_root.mkdir(parents=True, exist_ok=True)
        for path in source_root.glob("*.*"):
            if path.suffix.lower() not in {".png", ".pdf"}:
                continue
            target = target_root / path.name
            shutil.copy2(path, target)
            copied.append(target)
    return copied


def update_figure_notes(outputs: list[Path]) -> None:
    existing = FIGURE_NOTES_PATH.read_text(encoding="utf-8") if FIGURE_NOTES_PATH.exists() else "# Figure Notes\n"
    marker = "## Phase 7.5 Journal-Ready Figures"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n"
    section = phase7_figure_notes(outputs).replace("## Phase 7 Route A Journal-Draft Figures", marker)
    FIGURE_NOTES_PATH.write_text(existing.rstrip() + "\n\n" + section + "\n", encoding="utf-8")


def main() -> int:
    add_conda_dll_directories()
    for path in (ARCHIVE_ROOT, POSE_READY_ROOT, TABULAR_READY_ROOT, FIGURE_NOTES_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    archived = archive_phase7_figures()
    outputs = make_all_phase7_figures(
        humanm3_path=HUMANM3_PATH,
        epfl_path=EPFL_PATH,
        label_rules_path=LABEL_RULES_PATH,
        pose_output_root=POSE_READY_ROOT,
        tabular_output_root=TABULAR_READY_ROOT,
        tabular_result_root=TABULAR_RESULT_ROOT,
    )
    update_figure_notes(outputs)
    LOG_PATH.write_text(
        "Phase 7.5 journal-ready figures generated.\n"
        f"Archived files: {len(archived)}\n"
        + "\n".join(path.as_posix() for path in outputs)
        + "\n",
        encoding="utf-8",
    )
    print("Phase 7.5 journal-ready figures generated.")
    print(f"Archived files: {len(archived)}")
    for path in outputs:
        print(f"Wrote: {path}")
    print(f"Wrote: {FIGURE_NOTES_PATH}")
    print(f"Wrote: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
