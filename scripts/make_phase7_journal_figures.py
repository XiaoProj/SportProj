"""Generate Phase 7 Route A journal-draft figures.

The script reads only processed sample tables and tabular exploratory results.
It does not read raw videos, run ffmpeg, or load any raw dataset content.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.visualization.journal_figures import make_all_phase7_figures, phase7_figure_notes


DATASET_ROOT = PROJECT_ROOT / "dataset"
HUMANM3_PATH = PROJECT_ROOT / "outputs" / "processed" / "humanm3_pose_features_sample.csv"
EPFL_PATH = PROJECT_ROOT / "outputs" / "processed" / "epfl_multiview_pose_features_sample.csv"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
POSE_FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "pose_movement_quality"
TABULAR_FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "tabular_fatigue_sleep_performance"
TABULAR_RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance"
FIGURE_NOTES_PATH = PROJECT_ROOT / "docs" / "figure_notes.md"
POSE_QUALITY_REPORT = PROJECT_ROOT / "docs" / "pose_movement_quality_figure_quality_report.md"
LOG_PATH = PROJECT_ROOT / "logs" / "phase7_journal_figures.txt"


def _append_figure_notes(outputs: list[Path]) -> None:
    section = phase7_figure_notes(outputs)
    existing = FIGURE_NOTES_PATH.read_text(encoding="utf-8") if FIGURE_NOTES_PATH.exists() else "# Figure Notes\n"
    marker = "## Phase 7 Route A Journal-Draft Figures"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n"
    FIGURE_NOTES_PATH.write_text(existing + section + "\n", encoding="utf-8")


def _write_pose_quality_report(outputs: list[Path]) -> None:
    pose_outputs = [path for path in outputs if "pose_movement_quality" in path.as_posix()]
    lines = [
        "# Pose Movement Quality Figure Quality Report",
        "",
        "## Phase 7 Corrections",
        "",
        "- The previous missingness view is superseded by `pose_movement_quality_computational_missingness`.",
        "- Ankle-angle features are treated as structural unavailability because the current Human-M3 and EPFL multiview joint schemas do not include distal foot/toe joints.",
        "- `pose_movement_quality_feature_availability_matrix` separates available, proxy, and structurally unavailable feature states.",
        "- Knee valgus remains a proxy, not a validated frontal-plane biomechanical valgus measure.",
        "- PCA and radar figures are technical diagnostics only and should not be interpreted as fatigue or biological group-separation results.",
        "",
        "## Generated Pose Figures",
        "",
    ]
    lines.extend(f"- `{path.as_posix()}`" for path in pose_outputs)
    lines.extend(
        [
            "",
            "## Manuscript Placement",
            "",
            "- Core methods/results: feature availability, computational missingness, angle distributions, knee asymmetry, knee valgus proxy, pose completeness.",
            "- Exploratory appendix: PCA projection, radar comparison, correlation heatmap.",
            "",
        ]
    )
    POSE_QUALITY_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for path in (
        POSE_FIGURE_ROOT,
        TABULAR_FIGURE_ROOT,
        FIGURE_NOTES_PATH,
        POSE_QUALITY_REPORT,
        LOG_PATH,
    ):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    outputs = make_all_phase7_figures(
        humanm3_path=HUMANM3_PATH,
        epfl_path=EPFL_PATH,
        label_rules_path=LABEL_RULES_PATH,
        pose_output_root=POSE_FIGURE_ROOT,
        tabular_output_root=TABULAR_FIGURE_ROOT,
        tabular_result_root=TABULAR_RESULT_ROOT,
    )
    _append_figure_notes(outputs)
    _write_pose_quality_report(outputs)
    LOG_PATH.write_text(
        "Phase 7 journal figures generated from processed/results files only.\n"
        + "\n".join(path.as_posix() for path in outputs)
        + "\n",
        encoding="utf-8",
    )

    print("Phase 7 journal figures generated.")
    for path in outputs:
        print(f"Wrote: {path}")
    print(f"Wrote: {FIGURE_NOTES_PATH}")
    print(f"Wrote: {POSE_QUALITY_REPORT}")
    print(f"Wrote: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
