"""Run Phase 7.5 integrated modeling refresh and journal-ready figure rebuild."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_journal_ready_figures import main as make_journal_ready_main
from src.data.inventory import ensure_not_inside_dataset
from src.models.tabular_exploratory import run_exploratory_modeling


DATASET_ROOT = PROJECT_ROOT / "dataset"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance"
MODEL_ROOT = PROJECT_ROOT / "outputs" / "models" / "tabular_fatigue_sleep_performance"
DOC_PATH = PROJECT_ROOT / "docs" / "phase7_5_integrated_fix_report.md"
LOG_PATH = PROJECT_ROOT / "logs" / "phase7_5_integrated_fix.txt"


def main() -> int:
    for path in (RESULT_ROOT, MODEL_ROOT, DOC_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    dependency_check = subprocess.run(
        [sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "check_and_report_dependencies.py")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    summary = run_exploratory_modeling(
        label_rules_path=LABEL_RULES_PATH,
        result_root=RESULT_ROOT,
        model_root=MODEL_ROOT,
        save_models=True,
    )
    figure_status = make_journal_ready_main()
    lines = [
        "# Phase 7.5 Integrated Fix Report",
        "",
        "## Scope",
        "",
        "Route A remains enforced: pose movement-quality analysis and 5742821 tabular exploratory ML are separate.",
        "No pose-fatigue merge or pose-based fatigue classifier was run.",
        "",
        "## Dependency Check",
        "",
        "Dependency check was run at the start of the integrated fix.",
        f"- Return code: `{dependency_check.returncode}`",
        "",
        "## Tabular Exploratory Modeling Outputs",
        "",
    ]
    for key, path in summary["paths"].items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(
        [
            "",
            "## Saved Models",
            "",
        ]
    )
    for item in summary["saved_models"]:
        lines.append(f"- {item['task_name']} / {item['model_name']}: `{item['model_path']}`")
    lines.extend(
        [
            "",
            "## Optional Dependencies",
            "",
            f"- SHAP available after install attempt: `{summary['shap_available']}`",
            "- XGBoost is used when importable; LightGBM remains skipped if import is unavailable.",
            "",
            "## Figure Regeneration",
            "",
            f"- Journal-ready figure generation return code: `{figure_status}`",
            "- Old Phase 7 figures were copied to `outputs/figures/archive_phase7/` before new figures were written.",
            "",
            "## Limitations",
            "",
            "- Candidate labels remain exploratory.",
            "- Small-sample CV metrics are not final performance estimates.",
            "- Pose figures do not support fatigue prediction claims.",
            "",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    LOG_PATH.write_text(
        "Phase 7.5 integrated fix complete.\n"
        + json.dumps(summary, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("Phase 7.5 integrated fix complete.")
    print(f"Wrote: {DOC_PATH}")
    print(f"Wrote: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
