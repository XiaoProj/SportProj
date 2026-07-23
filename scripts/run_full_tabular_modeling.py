"""Run full Route A 5742821 tabular exploratory modeling outputs."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.models.tabular_exploratory import run_exploratory_modeling


DATASET_ROOT = PROJECT_ROOT / "dataset"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance"
MODEL_ROOT = PROJECT_ROOT / "outputs" / "models" / "tabular_fatigue_sleep_performance"
LOG_PATH = PROJECT_ROOT / "logs" / "full_tabular_modeling.txt"


FULL_NAMES = {
    "model_comparison": "tabular_fatigue_sleep_performance_model_comparison_full.csv",
    "ablation": "tabular_fatigue_sleep_performance_ablation_results_full.csv",
    "predictions": "tabular_fatigue_sleep_performance_predictions_full.csv",
    "feature_importance": "tabular_fatigue_sleep_performance_feature_importance_full.csv",
    "cv_summary": "tabular_fatigue_sleep_performance_cv_summary_full.csv",
    "skipped_models": "tabular_fatigue_sleep_performance_skipped_models_full.csv",
}


def run() -> dict:
    for path in (RESULT_ROOT, MODEL_ROOT, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    summary = run_exploratory_modeling(
        label_rules_path=LABEL_RULES_PATH,
        result_root=RESULT_ROOT,
        model_root=MODEL_ROOT,
        save_models=True,
    )
    full_paths = {}
    for key, filename in FULL_NAMES.items():
        source = Path(summary["paths"][key])
        target = RESULT_ROOT / filename
        shutil.copy2(source, target)
        full_paths[key] = target.as_posix()
    summary["full_paths"] = full_paths
    LOG_PATH.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run()
    print("Full tabular exploratory modeling complete.")
    print(json.dumps({"full_paths": summary["full_paths"], "saved_models": len(summary["saved_models"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
