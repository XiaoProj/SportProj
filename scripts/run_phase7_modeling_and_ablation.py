"""Run Phase 7 Route A tabular exploratory modeling and ablation.

This script only uses `outputs/processed/fatigue_sleep_label_rules.csv`.
It does not load pose features, read videos, read MP4 content, or touch
`dataset/`.
"""

from __future__ import annotations

import json
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
DOC_REPORT = PROJECT_ROOT / "docs" / "phase7_modeling_and_ablation_report.md"
DOC_MODEL_CARDS = PROJECT_ROOT / "docs" / "tabular_fatigue_sleep_performance_model_cards.md"
LOG_PATH = PROJECT_ROOT / "logs" / "phase7_modeling_and_ablation.txt"


def _write_report(summary: dict) -> None:
    lines = [
        "# Phase 7 Modeling and Ablation Report",
        "",
        "## Scope",
        "",
        "Phase 7 follows Route A. The modeling pipeline is limited to the 5742821 tabular fatigue/sleep/free-throw performance data.",
        "Human-M3 and EPFL pose features are not loaded or merged with 5742821 labels.",
        "",
        "All targets are exploratory candidate labels or candidate scores. No ground-truth fatigue classifier is claimed.",
        "",
        "## Inputs",
        "",
        f"- Label-rule table: `{LABEL_RULES_PATH.as_posix()}`",
        "",
        "## Outputs",
        "",
    ]
    for label, path in summary["paths"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Modeling Tasks",
            "",
            "- condition-group candidate classification",
            "- VAS MF continuous-score regression",
            "- VAS MF median split candidate classification",
            "- VAS MF tertile candidate classification, if class counts allow",
            "- Shot 2 minus Shot 1 performance-change candidate regression",
            "- performance-decline candidate classification, if class counts allow",
            "",
            "## Models",
            "",
            "Classification models include DummyClassifier, Logistic Regression, Random Forest, Gradient Boosting, SVM when sample size allows, and optional XGBoost/LightGBM when installed.",
            "Regression models include DummyRegressor, Linear Regression, Ridge, Random Forest Regressor, Gradient Boosting Regressor, and SVR when sample size allows.",
            "",
            "## Validation",
            "",
            "The default validation strategy is leave-one-subject-out cross-validation when `subject_id` is available.",
            "This is more conservative than row-level leave-one-out because each subject contributes multiple condition rows.",
            "",
            "## Leakage Controls",
            "",
            "- Direct condition encodings are excluded when predicting condition-group candidates.",
            "- VAS MF variables and their direct derivatives are excluded when predicting VAS MF score or VAS-threshold candidate labels.",
            "- Direct shot-change and performance-decline variables are excluded from performance-change targets.",
            "- Shot 2 variables are excluded for performance-change targets to avoid directly reconstructing Shot 2 minus Shot 1.",
            "",
            "## Model Persistence",
            "",
            "Only 5742821 tabular exploratory models are saved. No pose-fatigue model is saved.",
        ]
    )
    if summary["saved_models"]:
        for item in summary["saved_models"]:
            lines.append(f"- {item['task_name']} / {item['model_name']}: `{item['model_path']}`")
    else:
        lines.append("- No model artifacts were saved because no non-baseline task passed basic validation.")
    lines.extend(
        [
            "",
            "## SHAP / Importance",
            "",
            f"- SHAP installed: `{summary['shap_available']}`",
            "- Permutation importance is generated for successful exploratory models.",
            "- SHAP is optional and skipped without failing the main Phase 7 pipeline when unavailable.",
            "",
            "## Interpretation Limits",
            "",
            "- These outputs are suitable for exploratory appendix reporting and methods validation.",
            "- They are not final model-performance estimates.",
            "- They do not support a claim that pose features predict fatigue.",
            "",
        ]
    )
    DOC_REPORT.write_text("\n".join(lines), encoding="utf-8")


def _write_model_cards_index(summary: dict) -> None:
    lines = [
        "# Tabular Fatigue/Sleep/Performance Model Cards",
        "",
        "This index lists exploratory model-card artifacts saved during Phase 7.",
        "Every listed model is trained only on 5742821 tabular variables and is not a final fatigue classifier.",
        "",
    ]
    if not summary["saved_models"]:
        lines.append("No exploratory model bundles were saved.")
    for item in summary["saved_models"]:
        lines.extend(
            [
                f"## {item['task_name']} / {item['model_name']}",
                "",
                f"- Model: `{item['model_path']}`",
                f"- Feature columns: `{item['feature_columns_path']}`",
                f"- Metadata: `{item['metadata_path']}`",
                f"- Model card: `{item['model_card_path']}`",
                "",
            ]
        )
    DOC_MODEL_CARDS.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for path in (RESULT_ROOT, MODEL_ROOT, DOC_REPORT, DOC_MODEL_CARDS, LOG_PATH):
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
    _write_report(summary)
    _write_model_cards_index(summary)

    log_lines = [
        "Phase 7 Route A modeling complete.",
        "Data scope: 5742821 tabular processed data only.",
        "No pose features loaded or merged.",
        f"Summary: {json.dumps(summary, indent=2)}",
        f"Report: {DOC_REPORT}",
        f"Model cards index: {DOC_MODEL_CARDS}",
    ]
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("Phase 7 tabular exploratory modeling and ablation complete.")
    for label, path in summary["paths"].items():
        print(f"Wrote {label}: {path}")
    for item in summary["saved_models"]:
        print(f"Saved exploratory model: {item['model_path']}")
    print(f"Wrote: {DOC_REPORT}")
    print(f"Wrote: {DOC_MODEL_CARDS}")
    print(f"Wrote: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
