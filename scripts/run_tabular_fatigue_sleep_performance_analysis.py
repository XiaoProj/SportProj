"""Run Route A tabular fatigue/sleep/free-throw exploratory analysis.

Inputs are processed 5742821 candidate-label tables only. No pose features are
loaded or merged, and no final model artifacts are saved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset


DATASET_ROOT = PROJECT_ROOT / "dataset"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance"
DOC_PATH = PROJECT_ROOT / "docs" / "tabular_fatigue_sleep_performance_ml_notes.md"
LOG_PATH = PROJECT_ROOT / "logs" / "tabular_fatigue_sleep_performance_analysis.txt"

FEATURES_FOR_CONDITION = [
    "shot_1_accuracy",
    "shot_2_accuracy",
    "shot_accuracy_change_shot2_minus_shot1",
    "vas_mf_mean",
    "vas_mot_mean",
    "prontezza",
    "questions_accuracy",
]

FEATURES_FOR_VAS = [
    "shot_1_accuracy",
    "shot_2_accuracy",
    "shot_accuracy_change_shot2_minus_shot1",
    "vas_mot_mean",
    "prontezza",
    "questions_accuracy",
]


def _load_rules():
    import pandas as pd

    data = pd.read_csv(LABEL_RULES_PATH)
    numeric = set(FEATURES_FOR_CONDITION + FEATURES_FOR_VAS + ["vas_mf_median_split_candidate", "condition_group_rule_candidate", "performance_decline_candidate"])
    for column in numeric:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _condition_summary(data):
    rows = []
    grouped = data.groupby("condition_group", dropna=False)
    for condition, subset in grouped:
        rows.append(
            {
                "condition_group": condition,
                "condition_interpretation": "MF or SR+MF parsed column group; Control/SR reconstruction uses Shot 1 protocol interpretation",
                "n_rows": int(len(subset)),
                "n_subjects": int(subset["subject_id"].nunique()),
                "shot_1_accuracy_mean": subset["shot_1_accuracy"].mean(),
                "shot_2_accuracy_mean": subset["shot_2_accuracy"].mean(),
                "free_throw_accuracy_mean": None,
                "shot2_minus_shot1_mean": subset["shot_accuracy_change_shot2_minus_shot1"].mean(),
                "vas_mf_mean": subset["vas_mf_mean"].mean(),
                "vas_mot_mean": subset["vas_mot_mean"].mean(),
                "questions_accuracy_mean": subset["questions_accuracy"].mean(),
                "prontezza_mean": subset["prontezza"].mean(),
                "label_status": "candidate_protocol_condition_not_ground_truth_fatigue_label",
            }
        )

    reconstructed = []
    mapping = [
        ("mental_fatigue", "Control", "shot_1_accuracy"),
        ("mental_fatigue", "MF", "shot_2_accuracy"),
        ("mental_fatigue_plus_sleep_deprivation", "SR", "shot_1_accuracy"),
        ("mental_fatigue_plus_sleep_deprivation", "SR+MF", "shot_2_accuracy"),
    ]
    for source_group, condition, shot_column in mapping:
        subset = data[data["condition_group"] == source_group]
        reconstructed.append(
            {
                "condition_group": condition,
                "condition_interpretation": f"reconstructed free-throw condition from {source_group} {shot_column}; protocol-derived candidate",
                "n_rows": int(len(subset)),
                "n_subjects": int(subset["subject_id"].nunique()),
                "shot_1_accuracy_mean": subset[shot_column].mean(),
                "shot_2_accuracy_mean": None,
                "free_throw_accuracy_mean": subset[shot_column].mean(),
                "shot2_minus_shot1_mean": None,
                "vas_mf_mean": None,
                "vas_mot_mean": None,
                "questions_accuracy_mean": subset["questions_accuracy"].mean(),
                "prontezza_mean": subset["prontezza"].mean(),
                "label_status": "reconstructed_performance_condition_candidate",
            }
        )
    return rows + reconstructed


def _summary_by_columns(data, columns: list[str], group_column: str = "condition_group"):
    rows = []
    for group, subset in data.groupby(group_column, dropna=False):
        for column in columns:
            values = subset[column].dropna()
            rows.append(
                {
                    group_column: group,
                    "variable": column,
                    "n": int(values.shape[0]),
                    "mean": values.mean() if not values.empty else None,
                    "std": values.std() if values.shape[0] > 1 else None,
                    "median": values.median() if not values.empty else None,
                    "min": values.min() if not values.empty else None,
                    "max": values.max() if not values.empty else None,
                    "analysis_status": "exploratory",
                }
            )
    return rows


def _classification_cv(data, target: str, features: list[str], task_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    available = [feature for feature in features if feature in data.columns]
    frame = data.dropna(subset=[target]).copy()
    y = frame[target].astype(int)
    counts = y.value_counts().to_dict()
    if len(counts) < 2 or min(counts.values()) < 3:
        return (
            {
                "task_name": task_name,
                "task_type": "classification",
                "target": target,
                "status": "skipped",
                "reason": f"insufficient classes for exploratory CV: {counts}",
                "n_samples": int(len(frame)),
                "n_features": len(available),
            },
            [],
            [],
        )
    x = frame[available]
    groups = frame["subject_id"].astype(str)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    cv = LeaveOneGroupOut()
    y_pred = cross_val_predict(model, x, y, cv=cv, groups=groups, method="predict")
    y_score = cross_val_predict(model, x, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_score) if len(np.unique(y)) == 2 else None
    metric = {
        "task_name": task_name,
        "task_type": "classification",
        "target": target,
        "status": "ok_exploratory_candidate_label",
        "reason": "Leave-one-subject-out exploratory CV; not final model performance",
        "n_samples": int(len(frame)),
        "n_features": len(available),
        "auc": float(auc) if auc is not None else None,
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    preds = []
    for idx, true_value, pred_value, score in zip(frame.index, y, y_pred, y_score):
        preds.append(
            {
                "task_name": task_name,
                "row_index": int(idx),
                "subject_id": frame.loc[idx, "subject_id"],
                "condition_group": frame.loc[idx, "condition_group"],
                "target": target,
                "true_value": int(true_value),
                "predicted_value": int(pred_value),
                "prediction_score": float(score),
                "prediction_status": "exploratory_candidate_label_not_ground_truth",
            }
        )
    model.fit(x, y)
    coefs = model.named_steps["model"].coef_[0]
    importance = [
        {
            "task_name": task_name,
            "feature": feature,
            "importance": abs(float(coef)),
            "signed_coefficient": float(coef),
            "importance_status": "exploratory_full_sample_coefficient_not_final_model",
        }
        for feature, coef in zip(available, coefs)
    ]
    return metric, preds, importance


def _regression_cv(data, target: str, features: list[str], task_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    available = [feature for feature in features if feature in data.columns]
    frame = data.dropna(subset=[target]).copy()
    if len(frame) < 8:
        return (
            {
                "task_name": task_name,
                "task_type": "regression",
                "target": target,
                "status": "skipped",
                "reason": "fewer than 8 rows",
                "n_samples": int(len(frame)),
                "n_features": len(available),
            },
            [],
            [],
        )
    x = frame[available]
    y = frame[target].astype(float)
    groups = frame["subject_id"].astype(str)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    y_pred = cross_val_predict(model, x, y, cv=LeaveOneGroupOut(), groups=groups)
    metric = {
        "task_name": task_name,
        "task_type": "regression",
        "target": target,
        "status": "ok_exploratory_continuous_candidate",
        "reason": "Leave-one-subject-out exploratory CV; not final model performance",
        "n_samples": int(len(frame)),
        "n_features": len(available),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }
    preds = []
    for idx, true_value, pred_value in zip(frame.index, y, y_pred):
        preds.append(
            {
                "task_name": task_name,
                "row_index": int(idx),
                "subject_id": frame.loc[idx, "subject_id"],
                "condition_group": frame.loc[idx, "condition_group"],
                "target": target,
                "true_value": float(true_value),
                "predicted_value": float(pred_value),
                "prediction_score": "",
                "prediction_status": "exploratory_candidate_score_not_ground_truth",
            }
        )
    model.fit(x, y)
    coefs = model.named_steps["model"].coef_
    importance = [
        {
            "task_name": task_name,
            "feature": feature,
            "importance": abs(float(coef)),
            "signed_coefficient": float(coef),
            "importance_status": "exploratory_full_sample_coefficient_not_final_model",
        }
        for feature, coef in zip(available, coefs)
    ]
    return metric, preds, importance


def _write_ml_notes(metrics_path: Path, predictions_path: Path, importance_path: Path) -> None:
    lines = [
        "# Tabular Fatigue/Sleep/Performance ML Notes",
        "",
        "These analyses use only 5742821 processed tabular variables.",
        "They are exploratory and use candidate labels or candidate continuous scores.",
        "",
        "No pose features are used. No Human-M3 or EPFL rows are loaded or merged.",
        "No model file is saved.",
        "",
        "## Outputs",
        "",
        f"- Metrics: `{metrics_path.as_posix()}`",
        f"- Predictions: `{predictions_path.as_posix()}`",
        f"- Feature importance: `{importance_path.as_posix()}`",
        "",
        "## Label Status",
        "",
        "- condition-group labels are protocol-condition candidates",
        "- VAS MF median split is exploratory only",
        "- performance decline is exploratory and depends on protocol-derived Shot 1 / Shot 2 ordering",
        "- no ground-truth fatigue label is claimed",
        "",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import pandas as pd

    for path in (RESULT_ROOT, DOC_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_rules()
    condition_summary = pd.DataFrame(_condition_summary(data))
    vas_summary = pd.DataFrame(_summary_by_columns(data, ["vas_mf_mean", "vas_mot_mean", "vas_mf_1", "vas_mf_2", "vas_mf_3", "vas_mot_1", "vas_mot_2", "vas_mot_3"]))
    shot_summary = pd.DataFrame(_summary_by_columns(data, ["shot_1_accuracy", "shot_2_accuracy", "shot_accuracy_change_shot2_minus_shot1", "questions_accuracy", "prontezza"]))

    metrics = []
    predictions = []
    importances = []
    tasks = [
        _classification_cv(data, "condition_group_rule_candidate", FEATURES_FOR_CONDITION, "condition_group_candidate_classification"),
        _regression_cv(data, "vas_mf_mean", FEATURES_FOR_VAS, "vas_mf_continuous_score_regression"),
        _classification_cv(data, "vas_mf_median_split_candidate", FEATURES_FOR_VAS, "vas_mf_median_split_candidate_classification"),
        _classification_cv(data, "performance_decline_candidate", ["vas_mf_mean", "vas_mot_mean", "prontezza", "questions_accuracy"], "performance_decline_candidate_classification"),
    ]
    for metric, preds, imp in tasks:
        metrics.append(metric)
        predictions.extend(preds)
        importances.extend(imp)

    outputs = {
        "condition": RESULT_ROOT / "tabular_fatigue_sleep_performance_condition_summary.csv",
        "vas": RESULT_ROOT / "tabular_fatigue_sleep_performance_vas_summary.csv",
        "shot": RESULT_ROOT / "tabular_fatigue_sleep_performance_shot_change_summary.csv",
        "ml": RESULT_ROOT / "tabular_fatigue_sleep_performance_ml_results.csv",
        "pred": RESULT_ROOT / "tabular_fatigue_sleep_performance_predictions.csv",
        "imp": RESULT_ROOT / "tabular_fatigue_sleep_performance_feature_importance.csv",
    }
    for path in outputs.values():
        ensure_not_inside_dataset(path, DATASET_ROOT)

    condition_summary.to_csv(outputs["condition"], index=False)
    vas_summary.to_csv(outputs["vas"], index=False)
    shot_summary.to_csv(outputs["shot"], index=False)
    pd.DataFrame(metrics).to_csv(outputs["ml"], index=False)
    pd.DataFrame(predictions).to_csv(outputs["pred"], index=False)
    pd.DataFrame(importances).to_csv(outputs["imp"], index=False)
    _write_ml_notes(outputs["ml"], outputs["pred"], outputs["imp"])

    LOG_PATH.write_text(
        "\n".join(
            [
                "Route A tabular fatigue/sleep/performance analysis complete.",
                "No pose features were used.",
                "No model file was saved.",
                f"Condition summary: {outputs['condition']}",
                f"ML results: {outputs['ml']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("Tabular fatigue/sleep/performance analysis complete.")
    for path in outputs.values():
        print(f"Wrote: {path}")
    print(f"Wrote: {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
