"""Check optional dependencies and current skipped items for Phase 7.5.

This script does not install packages and does not read or write `dataset/`.
It only inspects Python import availability and existing result tables.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset


DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "results"
TABULAR_RESULT_ROOT = RESULT_ROOT / "tabular_fatigue_sleep_performance"
CSV_PATH = RESULT_ROOT / "dependency_status.csv"
DOC_PATH = PROJECT_ROOT / "docs" / "dependency_and_skipped_items_report.md"
LOG_PATH = PROJECT_ROOT / "logs" / "dependency_and_skipped_items.txt"


DEPENDENCIES = [
    {
        "name": "shap",
        "import_name": "shap",
        "role": "optional interpretability",
        "required_for_phase7_5": "optional",
    },
    {
        "name": "xgboost",
        "import_name": "xgboost",
        "role": "optional tree boosting models",
        "required_for_phase7_5": "optional",
    },
    {
        "name": "lightgbm",
        "import_name": "lightgbm",
        "role": "optional tree boosting models",
        "required_for_phase7_5": "optional",
    },
    {
        "name": "joblib",
        "import_name": "joblib",
        "role": "model persistence",
        "required_for_phase7_5": "required",
    },
    {
        "name": "scipy",
        "import_name": "scipy",
        "role": "statistics and correlations",
        "required_for_phase7_5": "recommended",
    },
    {
        "name": "statsmodels",
        "import_name": "statsmodels",
        "role": "optional statistical models",
        "required_for_phase7_5": "optional",
    },
    {
        "name": "seaborn",
        "import_name": "seaborn",
        "role": "optional plotting convenience",
        "required_for_phase7_5": "optional",
    },
    {
        "name": "openpyxl",
        "import_name": "openpyxl",
        "role": "Excel metadata reading",
        "required_for_phase7_5": "recommended",
    },
    {
        "name": "scikit-learn",
        "import_name": "sklearn",
        "role": "machine learning models and metrics",
        "required_for_phase7_5": "required",
    },
]


def _version(import_name: str) -> str:
    try:
        module = importlib.import_module(import_name)
    except Exception:
        return ""
    return str(getattr(module, "__version__", "unknown"))


def _dependency_rows() -> list[dict[str, Any]]:
    rows = []
    for item in DEPENDENCIES:
        installed = importlib.util.find_spec(item["import_name"]) is not None
        category = "available"
        action = "none"
        if not installed and item["required_for_phase7_5"] == "optional":
            category = "optional dependency missing"
            action = "install only after user approval"
        elif not installed:
            category = "required/recommended dependency missing"
            action = "install or skip dependent functionality after user approval"
        rows.append(
            {
                "record_type": "dependency",
                "name": item["name"],
                "status": "installed" if installed else "missing",
                "category": category,
                "version": _version(item["import_name"]) if installed else "",
                "reason": item["role"],
                "recommended_action": action,
            }
        )
    return rows


def _classify_skip(reason: str, model_name: str = "") -> str:
    text = f"{reason} {model_name}".lower()
    if "optional dependency" in text or "not installed" in text or model_name.lower() in {"xgboost", "lightgbm"}:
        return "optional dependency missing"
    if "class" in text and ("insufficient" in text or "count" in text or "imbalance" in text):
        return "class imbalance limitation"
    if "sample" in text or "fewer" in text or "too small" in text:
        return "sample size limitation"
    if "forbidden" in text or "methodolog" in text:
        return "methodologically forbidden item"
    return "deliberately skipped for Route A safety"


def _skipped_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_files = [
        TABULAR_RESULT_ROOT / "tabular_fatigue_sleep_performance_model_comparison.csv",
        TABULAR_RESULT_ROOT / "tabular_fatigue_sleep_performance_ablation_results.csv",
        TABULAR_RESULT_ROOT / "tabular_fatigue_sleep_performance_skipped_models.csv",
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        try:
            import pandas as pd

            data = pd.read_csv(path)
        except Exception as exc:
            rows.append(
                {
                    "record_type": "skipped_item",
                    "name": path.name,
                    "status": "read_failed",
                    "category": "sample size limitation",
                    "version": "",
                    "reason": str(exc),
                    "recommended_action": "inspect result table format",
                }
            )
            continue
        if "status" not in data.columns:
            continue
        skipped = data[data["status"].astype(str).str.lower().ne("ok_exploratory_candidate_not_final")].copy()
        for _, row in skipped.iterrows():
            reason = str(row.get("reason", ""))
            model_name = str(row.get("model_name", ""))
            category = _classify_skip(reason, model_name)
            name_parts = [str(row.get("task_name", path.stem))]
            if model_name and model_name != "nan":
                name_parts.append(model_name)
            rows.append(
                {
                    "record_type": "skipped_item",
                    "name": " / ".join(name_parts),
                    "status": str(row.get("status", "skipped")),
                    "category": category,
                    "version": "",
                    "reason": reason,
                    "recommended_action": "install optional dependency after approval"
                    if category == "optional dependency missing"
                    else "keep skipped unless data/label support improves",
                }
            )

    rows.extend(
        [
            {
                "record_type": "skipped_item",
                "name": "pose_based_fatigue_classifier",
                "status": "forbidden",
                "category": "methodologically forbidden item",
                "version": "",
                "reason": "Route A states Human-M3/EPFL pose data and 5742821 fatigue/sleep labels are not aligned subjects/sessions.",
                "recommended_action": "do not run unless a real alignment table and approved protocol are added",
            },
            {
                "record_type": "skipped_item",
                "name": "pose_tabular_fatigue_merge",
                "status": "forbidden",
                "category": "methodologically forbidden item",
                "version": "",
                "reason": "No verified subject/session correspondence exists between pose datasets and 5742821.",
                "recommended_action": "keep prohibited under Route A",
            },
            {
                "record_type": "skipped_item",
                "name": "raw_video_frame_extraction",
                "status": "deliberately_skipped",
                "category": "deliberately skipped for Route A safety",
                "version": "",
                "reason": "User prohibited MP4 reading, frame extraction, ffmpeg, and full 68GB processing in this stage.",
                "recommended_action": "continue using processed sample tables only",
            },
        ]
    )
    return rows


def _write_markdown(rows: list[dict[str, Any]]) -> None:
    dependency_rows = [row for row in rows if row["record_type"] == "dependency"]
    skipped_rows = [row for row in rows if row["record_type"] == "skipped_item"]
    missing_optional = [
        row
        for row in dependency_rows
        if row["name"] in {"shap", "xgboost", "lightgbm"} and row["status"] == "missing"
    ]
    lines = [
        "# Dependency and Skipped Items Report",
        "",
        "Phase 7.5 dependency check only inspects Python imports and existing output tables.",
        "It does not install packages and does not read or write `dataset/`.",
        "",
        "## Dependency Status",
        "",
        "| Package | Status | Version | Role | Action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in dependency_rows:
        lines.append(
            f"| {row['name']} | {row['status']} | {row['version'] or '-'} | {row['reason']} | {row['recommended_action']} |"
        )

    lines.extend(["", "## Skipped Item Categories", ""])
    for category in [
        "optional dependency missing",
        "sample size limitation",
        "class imbalance limitation",
        "methodologically forbidden item",
        "deliberately skipped for Route A safety",
    ]:
        items = [row for row in skipped_rows if row["category"] == category]
        lines.extend([f"### {category}", ""])
        if not items:
            lines.append("- None observed in the current outputs.")
        for row in items:
            lines.append(f"- `{row['name']}`: {row['status']}. {row['reason']}")
        lines.append("")

    if missing_optional:
        lines.extend(
            [
                "## Optional Install Recommendation",
                "",
                "The following optional packages are missing and would enable SHAP and boosted tree model variants:",
                "",
            ]
        )
        for row in missing_optional:
            lines.append(f"- `{row['name']}`")
        install_names = " ".join(row["name"] for row in missing_optional)
        lines.extend(
            [
                "",
                "Suggested command, only after user approval:",
                "",
                "```powershell",
                f"python -m pip install {install_names}",
                "```",
                "",
            ]
        )
    else:
        lines.extend(["## Optional Install Recommendation", "", "No SHAP/XGBoost/LightGBM install is needed based on current imports.", ""])

    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import pandas as pd

    for path in (CSV_PATH, DOC_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = _dependency_rows() + _skipped_rows()
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    _write_markdown(rows)

    missing_optional = [
        row["name"]
        for row in rows
        if row["record_type"] == "dependency"
        and row["name"] in {"shap", "xgboost", "lightgbm"}
        and row["status"] == "missing"
    ]
    log_lines = [
        "Phase 7.5 dependency and skipped-items check complete.",
        f"Output CSV: {CSV_PATH}",
        f"Report: {DOC_PATH}",
        "Missing optional packages: " + (", ".join(missing_optional) if missing_optional else "none"),
        "No packages were installed.",
        "No dataset files were modified.",
    ]
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("Dependency and skipped-items report complete.")
    print(f"Wrote: {CSV_PATH}")
    print(f"Wrote: {DOC_PATH}")
    print(f"Wrote: {LOG_PATH}")
    if missing_optional:
        print("Missing optional packages: " + ", ".join(missing_optional))
        print("Suggested command after approval: python -m pip install " + " ".join(missing_optional))
    else:
        print("No SHAP/XGBoost/LightGBM install needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
