"""Phase 8 final end-to-end completion for Route A.

Must be run with:
    E:\\Anaconda_envs\\envs\\spenv\\python.exe
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_5742821_subject_count import run as run_subject_audit
from scripts.extract_epfl_representative_frames import run as run_video_frames
from scripts.make_figure_contact_sheets import main as run_contact_sheets
from scripts.make_journal_ready_figures import main as run_journal_ready_figures
from scripts.run_full_pose_movement_quality_processing import run as run_pose_processing
from scripts.run_full_tabular_modeling import run as run_tabular_modeling
from src.data.inventory import ensure_not_inside_dataset
from src.reports.final_index import write_csv, write_phase8_final_docs, write_text
from src.runtime import add_conda_dll_directories


EXPECTED_PYTHON = Path(r"E:\Anaconda_envs\envs\spenv\python.exe")
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
RESULTS_ROOT = OUTPUTS_ROOT / "results"
DOCS_ROOT = PROJECT_ROOT / "docs"
LOG_PATH = PROJECT_ROOT / "logs" / "phase8_full_project_pipeline.txt"
AUDIT_PATH = RESULTS_ROOT / "phase8_full_project_audit.csv"
DEPENDENCY_CSV = RESULTS_ROOT / "dependency_status_phase8.csv"
DEPENDENCY_DOC = DOCS_ROOT / "dependency_and_skipped_items_report_phase8.md"
DEPENDENCY_LOG = PROJECT_ROOT / "logs" / "dependency_and_skipped_items_phase8.txt"


def _check_environment() -> dict[str, Any]:
    actual = Path(sys.executable)
    if actual.resolve() != EXPECTED_PYTHON.resolve():
        raise SystemExit(f"Wrong Python environment. Expected {EXPECTED_PYTHON}, got {actual}")
    rows = []
    for package in ["shap", "xgboost", "lightgbm", "numpy", "pandas", "sklearn", "matplotlib", "scipy", "openpyxl", "joblib", "yaml"]:
        try:
            module = importlib.import_module(package)
            rows.append(
                {
                    "record_type": "dependency",
                    "name": "pyyaml" if package == "yaml" else package,
                    "status": "available",
                    "category": "available",
                    "version": str(getattr(module, "__version__", "unknown")),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "record_type": "dependency",
                    "name": package,
                    "status": "failed",
                    "category": "dependency missing",
                    "version": "",
                    "error": repr(exc),
                }
            )
    return {"python_executable": str(actual), "dependency_rows": rows}


def _copy_phase8_dependency_report(env: dict[str, Any], skipped_rows: list[dict[str, Any]]) -> None:
    rows = [*env["dependency_rows"], *skipped_rows]
    write_csv(DEPENDENCY_CSV, rows)
    lines = [
        "# Dependency and Skipped Items Report Phase 8",
        "",
        f"- Python executable: `{env['python_executable']}`",
        "",
        "## Dependencies",
        "",
        "| Name | Status | Version | Category | Error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in env["dependency_rows"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['version']} | {row['category']} | {row['error'] or '-'} |")
    lines.extend(["", "## Skipped Items", "", "| Name | Category | Reason |", "| --- | --- | --- |"])
    for row in skipped_rows:
        lines.append(f"| {row['name']} | {row['category']} | {row['reason']} |")
    write_text(DEPENDENCY_DOC, lines)
    write_text(DEPENDENCY_LOG, [json.dumps({"dependencies": env["dependency_rows"], "skipped": skipped_rows}, indent=2)])


def _make_full_pose_figures() -> list[Path]:
    """Generate a small set of final full-pose figures without over-styling."""

    import pandas as pd

    combined_path = OUTPUTS_ROOT / "processed" / "pose_movement_quality_features_analysis_table_full.csv"
    if not combined_path.exists():
        return []
    data = pd.read_csv(combined_path)
    out_root = OUTPUTS_ROOT / "figures" / "pose_movement_quality"
    out_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    base = out_root / "pose_movement_quality_full_feature_distributions"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _make_full_pose_figures_cv2(data, base)

    groups = [
        ("Knee flexion", ["left_knee_flexion_angle", "right_knee_flexion_angle"]),
        ("Hip flexion", ["left_hip_flexion_angle", "right_hip_flexion_angle"]),
        ("Knee asymmetry", ["left_right_knee_angle_difference"]),
        ("Knee valgus proxy", ["left_knee_valgus_proxy", "right_knee_valgus_proxy"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.flatten()
    for ax, (title, cols) in zip(axes, groups):
        values = []
        labels = []
        for dataset, subset in data.groupby("dataset_name"):
            values.append(subset[cols].mean(axis=1).dropna())
            labels.append(dataset.replace("epfl_sportcenter_", "EPFL ").replace("humanm3", "Human-M3"))
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Degrees" if "valgus" not in title.lower() else "Proxy units")
    fig.suptitle("Full pose movement-quality feature distributions")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _make_full_pose_figures_cv2(data: Any, base: Path) -> list[Path]:
    try:
        import cv2
        import numpy as np
        from src.video.representative_frames import _write_single_image_pdf
    except Exception:
        return []

    rows = []
    for dataset, subset in data.groupby("dataset_name"):
        rows.append(
            {
                "dataset": str(dataset),
                "n": len(subset),
                "knee": float(subset[["left_knee_flexion_angle", "right_knee_flexion_angle"]].mean(numeric_only=True).mean()),
                "hip": float(subset[["left_hip_flexion_angle", "right_hip_flexion_angle"]].mean(numeric_only=True).mean()),
                "asym": float(subset["left_right_knee_angle_difference"].mean()),
                "valgus": float(subset[["left_knee_valgus_proxy", "right_knee_valgus_proxy"]].mean(numeric_only=True).mean()),
            }
        )
    width, height = 1250, 420
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Full pose movement-quality feature summary", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    headers = ["Dataset", "Rows", "Mean knee deg", "Mean hip deg", "Knee asym deg", "Valgus proxy"]
    x_positions = [30, 455, 585, 760, 925, 1090]
    y = 105
    for x, header in zip(x_positions, headers):
        cv2.putText(canvas, header, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.line(canvas, (25, 122), (1225, 122), (180, 180, 180), 1)
    for idx, row in enumerate(rows):
        y = 165 + idx * 58
        values = [
            row["dataset"],
            f"{row['n']}",
            f"{row['knee']:.2f}",
            f"{row['hip']:.2f}",
            f"{row['asym']:.2f}",
            f"{row['valgus']:.4f}",
        ]
        for x, value in zip(x_positions, values):
            cv2.putText(canvas, str(value)[:36], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (30, 30, 30), 1, cv2.LINE_AA)
    note = "Fallback rendering used because matplotlib/PIL was unavailable; CSV summaries contain full distributions."
    cv2.putText(canvas, note, (30, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    cv2.imwrite(str(png), canvas)
    ok, jpeg_buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        _write_single_image_pdf(pdf, jpeg_buffer.tobytes(), width=canvas.shape[1], height=canvas.shape[0])
        return [png, pdf]
    return [png]


def main() -> int:
    add_conda_dll_directories()
    for path in (LOG_PATH, AUDIT_PATH, DEPENDENCY_CSV, DEPENDENCY_DOC, DEPENDENCY_LOG):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)

    log: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = [
        {
            "record_type": "skipped_item",
            "name": "pose_based_fatigue_classifier",
            "status": "skipped",
            "category": "methodologically forbidden",
            "reason": "Route A forbids pose-fatigue modeling without verified subject/session alignment.",
        },
        {
            "record_type": "skipped_item",
            "name": "humanm3_epfl_5742821_feature_merge",
            "status": "skipped",
            "category": "Route A safety block",
            "reason": "Pose datasets and 5742821 are independent and not merged.",
        },
        {
            "record_type": "skipped_item",
            "name": "full_video_frame_extraction",
            "status": "skipped",
            "category": "runtime limitation",
            "reason": "Resource strategy: only start/middle/near-end representative frames are extracted by design.",
        },
        {
            "record_type": "skipped_item",
            "name": "Mental fatigue.AW5 parsing",
            "status": "skipped",
            "category": "file format unsupported",
            "reason": "AW5 is a proprietary/unsupported format in this project until an approved reader and schema are confirmed.",
        },
    ]

    env = _check_environment()
    log.append(f"Python environment ok: {env['python_executable']}")

    pose_summary = run_pose_processing()
    log.append("Pose processing complete.")
    video_summary = run_video_frames()
    log.append("Video metadata/representative frame step complete.")
    if video_summary.get("status") not in {"ok"}:
        skipped_rows.append(
            {
                "record_type": "skipped_item",
                "name": "epfl_multiview_representative_frame_extraction",
                "status": "skipped",
                "category": "dependency missing" if "unavailable" in str(video_summary.get("reason", "")).lower() else "runtime limitation",
                "reason": video_summary.get("reason") or f"Video step status: {video_summary.get('status')}",
            }
        )
    tabular_summary = run_tabular_modeling()
    log.append("Tabular exploratory modeling complete.")
    subject_audit = run_subject_audit()
    log.append("5742821 subject-count audit complete.")
    pose_figures = _make_full_pose_figures()
    log.append(f"Full-pose final figures generated: {len(pose_figures)} files.")

    try:
        journal_ready_status = run_journal_ready_figures()
        log.append(f"Journal-ready figure regeneration return code: {journal_ready_status}.")
        if journal_ready_status != 0:
            skipped_rows.append(
                {
                    "record_type": "skipped_item",
                    "name": "journal_ready_figure_regeneration",
                    "status": "skipped",
                    "category": "runtime limitation",
                    "reason": f"Journal-ready figure script returned non-zero status {journal_ready_status}.",
                }
            )
    except Exception as exc:
        skipped_rows.append(
            {
                "record_type": "skipped_item",
                "name": "journal_ready_figure_regeneration",
                "status": "skipped",
                "category": "runtime limitation",
                "reason": repr(exc),
            }
        )
        log.append(f"Journal-ready figure regeneration skipped: {exc!r}.")

    try:
        contact_sheet_status = run_contact_sheets()
        log.append(f"Contact-sheet regeneration return code: {contact_sheet_status}.")
        if contact_sheet_status != 0:
            skipped_rows.append(
                {
                    "record_type": "skipped_item",
                    "name": "figure_contact_sheet_regeneration",
                    "status": "skipped",
                    "category": "runtime limitation",
                    "reason": f"Contact-sheet script returned non-zero status {contact_sheet_status}.",
                }
            )
    except Exception as exc:
        skipped_rows.append(
            {
                "record_type": "skipped_item",
                "name": "figure_contact_sheet_regeneration",
                "status": "skipped",
                "category": "runtime limitation",
                "reason": repr(exc),
            }
        )
        log.append(f"Contact-sheet regeneration skipped: {exc!r}.")

    # Collect LightGBM or runtime skipped rows from tabular full outputs.
    skipped_models_path = RESULTS_ROOT / "tabular_fatigue_sleep_performance" / "tabular_fatigue_sleep_performance_skipped_models_full.csv"
    if skipped_models_path.exists():
        import pandas as pd
        from pandas.errors import EmptyDataError

        try:
            skipped_df = pd.read_csv(skipped_models_path)
        except EmptyDataError:
            skipped_df = pd.DataFrame()
        for _, row in skipped_df.iterrows():
            skipped_rows.append(
                {
                    "record_type": "skipped_item",
                    "name": f"{row.get('task_name')} / {row.get('model_name')}",
                    "status": row.get("status"),
                    "category": "sample size limitation" if "insufficient" in str(row.get("reason", "")).lower() else "runtime limitation",
                    "reason": row.get("reason"),
                }
            )

    _copy_phase8_dependency_report(env, skipped_rows)

    processed_files = sorted((OUTPUTS_ROOT / "processed").glob("*full*.csv")) + [
        OUTPUTS_ROOT / "processed" / "pose_movement_quality_epfl_single_metadata_summary.csv"
    ]
    result_files = sorted((OUTPUTS_ROOT / "results" / "pose_movement_quality").glob("*full*.csv"))
    result_files += sorted((OUTPUTS_ROOT / "results" / "pose_movement_quality").glob("epfl_multiview_*.csv"))
    result_files += sorted((OUTPUTS_ROOT / "results" / "tabular_fatigue_sleep_performance").glob("*full.csv"))
    result_files += [AUDIT_PATH, DEPENDENCY_CSV]
    model_files = sorted((OUTPUTS_ROOT / "models" / "tabular_fatigue_sleep_performance").glob("*"))
    figure_files = sorted((OUTPUTS_ROOT / "figures" / "pose_movement_quality").glob("*full*.*"))
    figure_files += sorted((OUTPUTS_ROOT / "figures" / "pose_movement_quality").glob("*epfl_multiview_camera_examples.*"))
    figure_files += sorted((OUTPUTS_ROOT / "figures" / "journal_ready").rglob("*.*"))

    outputs_summary = {
        "environment": env,
        "humanm3": pose_summary["humanm3"],
        "epfl_multiview": pose_summary["epfl_multiview"],
        "epfl_single": pose_summary["epfl_single"],
        "video": video_summary,
        "tabular": {
            "full_paths": tabular_summary.get("full_paths"),
            "saved_models": len(tabular_summary.get("saved_models", [])),
        },
        "subject_audit": subject_audit,
    }
    audit_rows.extend(
        [
            {"item": "python_executable", "value": env["python_executable"], "status": "ok"},
            {"item": "humanm3_pose_json_processed", "value": pose_summary["humanm3"]["processed_pose_json"], "status": pose_summary["humanm3"]["mode"]},
            {"item": "epfl_multiview_pose_json_processed", "value": pose_summary["epfl_multiview"]["processed_pose_json"], "status": pose_summary["epfl_multiview"]["mode"]},
            {"item": "epfl_single_metadata_sequences", "value": pose_summary["epfl_single"]["processed_sequences"], "status": "metadata_only"},
            {"item": "mp4_videos_processed", "value": video_summary.get("videos_processed"), "status": video_summary.get("status")},
            {"item": "representative_frames_extracted", "value": video_summary.get("frames_extracted"), "status": "limited_representative_frames"},
            {"item": "tabular_models_saved", "value": len(tabular_summary.get("saved_models", [])), "status": "exploratory_tabular_only"},
            {"item": "dataset_modified", "value": False, "status": "ok"},
            {"item": "pose_fatigue_merge", "value": False, "status": "ok"},
        ]
    )
    write_csv(AUDIT_PATH, audit_rows)
    write_phase8_final_docs(
        docs_root=DOCS_ROOT,
        outputs=outputs_summary,
        audit_rows=audit_rows,
        model_files=model_files,
        figure_files=figure_files,
        processed_files=processed_files,
        result_files=result_files,
        skipped_rows=skipped_rows,
    )

    LOG_PATH.write_text(
        "\n".join(log)
        + "\n\n"
        + json.dumps(outputs_summary, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    print("Phase 8 final end-to-end pipeline complete.")
    print(json.dumps(outputs_summary, indent=2, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
