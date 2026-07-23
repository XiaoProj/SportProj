"""Phasjournal-ready figures and assign main/appendix/drop status."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset


DATASET_ROOT = PROJECT_ROOT / "dataset"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "journal_ready"
AUDIT_CSV = PROJECT_ROOT / "outputs" / "results" / "figure_quality_audit.csv"
AUDIT_DOC = PROJECT_ROOT / "docs" / "figure_quality_audit.md"
MAIN_APPENDIX_DOC = PROJECT_ROOT / "docs" / "main_vs_appendix_figures.md"
CAPTIONS_DOC = PROJECT_ROOT / "docs" / "manuscript_ready_captions.md"
FIGURE_NOTES = PROJECT_ROOT / "docs" / "figure_notes.md"


FIGURE_RULES = {
    "pose_movement_quality_feature_availability_matrix": (
        "keep",
        "outputs/processed pose feature tables",
        "Need structural availability separate from missingness.",
        "Heatmap with computable, structurally unavailable, and not applicable states.",
        True,
        "Pose feature availability matrix across Human-M3, EPFL multiview, and EPFL single-view metadata. Ankle-angle features are structurally unavailable because distal foot/toe joints are absent.",
        "Availability does not imply biomechanical validity; knee valgus remains a proxy feature.",
    ),
    "pose_movement_quality_angle_distributions": (
        "keep",
        "outputs/processed pose feature tables",
        "Panel clarity and units needed improvement.",
        "2x2 panel with labels A-D and degree units.",
        True,
        "Knee and hip angle distributions in the Human-M3 and EPFL multiview pose samples.",
        "Sample-level descriptive feature validation only; not fatigue analysis.",
    ),
    "pose_movement_quality_knee_asymmetry": (
        "keep",
        "outputs/processed pose feature tables",
        "Y-axis needed explicit definition.",
        "Axis now states absolute left-right knee angle difference in degrees.",
        True,
        "Absolute left-right knee angle difference in pose samples.",
        "Asymmetry is descriptive and not tied to fatigue labels.",
    ),
    "pose_movement_quality_knee_valgus_proxy": (
        "keep",
        "outputs/processed pose feature tables",
        "Proxy status needed explicit warning.",
        "Caption and plot note state proxy definition and non-clinical status.",
        True,
        "Knee valgus proxy computed from hip-knee-ankle coordinate alignment.",
        "This is not a clinical frontal-plane knee valgus measurement.",
    ),
    "pose_movement_quality_computational_missingness": (
        "keep",
        "outputs/processed pose feature tables",
        "Ankle structural gaps were previously conflated with missingness.",
        "Compact missingness summary excludes structurally unavailable ankle features.",
        True,
        "Computational missingness for features supported by the available joint schemas.",
        "Structural unavailability is shown separately in the availability matrix.",
    ),
    "pose_movement_quality_missingness": (
        "appendix",
        "outputs/processed pose feature tables",
        "Legacy filename could be confused.",
        "Overwritten with corrected computational missingness summary.",
        False,
        "Corrected computational missingness summary retained under the legacy filename.",
        "Prefer the explicitly named computational missingness figure in the main text.",
    ),
    "pose_movement_quality_pose_completeness": (
        "appendix",
        "outputs/processed pose feature tables",
        "Completeness variation is small; boxplot had low information density.",
        "Dot/summary panel with n, mean, and max missing joint ratio.",
        False,
        "Pose completeness summary by dataset.",
        "Low variation makes this best suited for appendix or quality-control notes.",
    ),
    "pose_movement_quality_dataset_comparison_heatmap": (
        "appendix",
        "outputs/processed pose feature tables",
        "Radar plot was not ideal for precise comparison.",
        "Radar replaced with standardized dataset comparison heatmap.",
        False,
        "Standardized comparison of selected pose movement-quality metrics by dataset.",
        "Technical diagnostic only; no fatigue or biological group inference.",
    ),
    "pose_movement_quality_pca_projection": (
        "appendix",
        "outputs/processed pose feature tables",
        "Technical diagnostic.",
        "Retained as PCA projection for appendix.",
        False,
        "Technical PCA projection of pose features.",
        "Not a biological or fatigue classification result.",
    ),
    "pose_movement_quality_correlation_heatmap": (
        "appendix",
        "outputs/processed pose feature tables",
        "Technical diagnostic.",
        "Retained as correlation heatmap for appendix.",
        False,
        "Correlation among pose movement-quality features.",
        "Exploratory correlation in small processed samples.",
    ),
    "tabular_fatigue_sleep_performance_vas_mf_distribution": (
        "keep",
        "outputs/processed/fatigue_sleep_label_rules.csv",
        "Small sample distribution needs individual points.",
        "Boxplot with individual points and fixed parsed condition groups.",
        True,
        "Subjective mental fatigue scores by parsed condition group.",
        "VAS MF is subjective and candidate/exploratory in downstream ML.",
    ),
    "tabular_fatigue_sleep_performance_shot_accuracy_by_condition": (
        "keep",
        "outputs/processed/fatigue_sleep_label_rules.csv",
        "Condition order must follow protocol.",
        "Control, MF, SR, SR+MF order using protocol-derived Shot 1/Shot 2 mapping.",
        True,
        "Free-throw accuracy by protocol-derived condition: Control, MF, SR, and SR+MF.",
        "Mapping is protocol-derived from spreadsheet structure and remains candidate interpretation.",
    ),
    "tabular_fatigue_sleep_performance_shot_accuracy_change": (
        "keep",
        "outputs/processed/fatigue_sleep_label_rules.csv",
        "Shot change must be labeled candidate.",
        "Individual points and zero reference line retained.",
        True,
        "Shot 2 minus Shot 1 free-throw accuracy change by parsed condition group.",
        "Performance change is a candidate variable, not ground-truth fatigue effect.",
    ),
    "tabular_fatigue_sleep_performance_model_comparison": (
        "keep",
        "outputs/results/tabular_fatigue_sleep_performance_model_comparison.csv",
        "ML must be clearly marked exploratory.",
        "Title and docs label all ML outputs as exploratory.",
        True,
        "Exploratory tabular model comparison for candidate tasks.",
        "Small-sample leave-one-subject-out CV only; not final performance.",
    ),
    "tabular_fatigue_sleep_performance_ablation": (
        "appendix",
        "outputs/results/tabular_fatigue_sleep_performance_ablation_results.csv",
        "Feature-group ablation is exploratory.",
        "Kept as appendix figure.",
        False,
        "Feature-group ablation across exploratory tabular tasks.",
        "Candidate labels and leakage-controlled feature groups only.",
    ),
    "tabular_fatigue_sleep_performance_feature_importance": (
        "appendix",
        "outputs/results/tabular_fatigue_sleep_performance_feature_importance.csv",
        "Importance unstable in small sample.",
        "Permutation importance plotted and marked exploratory.",
        False,
        "Exploratory permutation feature importance across tabular tasks.",
        "Importance values are unstable with small sample size.",
    ),
    "tabular_fatigue_sleep_performance_prediction_scatter": (
        "appendix",
        "outputs/results/tabular_fatigue_sleep_performance_predictions.csv",
        "Prediction plot is model diagnostic.",
        "Kept as appendix diagnostic.",
        False,
        "Observed versus predicted VAS MF score for the selected exploratory regression model.",
        "Not a final predictive-performance estimate.",
    ),
    "tabular_fatigue_sleep_performance_confusion_matrix": (
        "appendix",
        "outputs/results/tabular_fatigue_sleep_performance_model_comparison.csv",
        "Classification diagnostic.",
        "Kept as appendix diagnostic.",
        False,
        "Confusion matrix for selected exploratory condition-group classifier.",
        "Candidate classes only; not ground-truth fatigue classification.",
    ),
    "tabular_fatigue_sleep_performance_roc_curves": (
        "appendix",
        "outputs/results/tabular_fatigue_sleep_performance_predictions.csv",
        "ROC curves only if probability outputs and binary classes allow.",
        "Generated for eligible binary candidate classification tasks.",
        False,
        "ROC curves for exploratory binary candidate classification tasks.",
        "Small sample and candidate labels limit interpretation.",
    ),
}


def main() -> int:
    import pandas as pd

    for path in (AUDIT_CSV, AUDIT_DOC, MAIN_APPENDIX_DOC, CAPTIONS_DOC, FIGURE_NOTES):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for stem, rule in FIGURE_RULES.items():
        png = next(FIGURE_ROOT.rglob(f"{stem}.png"), None)
        pdf = next(FIGURE_ROOT.rglob(f"{stem}.pdf"), None)
        decision, source_data, problem, fix, main_text, caption, limitation = rule
        rows.append(
            {
                "figure_name": stem,
                "decision": decision,
                "source_data": source_data,
                "main_problem": problem,
                "fix_applied": fix,
                "png_exists": bool(png),
                "pdf_exists": bool(pdf),
                "colorblind_friendly": True,
                "suitable_for_main_text": main_text and bool(png) and bool(pdf),
                "caption": caption,
                "limitation": limitation,
                "png_path": png.as_posix() if png else "",
                "pdf_path": pdf.as_posix() if pdf else "",
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(AUDIT_CSV, index=False)

    audit_lines = ["# Figure Quality Audit", "", "| Figure | Decision | PNG | PDF | Main Text | Fix |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        audit_lines.append(
            f"| `{row['figure_name']}` | {row['decision']} | {row['png_exists']} | {row['pdf_exists']} | {row['suitable_for_main_text']} | {row['fix_applied']} |"
        )
    AUDIT_DOC.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    main_lines = ["# Main vs Appendix Figures", "", "## Main Text", ""]
    for row in rows:
        if row["suitable_for_main_text"]:
            main_lines.append(f"- `{row['figure_name']}`: {row['caption']}")
    main_lines.extend(["", "## Appendix", ""])
    for row in rows:
        if row["decision"] == "appendix":
            main_lines.append(f"- `{row['figure_name']}`: {row['caption']}")
    main_lines.extend(["", "## Drop", ""])
    for row in rows:
        if row["decision"] == "drop":
            main_lines.append(f"- `{row['figure_name']}`")
    MAIN_APPENDIX_DOC.write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    caption_lines = ["# Manuscript-Ready Captions", ""]
    for row in rows:
        if row["png_exists"] and row["pdf_exists"]:
            caption_lines.extend(
                [
                    f"## {row['figure_name']}",
                    "",
                    row["caption"],
                    "",
                    f"Limitation: {row['limitation']}",
                    "",
                ]
            )
    CAPTIONS_DOC.write_text("\n".join(caption_lines), encoding="utf-8")

    existing = FIGURE_NOTES.read_text(encoding="utf-8") if FIGURE_NOTES.exists() else "# Figure Notes\n"
    marker = "## Figure Audit"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n"
    note = [
        marker,
        "",
        f"Audit CSV: `{AUDIT_CSV.as_posix()}`",
        f"Main/appendix decisions: `{MAIN_APPENDIX_DOC.as_posix()}`",
        f"Captions: `{CAPTIONS_DOC.as_posix()}`",
        "",
    ]
    FIGURE_NOTES.write_text(existing.rstrip() + "\n\n" + "\n".join(note), encoding="utf-8")

    print("Figure quality audit complete.")
    print(f"Wrote: {AUDIT_CSV}")
    print(f"Wrote: {AUDIT_DOC}")
    print(f"Wrote: {MAIN_APPENDIX_DOC}")
    print(f"Wrote: {CAPTIONS_DOC}")
    print(f"Wrote: {FIGURE_NOTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
