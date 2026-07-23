"""Make preliminary Phase 4 smoke-test figures from processed files only."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.visualization.style import OKABE_ITO, PALETTE, apply_matplotlib_style


DATASET_ROOT = PROJECT_ROOT / "dataset"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
HUMANM3_PATH = PROJECT_ROOT / "outputs" / "processed" / "humanm3_pose_features_sample.csv"
EPFL_PATH = PROJECT_ROOT / "outputs" / "processed" / "epfl_multiview_pose_features_sample.csv"
POSE_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "results" / "phase4_pose_feature_summary.csv"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "phase4_modeling_readiness.md"
LOG_PATH = PROJECT_ROOT / "logs" / "phase4_figures.txt"


def _condition_label(value: str) -> str:
    return {
        "mental_fatigue": "Mental fatigue",
        "mental_fatigue_plus_sleep_deprivation": "Mental fatigue + sleep deprivation",
    }.get(value, str(value))


def make_vas_distribution(label_data, output_path: Path) -> None:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    groups = ["mental_fatigue", "mental_fatigue_plus_sleep_deprivation"]
    data = [label_data.loc[label_data["condition_group"] == group, "vas_mf_mean"].dropna() for group in groups]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    box = ax.boxplot(data, patch_artist=True, tick_labels=[_condition_label(group) for group in groups])
    for patch, color in zip(box["boxes"], [OKABE_ITO["blue"], OKABE_ITO["orange"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for idx, values in enumerate(data, start=1):
        ax.scatter([idx] * len(values), values, color=OKABE_ITO["black"], s=18, alpha=0.65, zorder=3)
    ax.set_title("Preliminary VAS mental fatigue distribution")
    ax.set_ylabel("Mean VAS MF score")
    ax.set_xlabel("Candidate condition group")
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_shot_change(label_data, output_path: Path) -> None:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    groups = ["mental_fatigue", "mental_fatigue_plus_sleep_deprivation"]
    data = [
        label_data.loc[label_data["condition_group"] == group, "shot_accuracy_change_shot2_minus_shot1"].dropna()
        for group in groups
    ]
    means = [values.mean() for values in data]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar([0, 1], means, color=[OKABE_ITO["blue"], OKABE_ITO["orange"]], alpha=0.75)
    for idx, values in enumerate(data):
        ax.scatter([idx] * len(values), values, color=OKABE_ITO["black"], s=18, alpha=0.65, zorder=3)
    ax.axhline(0, color=OKABE_ITO["black"], linewidth=1)
    ax.set_xticks([0, 1], [_condition_label(group) for group in groups], rotation=15, ha="right")
    ax.set_title("Preliminary shot accuracy change")
    ax.set_ylabel("Shot 2 accuracy minus Shot 1 accuracy")
    ax.set_xlabel("Candidate condition group")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_pose_missingness(pose_summary, output_path: Path) -> None:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    plot_data = pose_summary[pose_summary["feature"].str.contains("angle|difference|valgus|missing_joint_ratio")].copy()
    features = sorted(plot_data["feature"].unique())
    datasets = ["humanm3", "epfl_multiview"]
    width = 0.38
    x_positions = list(range(len(features)))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for idx, dataset in enumerate(datasets):
        rates = []
        for feature in features:
            subset = plot_data[(plot_data["table_name"] == dataset) & (plot_data["feature"] == feature)]
            rates.append(float(subset["missing_rate"].iloc[0]) if not subset.empty else 1.0)
        offsets = [x + (idx - 0.5) * width for x in x_positions]
        ax.bar(offsets, rates, width=width, label=dataset, color=PALETTE[idx], alpha=0.78)
    ax.set_xticks(x_positions, features, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Missing rate")
    ax.set_title("Preliminary pose feature missingness")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_pose_distributions(humanm3, epfl, output_path: Path) -> None:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt
    import pandas as pd

    humanm3 = humanm3.copy()
    epfl = epfl.copy()
    humanm3["table_name"] = "Human-M3"
    epfl["table_name"] = "EPFL multiview"
    combined = pd.concat([humanm3, epfl], ignore_index=True)
    features = [
        "left_knee_flexion_angle",
        "right_knee_flexion_angle",
        "left_hip_flexion_angle",
        "right_hip_flexion_angle",
        "left_right_knee_angle_difference",
        "left_knee_valgus_proxy",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        data = [
            combined.loc[combined["table_name"] == "Human-M3", feature].dropna(),
            combined.loc[combined["table_name"] == "EPFL multiview", feature].dropna(),
        ]
        box = ax.boxplot(data, patch_artist=True, tick_labels=["Human-M3", "EPFL"])
        for patch, color in zip(box["boxes"], [OKABE_ITO["blue"], OKABE_ITO["orange"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(feature)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Preliminary pose feature distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def update_doc_with_figures(figures: list[Path]) -> None:
    marker = "## Preliminary Figures"
    section = [
        marker,
        "",
        "These figures are preliminary smoke-test figures generated from small processed samples only. They are not final paper results.",
        "",
    ]
    for figure in figures:
        section.append(f"- `{figure.as_posix()}`")
    section.append("")
    new_section = "\n".join(section)
    if DOC_PATH.exists():
        text = DOC_PATH.read_text(encoding="utf-8")
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n\n" + new_section
        else:
            text = text.rstrip() + "\n\n" + new_section
    else:
        text = "# Phase 4 Modeling Readiness\n\n" + new_section
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    for path in (FIGURE_ROOT, DOC_PATH, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    label_data = pd.read_csv(LABEL_RULES_PATH)
    humanm3 = pd.read_csv(HUMANM3_PATH)
    epfl = pd.read_csv(EPFL_PATH)
    pose_summary = pd.read_csv(POSE_SUMMARY_PATH)

    figures = [
        FIGURE_ROOT / "phase4_vas_mf_distribution.png",
        FIGURE_ROOT / "phase4_shot_accuracy_change.png",
        FIGURE_ROOT / "phase4_pose_feature_missingness.png",
        FIGURE_ROOT / "phase4_pose_feature_distributions.png",
    ]
    for figure in figures:
        ensure_not_inside_dataset(figure, DATASET_ROOT)

    make_vas_distribution(label_data, figures[0])
    make_shot_change(label_data, figures[1])
    make_pose_missingness(pose_summary, figures[2])
    make_pose_distributions(humanm3, epfl, figures[3])
    update_doc_with_figures(figures)
    LOG_PATH.write_text(
        "Phase 4 preliminary figures generated from processed sample files only.\n"
        + "\n".join(str(path) for path in figures)
        + "\n",
        encoding="utf-8",
    )
    print("Phase 4 preliminary figures generated.")
    for figure in figures:
        print(f"Wrote: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
