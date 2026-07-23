"""Generate Route A journal-draft figures from processed/results files only."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.visualization.style import OKABE_ITO, PALETTE, apply_matplotlib_style, save_journal_figure


DATASET_ROOT = PROJECT_ROOT / "dataset"
POSE_FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "pose_movement_quality"
TABULAR_FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "tabular_fatigue_sleep_performance"
POSE_RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "pose_movement_quality"
TABULAR_RESULT_ROOT = PROJECT_ROOT / "outputs" / "results" / "tabular_fatigue_sleep_performance"
HUMANM3_PATH = PROJECT_ROOT / "outputs" / "processed" / "humanm3_pose_features_sample.csv"
EPFL_PATH = PROJECT_ROOT / "outputs" / "processed" / "epfl_multiview_pose_features_sample.csv"
LABEL_RULES_PATH = PROJECT_ROOT / "outputs" / "processed" / "fatigue_sleep_label_rules.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "route_a_figures.txt"


def _condition_label(value: str) -> str:
    return {
        "mental_fatigue": "MF",
        "mental_fatigue_plus_sleep_deprivation": "SR+MF",
        "Control": "Control",
        "MF": "MF",
        "SR": "SR",
        "SR+MF": "SR+MF",
    }.get(str(value), str(value))


def _load_pose_data():
    import pandas as pd

    humanm3 = pd.read_csv(HUMANM3_PATH)
    epfl = pd.read_csv(EPFL_PATH)
    humanm3["pose_dataset_source"] = "Human-M3"
    epfl["pose_dataset_source"] = "EPFL multiview"
    return pd.concat([humanm3, epfl], ignore_index=True, sort=False)


def _save(fig, base: Path) -> None:
    ensure_not_inside_dataset(base.with_suffix(".png"), DATASET_ROOT)
    ensure_not_inside_dataset(base.with_suffix(".pdf"), DATASET_ROOT)
    save_journal_figure(fig, base)


def make_pose_missingness() -> list[Path]:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt
    import pandas as pd

    summary = pd.read_csv(POSE_RESULT_ROOT / "pose_movement_quality_missingness_summary.csv")
    plot_features = [
        "left_knee_flexion_angle",
        "right_knee_flexion_angle",
        "left_hip_flexion_angle",
        "right_hip_flexion_angle",
        "left_ankle_angle",
        "right_ankle_angle",
        "left_right_knee_angle_difference",
        "left_right_ankle_angle_difference",
        "left_knee_valgus_proxy",
        "right_knee_valgus_proxy",
    ]
    datasets = ["Human-M3", "EPFL multiview"]
    x = list(range(len(plot_features)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for idx, dataset in enumerate(datasets):
        values = []
        for feature in plot_features:
            row = summary[(summary["pose_dataset_source"] == dataset) & (summary["feature"] == feature)]
            values.append(float(row["missing_rate"].iloc[0]) if not row.empty else 1.0)
        positions = [value + (idx - 0.5) * width for value in x]
        ax.bar(positions, values, width=width, color=PALETTE[idx], alpha=0.82, label=dataset)
    ax.set_xticks(x, plot_features, rotation=35, ha="right")
    ax.set_ylabel("Missing rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pose feature missingness by dataset")
    ax.legend(frameon=False)
    fig.tight_layout()
    base = POSE_FIGURE_ROOT / "pose_movement_quality_missingness"
    _save(fig, base)
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def make_pose_angle_distributions() -> list[Path]:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    data = _load_pose_data()
    features = [
        "left_knee_flexion_angle",
        "right_knee_flexion_angle",
        "left_hip_flexion_angle",
        "right_hip_flexion_angle",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        groups = [
            data.loc[data["pose_dataset_source"] == "Human-M3", feature].dropna(),
            data.loc[data["pose_dataset_source"] == "EPFL multiview", feature].dropna(),
        ]
        box = ax.boxplot(groups, patch_artist=True, tick_labels=["Human-M3", "EPFL"])
        for patch, color in zip(box["boxes"], [OKABE_ITO["blue"], OKABE_ITO["orange"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(feature.replace("_", " "))
        ax.set_ylabel("Angle (degrees)")
    fig.suptitle("Knee and hip angle distributions", y=1.02)
    fig.tight_layout()
    base = POSE_FIGURE_ROOT / "pose_movement_quality_angle_distributions"
    _save(fig, base)
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def make_pose_knee_asymmetry() -> list[Path]:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    data = _load_pose_data()
    groups = [
        data.loc[data["pose_dataset_source"] == "Human-M3", "left_right_knee_angle_difference"].dropna(),
        data.loc[data["pose_dataset_source"] == "EPFL multiview", "left_right_knee_angle_difference"].dropna(),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    box = ax.boxplot(groups, patch_artist=True, tick_labels=["Human-M3", "EPFL"])
    for patch, color in zip(box["boxes"], [OKABE_ITO["blue"], OKABE_ITO["orange"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Left-right knee angle asymmetry")
    ax.set_ylabel("Absolute angle difference (degrees)")
    ax.set_xlabel("Pose dataset")
    fig.tight_layout()
    base = POSE_FIGURE_ROOT / "pose_movement_quality_knee_asymmetry"
    _save(fig, base)
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def make_pose_knee_valgus_proxy() -> list[Path]:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt

    data = _load_pose_data()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = []
    groups = []
    colors = []
    for dataset, color in [("Human-M3", OKABE_ITO["blue"]), ("EPFL multiview", OKABE_ITO["orange"])]:
        for side, feature in [("Left", "left_knee_valgus_proxy"), ("Right", "right_knee_valgus_proxy")]:
            labels.append(f"{dataset}\n{side}")
            groups.append(data.loc[data["pose_dataset_source"] == dataset, feature].dropna())
            colors.append(color)
    box = ax.boxplot(groups, patch_artist=True, tick_labels=labels)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Knee valgus proxy by dataset and side")
    ax.set_ylabel("Proxy distance in coordinate plane")
    ax.set_xlabel("Dataset and side")
    fig.tight_layout()
    base = POSE_FIGURE_ROOT / "pose_movement_quality_knee_valgus_proxy"
    _save(fig, base)
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def make_tabular_vas(variable: str, title: str, filename: str, ylabel: str) -> list[Path]:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv(LABEL_RULES_PATH)
    groups = ["mental_fatigue", "mental_fatigue_plus_sleep_deprivation"]
    values = [data.loc[data["condition_group"] == group, variable].dropna() for group in groups]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    box = ax.boxplot(values, patch_artist=True, tick_labels=[_condition_label(group) for group in groups])
    for patch, color in zip(box["boxes"], [OKABE_ITO["blue"], OKABE_ITO["orange"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for idx, series in enumerate(values, start=1):
        ax.scatter([idx] * len(series), series, color=OKABE_ITO["black"], s=18, alpha=0.65, zorder=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Protocol condition group")
    fig.tight_layout()
    base = TABULAR_FIGURE_ROOT / filename
    _save(fig, base)
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def make_tabular_shot_change() -> list[Path]:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv(LABEL_RULES_PATH)
    groups = ["mental_fatigue", "mental_fatigue_plus_sleep_deprivation"]
    values = [data.loc[data["condition_group"] == group, "shot_accuracy_change_shot2_minus_shot1"].dropna() for group in groups]
    means = [series.mean() for series in values]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar([0, 1], means, color=[OKABE_ITO["blue"], OKABE_ITO["orange"]], alpha=0.8)
    for idx, series in enumerate(values):
        ax.scatter([idx] * len(series), series, color=OKABE_ITO["black"], s=18, alpha=0.65, zorder=3)
    ax.axhline(0, color=OKABE_ITO["black"], linewidth=1)
    ax.set_xticks([0, 1], [_condition_label(group) for group in groups])
    ax.set_title("Free-throw accuracy change by condition group")
    ax.set_ylabel("Shot 2 minus Shot 1 accuracy")
    ax.set_xlabel("Protocol condition group")
    fig.tight_layout()
    base = TABULAR_FIGURE_ROOT / "tabular_fatigue_sleep_performance_shot_accuracy_change"
    _save(fig, base)
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def make_tabular_condition_summary() -> list[Path]:
    apply_matplotlib_style()
    import matplotlib.pyplot as plt
    import pandas as pd

    summary = pd.read_csv(TABULAR_RESULT_ROOT / "tabular_fatigue_sleep_performance_condition_summary.csv")
    reconstructed = summary[summary["label_status"] == "reconstructed_performance_condition_candidate"].copy()
    order = ["Control", "MF", "SR", "SR+MF"]
    reconstructed["order"] = reconstructed["condition_group"].map({name: idx for idx, name in enumerate(order)})
    reconstructed = reconstructed.sort_values("order")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(
        reconstructed["condition_group"].map(_condition_label),
        reconstructed["free_throw_accuracy_mean"],
        color=[OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"]],
        alpha=0.82,
    )
    ax.set_title("Free-throw accuracy by reconstructed condition")
    ax.set_ylabel("Mean free-throw accuracy")
    ax.set_xlabel("Protocol-derived condition")
    ax.set_ylim(0, max(100, reconstructed["free_throw_accuracy_mean"].max() * 1.12))
    fig.tight_layout()
    base = TABULAR_FIGURE_ROOT / "tabular_fatigue_sleep_performance_condition_summary"
    _save(fig, base)
    plt.close(fig)
    return [base.with_suffix(".png"), base.with_suffix(".pdf")]


def main() -> int:
    for path in (POSE_FIGURE_ROOT, TABULAR_FIGURE_ROOT, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.extend(make_pose_missingness())
    outputs.extend(make_pose_angle_distributions())
    outputs.extend(make_pose_knee_asymmetry())
    outputs.extend(make_pose_knee_valgus_proxy())
    outputs.extend(
        make_tabular_vas(
            "vas_mf_mean",
            "Subjective mental fatigue by condition group",
            "tabular_fatigue_sleep_performance_vas_mf_distribution",
            "Mean VAS MF score",
        )
    )
    outputs.extend(
        make_tabular_vas(
            "vas_mot_mean",
            "Motivation rating by condition group",
            "tabular_fatigue_sleep_performance_vas_mot_distribution",
            "Mean VAS Mot score",
        )
    )
    outputs.extend(make_tabular_shot_change())
    outputs.extend(make_tabular_condition_summary())

    LOG_PATH.write_text(
        "Route A figures generated from processed/results files only.\n"
        + "\n".join(path.as_posix() for path in outputs)
        + "\n",
        encoding="utf-8",
    )
    print("Route A figures generated.")
    for path in outputs:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
