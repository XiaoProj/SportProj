"""Ablation-study framework for feature groups."""

from __future__ import annotations

from typing import Any

from src.features.feature_groups import available_feature_groups, filter_existing_features
from src.models.training import available_model_specs, fit_and_evaluate_model


def planned_ablation_groups() -> dict[str, list[str]]:
    """Return the seven planned feature groups for the study."""

    return available_feature_groups()


def run_ablation(
    feature_table: Any,
    target_column: str,
    train_index: Any,
    test_index: Any,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Run ablation experiments on an already-built feature table."""

    groups = planned_ablation_groups()
    specs = available_model_specs(random_seed=random_seed)
    results: list[dict[str, Any]] = []
    columns = list(feature_table.columns)
    for group_name, requested_features in groups.items():
        features = filter_existing_features(columns, requested_features)
        if not features:
            results.append(
                {
                    "feature_group": group_name,
                    "model": None,
                    "status": "skipped_missing_features",
                }
            )
            continue
        x_train = feature_table.loc[train_index, features]
        x_test = feature_table.loc[test_index, features]
        y_train = feature_table.loc[train_index, target_column]
        y_test = feature_table.loc[test_index, target_column]
        for spec in specs:
            metrics = fit_and_evaluate_model(spec, x_train, y_train, x_test, y_test)
            metrics["feature_group"] = group_name
            metrics["n_features"] = len(features)
            results.append(metrics)
    return results
