"""Route A SHAP dry run.

SHAP is allowed only for the tabular exploratory pipeline unless future evidence
creates an approved pose-label dataset. This script does not compute SHAP.
"""

from __future__ import annotations


def main() -> int:
    print("Route A SHAP policy:")
    print("- tabular_fatigue_sleep_performance_*: exploratory SHAP/feature importance may be run later.")
    print("- pose_movement_quality_*: SHAP for fatigue labels is not applicable.")
    print("- no pose-fatigue merge or pose-based fatigue SHAP is allowed by default.")
    print("No SHAP computation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
