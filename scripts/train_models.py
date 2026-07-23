"""Route A model-training dry run.

This script intentionally does not train a formal model. It reports the active
analysis route and refuses pose-fatigue merge training by default.
"""

from __future__ import annotations


def main() -> int:
    print("Route A is active: dual independent pipelines.")
    print("pose_movement_quality_pipeline: descriptive movement-quality analysis only.")
    print("tabular_fatigue_sleep_performance_pipeline: exploratory tabular ML only after label-rule approval.")
    print("Safety: pose-fatigue merge training is disabled by default.")
    print("No formal model training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
