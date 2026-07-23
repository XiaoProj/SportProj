"""Route A ablation dry run.

No ablation is run here. The script reports which ablations are valid under the
dual-pipeline research scope.
"""

from __future__ import annotations


def main() -> int:
    print("Route A ablation policy:")
    print("- pose_movement_quality_*: feature-group descriptive comparisons are allowed.")
    print("- tabular_fatigue_sleep_performance_*: exploratory tabular feature ablations are allowed.")
    print("- pose features + 5742821 fatigue labels: prohibited without verified alignment.")
    print("No ablation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
