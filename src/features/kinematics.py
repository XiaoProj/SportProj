"""Generic kinematic feature functions for pose-derived movement analysis."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


Point = tuple[float, ...]
Pose = Mapping[str, Any]


def _to_point(value: Any) -> Point | None:
    """Convert a point-like object to a numeric tuple, preserving missing values."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        coords = [value.get(axis) for axis in ("x", "y", "z") if axis in value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        coords = list(value)
    else:
        return None
    numeric: list[float] = []
    for coord in coords:
        if coord is None:
            return None
        try:
            value_float = float(coord)
        except (TypeError, ValueError):
            return None
        if math.isnan(value_float):
            return None
        numeric.append(value_float)
    return tuple(numeric) if len(numeric) >= 2 else None


def _get_joint(pose: Pose, joint_name: str) -> Point | None:
    """Fetch a joint point by name and convert it to a numeric tuple."""

    return _to_point(pose.get(joint_name))


def _vector(start: Point, end: Point) -> list[float] | None:
    """Return end-start for equal-dimensional points."""

    if len(start) != len(end):
        return None
    return [b - a for a, b in zip(start, end)]


def angle_degrees(a: Any, b: Any, c: Any) -> float | None:
    """Compute the angle ABC in degrees, returning None when inputs are missing."""

    point_a = _to_point(a)
    point_b = _to_point(b)
    point_c = _to_point(c)
    if point_a is None or point_b is None or point_c is None:
        return None
    ba = _vector(point_b, point_a)
    bc = _vector(point_b, point_c)
    if ba is None or bc is None:
        return None
    dot = sum(x * y for x, y in zip(ba, bc))
    norm_ba = math.sqrt(sum(x * x for x in ba))
    norm_bc = math.sqrt(sum(x * x for x in bc))
    if norm_ba == 0 or norm_bc == 0:
        return None
    cosine = max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))
    return math.degrees(math.acos(cosine))


def knee_flexion_angle(pose: Pose, side: str) -> float | None:
    """Estimate knee flexion from hip-knee-ankle angle for the requested side."""

    return angle_degrees(
        _get_joint(pose, f"{side}_hip"),
        _get_joint(pose, f"{side}_knee"),
        _get_joint(pose, f"{side}_ankle"),
    )


def hip_flexion_angle(pose: Pose, side: str) -> float | None:
    """Estimate hip flexion from pelvis-hip-knee angle for the requested side."""

    return angle_degrees(
        _get_joint(pose, "pelvis"),
        _get_joint(pose, f"{side}_hip"),
        _get_joint(pose, f"{side}_knee"),
    )


def ankle_angle(pose: Pose, side: str) -> float | None:
    """Estimate ankle angle from knee-ankle-foot/toe angle when a distal foot point exists."""

    foot = _get_joint(pose, f"{side}_foot") or _get_joint(pose, f"{side}_toe")
    if foot is None:
        return None
    return angle_degrees(
        _get_joint(pose, f"{side}_knee"),
        _get_joint(pose, f"{side}_ankle"),
        foot,
    )


def left_right_knee_angle_difference(pose: Pose) -> float | None:
    """Compute absolute left-right knee-angle difference for symmetry analysis."""

    left = knee_flexion_angle(pose, "left")
    right = knee_flexion_angle(pose, "right")
    if left is None or right is None:
        return None
    return abs(left - right)


def left_right_ankle_angle_difference(pose: Pose) -> float | None:
    """Compute absolute left-right ankle-angle difference when ankle angles are available."""

    left = ankle_angle(pose, "left")
    right = ankle_angle(pose, "right")
    if left is None or right is None:
        return None
    return abs(left - right)


def knee_valgus_proxy(pose: Pose, side: str) -> float | None:
    """Estimate frontal-plane knee displacement relative to the hip-ankle line."""

    hip = _get_joint(pose, f"{side}_hip")
    knee = _get_joint(pose, f"{side}_knee")
    ankle = _get_joint(pose, f"{side}_ankle")
    if hip is None or knee is None or ankle is None:
        return None
    hx, hy = hip[:2]
    kx, ky = knee[:2]
    ax, ay = ankle[:2]
    line_dx = ax - hx
    line_dy = ay - hy
    denom = math.sqrt(line_dx * line_dx + line_dy * line_dy)
    if denom == 0:
        return None
    return abs(line_dy * kx - line_dx * ky + ax * hy - ay * hx) / denom


def knee_valgus_variation(frames: Sequence[Pose], side: str = "left") -> float | None:
    """Compute frame-to-frame variation of the knee valgus proxy."""

    values = [knee_valgus_proxy(frame, side) for frame in frames]
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return None
    return statistics.fstdev(clean)


def landing_phase_knee_angle_fluctuation(frames: Sequence[Pose], side: str = "left") -> float | None:
    """Compute knee-angle standard deviation across landing-phase frames."""

    values = [knee_flexion_angle(frame, side) for frame in frames]
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return None
    return statistics.fstdev(clean)


def takeoff_to_landing_duration(
    takeoff_frame: int | None,
    landing_frame: int | None,
    fps: float | None = None,
    takeoff_time: float | None = None,
    landing_time: float | None = None,
) -> float | None:
    """Return takeoff-to-landing duration from timestamps or frame indices."""

    if takeoff_time is not None and landing_time is not None:
        duration = landing_time - takeoff_time
        return duration if duration >= 0 else None
    if takeoff_frame is None or landing_frame is None or fps is None or fps <= 0:
        return None
    duration = (landing_frame - takeoff_frame) / fps
    return duration if duration >= 0 else None


def center_of_mass_height(pose: Pose) -> float | None:
    """Estimate center-of-mass height using pelvis first, then hip average."""

    pelvis = _get_joint(pose, "pelvis")
    if pelvis is not None and len(pelvis) >= 2:
        return pelvis[1]
    left_hip = _get_joint(pose, "left_hip")
    right_hip = _get_joint(pose, "right_hip")
    if left_hip is None or right_hip is None or len(left_hip) < 2 or len(right_hip) < 2:
        return None
    return (left_hip[1] + right_hip[1]) / 2.0


def lowest_center_of_mass_timing(frames: Sequence[Pose], timestamps: Sequence[float] | None = None) -> float | None:
    """Return the time or normalized frame index at which estimated COM is lowest."""

    heights = [center_of_mass_height(frame) for frame in frames]
    indexed = [(idx, value) for idx, value in enumerate(heights) if value is not None]
    if not indexed:
        return None
    lowest_idx, _ = min(indexed, key=lambda pair: pair[1])
    if timestamps is not None and lowest_idx < len(timestamps):
        return float(timestamps[lowest_idx])
    if len(frames) == 1:
        return 0.0
    return lowest_idx / float(len(frames) - 1)


def training_load(srpe: Any, duration_minutes: Any) -> float | None:
    """Compute session training load as sRPE multiplied by training duration."""

    try:
        srpe_value = float(srpe)
        duration_value = float(duration_minutes)
    except (TypeError, ValueError):
        return None
    if math.isnan(srpe_value) or math.isnan(duration_value) or duration_value < 0:
        return None
    return srpe_value * duration_value


def subjective_fatigue_features(record: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract subjective fatigue, sleep, and soreness fields without inventing missing values."""

    aliases = {
        "fatigue_score": ("fatigue_score", "fatigue", "vas_mf", "mental_fatigue"),
        "sleep_score": ("sleep_score", "sleep", "sleep_quality", "sleep_deprivation"),
        "muscle_soreness_score": ("muscle_soreness_score", "muscle_soreness", "soreness"),
    }
    result: dict[str, float | None] = {}
    lowered = {str(key).lower(): value for key, value in record.items()}
    for output_name, candidates in aliases.items():
        value = None
        for candidate in candidates:
            if candidate in lowered:
                try:
                    value = float(lowered[candidate])
                except (TypeError, ValueError):
                    value = None
                break
        result[output_name] = value
    return result


def valid_point_count(pose: Pose, joint_names: Sequence[str]) -> int:
    """Count joints that have at least two finite numeric coordinates."""

    return sum(1 for joint_name in joint_names if _get_joint(pose, joint_name) is not None)


def missing_joint_ratio(pose: Pose, joint_names: Sequence[str]) -> float | None:
    """Compute the fraction of expected joints missing from one pose."""

    if not joint_names:
        return None
    return 1.0 - (valid_point_count(pose, joint_names) / float(len(joint_names)))
