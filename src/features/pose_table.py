"""Build small pose-derived feature tables from confirmed schema samples."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.adapters.humanm3 import HUMANM3_JOINT_NAMES
from src.features.kinematics import (
    angle_degrees,
    knee_flexion_angle,
    knee_valgus_proxy,
    left_right_ankle_angle_difference,
    left_right_knee_angle_difference,
    missing_joint_ratio,
    valid_point_count,
)


EPFL_MULTIVIEW_JOINT_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


POSE_FEATURE_COLUMNS = [
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
    "pose_valid_joint_count",
    "pose_expected_joint_count",
    "pose_missing_joint_ratio",
]


POSE_TABLE_COLUMNS = [
    "dataset_name",
    "subject_id",
    "person_id",
    "session_id",
    "split",
    "scene_name",
    "camera_id",
    "view_id",
    "frame_id",
    "trial_id",
    "task_type",
    "source_file",
    "joint_schema",
    "hip_flexion_reference",
    "ankle_angle_status",
    *POSE_FEATURE_COLUMNS,
]


@dataclass(frozen=True)
class PoseTableSummary:
    """Summary for a generated small-sample pose feature table."""

    output_path: Path
    rows: int
    columns: int
    features_with_values: list[str]
    features_all_missing: list[str]
    source_files: list[str]


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number


def _joint_list_to_pose(joints: list[Any], joint_names: list[str]) -> dict[str, Any]:
    pose: dict[str, Any] = {}
    for idx, joint_name in enumerate(joint_names):
        if idx < len(joints) and isinstance(joints[idx], list):
            pose[joint_name] = joints[idx]
        else:
            pose[joint_name] = None
    return pose


def _hip_flexion(pose: dict[str, Any], side: str, reference: str) -> float | None:
    if reference == "pelvis":
        proximal = pose.get("pelvis")
    elif reference == "same_side_shoulder_proxy":
        proximal = pose.get(f"{side}_shoulder")
    else:
        proximal = None
    return angle_degrees(proximal, pose.get(f"{side}_hip"), pose.get(f"{side}_knee"))


def compute_pose_features(
    pose: dict[str, Any],
    expected_joint_names: list[str],
    hip_reference: str,
) -> dict[str, Any]:
    """Compute one row of pose features from named joints."""

    left_knee = knee_flexion_angle(pose, "left")
    right_knee = knee_flexion_angle(pose, "right")
    features = {
        "left_knee_flexion_angle": left_knee,
        "right_knee_flexion_angle": right_knee,
        "left_hip_flexion_angle": _hip_flexion(pose, "left", hip_reference),
        "right_hip_flexion_angle": _hip_flexion(pose, "right", hip_reference),
        "left_ankle_angle": None,
        "right_ankle_angle": None,
        "left_right_knee_angle_difference": left_right_knee_angle_difference(pose),
        "left_right_ankle_angle_difference": left_right_ankle_angle_difference(pose),
        "left_knee_valgus_proxy": knee_valgus_proxy(pose, "left"),
        "right_knee_valgus_proxy": knee_valgus_proxy(pose, "right"),
        "pose_valid_joint_count": valid_point_count(pose, expected_joint_names),
        "pose_expected_joint_count": len(expected_joint_names),
        "pose_missing_joint_ratio": missing_joint_ratio(pose, expected_joint_names),
    }
    return features


def _write_csv(rows: list[dict[str, Any]], output_path: Path, columns: list[str]) -> PoseTableSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})

    features_with_values: list[str] = []
    features_all_missing: list[str] = []
    for feature in POSE_FEATURE_COLUMNS:
        values = [row.get(feature) for row in rows]
        has_value = any(value is not None and value != "" and _is_number(value) for value in values)
        (features_with_values if has_value else features_all_missing).append(feature)
    source_files = sorted({str(row.get("source_file")) for row in rows if row.get("source_file")})
    return PoseTableSummary(
        output_path=output_path,
        rows=len(rows),
        columns=len(columns),
        features_with_values=features_with_values,
        features_all_missing=features_all_missing,
        source_files=source_files,
    )


def _safe_relpath(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _humanm3_sample_files(dataset_root: Path, per_scene_files: int) -> list[tuple[Path, str, str, str]]:
    specs = [
        ("train", "basketball1", dataset_root / "train" / "basketball1" / "split1" / "pose_calib"),
        ("train", "basketball2", dataset_root / "train" / "basketball2" / "pose_calib"),
        ("test", "basketball1", dataset_root / "test" / "basketball1" / "split1" / "pose_calib"),
        ("test", "basketball2", dataset_root / "test" / "basketball2" / "pose_calib"),
    ]
    result: list[tuple[Path, str, str, str]] = []
    for split, scene, pose_dir in specs:
        if not pose_dir.is_dir():
            continue
        session_id = pose_dir.parent.name if pose_dir.parent.name.startswith("split") else scene
        for path in sorted(pose_dir.glob("*.json"))[:per_scene_files]:
            result.append((path, split, scene, session_id))
    return result


def build_humanm3_pose_feature_sample(
    dataset_root: Path,
    output_path: Path,
    project_root: Path,
    per_scene_files: int = 2,
) -> PoseTableSummary:
    """Build a small Human-M3 15-joint 3D pose feature table."""

    rows: list[dict[str, Any]] = []
    for path, split, scene, session_id in _humanm3_sample_files(dataset_root, per_scene_files):
        data = json.loads(path.read_text(encoding="utf-8"))
        frame_id = path.stem
        for person_id, joints in data.items():
            if not isinstance(joints, list):
                continue
            pose = _joint_list_to_pose(joints, HUMANM3_JOINT_NAMES)
            row = {
                "dataset_name": "humanm3",
                "subject_id": person_id,
                "person_id": person_id,
                "session_id": session_id,
                "split": split,
                "scene_name": scene,
                "camera_id": "",
                "view_id": "3d_pose_calib",
                "frame_id": frame_id,
                "trial_id": "",
                "task_type": "basketball_pose",
                "source_file": _safe_relpath(path, project_root),
                "joint_schema": "humanm3_15_joint",
                "hip_flexion_reference": "pelvis",
                "ankle_angle_status": "missing_distal_foot_or_toe_joint",
            }
            row.update(compute_pose_features(pose, HUMANM3_JOINT_NAMES, hip_reference="pelvis"))
            rows.append(row)
    return _write_csv(rows, output_path, POSE_TABLE_COLUMNS)


def build_epfl_multiview_pose_feature_sample(
    dataset_root: Path,
    output_path: Path,
    project_root: Path,
    max_frames_per_subject: int = 20,
) -> PoseTableSummary:
    """Build a small EPFL multiview 13-joint 3D pose feature table."""

    rows: list[dict[str, Any]] = []
    pose_files = [
        dataset_root / "human_poses" / "21380_21440_3" / "pose_subject7.json",
        dataset_root / "human_poses" / "21380_21440_3" / "pose_subject12.json",
    ]
    for path in pose_files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        subject_id = path.stem.replace("pose_subject", "")
        split = "test" if subject_id in {"7", "12"} else "train"
        session_id = path.parent.name
        frames = data.get("3d", {}).get("idx_frame", [])[:max_frames_per_subject]
        poses = data.get("3d", {}).get("pose", [])[:max_frames_per_subject]
        available_views = ",".join(sorted(data.get("2d", {}).keys()))
        for frame_id, joints in zip(frames, poses):
            if not isinstance(joints, list):
                continue
            pose = _joint_list_to_pose(joints, EPFL_MULTIVIEW_JOINT_NAMES)
            row = {
                "dataset_name": "epfl_sportcenter_multiview",
                "subject_id": subject_id,
                "person_id": subject_id,
                "session_id": session_id,
                "split": split,
                "scene_name": "sportcenter_basketball",
                "camera_id": "",
                "view_id": "3d_triangulated",
                "frame_id": frame_id,
                "trial_id": "",
                "task_type": "basketball_pose",
                "source_file": _safe_relpath(path, project_root),
                "joint_schema": "epfl_multiview_13_joint",
                "hip_flexion_reference": "same_side_shoulder_proxy",
                "ankle_angle_status": "missing_distal_foot_or_toe_joint",
                "available_2d_views": available_views,
            }
            row.update(
                compute_pose_features(
                    pose,
                    EPFL_MULTIVIEW_JOINT_NAMES,
                    hip_reference="same_side_shoulder_proxy",
                )
            )
            rows.append(row)
    columns = [*POSE_TABLE_COLUMNS, "available_2d_views"]
    return _write_csv(rows, output_path, columns)
