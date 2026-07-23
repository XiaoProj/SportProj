"""Full or large-sample pose movement-quality processing for Route A."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.adapters.humanm3 import HUMANM3_JOINT_NAMES
from src.features.pose_table import (
    EPFL_MULTIVIEW_JOINT_NAMES,
    POSE_TABLE_COLUMNS,
    _joint_list_to_pose,
    compute_pose_features,
)


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_rows_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _humanm3_pose_files(dataset_root: Path) -> list[Path]:
    roots = [
        dataset_root / "train" / "basketball1",
        dataset_root / "train" / "basketball2",
        dataset_root / "test" / "basketball1",
        dataset_root / "test" / "basketball2",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("pose_calib/*.json")))
    return sorted(files)


def _humanm3_meta(path: Path, dataset_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(dataset_root)
    split = rel.parts[0]
    scene = rel.parts[1]
    session = scene
    if "split1" in rel.parts:
        session = "split1"
    elif "split2" in rel.parts:
        session = "split2"
    return split, scene, session


def process_humanm3_full(
    *,
    dataset_root: Path,
    output_path: Path,
    project_root: Path,
    max_runtime_seconds: float = 240.0,
) -> dict[str, Any]:
    """Process Human-M3 basketball pose_calib JSON files."""

    files = _humanm3_pose_files(dataset_root)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    start = time.monotonic()
    processed_files = 0
    mode = "full"
    sampling_rule = "all basketball1/basketball2 train/test pose_calib JSON files"

    for idx, path in enumerate(files):
        if time.monotonic() - start > max_runtime_seconds:
            mode = "large_sample"
            sampling_rule = f"processed first {processed_files} of {len(files)} sorted pose_calib JSON files before runtime guard"
            skipped.extend(
                {
                    "dataset": "humanm3",
                    "source_file": _safe_rel(remaining, project_root),
                    "reason": "runtime guard large-sample downgrade",
                }
                for remaining in files[idx:]
            )
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append({"dataset": "humanm3", "source_file": _safe_rel(path, project_root), "reason": repr(exc)})
            continue
        split, scene, session_id = _humanm3_meta(path, dataset_root)
        frame_id = path.stem
        if not isinstance(data, dict):
            skipped.append({"dataset": "humanm3", "source_file": _safe_rel(path, project_root), "reason": "top-level JSON is not dict"})
            continue
        processed_files += 1
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
                "source_file": _safe_rel(path, project_root),
                "joint_schema": "humanm3_15_joint",
                "hip_flexion_reference": "pelvis",
                "ankle_angle_status": "structural_unavailable_missing_distal_foot_or_toe_joint",
            }
            row.update(compute_pose_features(pose, HUMANM3_JOINT_NAMES, hip_reference="pelvis"))
            rows.append(row)

    _write_rows_csv(rows, output_path, POSE_TABLE_COLUMNS)
    return {
        "dataset": "humanm3",
        "mode": mode,
        "sampling_rule": sampling_rule,
        "candidate_files": len(files),
        "processed_pose_json": processed_files,
        "skipped_files": len(skipped),
        "feature_rows": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "persons": len({row["person_id"] for row in rows}),
        "frames": len({row["frame_id"] for row in rows}),
        "output_path": output_path.as_posix(),
        "skipped": skipped,
    }


def process_epfl_multiview_full(
    *,
    dataset_root: Path,
    output_path: Path,
    project_root: Path,
    max_runtime_seconds: float = 180.0,
) -> dict[str, Any]:
    """Process all EPFL multiview pose_subject JSON files."""

    pose_files = sorted((dataset_root / "human_poses").rglob("pose_subject*.json"))
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    start = time.monotonic()
    processed_files = 0
    mode = "full"
    sampling_rule = "all pose_subject*.json files under human_poses"

    for idx, path in enumerate(pose_files):
        if time.monotonic() - start > max_runtime_seconds:
            mode = "large_sample"
            sampling_rule = f"processed first {processed_files} of {len(pose_files)} pose_subject JSON files before runtime guard"
            skipped.extend(
                {
                    "dataset": "epfl_multiview",
                    "source_file": _safe_rel(remaining, project_root),
                    "reason": "runtime guard large-sample downgrade",
                }
                for remaining in pose_files[idx:]
            )
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append({"dataset": "epfl_multiview", "source_file": _safe_rel(path, project_root), "reason": repr(exc)})
            continue
        subject_id = path.stem.replace("pose_subject", "")
        split = "test" if subject_id in {"7", "12"} else "train"
        session_id = path.parent.name
        available_views = ",".join(sorted(data.get("2d", {}).keys())) if isinstance(data, dict) else ""
        frames = data.get("3d", {}).get("idx_frame", []) if isinstance(data, dict) else []
        poses = data.get("3d", {}).get("pose", []) if isinstance(data, dict) else []
        if not frames or not poses:
            skipped.append({"dataset": "epfl_multiview", "source_file": _safe_rel(path, project_root), "reason": "missing 3d idx_frame/pose arrays"})
            continue
        processed_files += 1
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
                "source_file": _safe_rel(path, project_root),
                "joint_schema": "epfl_multiview_13_joint",
                "hip_flexion_reference": "same_side_shoulder_proxy",
                "ankle_angle_status": "structural_unavailable_missing_distal_foot_or_toe_joint",
                "available_2d_views": available_views,
            }
            row.update(compute_pose_features(pose, EPFL_MULTIVIEW_JOINT_NAMES, hip_reference="same_side_shoulder_proxy"))
            rows.append(row)

    columns = [*POSE_TABLE_COLUMNS, "available_2d_views"]
    _write_rows_csv(rows, output_path, columns)
    return {
        "dataset": "epfl_multiview",
        "mode": mode,
        "sampling_rule": sampling_rule,
        "candidate_files": len(pose_files),
        "processed_pose_json": processed_files,
        "skipped_files": len(skipped),
        "feature_rows": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "persons": len({row["person_id"] for row in rows}),
        "frames": len({str(row["frame_id"]) for row in rows}),
        "output_path": output_path.as_posix(),
        "skipped": skipped,
    }


def process_epfl_single_metadata(
    *,
    dataset_root: Path,
    output_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Summarize EPFL single-view/camera-pose metadata without joint features."""

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    top_files = {path.name: path for path in dataset_root.glob("*") if path.is_file()}
    for seq_dir in sorted(path for path in dataset_root.glob("seq_*") if path.is_dir()):
        poses_path = seq_dir / "poses.json"
        image_count = len(list((seq_dir / "images_orig_blurred").glob("*.JPG"))) if (seq_dir / "images_orig_blurred").is_dir() else 0
        pose_records = None
        pose_keys = ""
        if poses_path.exists():
            try:
                data = json.loads(poses_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    pose_records = len(data)
                    pose_keys = ",".join(list(data.keys())[:8])
                elif isinstance(data, list):
                    pose_records = len(data)
                    pose_keys = "list"
            except Exception as exc:
                skipped.append({"dataset": "epfl_single", "source_file": _safe_rel(poses_path, project_root), "reason": repr(exc)})
        rows.append(
            {
                "dataset_name": "epfl_sportcenter_single_view_camerapose",
                "sequence_id": seq_dir.name,
                "poses_json_present": poses_path.exists(),
                "poses_json_records": pose_records,
                "poses_json_top_keys_or_type": pose_keys,
                "image_file_count_metadata_only": image_count,
                "player_positions_json_present": (seq_dir / "player_positions.json").exists(),
                "human_joint_keypoints_present": False,
                "movement_quality_joint_features_supported": False,
                "metadata_role": "camera_pose_player_position_court_geometry_only",
                "source_file": _safe_rel(poses_path, project_root) if poses_path.exists() else "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {
        "dataset": "epfl_single",
        "processed_sequences": len(rows),
        "top_level_metadata_files": ",".join(sorted(top_files)),
        "feature_rows": len(rows),
        "output_path": output_path.as_posix(),
        "skipped": skipped,
    }


def write_pose_analysis_outputs(
    *,
    human_path: Path,
    epfl_path: Path,
    combined_path: Path,
    result_root: Path,
    audit_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Write combined pose table, summaries, missingness, comparisons, and audit."""

    human = pd.read_csv(human_path)
    epfl = pd.read_csv(epfl_path)
    combined = pd.concat([human, epfl], ignore_index=True, sort=False)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    result_root.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "left_knee_flexion_angle",
        "right_knee_flexion_angle",
        "left_hip_flexion_angle",
        "right_hip_flexion_angle",
        "left_right_knee_angle_difference",
        "left_knee_valgus_proxy",
        "right_knee_valgus_proxy",
        "pose_missing_joint_ratio",
    ]
    summary = (
        combined.groupby("dataset_name")[feature_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary_path = result_root / "pose_movement_quality_feature_summary_full.csv"
    summary.to_csv(summary_path, index=False)

    missing_rows = []
    for dataset, subset in combined.groupby("dataset_name"):
        for feature in feature_cols + ["left_ankle_angle", "right_ankle_angle", "left_right_ankle_angle_difference"]:
            status = "structural_unavailable" if "ankle" in feature else "computable"
            missing_rows.append(
                {
                    "dataset_name": dataset,
                    "feature": feature,
                    "availability_status": status,
                    "missing_rate": subset[feature].isna().mean() if feature in subset else 1.0,
                    "n_rows": len(subset),
                }
            )
    missing_path = result_root / "pose_movement_quality_missingness_summary_full.csv"
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False)

    comparison_rows = []
    for dataset, subset in combined.groupby("dataset_name"):
        comparison_rows.append(
            {
                "dataset_name": dataset,
                "rows": len(subset),
                "subjects": subset["subject_id"].nunique(),
                "persons": subset["person_id"].nunique(),
                "frames": subset["frame_id"].nunique(),
                "mean_knee_angle": subset[["left_knee_flexion_angle", "right_knee_flexion_angle"]].mean(numeric_only=True).mean(),
                "mean_hip_angle": subset[["left_hip_flexion_angle", "right_hip_flexion_angle"]].mean(numeric_only=True).mean(),
                "mean_knee_asymmetry": subset["left_right_knee_angle_difference"].mean(),
                "mean_pose_missing_joint_ratio": subset["pose_missing_joint_ratio"].mean(),
            }
        )
    comparison_path = result_root / "pose_movement_quality_dataset_comparison_full.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    audit_path = result_root / "pose_movement_quality_processing_audit_full.csv"
    flat_audit = []
    for item in audit_rows:
        copy = {key: value for key, value in item.items() if key != "skipped"}
        copy["skipped_detail_count"] = len(item.get("skipped", []))
        flat_audit.append(copy)
        for skipped in item.get("skipped", []):
            flat_audit.append({**copy, "skipped_source_file": skipped.get("source_file"), "skipped_reason": skipped.get("reason")})
    pd.DataFrame(flat_audit).to_csv(audit_path, index=False)
    return {
        "combined": combined_path.as_posix(),
        "summary": summary_path.as_posix(),
        "missingness": missing_path.as_posix(),
        "comparison": comparison_path.as_posix(),
        "audit": audit_path.as_posix(),
    }
