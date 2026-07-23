"""Adapter for Human-M3 multi-person multi-view pose data."""

from __future__ import annotations

import os
from pathlib import Path

from src.data.adapters.base import BaseDatasetAdapter, DatasetManifest, SplitInfo
from src.data.schema_utils import first_matching_file, read_text_excerpt, summarize_json_file


HUMANM3_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


class HumanM3Adapter(BaseDatasetAdapter):
    """Read-only structural adapter for Human-M3."""

    name = "humanm3"

    def scenes(self) -> dict[str, list[str]]:
        self.validate_root()
        result: dict[str, list[str]] = {}
        for split in ("train", "test"):
            split_dir = self.root / split
            if split_dir.is_dir():
                result[split] = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        return result

    def split_info(self) -> SplitInfo:
        scenes = self.scenes()
        return SplitInfo(
            split_type="directory",
            train=[f"train/{name}" for name in scenes.get("train", [])],
            test=[f"test/{name}" for name in scenes.get("test", [])],
            notes=[
                "Explicit train/ and test/ directories are present.",
                "basketball1, basketball2, intersection, and plaza scenes are available when present.",
            ],
        )

    def manifest(self) -> DatasetManifest:
        files = self.iter_files(patterns=["*.json", "*.jpg", "*.jpeg", "*.pcd", "*.md"])
        directories = self.iter_directories()
        notes = [
            "pose_calib JSON files contain 15-joint pose annotations using the documented Human-M3 joint order.",
            "Point cloud files are left untouched; no conversion is performed.",
        ]
        return DatasetManifest(
            name=self.name,
            root=self.root,
            files=files,
            directories=directories,
            supported_tasks=[
                "3D joint angle feature engineering",
                "multi-person pose adapter validation",
                "movement symmetry and stability feature prototyping",
            ],
            split_info=self.split_info(),
            notes=notes,
        )

    @staticmethod
    def joint_names() -> list[str]:
        return list(HUMANM3_JOINT_NAMES)

    def _scene_sample(self, split: str, scene: str, project_root: Path) -> dict:
        scene_root = self.root / split / scene
        sample: dict = {"scene_path": str(scene_root), "exists": scene_root.is_dir()}
        if not scene_root.is_dir():
            return sample

        pose_file = first_matching_file(scene_root, "pose_calib", ".json")
        camera_file = first_matching_file(scene_root, "camera_calibration", ".json")
        if pose_file is not None:
            pose_summary = summarize_json_file(pose_file, project_root)
            sample["pose_calib_sample"] = pose_summary
            first_person = next(iter(pose_summary["summary"].get("keys", [])), None)
            sample["joint_order"] = HUMANM3_JOINT_NAMES
            sample["joint_count_expected_from_readme"] = len(HUMANM3_JOINT_NAMES)
            sample["first_person_id_in_sample"] = first_person
        if camera_file is not None:
            sample["camera_calibration_sample"] = summarize_json_file(camera_file, project_root)

        image_root = None
        for dirpath, dirnames, _filenames in os.walk(scene_root):
            current = Path(dirpath)
            if current.name == "images":
                image_root = current
                break
            if "images" in dirnames:
                image_root = current / "images"
                break
        image_dirs = sorted(path.name for path in image_root.iterdir() if path.is_dir()) if image_root else []
        if image_dirs:
            sample["image_view_directories"] = image_dirs[:8]
        return sample

    def read_schema_sample(self, project_root: str | Path | None = None) -> dict:
        """Read README plus one pose/camera sample from basketball train/test scenes."""

        project = Path(project_root) if project_root is not None else self.root.parent.parent
        samples: dict[str, dict] = {}
        readme = self.root / "README.md"
        if readme.exists():
            samples["README.md"] = read_text_excerpt(readme, project)
        for split in ("train", "test"):
            for scene in ("basketball1", "basketball2"):
                samples[f"{split}/{scene}"] = self._scene_sample(split, scene, project)

        return {
            "dataset_name": self.name,
            "root": str(self.root),
            "split_info": self.split_info_dict(),
            "samples": samples,
            "detected_schema": {
                "split": "Top-level train/ and test/ directories.",
                "scene_name": "Scene folders include basketball1 and basketball2, plus non-basketball scenes.",
                "camera_parameters": "camera_calibration/camera_*.json entries include intrinsic and extrinsic.",
                "view_id": "images/camera_* directories indicate camera/view IDs where present.",
                "person_id": "pose_calib JSON top-level keys are person identifiers.",
                "pose_3d": "pose_calib person entries are lists of 15 joint coordinate lists.",
                "joint_names": HUMANM3_JOINT_NAMES,
                "fatigue_label": "TODO: no fatigue label observed.",
                "before_after_label": "TODO: no before/after label observed.",
            },
        }
