"""Adapter for EPFL SportCenter multi-view basketball data."""

from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import BaseDatasetAdapter, DatasetManifest, SplitInfo
from src.data.schema_utils import list_immediate_files, read_text_excerpt, summarize_json_file


class EPFLMultiViewAdapter(BaseDatasetAdapter):
    """Read-only structural adapter for the multi-view SportCenter dataset."""

    name = "epfl_sportcenter_multi_view"

    def split_info(self) -> SplitInfo:
        return SplitInfo(
            split_type="subject",
            test=["subject_7", "subject_12"],
            notes=[
                "README states that subjects 7 and 12 are used for testing and the rest for training.",
                "Subject identifiers should be confirmed from pose_subject*.json files before modeling.",
            ],
        )

    def extraction_requirements(self) -> list[str]:
        return [
            "Raw videos are present as multicam/ace_*.mp4.",
            "Frame extraction is not performed by this adapter.",
            "If extraction is required later, write approved outputs only to outputs/extracted/.",
        ]

    def manifest(self) -> DatasetManifest:
        files = self.iter_files(patterns=["*.json", "*.txt", "*.py", "*.png", "*.jpg", "*.mp4"])
        directories = self.iter_directories()
        return DatasetManifest(
            name=self.name,
            root=self.root,
            files=files,
            directories=directories,
            supported_tasks=[
                "multi-view video inspection",
                "2D and 3D pose feature engineering",
                "trajectory-based movement analysis",
            ],
            split_info=self.split_info(),
            notes=self.extraction_requirements(),
        )

    def _pose_samples(self) -> list[Path]:
        samples: list[Path] = []
        for subject_id in ("7", "12"):
            matches = sorted((self.root / "human_poses").glob(f"*/pose_subject{subject_id}.json"))
            if matches:
                samples.append(matches[0])
        if not samples:
            samples = sorted((self.root / "human_poses").glob("*/pose_subject*.json"))[:2]
        return samples

    def read_schema_sample(self, project_root: str | Path | None = None) -> dict:
        """Read README, calibration, trajectories, and a few pose annotations."""

        project = Path(project_root) if project_root is not None else self.root.parent.parent
        samples: dict[str, dict | list[dict]] = {}
        readme = self.root / "README.txt"
        if readme.exists():
            samples["README.txt"] = read_text_excerpt(readme, project)
        for json_name in ("calibration.json", "trajectories.json"):
            path = self.root / json_name
            if path.exists():
                samples[json_name] = summarize_json_file(path, project)

        pose_samples = []
        for path in self._pose_samples():
            pose_samples.append(summarize_json_file(path, project))
        samples["pose_subject_samples"] = pose_samples
        samples["videos"] = list_immediate_files(self.root / "multicam", "ace_*.mp4", project)

        return {
            "dataset_name": self.name,
            "root": str(self.root),
            "split_info": self.split_info_dict(),
            "samples": samples,
            "detected_schema": {
                "camera_id": "Camera/video IDs use ace_0 through ace_7 in calibration and multicam video filenames.",
                "camera_parameters": "calibration.json entries include K, dist, R, and t.",
                "video_mapping": "multicam/ace_N.mp4 corresponds to calibration key ace_N.",
                "pose_2d": "pose_subject*.json contains a 2d section keyed by camera IDs such as ace_1.",
                "pose_3d": "pose_subject*.json contains a 3d section with idx_frame and pose.",
                "trajectories": "trajectories.json is present and sampled for top-level structure.",
                "test_subjects": "README states subjects 7 and 12 are used for testing.",
                "fatigue_label": "TODO: no fatigue label observed.",
                "before_after_label": "TODO: no before/after label observed.",
            },
        }
