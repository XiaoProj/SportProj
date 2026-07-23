"""Adapter for EPFL SportCenter single-view / camera-pose data."""

from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import BaseDatasetAdapter, DatasetManifest, SplitInfo
from src.data.schema_utils import read_text_excerpt, summarize_json_file


TRAINING_SEQUENCES = [
    "seq_9841",
    "seq_9842",
    "seq_9843",
    "seq_9849",
    "seq_9850",
    "seq_171305",
    "seq_171931",
    "seq_172146",
    "seq_172444",
    "seq_173321",
    "seq_173510",
    "seq_173833",
]

TESTING_SEQUENCES = [
    "seq_9844",
    "seq_9847",
    "seq_9851",
    "seq_9852",
    "seq_9853",
    "seq_9845",
    "seq_9854",
    "seq_9855",
    "seq_9856",
    "seq_9857",
    "seq_172318",
    "seq_172647",
    "seq_172730",
    "seq_173742",
    "seq_174006",
    "seq_174210",
]


class EPFLSingleViewAdapter(BaseDatasetAdapter):
    """Read-only structural adapter for the single-view SportCenter dataset."""

    name = "epfl_sportcenter_single_view"

    def discover_sequences(self) -> list[Path]:
        self.validate_root()
        return sorted(path for path in self.root.iterdir() if path.is_dir() and path.name.startswith("seq_"))

    def split_info(self) -> SplitInfo:
        return SplitInfo(
            split_type="sequence",
            train=TRAINING_SEQUENCES,
            test=TESTING_SEQUENCES,
            notes=["Split comes from the dataset README.txt."],
        )

    def manifest(self) -> DatasetManifest:
        sequences = self.discover_sequences()
        files = self.iter_files(patterns=["*.json", "*.txt", "*.py", "*.png"])
        notes = [
            "Sequence folders contain images_orig_blurred/ plus poses.json and/or player_positions.json.",
            "This adapter does not extract or rewrite images.",
        ]
        return DatasetManifest(
            name=self.name,
            root=self.root,
            files=files,
            directories=sequences,
            supported_tasks=[
                "single-view camera-pose inspection",
                "player position summary",
                "sequence-level train/test split evaluation",
            ],
            split_info=self.split_info(),
            notes=notes,
        )

    def read_schema_sample(self, project_root: str | Path | None = None) -> dict:
        """Read README and a small set of JSON schema samples."""

        project = Path(project_root) if project_root is not None else self.root.parent.parent
        preferred_sequence = self.root / "seq_9850"
        sequence = preferred_sequence if preferred_sequence.is_dir() else self.discover_sequences()[0]
        sample_files = [
            self.root / "README.txt",
            self.root / "ground_grid.json",
            self.root / "homography_rectified_template.json",
            self.root / "intrinsics_seq_17xxxx.json",
            self.root / "intrinsics_seq_98xx.json",
            sequence / "poses.json",
            sequence / "player_positions.json",
        ]

        samples: dict[str, dict] = {}
        for path in sample_files:
            if not path.exists():
                continue
            if path.suffix.lower() == ".txt":
                samples[path.name] = read_text_excerpt(path, project)
            else:
                samples[path.name] = summarize_json_file(path, project)

        return {
            "dataset_name": self.name,
            "root": str(self.root),
            "sample_sequence": sequence.name,
            "split_info": self.split_info_dict(),
            "samples": samples,
            "detected_schema": {
                "frame_id": "Top-level image filename keys in poses.json and player_positions.json.",
                "camera_pose": "poses.json frame entries include R, Hr, filename, and t.",
                "player_positions": "player_positions.json maps frame filenames to lists of player position entries.",
                "court_geometry": "ground_grid.json and homography_rectified_template.json are present.",
                "intrinsics": "intrinsics_seq_17xxxx.json and intrinsics_seq_98xx.json are present.",
                "ball": "TODO: no ball field observed in sampled files.",
                "fatigue_label": "TODO: no fatigue label observed.",
                "before_after_label": "TODO: no before/after label observed.",
            },
        }
