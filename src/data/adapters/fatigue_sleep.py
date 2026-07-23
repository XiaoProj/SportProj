"""Adapter for basketball mental fatigue / sleep restriction data."""

from __future__ import annotations

from pathlib import Path

from src.data.adapters.base import BaseDatasetAdapter, DatasetManifest, SplitInfo
from src.data.schema_utils import read_xlsx_preview, safe_relpath


class FatigueSleepAdapter(BaseDatasetAdapter):
    """Read-only structural adapter for the 5742821 fatigue/sleep dataset."""

    name = "basketball_fatigue_sleep_5742821"

    def split_info(self) -> SplitInfo:
        return SplitInfo(
            split_type="none_detected",
            notes=[
                "No train/test split files or directories were detected.",
                "Use cross-validation or subject-level splitting after labels and subject identifiers are confirmed.",
            ],
        )

    def manifest(self) -> DatasetManifest:
        files = self.iter_files(patterns=["*.xlsx", "*.AW5", "*.aw5"])
        notes = [
            "Data.xlsx can be inspected for spreadsheet headers without modifying the file.",
            "Mental fatigue.AW5 is an unknown proprietary format and must not be interpreted until a reader/schema is confirmed.",
        ]
        return DatasetManifest(
            name=self.name,
            root=self.root,
            files=files,
            directories=self.iter_directories(),
            supported_tasks=[
                "subjective fatigue variable design",
                "sleep restriction metadata integration",
                "shooting accuracy and VAS score analysis after schema confirmation",
            ],
            split_info=self.split_info(),
            notes=notes,
        )

    def read_schema_sample(self, project_root: str | Path | None = None) -> dict:
        """Read spreadsheet workbook metadata and keep AW5 as unknown."""

        project = Path(project_root) if project_root is not None else self.root.parent.parent
        xlsx_path = self.root / "Data.xlsx"
        aw5_path = self.root / "Mental fatigue.AW5"
        samples: dict[str, dict] = {}
        if xlsx_path.exists():
            samples["Data.xlsx"] = read_xlsx_preview(xlsx_path, project)
        if aw5_path.exists():
            samples["Mental fatigue.AW5"] = {
                "path": safe_relpath(aw5_path, project),
                "size_bytes": aw5_path.stat().st_size,
                "format_status": "unknown_proprietary",
                "fields": "TODO: unknown; do not infer fields until reader/schema is confirmed.",
            }

        return {
            "dataset_name": self.name,
            "root": str(self.root),
            "split_info": self.split_info_dict(),
            "samples": samples,
            "detected_schema": {
                "subject_id": "Data.xlsx row 2 includes Subject.",
                "training_history": "Data.xlsx row 2 includes frequency training, years of basketball, height, weight, and age.",
                "performance": "Data.xlsx includes Shot 1 and Shot 2 accuracy fields.",
                "fatigue_condition": "Data.xlsx row 1 includes Mental fatigue and Mental fatigue + Sleep deprivation condition groups.",
                "subjective_fatigue": "Data.xlsx row 2 includes VAS MF 1/2/3 and VAS Mot 1/2/3 fields.",
                "sleep_condition": "Sleep deprivation is present as an experimental condition group, not a direct numeric sleep score.",
                "fatigue_label": "TODO: no explicit binary fatigue_label column observed in the header preview.",
                "before_after_label": "TODO: no explicit before/after label observed in the header preview.",
                "aw5": "Unknown proprietary format; not parsed.",
            },
        }
