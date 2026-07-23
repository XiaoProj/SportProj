"""Base classes for read-only dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitInfo:
    """Describes available split metadata without changing raw files."""

    split_type: str
    train: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetManifest:
    """Read-only structural description of a dataset."""

    name: str
    root: Path
    files: list[Path]
    directories: list[Path]
    supported_tasks: list[str]
    split_info: SplitInfo
    notes: list[str] = field(default_factory=list)


class BaseDatasetAdapter:
    """Base adapter for inspecting raw data without writing into dataset/."""

    name = "base"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def validate_root(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

    def iter_files(self, patterns: Iterable[str] | None = None) -> list[Path]:
        """Return matching files under root using read-only glob traversal."""

        self.validate_root()
        if patterns is None:
            return sorted(path for path in self.root.rglob("*") if path.is_file())
        files: list[Path] = []
        for pattern in patterns:
            files.extend(path for path in self.root.rglob(pattern) if path.is_file())
        return sorted(set(files))

    def iter_directories(self) -> list[Path]:
        """Return directories under root using read-only traversal."""

        self.validate_root()
        return sorted(path for path in self.root.rglob("*") if path.is_dir())

    def split_info(self) -> SplitInfo:
        """Return known split information for this dataset."""

        return SplitInfo(split_type="unknown", notes=["No split metadata implemented for this adapter."])

    def extraction_requirements(self) -> list[str]:
        """Report extraction needs; adapters never extract data automatically."""

        return []

    def split_info_dict(self) -> dict:
        """Return split information as a serializable dictionary."""

        return asdict(self.split_info())

    def read_schema_sample(self, project_root: str | Path | None = None) -> dict:
        """Read a minimal schema sample for this dataset.

        Subclasses should override this method with small, explicit reads only.
        """

        _ = project_root
        return {
            "dataset_name": self.name,
            "root": str(self.root),
            "split_info": self.split_info_dict(),
            "notes": ["TODO: implement dataset-specific schema probing."],
        }

    def manifest(self) -> DatasetManifest:
        """Build a read-only manifest for this dataset."""

        return DatasetManifest(
            name=self.name,
            root=self.root,
            files=self.iter_files(),
            directories=self.iter_directories(),
            supported_tasks=[],
            split_info=self.split_info(),
            notes=self.extraction_requirements(),
        )
