"""Read-only dataset inventory utilities.

This module only scans paths and reads small metadata files such as README
documents. It never writes inside the raw dataset directory.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


COMPRESSED_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".rar", ".7z", ".bz2", ".xz"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOTATION_EXTENSIONS = {".json", ".txt", ".csv", ".xlsx", ".xls", ".mat", ".pkl", ".pickle"}
POINTCLOUD_EXTENSIONS = {".pcd", ".ply", ".las"}
README_NAMES = {"readme", "readme.txt", "readme.md"}
SPLIT_PATTERN = re.compile(r"(train|training|test|testing|val|valid|validation|split|fold)", re.I)


KNOWN_DATASETS = {
    "sportcenter_camerapose_dataset": {
        "display_name": "EPFL SportCenter single-view / camera-pose",
        "purpose": (
            "Single-view basketball imagery with camera pose, player positions, "
            "ground grid, homography, and intrinsics."
        ),
        "tasks": [
            "single-view pose/camera-pose inspection",
            "player-position and movement-quality feature prototyping",
            "train/test sequence split benchmarking",
        ],
    },
    "sportcenter_multiview_dataset": {
        "display_name": "EPFL SportCenter multi-view",
        "purpose": (
            "Eight synchronized calibrated basketball videos with trajectories, "
            "obstacle masks, and manually annotated 2D/3D human poses."
        ),
        "tasks": [
            "multi-view pose and 3D kinematics",
            "subject-level train/test split benchmarking",
            "video-to-pose pipeline validation when frame extraction is approved",
        ],
    },
    "humanm3": {
        "display_name": "Human-M3 multi-person multi-view pose",
        "purpose": (
            "Multi-person, multi-view scenes including basketball with images, "
            "point clouds, camera calibration, 15-joint pose labels, and SMPL estimates."
        ),
        "tasks": [
            "3D joint-angle feature engineering",
            "pose adapter validation with explicit train/test directories",
            "movement symmetry and stability feature prototyping",
        ],
    },
    "5742821": {
        "display_name": "Basketball mental fatigue / sleep restriction",
        "purpose": (
            "Small spreadsheet plus one AW5 proprietary file related to basketball "
            "mental fatigue and sleep-deprivation experiments."
        ),
        "tasks": [
            "subjective fatigue and sleep variables",
            "shooting-performance and VAS score analysis",
            "metadata source for fatigue-state labels after schema confirmation",
        ],
    },
}


@dataclass(frozen=True)
class ExtensionSummary:
    """File-count and size summary for one extension."""

    extension: str
    count: int
    total_bytes: int


@dataclass(frozen=True)
class DatasetInventory:
    """Inventory summary for one top-level dataset folder."""

    key: str
    display_name: str
    path: str
    purpose: str
    tasks: list[str]
    directory_count: int
    file_count: int
    total_bytes: int
    extension_summaries: list[ExtensionSummary]
    category_counts: dict[str, int]
    split_evidence: list[str] = field(default_factory=list)
    archive_files: list[str] = field(default_factory=list)
    notable_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectInventory:
    """Inventory for all datasets under the raw dataset root."""

    dataset_root: str
    dataset_count: int
    total_directories: int
    total_files: int
    total_bytes: int
    datasets: list[DatasetInventory]


def bytes_to_human(num_bytes: int) -> str:
    """Format bytes into a compact human-readable value."""

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def classify_extension(extension: str) -> str:
    """Map a filename extension to a broad data category."""

    ext = extension.lower()
    if ext in COMPRESSED_EXTENSIONS:
        return "compressed_archive"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in POINTCLOUD_EXTENSIONS:
        return "pointcloud"
    if ext in ANNOTATION_EXTENSIONS:
        return "annotation_or_table"
    if ext == ".aw5":
        return "unknown_proprietary"
    if ext in {".py", ".md"}:
        return "code_or_documentation"
    if ext == ".ds_store":
        return "system_file"
    return "other"


def is_relative_to(path: Path, parent: Path) -> bool:
    """Compatibility helper for checking whether path is under parent."""

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_not_inside_dataset(output_path: Path, dataset_root: Path) -> None:
    """Prevent report writers from accidentally targeting the raw dataset tree."""

    if is_relative_to(output_path, dataset_root):
        raise ValueError(f"Refusing to write inside dataset root: {output_path}")


def _read_small_text(path: Path, max_bytes: int = 250_000) -> str:
    """Read a small text file defensively for metadata detection."""

    data = path.read_bytes()[:max_bytes]
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _detect_split_evidence(dataset_path: Path, dir_names: Iterable[str], notable_files: list[str]) -> list[str]:
    """Detect train/test/split hints from folder names and README files."""

    evidence: list[str] = []
    matched_dirs = sorted({name for name in dir_names if SPLIT_PATTERN.search(name)})
    if matched_dirs:
        evidence.append("Directory names indicate split structure: " + ", ".join(matched_dirs[:12]))

    for rel_path in notable_files:
        name = Path(rel_path).name.lower()
        if name not in README_NAMES:
            continue
        text = _read_small_text(dataset_path / rel_path)
        compact = " ".join(text.split())
        lower = compact.lower()
        if "training" in lower and "testing" in lower:
            evidence.append("README declares training/testing split.")
        if "for testing we use" in lower:
            evidence.append("README describes subject-level testing split: " + compact[:220])

    if dataset_path.name.lower() == "humanm3":
        if (dataset_path / "train").is_dir() and (dataset_path / "test").is_dir():
            evidence.append("Explicit Human-M3 train/ and test/ directories are present.")

    return list(dict.fromkeys(evidence))


def scan_dataset(dataset_path: Path, dataset_root: Path) -> DatasetInventory:
    """Scan one top-level dataset directory without modifying it."""

    key = dataset_path.name
    meta = KNOWN_DATASETS.get(
        key,
        {
            "display_name": key,
            "purpose": "Unrecognized top-level dataset folder.",
            "tasks": ["manual inspection required"],
        },
    )

    extension_counts: Counter[str] = Counter()
    extension_sizes: defaultdict[str, int] = defaultdict(int)
    category_counts: Counter[str] = Counter()
    archive_files: list[str] = []
    notable_files: list[str] = []
    dir_names: list[str] = []
    directory_count = 0
    file_count = 0
    total_bytes = 0
    warnings: list[str] = []

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        directory_count += len(dirnames)
        dir_names.extend(dirnames)
        base = Path(dirpath)
        for filename in filenames:
            file_path = base / filename
            rel = file_path.relative_to(dataset_path).as_posix()
            suffix = file_path.suffix.lower() or "[none]"
            try:
                size = file_path.stat().st_size
            except OSError:
                warnings.append(f"Could not stat file: {rel}")
                continue

            file_count += 1
            total_bytes += size
            extension_counts[suffix] += 1
            extension_sizes[suffix] += size
            category_counts[classify_extension(suffix)] += 1

            lower_name = filename.lower()
            if suffix in COMPRESSED_EXTENSIONS:
                archive_files.append(rel)
            if lower_name in README_NAMES or suffix in {".xlsx", ".aw5"}:
                notable_files.append(rel)
            elif dataset_path.name == "sportcenter_multiview_dataset" and lower_name in {
                "calibration.json",
                "trajectories.json",
            }:
                notable_files.append(rel)
            elif dataset_path.name == "sportcenter_camerapose_dataset" and lower_name in {
                "ground_grid.json",
                "homography_rectified_template.json",
                "intrinsics_seq_17xxxx.json",
                "intrinsics_seq_98xx.json",
            }:
                notable_files.append(rel)

    extension_summaries = [
        ExtensionSummary(extension=ext, count=extension_counts[ext], total_bytes=extension_sizes[ext])
        for ext in sorted(extension_counts)
    ]

    if key == "5742821":
        aw5_files = [path for path in notable_files if path.lower().endswith(".aw5")]
        if aw5_files:
            warnings.append(
                "AW5 is treated as an unknown proprietary format. Field names and parsing method "
                "must be confirmed before analysis."
            )

    split_evidence = _detect_split_evidence(dataset_path, dir_names, notable_files)

    return DatasetInventory(
        key=key,
        display_name=str(meta["display_name"]),
        path=dataset_path.relative_to(dataset_root.parent).as_posix(),
        purpose=str(meta["purpose"]),
        tasks=list(meta["tasks"]),
        directory_count=directory_count,
        file_count=file_count,
        total_bytes=total_bytes,
        extension_summaries=extension_summaries,
        category_counts=dict(sorted(category_counts.items())),
        split_evidence=split_evidence,
        archive_files=sorted(archive_files),
        notable_files=sorted(notable_files),
        warnings=warnings,
    )


def build_inventory(dataset_root: Path) -> ProjectInventory:
    """Build a read-only inventory for every top-level folder in dataset_root."""

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    datasets = [
        scan_dataset(path, dataset_root)
        for path in sorted(dataset_root.iterdir(), key=lambda p: p.name.lower())
        if path.is_dir()
    ]
    return ProjectInventory(
        dataset_root=str(dataset_root),
        dataset_count=len(datasets),
        total_directories=sum(item.directory_count for item in datasets),
        total_files=sum(item.file_count for item in datasets),
        total_bytes=sum(item.total_bytes for item in datasets),
        datasets=datasets,
    )


def _format_extension_table(dataset: DatasetInventory) -> list[str]:
    lines = ["Extension | Count | Size", "--- | ---: | ---:"]
    for item in sorted(dataset.extension_summaries, key=lambda row: row.count, reverse=True):
        lines.append(f"{item.extension} | {item.count} | {bytes_to_human(item.total_bytes)}")
    return lines


def write_text_report(inventory: ProjectInventory, output_path: Path, dataset_root: Path) -> None:
    """Write a plain-text inventory report outside the dataset directory."""

    ensure_not_inside_dataset(output_path, dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "Basketball fatigue project dataset inventory",
        "=" * 52,
        f"Dataset root: {inventory.dataset_root}",
        "Scan mode: read-only metadata and filesystem traversal",
        f"Top-level datasets: {inventory.dataset_count}",
        f"Total directories: {inventory.total_directories}",
        f"Total files: {inventory.total_files}",
        f"Total size: {bytes_to_human(inventory.total_bytes)}",
        "",
        "Safety",
        "-" * 52,
        "No dataset files are modified by this script.",
        "Reports are written outside dataset/ only.",
        "",
    ]

    for dataset in inventory.datasets:
        lines.extend(
            [
                dataset.display_name,
                "-" * 52,
                f"Key: {dataset.key}",
                f"Path: {dataset.path}",
                f"Purpose: {dataset.purpose}",
                f"Directories: {dataset.directory_count}",
                f"Files: {dataset.file_count}",
                f"Size: {bytes_to_human(dataset.total_bytes)}",
                "Tasks: " + "; ".join(dataset.tasks),
                "Split evidence: "
                + ("; ".join(dataset.split_evidence) if dataset.split_evidence else "No train/test split found."),
                "Archive files: "
                + (", ".join(dataset.archive_files[:20]) if dataset.archive_files else "No compressed archives found."),
                "Notable files: "
                + (", ".join(dataset.notable_files[:30]) if dataset.notable_files else "None detected."),
                "Warnings: " + ("; ".join(dataset.warnings) if dataset.warnings else "None."),
                "",
                "File extensions:",
            ]
        )
        for summary in sorted(dataset.extension_summaries, key=lambda row: row.count, reverse=True):
            lines.append(
                f"  {summary.extension}: {summary.count} files, {bytes_to_human(summary.total_bytes)}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(inventory: ProjectInventory, output_path: Path, dataset_root: Path) -> None:
    """Write a Markdown dataset summary outside the dataset directory."""

    ensure_not_inside_dataset(output_path, dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_probe_path = dataset_root.parent / "outputs" / "metadata" / "schema_probe.json"
    schema_probe = None
    if schema_probe_path.exists():
        try:
            schema_probe = json.loads(schema_probe_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            schema_probe = None

    lines: list[str] = [
        "# Dataset Summary",
        "",
        "This summary is generated by `scripts/inspect_dataset.py` using read-only filesystem traversal.",
        "The script does not modify, extract, convert, or cache any file under `dataset/`.",
        "",
        "## Overview",
        "",
        "| Dataset | Path | Files | Size | Train/Test Split | Main Use |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for dataset in inventory.datasets:
        split = "Yes" if dataset.split_evidence else "Not detected"
        lines.append(
            f"| {dataset.display_name} | `{dataset.path}` | {dataset.file_count} | "
            f"{bytes_to_human(dataset.total_bytes)} | {split} | {dataset.purpose} |"
        )

    if schema_probe:
        lines.extend(
            [
                "",
                "## Schema Probe Highlights",
                "",
                "The table below summarizes field-level findings from `scripts/probe_schema.py`.",
                "",
                "| Dataset | Key Schema Fields | Train/Test Evidence | Fatigue Label | Before/After Label |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for dataset in inventory.datasets:
            schema_item = schema_probe.get("datasets", {}).get(dataset.key, {})
            detected = schema_item.get("detected_schema", {})
            key_fields = ", ".join(list(detected.keys())[:8]) if detected else "Not probed"
            split_notes = schema_item.get("split_info", {}).get("notes", [])
            split = "; ".join(split_notes[:2]) if split_notes else ("Yes" if dataset.split_evidence else "Not detected")
            fatigue = detected.get("fatigue_label", "Not observed")
            before_after = detected.get("before_after_label", "Not observed")
            lines.append(
                f"| {dataset.display_name} | {key_fields} | {split} | {fatigue} | {before_after} |"
            )

    lines.extend(
        [
            "",
            "## Dataset Details",
            "",
        ]
    )

    for dataset in inventory.datasets:
        lines.extend(
            [
                f"### {dataset.display_name}",
                "",
                f"- Raw path: `{dataset.path}`",
                f"- Directories: {dataset.directory_count}",
                f"- Files: {dataset.file_count}",
                f"- Size: {bytes_to_human(dataset.total_bytes)}",
                f"- Purpose: {dataset.purpose}",
                "- Available tasks: " + "; ".join(dataset.tasks),
                "- Train/test split: "
                + ("; ".join(dataset.split_evidence) if dataset.split_evidence else "No split detected yet."),
                "- Compressed archives: "
                + (", ".join(f"`{path}`" for path in dataset.archive_files) if dataset.archive_files else "none detected."),
                "- Notable files: "
                + (", ".join(f"`{path}`" for path in dataset.notable_files[:20]) if dataset.notable_files else "none detected."),
                "- Warnings: " + ("; ".join(dataset.warnings) if dataset.warnings else "none."),
                "",
                "File type summary:",
                "",
                *_format_extension_table(dataset),
                "",
            ]
        )

    lines.extend(
        [
            "## Schema Probe",
            "",
            "Field-level schema validation is generated separately by `scripts/probe_schema.py`.",
            "See `logs/schema_probe.txt`, `docs/schema_mapping.md`, and `docs/feature_table_design.md`.",
            "",
            "## Safety Notes",
            "",
            "- `dataset/` is treated as read-only raw data.",
            "- No extraction, frame sampling, feature computation, model training, or cache creation was performed.",
            "- If video frame extraction is needed later, it should be requested explicitly and written only to `outputs/extracted/`.",
            "- `.AW5` files are unknown proprietary files in this project and are not parsed until the expected reader/schema is confirmed.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
