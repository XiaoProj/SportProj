"""Create contact sheets for Phase 7.5 journal-ready figures."""

from __future__ import annotations

import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import ensure_not_inside_dataset
from src.visualization.journal_style import apply_journal_style


DATASET_ROOT = PROJECT_ROOT / "dataset"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures" / "journal_ready"
CONTACT_ROOT = PROJECT_ROOT / "outputs" / "figures" / "journal_ready" / "contact_sheets"
LOG_PATH = PROJECT_ROOT / "logs" / "phase7_5_contact_sheets.txt"


def _make_sheet(paths: list[Path], output_base: Path, title: str) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    if not paths:
        return []
    cols = 2
    rows = math.ceil(len(paths) / cols)
    thumb_w, thumb_h = 900, 650
    label_h = 70
    margin = 40
    header_h = 70
    canvas_w = cols * thumb_w + (cols + 1) * margin
    canvas_h = header_h + rows * (thumb_h + label_h + margin) + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("arial.ttf", 34)
        font_label = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    draw.text((margin, 18), title, fill="black", font=font_title)
    for index, path in enumerate(paths):
        row = index // cols
        col = index % cols
        x = margin + col * (thumb_w + margin)
        y = header_h + row * (thumb_h + label_h + margin)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            paste_x = x + (thumb_w - image.width) // 2
            paste_y = y + (thumb_h - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="#D9D9D9", width=2)
        label = path.stem
        if len(label) > 78:
            label = label[:75] + "..."
        draw.text((x, y + thumb_h + 12), label, fill="black", font=font_label)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)
    return [png, pdf]


def main() -> int:
    for path in (CONTACT_ROOT, LOG_PATH):
        ensure_not_inside_dataset(path, DATASET_ROOT)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    pose_paths = sorted((FIGURE_ROOT / "pose_movement_quality").glob("*.png"))
    tabular_paths = sorted((FIGURE_ROOT / "tabular_fatigue_sleep_performance").glob("*.png"))
    outputs = []
    outputs.extend(_make_sheet(pose_paths, CONTACT_ROOT / "phase7_5_pose_movement_quality_contact_sheet", "Pose Movement Quality Figures"))
    outputs.extend(_make_sheet(tabular_paths, CONTACT_ROOT / "phase7_5_tabular_fatigue_sleep_performance_contact_sheet", "Tabular Fatigue/Sleep/Performance Figures"))
    LOG_PATH.write_text(
        "Phase 7.5 contact sheets generated.\n" + "\n".join(path.as_posix() for path in outputs) + "\n",
        encoding="utf-8",
    )
    print("Figure contact sheets generated.")
    for path in outputs:
        print(f"Wrote: {path}")
    print(f"Wrote: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
