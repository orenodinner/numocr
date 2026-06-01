#!/usr/bin/env python3
"""
Prepare bbox crops for fine-tuning a lightweight digit recognizer.

The output follows the Hugging Face ImageFolder convention:

training_data/catia_ocr_recognition/
  train/metadata.csv
  train/images/000000.png
  validation/metadata.csv
  validation/images/000960.png
  test/metadata.csv
  test/images/001080.png

Each metadata row has `file_name`, `text`, and `normalized_digits`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "training_data" / "catia_2d_digits"
DEFAULT_OUTPUT = REPO_ROOT / "training_data" / "catia_ocr_recognition"


def normalize_split(split: str) -> str:
    return "validation" if split == "val" else split


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def prepare_dataset(args: argparse.Namespace) -> None:
    if args.clean:
        clean_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = args.source_dir / "labels.csv"
    rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}

    for row in rows:
        source_path = args.source_dir / row["file"]
        split = normalize_split(row["split"])
        image_dir = args.output_dir / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as image:
            image = image.convert("L")
            left = int(row["bbox_left"])
            top = int(row["bbox_top"])
            width = int(row["bbox_width"])
            height = int(row["bbox_height"])
            box = (
                max(0, left - args.pad),
                max(0, top - args.pad),
                min(image.width, left + width + args.pad),
                min(image.height, top + height + args.pad),
            )
            crop = image.crop(box)
            crop = ImageOps.autocontrast(crop)
            if args.height > 0:
                scale = args.height / crop.height
                resized_width = max(8, int(round(crop.width * scale)))
                crop = crop.resize((resized_width, args.height), Image.Resampling.BICUBIC)
            if args.max_width > 0 and crop.width > args.max_width:
                crop.thumbnail((args.max_width, args.height if args.height > 0 else crop.height), Image.Resampling.BICUBIC)

            filename = f"{Path(row['file']).stem}.png"
            output_path = image_dir / filename
            crop.save(output_path)

        split_rows[split].append(
            {
                "file_name": f"images/{filename}",
                "text": row["label"],
                "normalized_digits": row["normalized_digits"],
                "template": row["template"],
                "label_pattern": row["label_pattern"],
                "source_file": row["file"],
            }
        )

    for split, records in split_rows.items():
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "file_name",
                    "text",
                    "normalized_digits",
                    "template",
                    "label_pattern",
                    "source_file",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

    metadata = {
        "source_dir": str(args.source_dir),
        "count": len(rows),
        "pad": args.pad,
        "height": args.height,
        "max_width": args.max_width,
        "splits": {split: len(records) for split, records in split_rows.items()},
        "hf_load_example": "load_dataset('imagefolder', data_dir='training_data/catia_ocr_recognition')",
        "note": "Use `text` for sequence recognizer labels; `normalized_digits` is useful for digit-only evaluation.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OCR recognizer crop dataset.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pad", type=int, default=8)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--max-width", type=int, default=256)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_dataset(parse_args())
