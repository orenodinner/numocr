#!/usr/bin/env python3
"""
Generate synthetic digit OCR training data.

Put licensed CATIA/CTIS TTF or OTF files in training_data/fonts, or pass
--font-path explicitly. If no custom font is available, the script uses common
Windows fonts as a fallback and marks the dataset as surrogate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_DIR = REPO_ROOT / "training_data" / "fonts"
DEFAULT_OUT_DIR = REPO_ROOT / "training_data" / "ctis_digits"
CHARS = "0123456789.,:-/¥￥"


@dataclass(frozen=True)
class FontSpec:
    path: Path
    source: str


def find_fonts(custom_font_paths: list[str], font_dir: Path) -> tuple[list[FontSpec], bool]:
    fonts: list[FontSpec] = []

    for raw_path in custom_font_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.exists() and path.suffix.lower() in {".ttf", ".otf"}:
            fonts.append(FontSpec(path=path, source="custom"))

    if font_dir.exists():
        for path in sorted(font_dir.glob("*")):
            if path.suffix.lower() in {".ttf", ".otf"}:
                fonts.append(FontSpec(path=path.resolve(), source="training_data/fonts"))

    if fonts:
        return fonts, False

    fallback_names = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/courbd.ttf",
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    fallback = [
        FontSpec(path=Path(name), source="windows-surrogate")
        for name in fallback_names
        if Path(name).exists()
    ]
    if not fallback:
        raise SystemExit("No usable .ttf/.otf font found. Put a licensed font in training_data/fonts.")
    return fallback, True


def random_label(rng: random.Random) -> str:
    pattern = rng.choice(
        [
            "digits",
            "date_slash",
            "date_compact",
            "time",
            "hyphen",
            "comma",
            "decimal",
            "yen",
            "short_code",
        ]
    )

    if pattern == "digits":
        length = rng.randint(1, 10)
        return "".join(rng.choice(string.digits) for _ in range(length))
    if pattern == "date_slash":
        return f"{rng.randint(2020, 2035):04d}/{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}"
    if pattern == "date_compact":
        return f"{rng.randint(2020, 2035):04d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    if pattern == "time":
        return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if pattern == "hyphen":
        return f"{rng.randint(1, 999):03d}-{rng.randint(1, 999):03d}"
    if pattern == "comma":
        return f"{rng.randint(1, 999):,},{rng.randint(0, 999):03d}"
    if pattern == "decimal":
        return f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}"
    if pattern == "yen":
        symbol = rng.choice(["¥", "￥"])
        return f"{symbol}{rng.randint(1, 999999):,}"
    return f"{rng.randint(0, 99):02d}-{rng.randint(0, 9999):04d}"


def split_for_index(index: int, total: int) -> str:
    train_cutoff = int(total * 0.8)
    val_cutoff = int(total * 0.9)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def render_sample(label: str, font_spec: FontSpec, rng: random.Random, canvas_size: tuple[int, int]) -> Image.Image:
    width, height = canvas_size
    bg = rng.randint(225, 255)
    fg = rng.randint(0, 55)
    image = Image.new("L", canvas_size, bg)
    draw = ImageDraw.Draw(image)

    font_size = rng.randint(24, 46)
    font = ImageFont.truetype(str(font_spec.path), font_size)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(2, (width - text_w) // 2 + rng.randint(-14, 14))
    y = max(2, (height - text_h) // 2 + rng.randint(-8, 8))
    draw.text((x, y), label, font=font, fill=fg)

    if rng.random() < 0.45:
        angle = rng.uniform(-2.5, 2.5)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=bg)
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    if rng.random() < 0.60:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.45))
    if rng.random() < 0.35:
        image = add_noise(image, rng, amount=rng.randint(3, 18))

    return image


def add_noise(image: Image.Image, rng: random.Random, amount: int) -> Image.Image:
    pixels = image.load()
    width, height = image.size
    for _ in range(width * height // amount):
        x = rng.randrange(width)
        y = rng.randrange(height)
        delta = rng.randint(-28, 28)
        pixels[x, y] = max(0, min(255, pixels[x, y] + delta))
    return image


def write_charset(out_dir: Path) -> None:
    (out_dir / "charset.txt").write_text("\n".join(["<blank>", *CHARS]) + "\n", encoding="utf-8")


def generate_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    fonts, is_surrogate = find_fonts(args.font_path, args.font_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_charset(args.output_dir)

    manifest_path = args.output_dir / "labels.csv"
    rows: list[dict[str, str]] = []

    for index in range(args.count):
        split = split_for_index(index, args.count)
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        label = random_label(rng)
        font = rng.choice(fonts)
        image = render_sample(label, font, rng, (args.width, args.height))
        filename = f"{split}/{index:06d}.png"
        image.save(args.output_dir / filename)
        rows.append(
            {
                "file": filename,
                "label": label,
                "normalized_digits": "".join(ch for ch in label if ch.isdigit()),
                "font": str(font.path),
                "font_source": font.source,
                "split": split,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["file", "label", "normalized_digits", "font", "font_source", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "count": args.count,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "charset": CHARS,
        "surrogate_fonts": is_surrogate,
        "fonts": [{"path": str(font.path), "source": font.source} for font in fonts],
        "note": "Use licensed CATIA/CTIS fonts in training_data/fonts for target-specific data.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic digit OCR training data.")
    parser.add_argument("--font-path", action="append", default=[], help="Licensed TTF/OTF path. Can be repeated.")
    parser.add_argument("--font-dir", type=Path, default=DEFAULT_FONT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=1600)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
