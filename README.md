# Digit OCR Viewer

Rust desktop app for searching digit sequences inside one image. The MVP uses the Tesseract CLI and keeps the OCR boundary behind a trait so a future ONNX digit OCR engine can replace it.

## Features

- Open PNG, JPG/JPEG, WEBP, and BMP files
- Display and zoom the image
- Run Tesseract OCR with selectable PSM 6, 7, or 11
- Preprocess OCR input with grayscale conversion and 1x to 4x scaling
- Search digit sequences such as `235` or `20260601`
- Match split OCR output such as `2 3 5` when searching `235`
- Highlight all matches in red and the selected match in yellow
- Navigate multiple matches with Previous and Next
- Show OCR text, confidence, and match group in the right panel

## Windows Setup

1. Install Rust:

```powershell
winget install Rustlang.Rustup
```

2. Install Tesseract OCR and make sure `tesseract.exe` is on `PATH`.

```powershell
tesseract --version
```

If the command is not found, add the Tesseract install directory to `PATH` and restart the terminal or Codex app.

3. Build and run:

```powershell
cd A:\numocr
cargo run
```

## Usage

1. Click `Open Image`.
2. Enter digits in the `Digits` search box.
3. Choose OCR scale and PSM if needed.
4. Click `OCR Search`.
5. Use `Previous` and `Next` to switch between matched groups.

Changing only the search text reuses the existing OCR result. Opening another image clears OCR results.

## Architecture

```text
src/
  main.rs
  app.rs
  ocr/
    mod.rs
    tesseract_cli.rs
    types.rs
  search/
    mod.rs
    digit_sequence.rs
  image_proc/
    mod.rs
    preprocess.rs
```

`ocr::types::OcrEngine` is the replacement point for a future ONNX engine:

```rust
pub trait OcrEngine {
    fn recognize(&self, image: &DynamicImage, options: &OcrOptions) -> anyhow::Result<Vec<OcrItem>>;
}
```

The current implementation is `TesseractCliOcrEngine`.

## Synthetic Digit Training Data

Generate OCR training images for a future ONNX recognizer:

```powershell
python tools\generate_digit_dataset.py --count 1600
```

For CATIA/CTIS-specific data, put licensed `.ttf` or `.otf` files in `training_data\fonts` first, or pass `--font-path`:

```powershell
python tools\generate_digit_dataset.py --font-path "C:\path\to\CTIS.ttf" --count 1600
```

If no custom font is available, the generator uses local Windows fonts as surrogate data and records that in `training_data\ctis_digits\metadata.json`.
