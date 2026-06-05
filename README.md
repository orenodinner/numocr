# Digit OCR Viewer

Rust desktop app for searching digit sequences inside one image. The default path uses a lightweight ONNX Runtime CRNN digit recognizer over detected regions of interest, with Tesseract CLI engines kept as fallbacks.

## Features

- Open PNG, JPG/JPEG, WEBP, and BMP files
- Display and zoom the image
- Run ONNX ROI OCR, Tesseract ROI OCR, or whole-image Tesseract OCR
- Selectable Tesseract PSM 6, 7, or 11 for Tesseract modes
- ROI mode that extracts likely digit regions before recognition
- Downscale large input images to fit a 1920x1080 OCR canvas, then map result boxes back to the original image
- Preprocess OCR input with grayscale conversion and 1x to 4x scaling
- Search digit sequences such as `235` or `20260601`
- Match split OCR output such as `2 3 5` when searching `235`
- Highlight all matches in red and the selected match in yellow
- Navigate multiple matches with Previous and Next
- Show OCR text, confidence, and match group in the right panel

## Windows Setup

For the GitHub release package, download and run:

```powershell
.\numocr-windows-x64-installer.exe
```

This installs the app under `%LOCALAPPDATA%\NumOCR` with the bundled ONNX model.

1. Install Rust:

```powershell
winget install Rustlang.Rustup
```

2. Optional: install Tesseract OCR for the Tesseract fallback engines.

```powershell
tesseract --version
```

The ONNX ROI engine does not require Tesseract. If `tesseract` is not on `PATH`, the app also checks `TESSERACT_PATH` and common Windows install locations such as `C:\Program Files\Tesseract-OCR\tesseract.exe`.

3. Build and run:

```powershell
cd A:\numocr
cargo run
```

## Usage

1. Click `Open Image`.
2. Enter digits in the `Digits` search box.
3. Choose OCR scale and PSM if needed.
4. Choose `ONNX ROI` for the built-in recognizer, or a Tesseract fallback engine.
5. Click `OCR Search`.
6. Use `Previous` and `Next` to switch between matched groups.

Changing only the search text reuses the existing OCR result. Opening another image clears OCR results.

## Architecture

```text
src/
  main.rs
  app.rs
  ocr/
    mod.rs
    onnx_digit.rs
    roi_tesseract_cli.rs
    tesseract_cmd.rs
    tesseract_cli.rs
    types.rs
  search/
    mod.rs
    digit_sequence.rs
  image_proc/
    mod.rs
    preprocess.rs
```

`ocr::types::OcrEngine` keeps each recognizer behind the same interface:

```rust
pub trait OcrEngine {
    fn recognize(&self, image: &DynamicImage, options: &OcrOptions) -> anyhow::Result<Vec<OcrItem>>;
}
```

The current implementations are:

- `TesseractCliOcrEngine`: whole-image Tesseract OCR
- `RoiTesseractCliOcrEngine`: CPU ROI extraction, one atlas image, then Tesseract OCR
- `OnnxDigitOcrEngine`: CPU ROI extraction, then lightweight CRNN/CTC ONNX recognition

On Windows, Tesseract fallback engines first use `TESSERACT_PATH` when set, then check common install locations, then fall back to `tesseract` on `PATH`.

## Synthetic Digit Training Data

Generate OCR training images for recognizer experiments:

```powershell
python tools\generate_digit_dataset.py --count 1600
```

For CATIA/CTIS-specific data, put licensed `.ttf` or `.otf` files in `training_data\fonts` first, or pass `--font-path`:

```powershell
python tools\generate_digit_dataset.py --font-path "C:\path\to\CTIS.ttf" --count 1600
```

If no custom font is available, the generator uses local Windows fonts as surrogate data and records that in `training_data\ctis_digits\metadata.json`.

## CATIA-like 2D Drawing Training Data

Generate deterministic CATIA-like 2D drawing patches with dimension lines, leaders, title blocks, table cells, centerlines, light grid lines, and scan-style noise:

```powershell
python tools\generate_catia_2d_dataset.py --count 1200 --clean
```

The default output is `training_data\catia_2d_digits`:

- `train`, `val`, `test`: generated PNG samples
- `labels.csv`: file path, label, normalized digit string, template, bbox, font, and split
- `metadata.json`: generation settings, templates, font sources, and charset
- `preview_grid.png`: quick visual contact sheet
- `charset.txt`: `<blank>` plus supported OCR characters

The current generated dataset contains 1,200 images split as 960 train, 120 val, and 120 test. It uses Windows surrogate fonts unless licensed CATIA/CTIS-like fonts are placed in `training_data\fonts` or passed with `--font-path`.

## CATIA OCR Evaluation

Evaluate the current Rust OCR engines against the generated CATIA-like dataset:

```powershell
cargo run --bin evaluate_catia -- --engine full --psm 11 --scale 2 --output training_data\catia_2d_digits\tesseract_current_default_eval.json
cargo run --bin evaluate_catia -- --engine roi --psm 6 --scale 1 --output training_data\catia_2d_digits\tesseract_roi_eval.json
cargo run --bin evaluate_catia -- --engine onnx --output training_data\catia_2d_digits\onnx_roi_eval.json
```

Current measured results on the generated 1,200-image dataset:

- Whole-image Tesseract, PSM 11, 2x: 420 / 1200 digit-search hits, about 35.0%
- ROI Tesseract, PSM 6, 1x: 485 / 1200 digit-search hits, about 40.4%, average 135.7 ms/image
- ONNX ROI recognizer: 899 / 1200 digit-search hits, about 74.9%, average 5.1 ms/image in release build

Tesseract remains below practical accuracy for dense 2D drawing patches. The ONNX ROI path is the practical default on CPU; remaining misses are mostly ROI segmentation and line-ordering cases rather than raw crop recognition speed.

## Fine-tuning and ONNX Recognizer

Prepare crop-level data in Hugging Face ImageFolder format:

```powershell
python tools\prepare_hf_recognition_dataset.py --clean
```

Output:

- `training_data\catia_ocr_recognition\train`
- `training_data\catia_ocr_recognition\validation`
- `training_data\catia_ocr_recognition\test`

Train the lightweight CRNN+CTC digit recognizer and export ONNX:

```powershell
python tools\train_crnn_digit_ocr.py --epochs 20 --batch-size 64 --output-dir models\catia_crnn_digit_ocr_digits
```

Current measured recognizer result:

- `models\catia_crnn_digit_ocr_digits\model.pt`: trained PyTorch checkpoint
- `models\catia_crnn_digit_ocr_digits\model.onnx`: ONNX export
- `models\catia_crnn_digit_ocr_digits\metadata.json`: training history
- `models\catia_crnn_digit_ocr_digits\onnx_cpu_eval.json`: ONNX Runtime CPU evaluation
- Test accuracy: 116 / 120 exact digit sequences, about 96.7%
- ONNX Runtime CPU speed: about 1.7 ms/crop on this machine

This proves the recognizer side is fast enough for the target CPU class. The Rust app loads `models\catia_crnn_digit_ocr_digits\model.onnx` by default, either relative to the working directory or next to the installed executable.
