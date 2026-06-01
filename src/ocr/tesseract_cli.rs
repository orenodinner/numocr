use std::process::Command;

use anyhow::{anyhow, Context, Result};
use egui::{pos2, Rect};
use image::DynamicImage;
use tempfile::Builder;

use crate::image_proc::preprocess::preprocess_for_ocr;
use crate::search::digit_sequence::normalize_digits;

use super::{OcrEngine, OcrItem, OcrOptions};

pub struct TesseractCliOcrEngine;

impl OcrEngine for TesseractCliOcrEngine {
    fn recognize(&self, image: &DynamicImage, options: &OcrOptions) -> Result<Vec<OcrItem>> {
        let processed = preprocess_for_ocr(image, options.scale);
        let temp_file = Builder::new()
            .prefix("digit-ocr-viewer-")
            .suffix(".png")
            .tempfile()
            .context("failed to create temporary OCR image")?;

        processed
            .save(temp_file.path())
            .context("failed to save temporary OCR image")?;

        let whitelist = "tessedit_char_whitelist=0123456789.,:-/¥￥";
        let output = Command::new("tesseract")
            .arg(temp_file.path())
            .arg("stdout")
            .arg("-l")
            .arg("eng")
            .arg("--psm")
            .arg(options.psm.to_string())
            .arg("-c")
            .arg(whitelist)
            .arg("tsv")
            .output()
            .context(
                "failed to run tesseract. Check that tesseract.exe is installed and on PATH",
            )?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("tesseract failed: {}", stderr.trim()));
        }

        let stdout = String::from_utf8(output.stdout).context("tesseract output was not UTF-8")?;
        parse_tesseract_tsv(&stdout, options.scale)
    }
}

fn parse_tesseract_tsv(tsv: &str, scale: u32) -> Result<Vec<OcrItem>> {
    let mut lines = tsv.lines();
    let header = lines.next().context("tesseract TSV output was empty")?;
    let columns: Vec<&str> = header.split('\t').collect();

    let text_col = column_index(&columns, "text")?;
    let conf_col = column_index(&columns, "conf")?;
    let left_col = column_index(&columns, "left")?;
    let top_col = column_index(&columns, "top")?;
    let width_col = column_index(&columns, "width")?;
    let height_col = column_index(&columns, "height")?;

    let scale = scale.max(1) as f32;
    let mut items = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        let Some(text) = fields.get(text_col).map(|s| s.trim()) else {
            continue;
        };

        if text.is_empty() {
            continue;
        }

        let normalized = normalize_digits(text);
        if normalized.is_empty() {
            continue;
        }

        let confidence = parse_f32(fields.get(conf_col).copied()).unwrap_or(-1.0);
        let left = parse_f32(fields.get(left_col).copied()).unwrap_or(0.0) / scale;
        let top = parse_f32(fields.get(top_col).copied()).unwrap_or(0.0) / scale;
        let width = parse_f32(fields.get(width_col).copied()).unwrap_or(0.0) / scale;
        let height = parse_f32(fields.get(height_col).copied()).unwrap_or(0.0) / scale;

        if width <= 0.0 || height <= 0.0 {
            continue;
        }

        items.push(OcrItem {
            text: text.to_owned(),
            normalized,
            confidence,
            rect_original: Rect::from_min_size(pos2(left, top), egui::vec2(width, height)),
            matched: false,
            match_group: None,
        });
    }

    Ok(items)
}

fn column_index(columns: &[&str], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|column| *column == name)
        .with_context(|| format!("missing TSV column: {name}"))
}

fn parse_f32(value: Option<&str>) -> Option<f32> {
    value?.trim().parse::<f32>().ok()
}
