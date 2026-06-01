use anyhow::{anyhow, Context, Result};
use egui::{pos2, Rect};
use image::{imageops, DynamicImage, GrayImage, Luma};
use tempfile::Builder;

use crate::image_proc::preprocess::preprocess_for_ocr;
use crate::image_proc::roi::{detect_digit_rois, RoiCandidate};
use crate::search::digit_sequence::normalize_digits;

use super::tesseract_cmd::new_tesseract_command;
use super::{OcrEngine, OcrItem, OcrOptions};

pub struct RoiTesseractCliOcrEngine;

#[derive(Debug, Clone)]
struct AtlasEntry {
    original_rect: Rect,
    atlas_rect: Rect,
    scale: f32,
}

impl OcrEngine for RoiTesseractCliOcrEngine {
    fn recognize(&self, image: &DynamicImage, options: &OcrOptions) -> Result<Vec<OcrItem>> {
        let rois = detect_digit_rois(image);
        if rois.is_empty() {
            return Ok(Vec::new());
        }

        let (atlas, entries) = build_roi_atlas(image, &rois, options.scale)?;
        let temp_file = Builder::new()
            .prefix("digit-ocr-roi-atlas-")
            .suffix(".png")
            .tempfile()
            .context("failed to create temporary ROI OCR atlas")?;

        atlas
            .save(temp_file.path())
            .context("failed to save temporary ROI OCR atlas")?;

        let whitelist = "tessedit_char_whitelist=0123456789.,:-/¥￥";
        let output = new_tesseract_command()
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
        parse_roi_tesseract_tsv(&stdout, &entries)
    }
}

fn build_roi_atlas(
    image: &DynamicImage,
    rois: &[RoiCandidate],
    scale: u32,
) -> Result<(GrayImage, Vec<AtlasEntry>)> {
    let scale = scale.max(1);
    let margin = 14u32;
    let max_width = rois
        .iter()
        .map(|roi| ((roi.rect_original.width().ceil() as u32) * scale) + margin * 2)
        .max()
        .unwrap_or(1)
        .max(64);
    let total_height = rois
        .iter()
        .map(|roi| ((roi.rect_original.height().ceil() as u32) * scale) + margin * 2)
        .sum::<u32>()
        .max(64);

    let mut atlas = GrayImage::from_pixel(max_width, total_height, Luma([255]));
    let mut entries = Vec::new();
    let mut cursor_y = margin;

    for roi in rois {
        let source_rect = clamp_rect_to_image(roi.rect_original, image.width(), image.height());
        let x = source_rect.left().floor() as u32;
        let y = source_rect.top().floor() as u32;
        let width = source_rect.width().ceil().max(1.0) as u32;
        let height = source_rect.height().ceil().max(1.0) as u32;
        let crop = image.crop_imm(x, y, width, height);
        let processed = preprocess_for_ocr(&crop, scale).to_luma8();
        imageops::overlay(&mut atlas, &processed, margin.into(), cursor_y.into());

        let atlas_rect = Rect::from_min_size(
            pos2(margin as f32, cursor_y as f32),
            egui::vec2(processed.width() as f32, processed.height() as f32),
        );
        entries.push(AtlasEntry {
            original_rect: source_rect,
            atlas_rect,
            scale: scale as f32,
        });
        cursor_y += processed.height() + margin * 2;
    }

    Ok((atlas, entries))
}

fn parse_roi_tesseract_tsv(tsv: &str, entries: &[AtlasEntry]) -> Result<Vec<OcrItem>> {
    let mut lines = tsv.lines();
    let header = lines.next().context("tesseract TSV output was empty")?;
    let columns: Vec<&str> = header.split('\t').collect();

    let text_col = column_index(&columns, "text")?;
    let conf_col = column_index(&columns, "conf")?;
    let left_col = column_index(&columns, "left")?;
    let top_col = column_index(&columns, "top")?;
    let width_col = column_index(&columns, "width")?;
    let height_col = column_index(&columns, "height")?;

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
        let left = parse_f32(fields.get(left_col).copied()).unwrap_or(0.0);
        let top = parse_f32(fields.get(top_col).copied()).unwrap_or(0.0);
        let width = parse_f32(fields.get(width_col).copied()).unwrap_or(0.0);
        let height = parse_f32(fields.get(height_col).copied()).unwrap_or(0.0);
        if width <= 0.0 || height <= 0.0 {
            continue;
        }

        let atlas_rect = Rect::from_min_size(pos2(left, top), egui::vec2(width, height));
        let Some(entry) = entries.iter().find(|entry| {
            let center = atlas_rect.center();
            center.y >= entry.atlas_rect.top() && center.y <= entry.atlas_rect.bottom()
        }) else {
            continue;
        };

        let original_min = pos2(
            entry.original_rect.left()
                + (atlas_rect.left() - entry.atlas_rect.left()) / entry.scale,
            entry.original_rect.top() + (atlas_rect.top() - entry.atlas_rect.top()) / entry.scale,
        );
        let original_max = pos2(
            entry.original_rect.left()
                + (atlas_rect.right() - entry.atlas_rect.left()) / entry.scale,
            entry.original_rect.top()
                + (atlas_rect.bottom() - entry.atlas_rect.top()) / entry.scale,
        );

        items.push(OcrItem {
            text: text.to_owned(),
            normalized,
            confidence,
            rect_original: Rect::from_min_max(original_min, original_max),
            matched: false,
            match_group: None,
        });
    }

    Ok(items)
}

fn clamp_rect_to_image(rect: Rect, image_width: u32, image_height: u32) -> Rect {
    Rect::from_min_max(
        pos2(rect.left().floor().max(0.0), rect.top().floor().max(0.0)),
        pos2(
            rect.right().ceil().min(image_width as f32),
            rect.bottom().ceil().min(image_height as f32),
        ),
    )
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
