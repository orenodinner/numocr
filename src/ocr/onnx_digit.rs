use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use egui::Rect;
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

use crate::image_proc::roi::detect_digit_rois;

use super::{OcrEngine, OcrItem, OcrOptions};

const CHARSET: &[u8] = b"0123456789";
const MODEL_HEIGHT: u32 = 48;
const MAX_MODEL_WIDTH: u32 = 256;

pub struct OnnxDigitOcrEngine {
    session: Mutex<Session>,
}

impl OnnxDigitOcrEngine {
    pub fn from_default_model() -> Result<Self> {
        Self::from_model_path(default_model_path()?)
    }

    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(anyhow!("ONNX model not found: {}", path.display()));
        }

        let session = Session::builder()
            .map_err(|err| anyhow!("failed to create ONNX session builder: {err}"))?
            .with_intra_threads(2)
            .map_err(|err| anyhow!("failed to set ONNX thread count: {err}"))?
            .commit_from_file(path)
            .map_err(|err| anyhow!("failed to commit ONNX session: {err}"))?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    fn recognize_crop(&self, crop: &DynamicImage) -> Result<String> {
        let tensor = crop_to_tensor(crop);
        let input = TensorRef::from_array_view(&tensor)
            .map_err(|err| anyhow!("failed to create ONNX input tensor: {err}"))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("ONNX session lock was poisoned"))?;
        let outputs = session
            .run(inputs![input])
            .map_err(|err| anyhow!("ONNX inference failed: {err}"))?;
        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|err| anyhow!("failed to extract ONNX logits: {err}"))?;
        decode_logits(&shape[..], logits)
    }
}

impl OcrEngine for OnnxDigitOcrEngine {
    fn recognize(&self, image: &DynamicImage, _options: &OcrOptions) -> Result<Vec<OcrItem>> {
        let rois = detect_digit_rois(image);
        let mut items = Vec::new();

        for roi in rois {
            let rect = clamp_rect(roi.rect_original, image.width(), image.height());
            let crop = crop_rect(image, rect);
            let text = self.recognize_crop(&crop)?;
            if text.is_empty() {
                continue;
            }

            items.push(OcrItem {
                normalized: text.clone(),
                text,
                confidence: 100.0,
                rect_original: rect,
                matched: false,
                match_group: None,
            });
        }

        Ok(items)
    }
}

fn default_model_path() -> Result<PathBuf> {
    let relative = PathBuf::from("models")
        .join("catia_crnn_digit_ocr_digits")
        .join("model.onnx");
    if relative.exists() {
        return Ok(relative);
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let sibling = parent
                .join("models")
                .join("catia_crnn_digit_ocr_digits")
                .join("model.onnx");
            if sibling.exists() {
                return Ok(sibling);
            }
        }
    }

    Ok(relative)
}

fn crop_rect(image: &DynamicImage, rect: Rect) -> DynamicImage {
    image.crop_imm(
        rect.left().floor().max(0.0) as u32,
        rect.top().floor().max(0.0) as u32,
        rect.width().ceil().max(1.0) as u32,
        rect.height().ceil().max(1.0) as u32,
    )
}

fn clamp_rect(rect: Rect, image_width: u32, image_height: u32) -> Rect {
    Rect::from_min_max(
        egui::pos2(rect.left().max(0.0), rect.top().max(0.0)),
        egui::pos2(
            rect.right().min(image_width as f32),
            rect.bottom().min(image_height as f32),
        ),
    )
}

fn crop_to_tensor(crop: &DynamicImage) -> Array4<f32> {
    let mut gray = crop.to_luma8();
    autocontrast(&mut gray);
    let scale = MODEL_HEIGHT as f32 / gray.height().max(1) as f32;
    let mut width = ((gray.width() as f32 * scale).round() as u32).max(8);
    if width > MAX_MODEL_WIDTH {
        width = MAX_MODEL_WIDTH;
    }
    let resized = image::imageops::resize(
        &gray,
        width,
        MODEL_HEIGHT,
        image::imageops::FilterType::CatmullRom,
    );

    let mut tensor = Array4::<f32>::zeros((1, 1, MODEL_HEIGHT as usize, width as usize));
    for (x, y, pixel) in resized.enumerate_pixels() {
        tensor[[0, 0, y as usize, x as usize]] = (255.0 - pixel[0] as f32) / 255.0;
    }
    tensor
}

fn autocontrast(image: &mut GrayImage) {
    let mut min = u8::MAX;
    let mut max = u8::MIN;
    for pixel in image.pixels() {
        min = min.min(pixel[0]);
        max = max.max(pixel[0]);
    }
    if max <= min {
        return;
    }
    let range = (max - min) as f32;
    for pixel in image.pixels_mut() {
        let stretched = ((pixel[0].saturating_sub(min)) as f32 * 255.0 / range).round() as u8;
        *pixel = Luma([stretched]);
    }
}

fn decode_logits(shape: &[i64], logits: &[f32]) -> Result<String> {
    if shape.len() != 3 {
        return Err(anyhow!("unexpected ONNX logits shape: {shape:?}"));
    }
    let time = shape[0] as usize;
    let batch = shape[1] as usize;
    let classes = shape[2] as usize;
    if batch != 1 || classes != CHARSET.len() + 1 {
        return Err(anyhow!("unexpected ONNX logits shape: {shape:?}"));
    }

    let mut output = String::new();
    let mut previous = 0usize;
    for t in 0..time {
        let offset = t * batch * classes;
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for class in 0..classes {
            let score = logits[offset + class];
            if score > best_score {
                best = class;
                best_score = score;
            }
        }
        if best != 0 && best != previous {
            output.push(CHARSET[best - 1] as char);
        }
        previous = best;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::decode_logits;

    #[test]
    fn decodes_ctc_logits() {
        let shape = [4, 1, 11];
        let mut logits = vec![-10.0f32; 4 * 11];
        logits[2] = 4.0;
        logits[11 + 2] = 5.0;
        logits[22] = 6.0;
        logits[33 + 3] = 7.0;

        assert_eq!(decode_logits(&shape, &logits).unwrap(), "12");
    }
}
