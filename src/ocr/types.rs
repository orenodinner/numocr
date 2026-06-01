use anyhow::Result;
use egui::Rect;
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct OcrItem {
    pub text: String,
    pub normalized: String,
    pub confidence: f32,
    pub rect_original: Rect,
    pub matched: bool,
    pub match_group: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct OcrOptions {
    pub psm: u8,
    pub scale: u32,
}

pub trait OcrEngine {
    fn recognize(&self, image: &DynamicImage, options: &OcrOptions) -> Result<Vec<OcrItem>>;
}
