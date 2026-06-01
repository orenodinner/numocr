use std::path::PathBuf;

use eframe::egui;
use egui::{Color32, ColorImage, Stroke, TextureHandle};
use image::DynamicImage;

use crate::ocr::{
    OcrEngine, OcrItem, OcrOptions, OnnxDigitOcrEngine, RoiTesseractCliOcrEngine,
    TesseractCliOcrEngine,
};
use crate::search::digit_sequence::apply_digit_sequence_search;

pub struct DigitOcrViewerApp {
    image_path: Option<PathBuf>,
    image: Option<DynamicImage>,
    texture: Option<TextureHandle>,
    search_text: String,
    status: String,
    ocr_items: Vec<OcrItem>,
    matched_indices: Vec<usize>,
    current_match: usize,
    zoom: f32,
    ocr_scale: u32,
    psm: u8,
    engine_mode: OcrEngineMode,
    onnx_engine: Option<OnnxDigitOcrEngine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OcrEngineMode {
    OnnxRoi,
    TesseractRoi,
    TesseractFull,
}

impl OcrEngineMode {
    fn label(self) -> &'static str {
        match self {
            Self::OnnxRoi => "ONNX ROI",
            Self::TesseractRoi => "Tesseract ROI",
            Self::TesseractFull => "Tesseract full",
        }
    }
}

impl DigitOcrViewerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            image_path: None,
            image: None,
            texture: None,
            search_text: String::new(),
            status: "Open an image to start.".to_owned(),
            ocr_items: Vec::new(),
            matched_indices: Vec::new(),
            current_match: 0,
            zoom: 1.0,
            ocr_scale: 1,
            psm: 6,
            engine_mode: OcrEngineMode::OnnxRoi,
            onnx_engine: None,
        }
    }

    fn open_image(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
            .pick_file()
        else {
            return;
        };

        match image::open(&path) {
            Ok(image) => match dynamic_image_to_color_image(&image) {
                Ok(color_image) => {
                    self.texture = Some(ctx.load_texture(
                        path.display().to_string(),
                        color_image,
                        egui::TextureOptions::LINEAR,
                    ));
                    self.image_path = Some(path);
                    self.image = Some(image);
                    self.ocr_items.clear();
                    self.matched_indices.clear();
                    self.current_match = 0;
                    self.status = "Image loaded. Run OCR search.".to_owned();
                }
                Err(err) => {
                    self.status = format!("Failed to prepare image texture: {err:#}");
                }
            },
            Err(err) => {
                self.status = format!("Failed to open image: {err:#}");
            }
        }
    }

    fn run_ocr_search(&mut self) {
        let Some(image) = &self.image else {
            self.status = "Open an image first.".to_owned();
            return;
        };

        if self.search_text.chars().all(|ch| !ch.is_ascii_digit()) {
            self.status = "Enter digits to search.".to_owned();
            return;
        }

        self.status = format!("Running {} OCR...", self.engine_mode.label());
        let options = OcrOptions {
            psm: self.psm,
            scale: self.ocr_scale,
            use_roi: matches!(
                self.engine_mode,
                OcrEngineMode::OnnxRoi | OcrEngineMode::TesseractRoi
            ),
        };

        let full_engine = TesseractCliOcrEngine;
        let roi_engine = RoiTesseractCliOcrEngine;
        let result = match self.engine_mode {
            OcrEngineMode::OnnxRoi => {
                if self.onnx_engine.is_none() {
                    match OnnxDigitOcrEngine::from_default_model() {
                        Ok(engine) => self.onnx_engine = Some(engine),
                        Err(err) => {
                            self.status = format!("{err:#}");
                            return;
                        }
                    }
                }
                self.onnx_engine
                    .as_ref()
                    .expect("ONNX engine initialized")
                    .recognize(image, &options)
            }
            OcrEngineMode::TesseractRoi => roi_engine.recognize(image, &options),
            OcrEngineMode::TesseractFull => full_engine.recognize(image, &options),
        };

        match result {
            Ok(mut items) => {
                if items.is_empty() {
                    self.ocr_items = items;
                    self.matched_indices.clear();
                    self.current_match = 0;
                    self.status = "OCR finished, but no digit text was found.".to_owned();
                    return;
                }

                self.matched_indices = apply_digit_sequence_search(&mut items, &self.search_text);
                self.current_match = 0;
                let item_count = items.len();
                let match_count = self.match_group_count(&items);
                self.ocr_items = items;

                if match_count == 0 {
                    self.status = format!("OCR found {item_count} digit items. No matches.");
                } else {
                    self.status =
                        format!("OCR found {item_count} digit items. {match_count} matches.");
                }
            }
            Err(err) => {
                self.status = format!("{err:#}");
            }
        }
    }

    fn reapply_search(&mut self) {
        self.matched_indices = apply_digit_sequence_search(&mut self.ocr_items, &self.search_text);
        self.current_match = 0;

        if self.ocr_items.is_empty() {
            return;
        }

        let match_count = self.match_group_count(&self.ocr_items);
        if self.search_text.chars().all(|ch| !ch.is_ascii_digit()) {
            self.status = "Enter digits to search.".to_owned();
        } else if match_count == 0 {
            self.status = "No matches in existing OCR results.".to_owned();
        } else {
            self.status = format!("{match_count} matches in existing OCR results.");
        }
    }

    fn match_group_count(&self, items: &[OcrItem]) -> usize {
        items
            .iter()
            .filter_map(|item| item.match_group)
            .max()
            .map(|max_group| max_group + 1)
            .unwrap_or(0)
    }

    fn next_match(&mut self) {
        let count = self.match_group_count(&self.ocr_items);
        if count > 0 {
            self.current_match = (self.current_match + 1) % count;
        }
    }

    fn previous_match(&mut self) {
        let count = self.match_group_count(&self.ocr_items);
        if count > 0 {
            self.current_match = (self.current_match + count - 1) % count;
        }
    }

    fn draw_top_toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open Image").clicked() {
                    self.open_image(ctx);
                }

                ui.label("Digits");
                let search_response = ui.add(
                    egui::TextEdit::singleline(&mut self.search_text)
                        .desired_width(160.0)
                        .hint_text("235"),
                );
                if search_response.changed() && !self.ocr_items.is_empty() {
                    self.reapply_search();
                }

                if ui.button("OCR Search").clicked() {
                    self.run_ocr_search();
                }

                let has_matches = self.match_group_count(&self.ocr_items) > 0;
                if ui
                    .add_enabled(has_matches, egui::Button::new("Previous"))
                    .clicked()
                {
                    self.previous_match();
                }
                if ui
                    .add_enabled(has_matches, egui::Button::new("Next"))
                    .clicked()
                {
                    self.next_match();
                }

                egui::ComboBox::from_label("OCR Scale")
                    .selected_text(format!("{}x", self.ocr_scale))
                    .show_ui(ui, |ui| {
                        for scale in 1..=4 {
                            ui.selectable_value(&mut self.ocr_scale, scale, format!("{scale}x"));
                        }
                    });

                egui::ComboBox::from_label("PSM")
                    .selected_text(self.psm.to_string())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.psm, 6, "6 block");
                        ui.selectable_value(&mut self.psm, 7, "7 line");
                        ui.selectable_value(&mut self.psm, 11, "11 sparse");
                    });

                egui::ComboBox::from_label("Engine")
                    .selected_text(self.engine_mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.engine_mode,
                            OcrEngineMode::OnnxRoi,
                            "ONNX ROI",
                        );
                        ui.selectable_value(
                            &mut self.engine_mode,
                            OcrEngineMode::TesseractRoi,
                            "Tesseract ROI",
                        );
                        ui.selectable_value(
                            &mut self.engine_mode,
                            OcrEngineMode::TesseractFull,
                            "Tesseract full",
                        );
                    });

                ui.add(egui::Slider::new(&mut self.zoom, 0.25..=6.0).text("Zoom"));

                let count = self.match_group_count(&self.ocr_items);
                if count > 0 {
                    ui.label(format!("Match {}/{}", self.current_match + 1, count));
                }

                ui.separator();
                ui.label(&self.status);
            });
        });
    }

    fn draw_right_panel(&self, ctx: &egui::Context) {
        egui::SidePanel::right("ocr_results")
            .resizable(true)
            .default_width(310.0)
            .show(ctx, |ui| {
                ui.heading("OCR Results");
                ui.label(format!("Items: {}", self.ocr_items.len()));
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (index, item) in self.ocr_items.iter().enumerate() {
                        let is_current = item.match_group == Some(self.current_match);
                        let color = if is_current {
                            Color32::YELLOW
                        } else if item.matched {
                            Color32::LIGHT_RED
                        } else {
                            ui.visuals().text_color()
                        };

                        ui.horizontal(|ui| {
                            ui.colored_label(color, format!("#{index}"));
                            ui.colored_label(color, &item.text);
                        });
                        ui.small(format!(
                            "conf: {:.1}  match: {}  group: {}",
                            item.confidence,
                            item.matched,
                            item.match_group
                                .map(|group| group.to_string())
                                .unwrap_or_else(|| "-".to_owned())
                        ));
                        ui.add_space(6.0);
                    }
                });
            });
    }

    fn draw_image_view(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let Some(texture) = &self.texture else {
                        ui.centered_and_justified(|ui| {
                            ui.label("Open an image file.");
                        });
                        return;
                    };

                    let image_size = texture.size_vec2() * self.zoom;
                    let response = ui.add(egui::Image::new((texture.id(), image_size)));
                    let image_rect = response.rect;
                    let painter = ui.painter_at(image_rect);

                    for item in &self.ocr_items {
                        if !item.matched {
                            continue;
                        }

                        let color = if item.match_group == Some(self.current_match) {
                            Color32::YELLOW
                        } else {
                            Color32::RED
                        };
                        let rect = egui::Rect::from_min_max(
                            image_rect.min + item.rect_original.min.to_vec2() * self.zoom,
                            image_rect.min + item.rect_original.max.to_vec2() * self.zoom,
                        );
                        painter.rect_stroke(rect, 0.0, Stroke::new(2.0, color));
                    }
                });
        });
    }
}

impl eframe::App for DigitOcrViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.draw_top_toolbar(ctx);
        self.draw_right_panel(ctx);
        self.draw_image_view(ctx);
    }
}

fn dynamic_image_to_color_image(image: &DynamicImage) -> anyhow::Result<ColorImage> {
    let rgba = image.to_rgba8();
    let width = usize::try_from(rgba.width())?;
    let height = usize::try_from(rgba.height())?;
    Ok(ColorImage::from_rgba_unmultiplied(
        [width, height],
        rgba.as_raw(),
    ))
}
