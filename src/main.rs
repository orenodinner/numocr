mod app;
mod image_proc;
mod ocr;
mod search;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Digit OCR Viewer",
        options,
        Box::new(|cc| Ok(Box::new(app::DigitOcrViewerApp::new(cc)))),
    )
}
