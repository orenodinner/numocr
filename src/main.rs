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
        Box::new(|cc| Ok(Box::new(digit_ocr_viewer::app::DigitOcrViewerApp::new(cc)))),
    )
}
