use image::{imageops::FilterType, DynamicImage, GenericImageView};

pub fn preprocess_for_ocr(image: &DynamicImage, scale: u32) -> DynamicImage {
    let grayscale = image.grayscale();
    let scale = scale.max(1);

    if scale == 1 {
        return grayscale;
    }

    let (width, height) = grayscale.dimensions();
    grayscale.resize_exact(
        width.saturating_mul(scale),
        height.saturating_mul(scale),
        FilterType::CatmullRom,
    )
}
