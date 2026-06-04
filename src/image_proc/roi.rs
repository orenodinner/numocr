use egui::{pos2, Rect};
use image::{DynamicImage, GrayImage};

const MAX_ANALYSIS_PIXELS: u64 = 4_000_000;
const MAX_ANALYSIS_SIDE: u32 = 2400;
const MAX_ROI_CANDIDATES: usize = 768;
const MIN_LINE_RUN: usize = 8;
const BASE_LINE_RUN: f32 = 28.0;

#[derive(Debug, Clone)]
pub struct RoiCandidate {
    pub rect_original: Rect,
}

#[derive(Debug, Clone)]
struct Component {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    pixels: u32,
}

impl Component {
    fn width(&self) -> u32 {
        self.max_x - self.min_x + 1
    }

    fn height(&self) -> u32 {
        self.max_y - self.min_y + 1
    }

    fn rect(&self) -> Rect {
        Rect::from_min_max(
            pos2(self.min_x as f32, self.min_y as f32),
            pos2((self.max_x + 1) as f32, (self.max_y + 1) as f32),
        )
    }
}

pub fn detect_digit_rois(image: &DynamicImage) -> Vec<RoiCandidate> {
    if image.width() == 0 || image.height() == 0 {
        return Vec::new();
    }

    let (gray, scale_x, scale_y, analysis_scale) = analysis_gray(image);
    let line_run = (BASE_LINE_RUN * analysis_scale)
        .round()
        .max(MIN_LINE_RUN as f32) as usize;
    let foreground = foreground_without_long_lines(&gray, 175, line_run);
    let mut components = connected_components(&foreground, gray.width(), gray.height());
    components.retain(is_text_like_component);

    let mut rects: Vec<Rect> = components
        .into_iter()
        .map(|component| component.rect())
        .collect();
    rects.sort_by(|a, b| {
        a.center()
            .y
            .partial_cmp(&b.center().y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.left()
                    .partial_cmp(&b.left())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mut groups: Vec<Rect> = Vec::new();
    for rect in rects {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| should_merge_text_rects(**group, rect))
        {
            *group = group.union(rect);
        } else {
            groups.push(rect);
        }
    }

    groups
        .into_iter()
        .take(MAX_ROI_CANDIDATES)
        .map(|rect| scale_rect_to_original(rect, scale_x, scale_y))
        .map(|rect| pad_and_clamp(rect, image.width(), image.height(), 8.0))
        .filter(|rect| rect.width() >= 8.0 && rect.height() >= 8.0)
        .map(|rect_original| RoiCandidate { rect_original })
        .collect()
}

fn analysis_gray(image: &DynamicImage) -> (GrayImage, f32, f32, f32) {
    let original_width = image.width();
    let original_height = image.height();
    let original_pixels = original_width as u64 * original_height as u64;
    let pixel_scale = if original_pixels > MAX_ANALYSIS_PIXELS {
        (MAX_ANALYSIS_PIXELS as f32 / original_pixels as f32).sqrt()
    } else {
        1.0
    };
    let side_scale = if original_width.max(original_height) > MAX_ANALYSIS_SIDE {
        MAX_ANALYSIS_SIDE as f32 / original_width.max(original_height) as f32
    } else {
        1.0
    };
    let analysis_scale = pixel_scale.min(side_scale).min(1.0);

    if analysis_scale >= 0.999 {
        return (image.to_luma8(), 1.0, 1.0, 1.0);
    }

    let analysis_width = ((original_width as f32 * analysis_scale).round() as u32).max(1);
    let analysis_height = ((original_height as f32 * analysis_scale).round() as u32).max(1);
    let gray = image.to_luma8();
    let resized = image::imageops::resize(
        &gray,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Triangle,
    );
    (
        resized,
        original_width as f32 / analysis_width as f32,
        original_height as f32 / analysis_height as f32,
        analysis_scale,
    )
}

fn foreground_without_long_lines(
    gray: &GrayImage,
    threshold: u8,
    line_run_threshold: usize,
) -> Vec<bool> {
    let width = gray.width() as usize;
    let height = gray.height() as usize;
    let mut foreground = vec![false; width * height];

    for y in 0..height {
        for x in 0..width {
            foreground[y * width + x] = gray.get_pixel(x as u32, y as u32)[0] <= threshold;
        }
    }

    let mut line_mask = vec![false; width * height];
    for y in 0..height {
        let mut x = 0;
        while x < width {
            while x < width && !foreground[y * width + x] {
                x += 1;
            }
            let start = x;
            while x < width && foreground[y * width + x] {
                x += 1;
            }
            if x - start >= line_run_threshold {
                for run_x in start..x {
                    line_mask[y * width + run_x] = true;
                }
            }
        }
    }

    for x in 0..width {
        let mut y = 0;
        while y < height {
            while y < height && !foreground[y * width + x] {
                y += 1;
            }
            let start = y;
            while y < height && foreground[y * width + x] {
                y += 1;
            }
            if y - start >= line_run_threshold {
                for run_y in start..y {
                    line_mask[run_y * width + x] = true;
                }
            }
        }
    }

    for (index, is_line) in line_mask.into_iter().enumerate() {
        if is_line {
            foreground[index] = false;
        }
    }

    foreground
}

fn scale_rect_to_original(rect: Rect, scale_x: f32, scale_y: f32) -> Rect {
    Rect::from_min_max(
        pos2(rect.left() * scale_x, rect.top() * scale_y),
        pos2(rect.right() * scale_x, rect.bottom() * scale_y),
    )
}

fn connected_components(foreground: &[bool], width: u32, height: u32) -> Vec<Component> {
    let len = (width as usize).saturating_mul(height as usize);
    let mut visited = vec![false; len];
    let mut components = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if visited[index] || !foreground[index] {
                continue;
            }

            let mut stack = vec![(x, y)];
            visited[index] = true;
            let mut component = Component {
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
                pixels: 0,
            };

            while let Some((cx, cy)) = stack.pop() {
                component.min_x = component.min_x.min(cx);
                component.min_y = component.min_y.min(cy);
                component.max_x = component.max_x.max(cx);
                component.max_y = component.max_y.max(cy);
                component.pixels += 1;

                let x0 = cx.saturating_sub(1);
                let y0 = cy.saturating_sub(1);
                let x1 = (cx + 1).min(width - 1);
                let y1 = (cy + 1).min(height - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let next_index = (ny * width + nx) as usize;
                        if visited[next_index] || !foreground[next_index] {
                            continue;
                        }
                        visited[next_index] = true;
                        stack.push((nx, ny));
                    }
                }
            }

            components.push(component);
        }
    }

    components
}

fn is_text_like_component(component: &Component) -> bool {
    let width = component.width();
    let height = component.height();
    if component.pixels < 4 || width < 2 || height < 4 {
        return false;
    }

    let aspect = width as f32 / height as f32;
    let fill = component.pixels as f32 / (width * height) as f32;

    if height <= 3 && width > 18 {
        return false;
    }
    if width <= 3 && height > 18 {
        return false;
    }
    if aspect > 18.0 || aspect < 0.05 {
        return false;
    }
    if fill < 0.04 && (width > 40 || height > 40) {
        return false;
    }

    true
}

fn should_merge_text_rects(a: Rect, b: Rect) -> bool {
    let same_line = (a.center().y - b.center().y).abs() <= a.height().max(b.height()).max(12.0);
    let horizontal_gap = if a.right() < b.left() {
        b.left() - a.right()
    } else if b.right() < a.left() {
        a.left() - b.right()
    } else {
        0.0
    };
    let vertical_overlap = a.intersect(b).height().max(0.0);
    let min_height = a.height().min(b.height()).max(1.0);

    same_line && horizontal_gap <= 18.0 && vertical_overlap / min_height >= 0.10
}

fn pad_and_clamp(rect: Rect, image_width: u32, image_height: u32, pad: f32) -> Rect {
    Rect::from_min_max(
        pos2((rect.left() - pad).max(0.0), (rect.top() - pad).max(0.0)),
        pos2(
            (rect.right() + pad).min(image_width as f32),
            (rect.bottom() + pad).min(image_height as f32),
        ),
    )
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GrayImage, Luma};

    use super::detect_digit_rois;

    #[test]
    fn detects_dark_text_like_blob() {
        let mut image = GrayImage::from_pixel(120, 60, Luma([255]));
        for y in 20..36 {
            for x in 30..40 {
                image.put_pixel(x, y, Luma([20]));
            }
            for x in 48..58 {
                image.put_pixel(x, y, Luma([20]));
            }
            for x in 66..76 {
                image.put_pixel(x, y, Luma([20]));
            }
        }

        let rois = detect_digit_rois(&DynamicImage::ImageLuma8(image));

        assert!(!rois.is_empty());
        assert!(rois[0].rect_original.left() <= 30.0);
        assert!(rois[0].rect_original.right() >= 76.0);
    }

    #[test]
    fn ignores_empty_images() {
        let image = GrayImage::new(0, 0);
        let rois = detect_digit_rois(&DynamicImage::ImageLuma8(image));

        assert!(rois.is_empty());
    }

    #[test]
    fn caps_large_noisy_images() {
        let mut image = GrayImage::from_pixel(2600, 1800, Luma([255]));
        for y in (20..1780).step_by(37) {
            for x in (20..2580).step_by(41) {
                image.put_pixel(x, y, Luma([20]));
                image.put_pixel((x + 1).min(2599), y, Luma([20]));
                image.put_pixel(x, (y + 1).min(1799), Luma([20]));
            }
        }

        let rois = detect_digit_rois(&DynamicImage::ImageLuma8(image));

        assert!(rois.len() <= super::MAX_ROI_CANDIDATES);
    }
}
