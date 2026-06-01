use egui::{pos2, Rect};
use image::{DynamicImage, GrayImage};

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
    let gray = image.to_luma8();
    let mut components = connected_components(&gray, 110);
    components.retain(is_text_like_component);

    let mut rects: Vec<Rect> = components.into_iter().map(|component| component.rect()).collect();
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
        .map(|rect| pad_and_clamp(rect, gray.width(), gray.height(), 8.0))
        .filter(|rect| rect.width() >= 8.0 && rect.height() >= 8.0)
        .map(|rect_original| RoiCandidate { rect_original })
        .collect()
}

fn connected_components(gray: &GrayImage, threshold: u8) -> Vec<Component> {
    let width = gray.width();
    let height = gray.height();
    let len = (width as usize).saturating_mul(height as usize);
    let mut visited = vec![false; len];
    let mut components = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if visited[index] || gray.get_pixel(x, y)[0] > threshold {
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
                        if visited[next_index] || gray.get_pixel(nx, ny)[0] > threshold {
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
            for x in 30..70 {
                image.put_pixel(x, y, Luma([20]));
            }
        }

        let rois = detect_digit_rois(&DynamicImage::ImageLuma8(image));

        assert!(!rois.is_empty());
        assert!(rois[0].rect_original.left() <= 30.0);
        assert!(rois[0].rect_original.right() >= 70.0);
    }
}
