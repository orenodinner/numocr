use std::collections::BTreeSet;

use crate::ocr::OcrItem;

const SAME_LINE_CENTER_Y_THRESHOLD: f32 = 12.0;
pub fn normalize_digits(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_digit()).collect()
}

pub fn apply_digit_sequence_search(items: &mut [OcrItem], query: &str) -> Vec<usize> {
    let normalized_query = normalize_digits(query);

    for item in items.iter_mut() {
        item.matched = false;
        item.match_group = None;
    }

    if normalized_query.is_empty() {
        return Vec::new();
    }

    for item in items.iter_mut() {
        item.normalized = normalize_digits(&item.text);
    }

    let mut ordered_indices: Vec<usize> = (0..items.len()).collect();
    ordered_indices.sort_by(|&a, &b| compare_ocr_items(&items[a], &items[b]));

    let mut joined = String::new();
    let mut char_to_item_index = Vec::new();
    for index in ordered_indices {
        for ch in items[index].normalized.chars() {
            joined.push(ch);
            char_to_item_index.push(index);
        }
    }

    if joined.is_empty() {
        return Vec::new();
    }

    let mut matched_groups = Vec::new();
    let mut group_id = 0usize;
    let mut search_from = 0usize;

    while let Some(relative_start) = joined[search_from..].find(&normalized_query) {
        let start = search_from + relative_start;
        let end = start + normalized_query.len();
        let mut group_items = BTreeSet::new();

        for char_index in start..end {
            if let Some(item_index) = char_to_item_index.get(char_index).copied() {
                group_items.insert(item_index);
            }
        }

        if let Some(first_index) = group_items.iter().next().copied() {
            for item_index in group_items {
                items[item_index].matched = true;
                items[item_index].match_group = Some(group_id);
            }
            matched_groups.push(first_index);
            group_id += 1;
        }

        search_from = start + 1;
    }

    matched_groups
}

fn compare_ocr_items(a: &OcrItem, b: &OcrItem) -> std::cmp::Ordering {
    let ay = a.rect_original.center().y;
    let by = b.rect_original.center().y;

    if (ay - by).abs() < SAME_LINE_CENTER_Y_THRESHOLD {
        a.rect_original
            .left()
            .partial_cmp(&b.rect_original.left())
            .unwrap_or(std::cmp::Ordering::Equal)
    } else {
        ay.partial_cmp(&by).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use egui::{pos2, Rect};

    use super::*;

    fn item(text: &str, x: f32, y: f32) -> OcrItem {
        OcrItem {
            text: text.to_owned(),
            normalized: normalize_digits(text),
            confidence: 90.0,
            rect_original: Rect::from_min_size(pos2(x, y), egui::vec2(10.0, 10.0)),
            matched: false,
            match_group: None,
        }
    }

    #[test]
    fn finds_split_digit_sequence() {
        let mut items = vec![
            item("2", 0.0, 0.0),
            item("3", 12.0, 0.0),
            item("5", 24.0, 0.0),
        ];
        let matches = apply_digit_sequence_search(&mut items, "235");

        assert_eq!(matches.len(), 1);
        assert!(items.iter().all(|item| item.matched));
        assert!(items.iter().all(|item| item.match_group == Some(0)));
    }

    #[test]
    fn ignores_non_digits_in_ocr_and_query() {
        let mut items = vec![item("2026/06/01", 0.0, 0.0)];
        let matches = apply_digit_sequence_search(&mut items, "20260601");

        assert_eq!(matches.len(), 1);
        assert!(items[0].matched);
    }
}
