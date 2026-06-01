use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use csv::Reader;
use digit_ocr_viewer::ocr::{
    OcrEngine, OcrOptions, OnnxDigitOcrEngine, RoiTesseractCliOcrEngine, TesseractCliOcrEngine,
};
use digit_ocr_viewer::search::digit_sequence::apply_digit_sequence_search;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "training_data/catia_2d_digits")]
    dataset_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = EngineMode::Roi)]
    engine: EngineMode,
    #[arg(long, default_value_t = 11)]
    psm: u8,
    #[arg(long, default_value_t = 2)]
    scale: u32,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EngineMode {
    Onnx,
    Full,
    Roi,
}

#[derive(Debug, Deserialize)]
struct Row {
    file: String,
    label: String,
    normalized_digits: String,
    template: String,
    label_pattern: String,
}

#[derive(Debug, Serialize)]
struct Failure {
    file: String,
    label: String,
    expected: String,
    recognized: Vec<String>,
    template: String,
    label_pattern: String,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    engine: String,
    psm: u8,
    scale: u32,
    count: usize,
    hits: usize,
    hit_rate: f32,
    recognized_any: usize,
    recognized_rate: f32,
    elapsed_ms: u128,
    avg_ms_per_image: f32,
    template_hit_rate: BTreeMap<String, String>,
    pattern_hit_rate: BTreeMap<String, String>,
    failures: Vec<Failure>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rows = read_rows(&args.dataset_dir, args.limit)?;
    let full_engine = TesseractCliOcrEngine;
    let roi_engine = RoiTesseractCliOcrEngine;
    let onnx_engine = if matches!(args.engine, EngineMode::Onnx) {
        Some(OnnxDigitOcrEngine::from_default_model()?)
    } else {
        None
    };
    let options = OcrOptions {
        psm: args.psm,
        scale: args.scale,
        use_roi: matches!(args.engine, EngineMode::Onnx | EngineMode::Roi),
    };

    let started = Instant::now();
    let mut hits = 0usize;
    let mut recognized_any = 0usize;
    let mut template_total: BTreeMap<String, usize> = BTreeMap::new();
    let mut template_hit: BTreeMap<String, usize> = BTreeMap::new();
    let mut pattern_total: BTreeMap<String, usize> = BTreeMap::new();
    let mut pattern_hit: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();

    for row in &rows {
        let image_path = args.dataset_dir.join(&row.file);
        let image = image::open(&image_path)
            .with_context(|| format!("failed to open {}", image_path.display()))?;
        let mut items = match args.engine {
            EngineMode::Onnx => onnx_engine
                .as_ref()
                .context("ONNX engine was not initialized")?
                .recognize(&image, &options)?,
            EngineMode::Full => full_engine.recognize(&image, &options)?,
            EngineMode::Roi => roi_engine.recognize(&image, &options)?,
        };
        let matched = !apply_digit_sequence_search(&mut items, &row.normalized_digits).is_empty();
        let has_recognition = !items.is_empty();

        *template_total.entry(row.template.clone()).or_default() += 1;
        *pattern_total.entry(row.label_pattern.clone()).or_default() += 1;
        if has_recognition {
            recognized_any += 1;
        }
        if matched {
            hits += 1;
            *template_hit.entry(row.template.clone()).or_default() += 1;
            *pattern_hit.entry(row.label_pattern.clone()).or_default() += 1;
        } else if failures.len() < 30 {
            failures.push(Failure {
                file: row.file.clone(),
                label: row.label.clone(),
                expected: row.normalized_digits.clone(),
                recognized: items.into_iter().map(|item| item.text).collect(),
                template: row.template.clone(),
                label_pattern: row.label_pattern.clone(),
            });
        }
    }

    let elapsed_ms = started.elapsed().as_millis();
    let report = EvalReport {
        engine: format!("{:?}", args.engine).to_lowercase(),
        psm: args.psm,
        scale: args.scale,
        count: rows.len(),
        hits,
        hit_rate: hits as f32 / rows.len().max(1) as f32,
        recognized_any,
        recognized_rate: recognized_any as f32 / rows.len().max(1) as f32,
        elapsed_ms,
        avg_ms_per_image: elapsed_ms as f32 / rows.len().max(1) as f32,
        template_hit_rate: ratio_map(&template_total, &template_hit),
        pattern_hit_rate: ratio_map(&pattern_total, &pattern_hit),
        failures,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = args.output {
        std::fs::write(&output, format!("{json}\n"))
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn read_rows(dataset_dir: &Path, limit: Option<usize>) -> Result<Vec<Row>> {
    let labels_path = dataset_dir.join("labels.csv");
    let mut reader = Reader::from_path(&labels_path)
        .with_context(|| format!("failed to read {}", labels_path.display()))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row?);
        if limit.is_some_and(|limit| rows.len() >= limit) {
            break;
        }
    }
    Ok(rows)
}

fn ratio_map(
    total: &BTreeMap<String, usize>,
    hit: &BTreeMap<String, usize>,
) -> BTreeMap<String, String> {
    total
        .iter()
        .map(|(key, total)| {
            let hit = hit.get(key).copied().unwrap_or_default();
            (key.clone(), format!("{hit}/{total}"))
        })
        .collect()
}
