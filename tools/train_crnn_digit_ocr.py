#!/usr/bin/env python3
"""
Train a small CRNN+CTC digit OCR recognizer and export ONNX.

This is intentionally lightweight for CPU inference on machines like
Core Ultra 7 / 16GB RAM. It trains on crops created by
tools/prepare_hf_recognition_dataset.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "training_data" / "catia_ocr_recognition"
DEFAULT_OUT = REPO_ROOT / "models" / "catia_crnn_digit_ocr"
CHARS = "0123456789"


class CrnnOcr(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 96):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.10,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images).mean(dim=2)
        sequence = features.permute(2, 0, 1)
        sequence, _ = self.rnn(sequence)
        return self.classifier(sequence)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    text: str
    normalized_digits: str


class OcrCropDataset(Dataset):
    def __init__(self, root: Path, split: str, charset: str, image_height: int, label_mode: str):
        self.root = root / split
        self.charset = charset
        self.image_height = image_height
        self.char_to_id = {ch: index + 1 for index, ch in enumerate(charset)}
        self.samples: list[Sample] = []
        with (self.root / "metadata.csv").open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                raw_text = row["normalized_digits"] if label_mode == "digits" else row["text"]
                text = "".join(ch for ch in raw_text if ch in self.char_to_id)
                if not text:
                    continue
                self.samples.append(
                    Sample(
                        image_path=self.root / row["file_name"],
                        text=text,
                        normalized_digits=row["normalized_digits"],
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("L")
        if image.height != self.image_height:
            scale = self.image_height / image.height
            image = image.resize((max(8, round(image.width * scale)), self.image_height), Image.Resampling.BICUBIC)
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(1, image.height, image.width)
        tensor = (255.0 - tensor) / 255.0
        target = torch.tensor([self.char_to_id[ch] for ch in sample.text], dtype=torch.long)
        return tensor, target, sample.text, sample.normalized_digits, image.width


def collate_batch(batch):
    images, targets, texts, normalized, widths = zip(*batch)
    height = images[0].shape[1]
    max_width = max(image.shape[2] for image in images)
    padded = torch.zeros((len(images), 1, height, max_width), dtype=torch.float32)
    for index, image in enumerate(images):
        padded[index, :, :, : image.shape[2]] = image
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    input_lengths = torch.tensor([max(1, width // 4) for width in widths], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return padded, flat_targets, target_lengths, input_lengths, texts, normalized


def decode_greedy(logits: torch.Tensor, charset: str) -> list[str]:
    pred = logits.argmax(dim=2).permute(1, 0).cpu().tolist()
    decoded = []
    for row in pred:
        chars = []
        prev = 0
        for token in row:
            if token != 0 and token != prev:
                chars.append(charset[token - 1])
            prev = token
        decoded.append("".join(chars))
    return decoded


def decode_greedy_with_lengths(logits: torch.Tensor, input_lengths: torch.Tensor, charset: str) -> list[str]:
    pred = logits.argmax(dim=2).permute(1, 0).cpu().tolist()
    decoded = []
    for row, length in zip(pred, input_lengths.cpu().tolist()):
        chars = []
        prev = 0
        for token in row[:length]:
            if token != 0 and token != prev:
                chars.append(charset[token - 1])
            prev = token
        decoded.append("".join(chars))
    return decoded


def normalize_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


@torch.no_grad()
def evaluate(model, loader, device, charset):
    model.eval()
    exact = 0
    digit_exact = 0
    total = 0
    for images, _targets, _target_lengths, input_lengths, texts, normalized in loader:
        images = images.to(device)
        logits = model(images)
        decoded = decode_greedy_with_lengths(logits, input_lengths, charset)
        for pred, text, digits in zip(decoded, texts, normalized):
            exact += int(pred == text)
            digit_exact += int(normalize_digits(pred) == digits)
            total += 1
    return {
        "exact": exact / max(1, total),
        "digit_exact": digit_exact / max(1, total),
        "total": total,
    }


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    charset = args.charset
    train_ds = OcrCropDataset(args.dataset_dir, "train", charset, args.image_height, args.label_mode)
    val_ds = OcrCropDataset(args.dataset_dir, "validation", charset, args.image_height, args.label_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = CrnnOcr(num_classes=len(charset) + 1, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best_digit_exact = -1.0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for images, targets, target_lengths, input_lengths, _texts, _normalized in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=2)
            loss = criterion(log_probs, targets, input_lengths.to(device), target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        metrics = evaluate(model, val_loader, device, charset)
        metrics["epoch"] = epoch
        metrics["loss"] = total_loss / max(1, total_batches)
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        if metrics["digit_exact"] > best_digit_exact:
            best_digit_exact = metrics["digit_exact"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "charset": charset,
                    "image_height": args.image_height,
                    "hidden_size": args.hidden_size,
                    "metrics": metrics,
                },
                args.output_dir / "model.pt",
            )

    checkpoint = torch.load(args.output_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_ds = OcrCropDataset(args.dataset_dir, "test", charset, args.image_height, args.label_mode)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_metrics = evaluate(model, test_loader, device, charset)
    test_metrics["elapsed_sec"] = round(time.time() - started, 3)

    metadata = {
        "charset": charset,
        "blank_id": 0,
        "image_height": args.image_height,
        "hidden_size": args.hidden_size,
        "device": str(device),
        "label_mode": args.label_mode,
        "best_validation": checkpoint["metrics"],
        "test": test_metrics,
        "history": history,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        export_onnx(model, args.output_dir / "model.onnx", args.image_height, device)
    except Exception as exc:
        (args.output_dir / "onnx_export_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\nInstall the `onnx` Python package and rerun training or export manually.\n",
            encoding="utf-8",
        )
        print(f"ONNX export skipped: {type(exc).__name__}: {exc}", flush=True)


def export_onnx(model: nn.Module, output_path: Path, image_height: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.zeros((1, 1, image_height, 160), dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch", 3: "width"}, "logits": {0: "time", 1: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight CRNN digit OCR model.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--charset", default=CHARS)
    parser.add_argument("--label-mode", choices=["digits", "text"], default="digits")
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
