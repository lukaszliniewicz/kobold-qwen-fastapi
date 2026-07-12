"""Verified KoboldCpp Qwen3-TTS model catalogue and lazy downloader."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


HF_BASE_URL = "https://huggingface.co/koboldcpp/tts/resolve/main"
MODEL_SIZES = ("0.6b", "1.7b")
QUANTIZATIONS = ("q8_0", "f16")

MODEL_FILES = {
    ("base", "0.6b", "q8_0"): "qwen3-tts-0.6b-q8_0.gguf",
    ("base", "0.6b", "f16"): "qwen3-tts-0.6b-f16.gguf",
    ("base", "1.7b", "q8_0"): "Qwen3-TTS-12Hz-1.7B-Base-q8_0.gguf",
    ("base", "1.7b", "f16"): "Qwen3-TTS-12Hz-1.7B-Base-f16.gguf",
    ("customvoice", "1.7b", "q8_0"): "Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf",
    ("customvoice", "1.7b", "f16"): "Qwen3-TTS-12Hz-1.7B-CustomVoice-F16.gguf",
}
TOKENIZER_FILES = {
    "q8_0": "qwen3-tts-tokenizer-q8_0.gguf",
    "f16": "qwen3-tts-tokenizer-f16.gguf",
}


def normalize_size(value: str | None) -> str:
    normalized = str(value or "0.6b").strip().lower().replace(" ", "")
    return normalized if normalized in MODEL_SIZES else "0.6b"


def normalize_quantization(value: str | None) -> str:
    normalized = str(value or "q8_0").strip().lower().replace("-", "_")
    normalized = {"q8": "q8_0", "fp16": "f16", "float16": "f16"}.get(normalized, normalized)
    return normalized if normalized in QUANTIZATIONS else "q8_0"


def model_filename(model_type: str, size: str, quantization: str) -> str:
    normalized_type = str(model_type or "base").strip().lower()
    normalized_size = normalize_size(size)
    normalized_quant = normalize_quantization(quantization)
    if normalized_type == "customvoice":
        normalized_size = "1.7b"
    try:
        return MODEL_FILES[(normalized_type, normalized_size, normalized_quant)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Qwen3-TTS model combination: {normalized_type}/{normalized_size}/{normalized_quant}"
        ) from error


def tokenizer_filename(quantization: str) -> str:
    return TOKENIZER_FILES[normalize_quantization(quantization)]


def download_file(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "Pandrator-Kobold-Qwen/1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def ensure_model(models_dir: Path, model_type: str, size: str, quantization: str) -> Path:
    filename = model_filename(model_type, size, quantization)
    return download_file(f"{HF_BASE_URL}/{filename}?download=true", models_dir / filename)


def ensure_tokenizer(models_dir: Path, quantization: str) -> Path:
    filename = tokenizer_filename(quantization)
    return download_file(f"{HF_BASE_URL}/{filename}?download=true", models_dir / filename)
