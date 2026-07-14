"""Verified KoboldCpp Qwen3-TTS model catalogue and lazy downloader."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


HF_REPO_ID = "koboldcpp/tts"
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


def _hf_hub_download(*, filename: str, local_dir: Path) -> Path:
    """Download through the Hub client using its reliable HTTP path by default."""
    # The public Xet CAS can return unauthorised responses for public files.
    # This must be set before importing huggingface_hub because its constants
    # are resolved at import time. Power users can explicitly set this to 0 to
    # opt back into Xet.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(local_dir),
        )
    )


def download_file(filename: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        resolved = _hf_hub_download(filename=filename, local_dir=destination.parent)
        if resolved.resolve() != destination.resolve():
            shutil.copyfile(resolved, temporary)
            os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not destination.is_file():
        raise RuntimeError(f"Hugging Face download did not create the expected file: {destination}")
    return destination


def ensure_model(models_dir: Path, model_type: str, size: str, quantization: str) -> Path:
    filename = model_filename(model_type, size, quantization)
    return download_file(filename, models_dir / filename)


def ensure_tokenizer(models_dir: Path, quantization: str) -> Path:
    filename = tokenizer_filename(quantization)
    return download_file(filename, models_dir / filename)
