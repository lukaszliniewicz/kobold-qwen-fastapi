import os
import sys
from pathlib import Path
from types import ModuleType

from model_catalog import (
    _hf_hub_download,
    ensure_tokenizer,
    ensure_model,
    model_filename,
    normalize_quantization,
    normalize_size,
    tokenizer_filename,
)
from run import parse_args


def test_catalog_resolves_supported_base_variants():
    assert model_filename("base", "0.6b", "q8_0") == "qwen3-tts-0.6b-q8_0.gguf"
    assert model_filename("base", "0.6b", "f16") == "qwen3-tts-0.6b-f16.gguf"
    assert model_filename("base", "1.7b", "q8_0") == "Qwen3-TTS-12Hz-1.7B-Base-q8_0.gguf"
    assert model_filename("base", "1.7b", "f16") == "Qwen3-TTS-12Hz-1.7B-Base-f16.gguf"


def test_custom_voice_uses_its_available_1_7b_variant():
    assert model_filename("customvoice", "0.6b", "q8_0") == "Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf"
    assert model_filename("customvoice", "1.7b", "f16") == "Qwen3-TTS-12Hz-1.7B-CustomVoice-F16.gguf"


def test_quantization_and_size_aliases_are_normalized():
    assert normalize_quantization("FP16") == "f16"
    assert normalize_quantization("q8") == "q8_0"
    assert normalize_size("1.7B") == "1.7b"
    assert tokenizer_filename("q8") == "qwen3-tts-tokenizer-q8_0.gguf"


def test_existing_model_is_not_downloaded(tmp_path: Path):
    expected = tmp_path / "qwen3-tts-0.6b-q8_0.gguf"
    expected.write_bytes(b"ready")
    resolved = ensure_model(tmp_path, "base", "0.6b", "q8_0")
    assert resolved == expected


def test_model_download_uses_hugging_face_hub(tmp_path: Path, monkeypatch):
    expected = tmp_path / "Qwen3-TTS-12Hz-1.7B-Base-f16.gguf"
    calls = []

    def fake_download(*, filename, local_dir):
        calls.append((filename, local_dir))
        destination = local_dir / filename
        destination.write_bytes(b"model")
        return destination

    monkeypatch.setattr("model_catalog._hf_hub_download", fake_download)

    resolved = ensure_model(tmp_path, "base", "1.7b", "f16")

    assert resolved == expected
    assert calls == [(expected.name, tmp_path)]


def test_tokenizer_download_uses_hugging_face_hub(tmp_path: Path, monkeypatch):
    expected = tmp_path / "qwen3-tts-tokenizer-f16.gguf"

    def fake_download(*, filename, local_dir):
        destination = local_dir / filename
        destination.write_bytes(b"tokenizer")
        return destination

    monkeypatch.setattr("model_catalog._hf_hub_download", fake_download)

    assert ensure_tokenizer(tmp_path, "f16") == expected


def test_hub_download_disables_xet_before_import(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    fake_hub = ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"model")
        return str(destination)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved = _hf_hub_download(filename="model.gguf", local_dir=tmp_path)

    assert resolved == tmp_path / "model.gguf"


def test_hub_download_respects_explicit_xet_opt_in(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    fake_hub = ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        assert os.environ["HF_HUB_DISABLE_XET"] == "0"
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"model")
        return str(destination)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    _hf_hub_download(filename="model.gguf", local_dir=tmp_path)


def test_launcher_defaults_to_lower_memory_quantized_model():
    args = parse_args([])
    assert args.model_size == "0.6b"
    assert args.quantization == "q8_0"
    assert args.initial_model == "base"


def test_launcher_accepts_full_precision_large_custom_voice_model():
    args = parse_args(
        ["--model-size", "1.7b", "--quantization", "f16", "--initial-model", "customvoice"]
    )
    assert args.model_size == "1.7b"
    assert args.quantization == "f16"
    assert args.initial_model == "customvoice"
