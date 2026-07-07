# Kobold Qwen FastAPI

Kobold Qwen FastAPI is an OpenAI-compatible speech synthesis (TTS) API wrapper for [KoboldCpp](https://github.com/LostRuins/koboldcpp) running the [Qwen3-TTS](https://huggingface.co/koboldcpp/tts) engine. It provides a drop-in OpenAI-compatible speech generation endpoint and was developed to serve as a backend engine for the [Pandrator](https://github.com/lukaszliniewicz/Pandrator) application.

It automatically detects the host machine's hardware (NVIDIA CUDA, Vulkan AMD/Intel, Apple Silicon Metal, or CPU fallback), downloads the appropriate platform-specific binaries and model weights, and manages the background lifecycle of the KoboldCpp inference process.

---

## Features

- **OpenAI-Compatible Speech API**: Exposes standard speech endpoints (`/v1/audio/speech` / `/audio/speech`) for drop-in integration.
- **Auto-Hardware Detection**: Detects OS and graphics acceleration to automatically select CUDA, Vulkan, Metal, or CPU backend.
- **Dependency Automation**: Automatically downloads the latest platform-appropriate KoboldCpp executable and Qwen3-TTS model weights if not present.
- **CPU Thread Optimization**: Dynamically queries physical CPU cores (using `psutil`) to set optimal thread count (`--ttsthreads`), avoiding hyperthreading overhead.
- **Dynamic Reference Cloning**: Saves uploaded audio clips to the reference folder, instantly enabling custom speaker cloning without server restarts.
- **Speed Post-Processing**: Applies speed changes dynamically via `pydub` (speeding up or slowing down output audio based on the `speed` parameter) before streaming it back.

---

## Prerequisites

- [Pixi](https://pixi.sh) package manager.
- FFmpeg (automatically managed via conda-forge inside the Pixi environment).
- GPU support for Vulkan or CUDA (optional; falls back to physical CPU cores if no GPU acceleration is found).

---

## Installation & Running

Initialize the Pixi environment and run the server (listening on port `8040` by default):

```bash
# Cross-platform direct bootstrap. This installs/uses the local Pixi env.
python run.py

# Or run through Pixi explicitly
pixi run python run.py

# Or run the batch helper on Windows
run.bat
```

To force the server into CPU-only mode:

```bash
python run.py --backend cpu
```

To prepare the local Pixi environment and download models without starting the API server:

```bash
python run.py --prepare-only
```

The bootstrapper keeps Python packages, temporary files, model weights, and binaries isolated under the local folder. It does not install Python packages system-wide.

---

## API Endpoints

### 1. Generate Speech
- **URL**: `POST /v1/audio/speech` (or `POST /audio/speech`)
- **Format**: JSON payload compatible with OpenAI's audio request schema.
- **Supported parameters**: `model`, `input`, `voice` (speaker name), `speed`.

### 2. List Models
- **URL**: `GET /v1/models`
- **Output**: JSON list of available models:
  - `qwen3-tts`

### 3. List Voices
- **URL**: `GET /v1/audio/voices` / `GET /v1/files`
- **Output**: JSON list of uploaded reference voices in the `voices/` directory.

### 4. Upload Reference Voice
- **URL**: `POST /v1/audio/voices` / `POST /v1/files`
- **Format**: Multipart form data with `audio_sample` or `files`, plus optional `voice_id` / `name`.
