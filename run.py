#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import subprocess
import sys
import platform
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
log = logging.getLogger("run")

PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent
DEFAULT_PIXI = PARENT_DIR / "bin" / ("pixi.exe" if os.name == "nt" else "pixi")

# Download URLs
KOBOLD_BASE_URL = "https://github.com/LostRuins/koboldcpp/releases/latest/download/"
QWEN_MODEL_URL = "https://huggingface.co/koboldcpp/tts/resolve/main/Qwen3-TTS-12Hz-1.7B-Base-q8_0.gguf?download=true"
QWEN_TOKENIZER_URL = "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-tokenizer-f16.gguf?download=true"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Kobold Qwen TTS FastAPI wrapper bootstrapper")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8040, help="Port number")
    parser.add_argument("--backend", choices=["auto", "cuda", "vulkan", "metal", "cpu"], default="auto", help="Backend accelerator target")
    parser.add_argument("--threads", type=int, default=None, help="Force specific number of CPU threads")
    parser.add_argument("--pixi-path", default=None, help="Pixi executable to use when bootstrapping")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare environment and download models without starting server")
    parser.add_argument("--inside-pixi", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def configure_portable_environment():
    cache_root = PARENT_DIR / "cache"
    pixi_cache = PARENT_DIR / ".pixi-cache"
    temp_dir = pixi_cache / "tmp"

    for directory in (cache_root, pixi_cache, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PIXI_CACHE_DIR", str(pixi_cache))
    os.environ.setdefault("RATTLER_CACHE_DIR", str(pixi_cache / "rattler"))
    os.environ.setdefault("PIP_CACHE_DIR", str(pixi_cache / "pip"))
    os.environ.setdefault("UV_CACHE_DIR", str(pixi_cache / "uv-cache"))
    os.environ.setdefault("TMP", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMPDIR", str(temp_dir))

    # Model caches
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("TTS_HOME", str(cache_root / "tts"))


def resolve_pixi(pixi_path=None):
    candidates = []
    if pixi_path:
        candidates.append(Path(pixi_path))
    candidates.append(DEFAULT_PIXI)
    
    # Check user home directory
    home_pixi = Path.home() / ".pixi" / "bin" / ("pixi.exe" if os.name == "nt" else "pixi")
    candidates.append(home_pixi)

    path_pixi = shutil.which("pixi.exe" if os.name == "nt" else "pixi")
    if path_pixi:
        candidates.append(Path(path_pixi))

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError("Pixi was not found. Install Pixi or configure path.")


def in_project_pixi_environment():
    env_root = PROJECT_DIR / ".pixi" / "envs" / "default"
    try:
        executable = Path(sys.executable).resolve()
        env_root = env_root.resolve()
        return os.path.commonpath((str(executable), str(env_root))) == str(env_root)
    except (OSError, ValueError):
        return False


def ensure_running_inside_pixi(args, argv):
    if args.inside_pixi or in_project_pixi_environment():
        return

    pixi = resolve_pixi(args.pixi_path)
    log.info("Bootstrapping Pixi environment dependencies...")
    subprocess.run([str(pixi), "install"], cwd=PROJECT_DIR, check=True)
    command = [pixi, "run", "python", str(PROJECT_DIR / "run.py"), "--inside-pixi", *argv]
    raise SystemExit(subprocess.call([str(part) for part in command], cwd=PROJECT_DIR))


def detect_hardware():
    sys_name = platform.system()
    machine = platform.machine()

    if sys_name == "Darwin" and (machine == "arm64" or "M" in platform.processor() or "Apple" in platform.processor()):
        log.info("Detected Apple Silicon (macOS M-series). Target backend: metal")
        return "metal"

    if sys_name == "Windows":
        # Check for NVIDIA CUDA via nvidia-smi
        if shutil.which("nvidia-smi") is not None:
            log.info("Detected NVIDIA GPU (nvidia-smi found). Target backend: cuda")
            return "cuda"
        # Check for Vulkan supported GPUs
        try:
            out = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode(errors="ignore")
            out_lower = out.lower()
            if "nvidia" in out_lower:
                return "cuda"
            if "amd" in out_lower or "radeon" in out_lower or "intel" in out_lower:
                log.info("Detected AMD or Intel GPU (vulkan supported). Target backend: vulkan")
                return "vulkan"
        except Exception:
            pass

    elif sys_name == "Linux":
        if shutil.which("nvidia-smi") is not None:
            log.info("Detected NVIDIA GPU (nvidia-smi found). Target backend: cuda")
            return "cuda"
        try:
            out = subprocess.check_output("lspci", shell=True).decode(errors="ignore")
            out_lower = out.lower()
            if "nvidia" in out_lower:
                return "cuda"
            if "amd" in out_lower or "radeon" in out_lower or "intel" in out_lower:
                log.info("Detected AMD or Intel GPU (vulkan supported). Target backend: vulkan")
                return "vulkan"
        except Exception:
            pass

    log.info("No supported GPU accelerator detected. Target backend: cpu")
    return "cpu"


def download_file(url, dest_path):
    dest_path = Path(dest_path)
    if dest_path.exists():
        return

    log.info(f"Downloading {url} -> {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".download")

    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            # Simple progress bar logic
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 1024
            downloaded = 0
            while True:
                block = response.read(block_size)
                if not block:
                    break
                out_file.write(block)
                downloaded += len(block)
                if total_size > 0:
                    percent = int(100 * downloaded / total_size)
                    sys.stdout.write(f"\rProgress: {percent}% ({downloaded // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB)")
                    sys.stdout.flush()
            sys.stdout.write("\n")
        
        os.replace(temp_path, dest_path)
        log.info(f"Successfully downloaded {dest_path.name}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed downloading {url}: {e}")


def ensure_kobold_binary(backend):
    bin_dir = PROJECT_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    sys_name = platform.system()

    if sys_name == "Windows":
        binary_path = bin_dir / "koboldcpp.exe"
        if not binary_path.exists():
            download_file(KOBOLD_BASE_URL + "koboldcpp.exe", binary_path)
        return binary_path

    binary_path = bin_dir / "koboldcpp"
    if not binary_path.exists():
        if sys_name == "Darwin":
            download_file(KOBOLD_BASE_URL + "koboldcpp-mac-arm64", binary_path)
        elif sys_name == "Linux":
            if backend == "cuda":
                download_file(KOBOLD_BASE_URL + "koboldcpp-linux-x64", binary_path)
            else:
                download_file(KOBOLD_BASE_URL + "koboldcpp-linux-x64-nocuda", binary_path)
        
        # Set execute permissions
        os.chmod(binary_path, 0o755)

    return binary_path


def ensure_qwen_models():
    models_dir = PROJECT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "Qwen3-TTS-12Hz-1.7B-Base-q8_0.gguf"
    tokenizer_path = models_dir / "qwen3-tts-tokenizer-f16.gguf"

    if not model_path.exists():
        download_file(QWEN_MODEL_URL, model_path)
    if not tokenizer_path.exists():
        download_file(QWEN_TOKENIZER_URL, tokenizer_path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    os.chdir(PROJECT_DIR)
    configure_portable_environment()
    ensure_running_inside_pixi(args, argv)

    # 1. Hardware Detection
    backend = args.backend
    if backend == "auto":
        backend = detect_hardware()

    # 2. Ensure dependencies are satisfied
    log.info("Validating local dependencies...")
    ensure_kobold_binary(backend)
    ensure_qwen_models()

    # Expose variables to FastAPI server
    os.environ["KOBOLD_QWEN_BACKEND"] = backend
    if args.threads:
        os.environ["KOBOLD_QWEN_THREADS"] = str(args.threads)

    # Ensure voices folder is created
    (PROJECT_DIR / "voices").mkdir(parents=True, exist_ok=True)

    if args.prepare_only:
        log.info("Environment preparation complete. Ready to launch.")
        return

    # Start FastAPI server
    import uvicorn
    log.info(f"Starting server on {args.host}:{args.port} using backend: {backend}")
    uvicorn.run("main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
