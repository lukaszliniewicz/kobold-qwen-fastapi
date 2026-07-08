import os
import io
import sys
import shutil
import socket
import threading
import logging
import subprocess
import time
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.effects import speedup
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("kobold-qwen-api")

app = FastAPI(
    title="Kobold Qwen TTS API Wrapper", 
    description="OpenAI/XTTS compatible FastAPI wrapper for KoboldCpp running Qwen3-TTS"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(PROJECT_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Shared server state
class ServerState:
    def __init__(self):
        self.kobold_port = None
        self.kobold_process = None
        self.kobold_log_handle = None
        self.backend = os.environ.get("KOBOLD_QWEN_BACKEND", "cpu")
        self.threads = os.environ.get("KOBOLD_QWEN_THREADS", None)
        self.active_model = "base"

state = ServerState()


def get_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def apply_speed(sound: AudioSegment, speed: float) -> AudioSegment:
    if speed <= 0 or abs(speed - 1.0) < 0.001:
        return sound
    if speed > 1.0:
        return speedup(sound, playback_speed=speed)

    slowed = sound._spawn(
        sound.raw_data,
        overrides={"frame_rate": max(1, int(sound.frame_rate * speed))},
    )
    return slowed.set_frame_rate(sound.frame_rate)


def start_koboldcpp(model_type=None):
    if model_type is None:
        model_type = state.active_model
    else:
        state.active_model = model_type

    state.kobold_port = get_free_port()
    binary_name = "koboldcpp.exe" if os.name == "nt" else "koboldcpp"
    binary_path = os.path.join(PROJECT_DIR, "bin", binary_name)

    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"KoboldCpp binary not found at: {binary_path}")

    model_filename = (
        "Qwen3-TTS-12Hz-1.7B-Base-q8_0.gguf"
        if model_type == "base"
        else "Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf"
    )
    model_path = os.path.join(PROJECT_DIR, "models", model_filename)

    # Build command line arguments
    command = [
        binary_path,
        "--nomodel",
        "--ttsmodel", model_path,
        "--ttswavtokenizer", os.path.join(PROJECT_DIR, "models", "qwen3-tts-tokenizer-f16.gguf"),
        "--ttsdir", VOICES_DIR,
        "--port", str(state.kobold_port),
        "--quiet"
    ]

    # Add backend-specific flags
    backend = state.backend.lower()
    if backend == "cuda":
        command.append("--ttsgpu")
    elif backend == "vulkan":
        command.extend(["--usevulkan", "0", "--ttsgpu"])
    elif backend == "metal":
        command.append("--ttsgpu")
    else:  # cpu
        command.append("--usecpu")
        if state.threads:
            command.extend(["--ttsthreads", str(state.threads)])
        else:
            try:
                import psutil
                cores = psutil.cpu_count(logical=False) or 4
                command.extend(["--ttsthreads", str(max(1, cores))])
            except ImportError:
                command.extend(["--ttsthreads", "4"])

    log_file = os.path.join(PROJECT_DIR, "koboldcpp.log")
    logger.info(f"Launching KoboldCpp on port {state.kobold_port} using command: {' '.join(command)}")
    logger.info(f"KoboldCpp subprocess logs redirected to {log_file}")

    log_handle = open(log_file, "a", encoding="utf-8")
    state.kobold_log_handle = log_handle
    
    # Hidden console on Windows to run cleanly in background
    kwargs = {}
    if os.name == "nt":
        # CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = 0x08000000

    try:
        state.kobold_process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_DIR,
            **kwargs
        )
    except Exception:
        log_handle.close()
        state.kobold_log_handle = None
        raise


def stop_koboldcpp():
    if state.kobold_process:
        logger.info(f"Terminating KoboldCpp subprocess (PID: {state.kobold_process.pid})...")
        try:
            state.kobold_process.terminate()
            state.kobold_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Subprocess did not terminate in time. Killing it...")
            state.kobold_process.kill()
        except Exception as e:
            logger.error(f"Error terminating KoboldCpp process: {e}")
        state.kobold_process = None
    if state.kobold_log_handle:
        state.kobold_log_handle.close()
        state.kobold_log_handle = None


def is_koboldcpp_online(timeout=60):
    url = f"http://127.0.0.1:{state.kobold_port}/api/extra/speakers_list"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                logger.info("KoboldCpp backend is online!")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


@app.on_event("startup")
async def startup_event():
    start_koboldcpp()
    if not is_koboldcpp_online():
        stop_koboldcpp()
        raise RuntimeError("Failed to verify that KoboldCpp startup succeeded within 60s.")


@app.on_event("shutdown")
async def shutdown_event():
    stop_koboldcpp()


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    language: Optional[str] = "en"
    speed: Optional[float] = 1.0
    temperature: Optional[float] = 0.8
    response_format: Optional[str] = "wav"


@app.get("/health")
@app.get("/")
async def health_check():
    kobold_online = False
    if state.kobold_port:
        try:
            resp = requests.get(f"http://127.0.0.1:{state.kobold_port}/api/extra/speakers_list", timeout=1)
            kobold_online = (resp.status_code == 200)
        except requests.RequestException:
            pass
            
    return {
        "status": "ok",
        "backend": state.backend,
        "kobold_online": kobold_online,
        "voices_count": len([name for name in os.listdir(VOICES_DIR) if name.endswith(".wav")]),
    }


# Model type constants
MODEL_BASE = "base"
MODEL_CUSTOMVOICE = "customvoice"
MODEL_ID_BASE = "qwen3-tts-base"
MODEL_ID_CUSTOMVOICE = "qwen3-tts-customvoice"
# Legacy alias — treat bare "qwen3-tts" as base
MODEL_ID_LEGACY = "qwen3-tts"


@app.get("/v1/models")
@app.get("/v1/audio/models")
async def list_models():
    return {
        "data": [
            {
                "id": MODEL_ID_BASE,
                "object": "model",
                "owned_by": "alibaba",
                "description": "Qwen3-TTS Base model — voice cloning from a WAV reference file",
            },
            {
                "id": MODEL_ID_CUSTOMVOICE,
                "object": "model",
                "owned_by": "alibaba",
                "description": "Qwen3-TTS CustomVoice model — pre-built named voices (Vivian, Serena, Ryan, …)",
            },
        ]
    }


# Preset voices are baked into the CustomVoice model; stored by canonical name
PRESET_VOICE_DATA = [
    {"id": "Aiden",    "voice_id": "Aiden",    "name": "Aiden"},
    {"id": "Dylan",    "voice_id": "Dylan",    "name": "Dylan"},
    {"id": "Eric",     "voice_id": "Eric",     "name": "Eric"},
    {"id": "Ono_Anna", "voice_id": "Ono_Anna", "name": "Ono_Anna"},
    {"id": "Ryan",     "voice_id": "Ryan",     "name": "Ryan"},
    {"id": "Serena",   "voice_id": "Serena",   "name": "Serena"},
    {"id": "Sohee",    "voice_id": "Sohee",    "name": "Sohee"},
    {"id": "Uncle_Fu", "voice_id": "Uncle_Fu", "name": "Uncle_Fu"},
    {"id": "Vivian",   "voice_id": "Vivian",   "name": "Vivian"},
]
# Fast lookup set (lowercase) for heuristic matching
PRESET_NAMES_LOWER = {p["name"].lower() for p in PRESET_VOICE_DATA}


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
@app.get("/v1/files")
async def list_voices():
    voices = []
    # User-uploaded cloning references — served by the Base model
    for name in os.listdir(VOICES_DIR):
        if name.endswith(".wav"):
            voice_id = os.path.splitext(name)[0]
            voices.append({
                "id": voice_id,
                "voice_id": voice_id,
                "name": voice_id,
                "type": "cloned",
                "model": MODEL_ID_BASE,
            })
    # Pre-built voices — served by the CustomVoice model
    for preset in PRESET_VOICE_DATA:
        voices.append({
            **preset,
            "type": "preset",
            "model": MODEL_ID_CUSTOMVOICE,
        })
    return {"data": voices}


@app.post("/v1/audio/voices")
@app.post("/v1/voices")
@app.post("/v1/files")
async def upload_voice(
    files: Optional[List[UploadFile]] = File(None),
    audio_sample: Optional[UploadFile] = File(None),
    voice_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    purpose: Optional[str] = Form(None)
):
    resolved_id = voice_id or name
    target_file = None

    if audio_sample:
        target_file = audio_sample
    elif files and len(files) > 0:
        target_file = files[0]

    if not target_file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    if not resolved_id:
        filename = target_file.filename
        resolved_id = os.path.splitext(filename)[0]

    # Sanitize voice ID
    resolved_id = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in resolved_id])
    target_path = os.path.join(VOICES_DIR, f"{resolved_id}.wav")
    temp_path = target_path + ".tmp"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(target_file.file, buffer)

        # Convert to PCM WAV using pydub
        try:
            sound = AudioSegment.from_file(temp_path)
            sound.export(target_path, format="wav")
            os.remove(temp_path)
            logger.info(f"Successfully uploaded and converted voice: {resolved_id}")
        except Exception as e:
            logger.warning(f"Pydub conversion failed: {e}. Saving file directly.")
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)

        # Restart KoboldCpp so it re-scans VOICES_DIR, preserving whichever model was active.
        # If KoboldCpp is already running we do a graceful restart; if it never started we boot
        # into Base mode (the natural default for voice-cloning use-cases).
        active_before = state.active_model if state.kobold_process else MODEL_BASE
        stop_koboldcpp()
        start_koboldcpp(model_type=active_before)
        if not is_koboldcpp_online():
            raise RuntimeError("Failed to verify that KoboldCpp startup succeeded within 60s.")

        return {
            "id": resolved_id,
            "voice_id": resolved_id,
            "name": resolved_id,
            "purpose": purpose or "user_data"
        }
    except Exception as e:
        logger.error(f"Failed to save voice file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def generate_speech(request: SpeechRequest):
    logger.info(f"Speech request: text_len={len(request.input)}, voice={request.voice}, model={request.model}")

    requested_voice = str(request.voice or "").strip()
    voice_lower = requested_voice.lower()
    requested_model = str(request.model or "").strip().lower()

    # ------------------------------------------------------------------ #
    # Step 1: Determine target model                                       #
    # Priority: explicit model field > voice-name heuristic               #
    # ------------------------------------------------------------------ #
    if requested_model == MODEL_ID_CUSTOMVOICE:
        target_model = MODEL_CUSTOMVOICE
    elif requested_model in (MODEL_ID_BASE, MODEL_ID_LEGACY, ""):
        # Explicit base selection or legacy bare name — still apply heuristic
        # so a preset voice name with model="qwen3-tts" works naturally.
        target_model = MODEL_CUSTOMVOICE if voice_lower in PRESET_NAMES_LOWER else MODEL_BASE
    else:
        # Unknown model value — fall back to voice heuristic
        target_model = MODEL_CUSTOMVOICE if voice_lower in PRESET_NAMES_LOWER else MODEL_BASE

    # ------------------------------------------------------------------ #
    # Step 2: Resolve voice filename                                       #
    # ------------------------------------------------------------------ #
    if target_model == MODEL_CUSTOMVOICE:
        # Resolve canonical preset name (preserves original casing for KoboldCpp)
        preset_name = next(
            (p["name"] for p in PRESET_VOICE_DATA if p["name"].lower() == voice_lower),
            PRESET_VOICE_DATA[0]["name"],
        )
        voice_filename = preset_name
        logger.info(f"Preset voice: '{voice_filename}' — using CustomVoice model")
    else:
        # Base model: use uploaded WAV reference, fall back to koboldcpp default
        voice_filename = "kobo"
        if requested_voice and requested_voice not in ("default", ""):
            voice_raw = requested_voice
            if voice_raw.endswith(".wav"):
                voice_raw = os.path.splitext(voice_raw)[0]
            wav_path = os.path.join(VOICES_DIR, f"{voice_raw}.wav")
            if os.path.exists(wav_path):
                voice_filename = f"{voice_raw}.wav"
                logger.info(f"Cloned voice reference: '{voice_filename}' — using Base model")
            else:
                voice_filename = voice_raw
                logger.warning(
                    f"Voice reference '{voice_raw}.wav' not found in voices/; "
                    f"passing '{voice_raw}' directly to KoboldCpp. Using Base model."
                )

    # ------------------------------------------------------------------ #
    # Step 3: Switch KoboldCpp model if needed                            #
    # ------------------------------------------------------------------ #
    if state.active_model != target_model:
        logger.info(f"Model switch: {state.active_model} → {target_model}")
        stop_koboldcpp()
        start_koboldcpp(model_type=target_model)
        if not is_koboldcpp_online():
            logger.error("KoboldCpp did not come online after model switch.")
            raise HTTPException(status_code=500, detail=f"Failed to start KoboldCpp with model: {target_model}")

    # Call KoboldCpp native TTS API
    url = f"http://127.0.0.1:{state.kobold_port}/api/extra/tts"
    payload = {
        "text": request.input,
        "voice": voice_filename
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            logger.error(f"KoboldCpp generation failed with code {resp.status_code}: {resp.text}")
            raise HTTPException(status_code=500, detail=f"KoboldCpp generation failed: {resp.text}")
        
        audio_content = resp.content
        buffer = io.BytesIO(audio_content)

        # Apply speed adjustment if requested
        if request.speed and request.speed != 1.0:
            try:
                sound = AudioSegment.from_file(buffer)
                sound = apply_speed(sound, request.speed)
                buffer = io.BytesIO()
                sound.export(buffer, format="wav")
                buffer.seek(0)
                logger.info(f"Applied speed adjustment: {request.speed}x")
            except Exception as speed_err:
                logger.error(f"Failed to adjust speed: {speed_err}")
                buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")
    except requests.RequestException as e:
        logger.error(f"Failed to contact KoboldCpp backend: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to contact KoboldCpp backend: {str(e)}")
    except Exception as e:
        logger.error(f"TTS generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")
