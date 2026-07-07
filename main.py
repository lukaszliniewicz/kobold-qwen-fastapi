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


def start_koboldcpp():
    state.kobold_port = get_free_port()
    binary_name = "koboldcpp.exe" if os.name == "nt" else "koboldcpp"
    binary_path = os.path.join(PROJECT_DIR, "bin", binary_name)

    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"KoboldCpp binary not found at: {binary_path}")

    # Build command line arguments
    command = [
        binary_path,
        "--nomodel",
        "--ttsmodel", os.path.join(PROJECT_DIR, "models", "Qwen3-TTS-12Hz-1.7B-Base-q8_0.gguf"),
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


@app.get("/v1/models")
@app.get("/v1/audio/models")
async def list_models():
    return {
        "data": [
            {"id": "qwen3-tts", "object": "model", "owned_by": "alibaba"}
        ]
    }


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
@app.get("/v1/files")
async def list_voices():
    voices = []
    for name in os.listdir(VOICES_DIR):
        if name.endswith(".wav"):
            voice_id = os.path.splitext(name)[0]
            voices.append({
                "id": voice_id,
                "voice_id": voice_id,
                "name": voice_id
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

        # Restart KoboldCpp to force it to re-scan the VOICES_DIR directory
        stop_koboldcpp()
        start_koboldcpp()
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
    logger.info(f"Speech request: text_len={len(request.input)}, voice={request.voice}")

    # Resolve reference voice filename in voices dir
    voice_filename = "kobo"
    if request.voice and request.voice != "default":
        # Handle cases where the request voice contains the extension
        voice_raw = request.voice
        if voice_raw.endswith(".wav"):
            voice_raw = os.path.splitext(voice_raw)[0]
        
        if os.path.exists(os.path.join(VOICES_DIR, f"{voice_raw}.wav")):
            voice_filename = f"{voice_raw}.wav"
            logger.info(f"Using cloned voice reference file: {voice_filename}")
        else:
            voice_filename = voice_raw
            logger.warning(f"Voice reference file '{voice_raw}.wav' not found, passing voice parameter directly.")

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
