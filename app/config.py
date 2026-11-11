"""Paths and user settings for Murmur Notetaker."""

import json
import os
import site
import sys
import threading
from pathlib import Path

# Packaged (PyInstaller) vs. running from source. In the packaged app the read-only resources
# (models, ffmpeg) sit next to the .exe, while everything we write (settings, meetings, the copilot
# library) must go to a per-user, writable folder — Program Files is read-only.
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    APP_DIR = Path(sys.executable).resolve().parent
    MODELS_DIR = APP_DIR / "models"
    BIN_DIR = APP_DIR / "bin"
    DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Murmur" / "data"
else:
    ROOT = Path(__file__).resolve().parent.parent
    APP_DIR = ROOT
    MODELS_DIR = ROOT / "models"
    BIN_DIR = ROOT / "bin"
    DATA_DIR = ROOT / "data"

# Optional overrides (useful to relocate the large models folder, or for testing the packaged build
# without copying gigabytes). Empty / unset keeps the defaults above.
if os.environ.get("MURMUR_MODELS"):
    MODELS_DIR = Path(os.environ["MURMUR_MODELS"])
if os.environ.get("MURMUR_DATA"):
    DATA_DIR = Path(os.environ["MURMUR_DATA"])

FFMPEG = BIN_DIR / "ffmpeg.exe"
MEETINGS_DIR = DATA_DIR / "meetings"
SETTINGS_FILE = DATA_DIR / "settings.json"
TEAM_TERMS_FILE = DATA_DIR / "team_terms.json"


def load_team_terms() -> dict:
    """Private, team-specific vocabulary (kept in data/, never committed):
    {"heard_as": {misheard: right}, "glossary": {German: English}, "polish_context": "The team builds ..."}"""
    try:
        data = json.loads(TEAM_TERMS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

WHISPER_DIR = MODELS_DIR / "faster-whisper"
PARAKEET_DIR = MODELS_DIR / "parakeet-tdt-0.6b-v3-onnx"
DIARIZATION_DIR = MODELS_DIR / "diarization"
NLLB_DIR = MODELS_DIR / "nllb-200-distilled-1.3B-ct2-int8"   # local translation (German -> English ...)
SILERO_VAD = MODELS_DIR / "silero_vad.onnx"

SAMPLE_RATE = 16000      # speech recognition and diarization
PLAYBACK_RATE = 48000    # what you listen to: devices are captured at this rate
HOST = "127.0.0.1"
PORT = 8765

DEFAULTS = {
    # "whisper" (faster-whisper, AZ/DE/EN) or "parakeet" (DE/EN only, fast on CPU)
    "engine": "whisper",
    "whisper_model": "large-v3-turbo",
    # "auto" -> CUDA if available, else CPU
    "device": "auto",
    # "auto" picks the best of the allowed languages per segment; or a fixed code
    "language_mode": "auto",
    "allowed_languages": ["az", "de", "en"],
    "mic_device": "",
    "speaker_device": "",
    # Summary + sentiment: "claude" or "openai"; both keys can be stored, the chosen one is used.
    "summary_provider": "claude",
    "anthropic_api_key": "",
    "anthropic_workspace_id": "",
    "openai_api_key": "",
    "openai_model": "",
    "nvidia_api_key": "",
    "nvidia_model": "",
    # "auto" writes the summary in the language most of the meeting was spoken in
    "summary_language": "auto",
    "auto_summary": False,
    # After a live recording, transcribe the saved audio again with slower, more accurate settings.
    "refine_after_recording": True,
    # While recording: every sentence appears at once, then the AI provider checks it a moment later (with the lines
    # before it as context) and fixes misheard terms and names in place, like the summary does after the meeting.
    "live_polish": True,
    # In a call the microphone is you: your lines get your name, the other people are told apart by
    # voice on the system channel, and anything on the mic that is not your voice (the others leaking
    # from loudspeakers, text invented on noise) is dropped. In-person meetings are detected (nothing
    # on the system channel) and then every voice on the mic is separated.
    "mic_is_me": True,
    "my_name": "",                 # shown for your lines; empty = "Me"
    "copilot_defaults": {},     # Murmur Copilot: last used session preferences (language, model, answer style, instructions)
    # Names, companies, products, jargon: recognition prefers these spellings (comma separated)
    "vocabulary": "",
    "ui_language": "en",
    # Translation next to the transcript, automatic: every sentence that is not in `translate_to` gets one.
    # "local" = NLLB model on this machine (nothing is sent anywhere), "ai" = your AI provider (text only)
    "translate_live": True,
    "translate_to": "en",
    "translate_also": "",          # optional second translation column, e.g. "az" next to "en"
    "translate_refine": True,      # local engine: re-check non-English lines through English, then pick the better one
    "translate_engine": "local",
    "translate_glossary": "",      # extra terms: "Termin=appointment, Rechnung=invoice" (team terms can also go in data/team_terms.json)
}

_lock = threading.Lock()


BACKUP_FILE = DATA_DIR / "settings.backup.json"


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_settings() -> dict:
    """Settings (including API keys) live in data/settings.json and survive restarts. If that
    file is ever damaged, the backup written next to it on every save is used instead."""
    data = _read(SETTINGS_FILE) if SETTINGS_FILE.exists() else {}
    if data is None:
        data = _read(BACKUP_FILE) or {}
    return {**DEFAULTS, **data}


def save_settings(updates: dict) -> dict:
    with _lock:
        current = load_settings()
        for key, value in updates.items():
            if key in DEFAULTS:
                current[key] = value
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        text = json.dumps(current, indent=2, ensure_ascii=False)
        # atomic: write a temp file, then swap it in, so a crash can never leave a half-written file
        tmp = SETTINGS_FILE.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(SETTINGS_FILE)
        BACKUP_FILE.write_text(text, encoding="utf-8")
        return current


def add_nvidia_dll_dirs() -> None:
    """Make pip-installed CUDA libraries (cuBLAS, cuDNN) visible to CTranslate2, where they exist.
    The packaged app ships CPU-only; this then finds nothing and CTranslate2 runs on the CPU."""
    roots = []
    try:
        roots += [Path(sp) / "nvidia" for sp in site.getsitepackages()]
    except Exception:
        pass
    if FROZEN:
        roots += [Path(getattr(sys, "_MEIPASS", APP_DIR)) / "nvidia", APP_DIR / "nvidia"]
    for base in roots:
        if not base.exists():
            continue
        for bin_dir in base.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass
