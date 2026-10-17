"""Download the models Murmur Notetaker needs into ./models (run once; safe to re-run)."""

import sys
import tarfile
import urllib.request

from .config import DIARIZATION_DIR, PARAKEET_DIR, SILERO_VAD, WHISPER_DIR, load_settings

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
# Parakeet TDT 0.6B v3 (ONNX, by istupakov). We only pull the int8 files — the fp32 weights add a
# 2.4 GB external-data file the int8 engine never touches.
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
PARAKEET_FILES = ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]
SEGMENTATION_URL = f"{SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SILERO_URL = f"{SHERPA}/asr-models/silero_vad.onnx"
# DTLN-aec ONNX conversions (MIT-licensed weights by Westhausen & Meyer), as shipped by anarlog
DTLN_URL = "https://raw.githubusercontent.com/fastrepl/anarlog/main/crates/aec/data/models/model_128_{part}.onnx"
EMBEDDING_URL = f"{SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx"


def fetch(url: str, dest) -> None:
    print(f"  ↓ {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def whisper(model_name: str) -> None:
    from faster_whisper.utils import download_model

    target = WHISPER_DIR / model_name
    if whisper_present(model_name):
        print(f"✓ Whisper {model_name} already present")
        return
    print(f"Downloading Whisper {model_name} (~1.6 GB)…")
    download_model(model_name, output_dir=str(target))
    print(f"✓ Whisper {model_name}")


def whisper_present(model_name: str) -> bool:
    return (WHISPER_DIR / model_name / "model.bin").exists()


def parakeet_present() -> bool:
    return (PARAKEET_DIR / "encoder-model.int8.onnx").exists() and \
        (PARAKEET_DIR / "decoder_joint-model.int8.onnx").exists()


def parakeet() -> None:
    from huggingface_hub import snapshot_download

    if parakeet_present():
        print("✓ Parakeet already present")
        return
    PARAKEET_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading Parakeet TDT 0.6B v3 int8 (~0.7 GB)…")
    snapshot_download(PARAKEET_REPO, local_dir=str(PARAKEET_DIR), allow_patterns=PARAKEET_FILES)
    print("✓ Parakeet")


def diarization() -> None:
    DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
    seg_dir = DIARIZATION_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
    if not (seg_dir / "model.onnx").exists():
        print("Downloading speaker segmentation model…")
        archive = DIARIZATION_DIR / "segmentation.tar.bz2"
        fetch(SEGMENTATION_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(DIARIZATION_DIR, filter="data")
        archive.unlink()
    print("✓ Segmentation model")
    emb = DIARIZATION_DIR / "wespeaker_en_voxceleb_resnet34.onnx"
    if not emb.exists():
        print("Downloading speaker embedding model…")
        fetch(EMBEDDING_URL, emb)
    print("✓ Speaker embedding model")


if __name__ == "__main__":
    settings = load_settings()
    models = sys.argv[1:] or [settings["whisper_model"]]
    for name in models:
        whisper(name)
    if settings.get("engine") == "parakeet" or "parakeet" in models:
        parakeet()
    diarization()
    if not SILERO_VAD.exists():
        print("Downloading Silero VAD…")
        fetch(SILERO_URL, SILERO_VAD)
    print("✓ Silero VAD")
    from .dtln import AEC_DIR
    AEC_DIR.mkdir(parents=True, exist_ok=True)
    for part in (1, 2):
        dest = AEC_DIR / f"dtln_aec_128_{part}.onnx"
        if not dest.exists():
            print("Downloading DTLN-aec…")
            fetch(DTLN_URL.format(part=part), dest)
    print("✓ DTLN-aec (echo + noise)")
    print("\nDone. Start the app with run.bat.")
