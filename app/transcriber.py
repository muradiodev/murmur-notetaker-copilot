"""Speech-to-text engines and the single GPU worker that runs them."""

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import PARAKEET_DIR, SAMPLE_RATE, WHISPER_DIR, add_nvidia_dll_dirs

log = logging.getLogger(__name__)

# Phrases Whisper tends to invent on silence or noise.
HALLUCINATIONS = re.compile(
    r"^(thanks for watching|thank you for watching|subtitles by|untertitel|vielen dank fürs zuschauen|"
    r"altyazı|izlədiyiniz üçün|abunə olun|\.+|you)[\s!.]*$",
    re.IGNORECASE,
)


# Meetily's clean_repetitive_text, ported: Whisper sometimes loops ("the the the", "I mean I mean").
_REPEAT_PHRASES = ("thank you for watching", "thanks for watching", "like and subscribe", "music playing",
                   "applause", "laughter", "um um um", "uh uh uh", "ah ah ah")


def clean_repetitive_text(text: str) -> str:
    low = text.lower()
    if any(p in low for p in _REPEAT_PHRASES):
        return ""
    if len(text) > 10 and len(set(text.replace(" ", ""))) <= 3:
        return ""
    words = text.split()
    if len(words) >= 3:  # collapse consecutive duplicate words
        out = [words[0]]
        for w in words[1:]:
            if w.lower().strip(".,!?") != out[-1].lower().strip(".,!?"):
                out.append(w)
        words = out
    for n in range(5, 1, -1):  # collapse consecutive repeated 2-5 word phrases
        i, out = 0, []
        while i < len(words):
            chunk = [w.lower() for w in words[i:i + n]]
            out.extend(words[i:i + n])
            j = i + n
            while len(chunk) == n and [w.lower() for w in words[j:j + n]] == chunk:
                j += n
            i = j
        words = out
    original = text.split()
    if len(original) >= 4 and 1 - len(words) / len(original) > 0.7:
        return ""
    return " ".join(words)


@dataclass
class Piece:
    start: float
    end: float
    text: str
    lang: str | None
    words: list = field(default_factory=list)  # [(start, end, word)] relative to the clip
    logprob: float = 0.0       # Whisper's average log-probability for the line
    no_speech: float = 0.0     # Whisper's probability that the clip is not speech


class WhisperEngine:
    name = "whisper"

    def __init__(self, model_name: str, device: str):
        add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        local = WHISPER_DIR / model_name
        model_path = str(local) if (local / "model.bin").exists() else model_name
        attempts = [("cuda", "int8_float16"), ("cuda", "float16"), ("cpu", "int8")]
        if device == "cpu":
            attempts = [("cpu", "int8")]
        last_error = None
        for dev, compute in attempts:
            try:
                self.model = WhisperModel(model_path, device=dev, compute_type=compute,
                                          download_root=str(WHISPER_DIR))
                # Run once so CUDA/cuDNN problems surface now, not mid-meeting.
                self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")
                self.device = f"{dev}/{compute}"
                log.info("Whisper %s loaded on %s", model_name, self.device)
                return
            except Exception as exc:
                last_error = exc
                log.warning("Whisper on %s/%s failed: %s", dev, compute, exc)
        raise RuntimeError(f"Could not load the Whisper model: {last_error}")

    def pick_language(self, audio: np.ndarray, allowed: list[str]) -> tuple[str | None, float]:
        try:
            _, _, probs = self.model.detect_language(audio)
        except Exception:
            return None, 0.0
        ranked = [(lang, p) for lang, p in probs if not allowed or lang in allowed]
        if not ranked:
            return None, 0.0
        lang, p = max(ranked, key=lambda x: x[1])
        total = sum(q for _, q in ranked) or 1.0
        return lang, p / total  # confidence among the allowed languages only

    def transcribe(self, audio: np.ndarray, language: str | None, quality: bool = False, fast: bool = False,
                   hotwords: str | None = None) -> list[Piece]:
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1 if fast else 5 if quality else 3,
            vad_filter=not fast,
            condition_on_previous_text=False,
            word_timestamps=not fast,
            hotwords=hotwords or None,                      # names and terms (OpenWhispr's custom dictionary)
            hallucination_silence_threshold=2.0 if quality and not fast else None,
        )
        pieces = []
        for s in segments:
            text = clean_repetitive_text(s.text.strip())
            if not text or s.no_speech_prob > 0.6 or s.avg_logprob < -1.2 or HALLUCINATIONS.match(text):
                continue
            if s.words and len(s.words) <= 3 and sum(w.probability for w in s.words) / len(s.words) < 0.45:
                continue  # a few low-confidence words on noise: typical Whisper hallucination
            words = [(w.start, w.end, w.word.strip()) for w in (s.words or []) if w.word.strip()]
            pieces.append(Piece(s.start, s.end, text, language or info.language, words, s.avg_logprob, s.no_speech_prob))
        return pieces


class ParakeetEngine:
    """NVIDIA Parakeet TDT 0.6B v3 (ONNX). German/English, no Azerbaijani."""

    name = "parakeet"

    def __init__(self):
        import onnx_asr

        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(PARAKEET_DIR), quantization="int8")
        self.device = "cpu/int8"

    def pick_language(self, audio, allowed):
        return None, 0.0

    def transcribe(self, audio: np.ndarray, language: str | None, quality: bool = False, fast: bool = False,
                   hotwords: str | None = None) -> list[Piece]:
        text = clean_repetitive_text(str(self.model.recognize(audio, sample_rate=SAMPLE_RATE)).strip())
        if not text:
            return []
        return [Piece(0.0, len(audio) / SAMPLE_RATE, text, None)]


@dataclass
class Job:
    channel: str
    offset: float
    audio: np.ndarray
    callback: Callable[[str, float, list[Piece]], None]
    context: dict = field(default_factory=dict)


class EngineManager:
    """Owns the loaded model and processes jobs one at a time on a worker thread."""

    def __init__(self, on_status: Callable[[dict], None]):
        self.on_status = on_status
        self.engine = None
        self.state = "idle"
        self.error = ""
        self.settings: dict = {}
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.last_lang: dict[str, str] = {}
        self._load_lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="stt-worker").start()

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "engine": getattr(self.engine, "name", None),
            "device": getattr(self.engine, "device", None),
            "queue": self.jobs.qsize(),
        }

    def load_async(self, settings: dict) -> None:
        threading.Thread(target=self.load, args=(settings,), daemon=True, name="stt-load").start()

    def load(self, settings: dict) -> None:
        with self._load_lock:
            self.settings = dict(settings)
            self.state, self.error = "loading", ""
            self.on_status(self.status())
            try:
                self.engine = None
                if settings["engine"] == "parakeet":
                    self.engine = ParakeetEngine()
                else:
                    self.engine = WhisperEngine(settings["whisper_model"], settings["device"])
                self.state = "ready"
            except Exception as exc:
                log.exception("Model load failed")
                self.state, self.error = "error", str(exc)
            self.on_status(self.status())

    def submit(self, job: Job) -> None:
        job.context.setdefault("t_submit", time.monotonic())
        self.jobs.put(job)

    def wait_idle(self) -> None:
        self.jobs.join()

    def _language_for(self, job: Job) -> str | None:
        mode = job.context.get("language") or self.settings.get("language_mode", "auto")
        if mode != "auto":
            return mode
        key = job.context.get("lang_key", job.channel)
        if job.context.get("interim") and key in self.last_lang:
            return self.last_lang[key]  # interim text must be fast; the final pass decides properly
        lang, confidence = self.engine.pick_language(job.audio, self.settings.get("allowed_languages", []))
        # Detection on very short clips is unreliable: keep it only when it is confident,
        # otherwise reuse this channel's last language (a short "Thank you all" stays English).
        if len(job.audio) < SAMPLE_RATE * 1.5 and key in self.last_lang and confidence < 0.8:
            return self.last_lang[key]
        if lang:
            self.last_lang[key] = lang
        return lang

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                while self.engine is None and self.state in ("idle", "loading"):
                    threading.Event().wait(0.5)
                if self.engine is None:
                    continue
                interim = job.context.get("interim", False)
                if interim and any(not j.context.get("interim") for j in list(self.jobs.queue)):
                    job.callback(job.channel, job.offset, [])   # a finished sentence waits: it goes first
                    continue
                t_start = time.monotonic()
                pieces = self.engine.transcribe(job.audio, self._language_for(job),
                                                job.context.get("quality", False),
                                                fast=interim or job.context.get("fast", False),
                                                hotwords=self.hotwords(job))
                if not interim and job.context.get("fast"):
                    log.info("STT %s: waited %.2fs, ran %.2fs for %.1fs of audio", job.channel,
                             t_start - job.context.get("t_submit", t_start), time.monotonic() - t_start,
                             len(job.audio) / SAMPLE_RATE)
                job.callback(job.channel, job.offset, pieces)
            except Exception:
                log.exception("Transcription job failed")
                try:
                    job.callback(job.channel, job.offset, [])   # let the caller clear its "waiting" state
                except Exception:
                    pass
            finally:
                self.jobs.task_done()
                if not job.context.get("interim"):
                    self.on_status(self.status())

    def hotwords(self, job: Job) -> str:
        """Custom vocabulary from Settings plus names known for this meeting."""
        words = [w.strip() for w in str(self.settings.get("vocabulary", "")).replace("\n", ",").split(",")]
        words += job.context.get("names", [])
        seen, out = set(), []
        for w in words:
            if w and w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
        return ", ".join(out[:60])

    def idle(self) -> bool:
        return self.jobs.unfinished_tasks == 0
