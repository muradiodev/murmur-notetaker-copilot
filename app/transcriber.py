"""Speech-to-text engines and the single GPU worker that runs them."""

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import PARAKEET_DIR, SAMPLE_RATE, WHISPER_DIR, add_nvidia_dll_dirs, load_team_terms

log = logging.getLogger(__name__)

# Phrases Whisper tends to invent on silence or noise.
HALLUCINATIONS = re.compile(
    r"^(thanks for watching|thank you for watching|subtitles by|untertitel|vielen dank fürs zuschauen|"
    r"altyazı|izlədiyiniz üçün|abunə olun|\.+|you)[\s!.]*$",
    re.IGNORECASE,
)


# Words the recognizer keeps mishearing. The team's own words come from data/team_terms.json. Whole words, case-insensitive. Ordinary words
# that are only sometimes wrong ("Cloud" for "Claude", "Cash" for "Cache") are NOT here, they are handled
# by the vocabulary hint in Settings instead.
HEARD_AS = {
    "koffein": "Caffeine",
    "kaffein": "Caffeine",
    "kaffeine": "Caffeine",
    "koffein-cache": "Caffeine-Cache",
    "intellijay": "IntelliJ",
    "intelli-j": "IntelliJ",
    "hafke": "Kafka",
    "kafke": "Kafka",
    "lock-pair": "Log-Pair",
    "lockzeile": "Logzeile",
    "login-framework": "Logging-Framework",
    **load_team_terms().get("heard_as", {}),   # team words, data/team_terms.json
}
_TERM = re.compile(r"[\w\-]+", re.UNICODE)


def collapse_loops(text: str) -> str:
    """Whisper sometimes gets stuck inside a word: "Intententententent", "Routing-Routing-Routing-...",
    "kafkafkafkafka". Each such run becomes one copy ("Intent", "Routing", "kafka")."""
    def fix(match: re.Match) -> str:
        parts = match.group(0).split("-")
        out = [parts[0]]
        for part in parts[1:]:
            if part.lower() != out[-1].lower():
                out.append(part)
        return re.sub(r"(\w{2,8}?)\1{3,}", r"\1", "-".join(out))
    return re.sub(r"\S+", fix, text)


def fix_terms(text: str) -> str:
    return _TERM.sub(lambda m: HEARD_AS.get(m.group(0).lower(), m.group(0)), text)


def clean_text(text: str) -> str:
    """Loops collapsed, team vocabulary fixed. Safe to run twice."""
    return fix_terms(collapse_loops(text))


# Meetily's clean_repetitive_text, ported: Whisper sometimes loops ("the the the", "I mean I mean").
_REPEAT_PHRASES = ("thank you for watching", "thanks for watching", "like and subscribe", "music playing",
                   "applause", "laughter", "um um um", "uh uh uh", "ah ah ah")


def clean_repetitive_text(text: str) -> str:
    text = collapse_loops(text)
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
            text = fix_terms(clean_repetitive_text(s.text.strip()))
            if not text or s.no_speech_prob > 0.6 or s.avg_logprob < -1.2 or HALLUCINATIONS.match(text):
                continue
            if s.words and len(s.words) <= 3 and sum(w.probability for w in s.words) / len(s.words) < 0.45:
                continue  # a few low-confidence words on noise: typical Whisper hallucination
            words = [(w.start, w.end, clean_text(w.word.strip())) for w in (s.words or []) if w.word.strip()]
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
        text = fix_terms(clean_repetitive_text(str(self.model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()))
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
    group: str = ""            # the meeting this job belongs to: wait_group() waits for exactly these jobs


class EngineManager:
    """Owns the loaded model and processes jobs one at a time on a worker thread.

    Two lanes share the one GPU worker. The LIVE lane (the sentence being spoken right now in a
    recording) always goes first. The BATCH lane (refine after a meeting, "transcribe again", file
    import) only runs when the live lane is empty AND no recording is running (`batch_paused`).
    An old meeting that is still being processed can therefore never starve a new recording; its
    work simply continues, in order, after the recording ends.
    """

    def __init__(self, on_status: Callable[[dict], None]):
        self.on_status = on_status
        self.engine = None
        self.state = "idle"
        self.error = ""
        self.settings: dict = {}
        self.last_lang: dict[str, str] = {}
        self._load_lock = threading.Lock()
        self._cv = threading.Condition()
        self._live: "deque[Job]" = deque()
        self._batch: "deque[Job]" = deque()
        self._running: Job | None = None
        self._pending: dict[str, int] = {}     # group -> jobs queued or running
        self._batch_paused = False
        threading.Thread(target=self._worker, daemon=True, name="stt-worker").start()

    def status(self) -> dict:
        with self._cv:
            queued = len(self._live) + len(self._batch)
            paused = self._batch_paused and bool(self._batch)
        return {
            "state": self.state,
            "error": self.error,
            "engine": getattr(self.engine, "name", None),
            "device": getattr(self.engine, "device", None),
            "queue": queued,
            "paused": paused,
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
        with self._cv:
            (self._batch if job.context.get("batch") else self._live).append(job)
            self._pending[job.group] = self._pending.get(job.group, 0) + 1
            self._cv.notify_all()

    def set_batch_paused(self, paused: bool) -> None:
        """True while a recording runs: batch jobs wait (the one already running finishes first)."""
        with self._cv:
            self._batch_paused = paused
            self._cv.notify_all()
        self.on_status(self.status())

    def wait_group(self, group: str) -> None:
        """Block until every job submitted for this meeting has finished (other meetings do not matter)."""
        with self._cv:
            while self._pending.get(group, 0) > 0:
                self._cv.wait(timeout=1.0)

    def idle(self) -> bool:
        """Nothing live left to do: only then are interim (still being spoken) captions worth computing."""
        with self._cv:
            return not self._live and (self._running is None or bool(self._running.context.get("batch")))

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

    def _next(self) -> Job:
        with self._cv:
            while True:
                if self._live:
                    job = self._live.popleft()
                elif self._batch and not self._batch_paused:
                    job = self._batch.popleft()
                else:
                    self._cv.wait()
                    continue
                self._running = job
                return job

    def _worker(self) -> None:
        while True:
            job = self._next()
            try:
                while self.engine is None and self.state in ("idle", "loading"):
                    threading.Event().wait(0.5)
                if self.engine is None:
                    continue
                interim = job.context.get("interim", False)
                if interim:
                    with self._cv:
                        final_waiting = any(not j.context.get("interim") for j in self._live)
                    if final_waiting:
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
                with self._cv:
                    self._running = None
                    left = self._pending.get(job.group, 1) - 1
                    if left > 0:
                        self._pending[job.group] = left
                    else:
                        self._pending.pop(job.group, None)
                    self._cv.notify_all()
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
