"""Murmur Notetaker local web server: recording, live transcript over WebSocket, meetings API."""

import asyncio
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import aec, copilot, diarize, dtln, livemic, localmt, names, polish, setup_models, storage, summary, translate
from .audio import Capture, list_devices, make_segmenter
import soxr

from .config import (DATA_DIR, FFMPEG, HOST, PORT, PLAYBACK_RATE, PARAKEET_DIR, SAMPLE_RATE, WHISPER_DIR,
                     load_settings, save_settings)
from .diarize import ME
from .transcriber import EngineManager, Job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("murmur")

STATIC = Path(__file__).parent / "static"
CHANNELS = ("mic", "system")

app = FastAPI(title="Murmur Notetaker")


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """The UI files change with every update: the browser must check them instead of using an old copy."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------------------------------------------------------- broadcasting

class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def send(self, message: dict) -> None:
        """Thread-safe broadcast to every connected browser tab."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._send(message), self.loop)

    async def _send(self, message: dict) -> None:
        data = json.dumps(message, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                self.clients.discard(ws)


hub = Hub()
engine = EngineManager(on_status=lambda s: hub.send({"type": "engine", **s}))


class PostQueue:
    """Everything that happens to a meeting AFTER it was recorded (clean-up, refine, speaker separation,
    file import, "transcribe again", crash recovery) runs here: one meeting at a time, in order.

    While a recording is running the queue waits (and the engine pauses its batch lane), so a new live
    meeting always has the machine to itself. When the recording ends, the waiting meetings continue
    one after the other. Long jobs call `checkpoint()` between their steps to honour the pause.
    """

    def __init__(self):
        self._q: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._free = threading.Event()      # set = no recording is running
        self._free.set()
        self._busy = False
        threading.Thread(target=self._run, daemon=True, name="post-processing").start()

    def recording(self, active: bool) -> None:
        (self._free.clear if active else self._free.set)()
        engine.set_batch_paused(active)

    def submit(self, meeting_id: str, fn) -> None:
        waiting = not self._free.is_set() or self._busy or not self._q.empty()
        self._q.put((meeting_id, fn))
        if waiting:
            hub.send({"type": "meeting", "meeting_id": meeting_id, "status": "processing", "step": "queued"})

    def checkpoint(self, meeting_id: str) -> None:
        """Block while a recording runs; the meeting shows "waiting" and then carries on by itself."""
        if self._free.is_set():
            return
        hub.send({"type": "meeting", "meeting_id": meeting_id, "status": "processing", "step": "queued"})
        self._free.wait()
        hub.send({"type": "meeting", "meeting_id": meeting_id, "status": "processing", "step": "finishing"})

    def _run(self) -> None:
        while True:
            meeting_id, fn = self._q.get()
            self._busy = True
            try:
                self.checkpoint(meeting_id)
                fn()
            except Exception:
                log.exception("Post-processing of %s failed", meeting_id)
            finally:
                self._busy = False
                self._q.task_done()


postq = PostQueue()


def translation_targets(primary: str, also: str) -> list[tuple[str, str]]:
    """[(language, segment field)]: the main translation column and the optional second one."""
    out = [(primary, "translation")] if primary else []
    if also and also != primary:
        out.append((also, "translation2"))
    return out


def _translation_field(meeting: dict, target: str | None) -> str:
    also = meeting.get("translate_also")
    if target and also and target == also and target != meeting.get("translate_to"):
        return "translation2"
    return "translation"


_translations_lock = threading.Lock()


def apply_translations(meeting_id: str, items: dict[str, str], target: str | None = None,
                       refined: bool = False) -> None:
    """Store translated lines on their segments (live meeting or saved one) and tell the browser.
    Live, a fast direct line is checked again through English afterwards (`refiner`)."""
    live = session
    if live and live.meeting["id"] == meeting_id:
        with live.lock:
            field = _translation_field(live.meeting, target)
            recheck = []
            for seg in live.meeting["segments"]:
                if seg["id"] in items:
                    seg[field] = items[seg["id"]]
                    if not refined and target:
                        recheck.append((seg["id"], seg["text"], seg.get("lang"), items[seg["id"]]))
            storage.save(live.meeting)
        for seg_id, text, lang, shown in recheck:
            refiner.submit(meeting_id, seg_id, text, target, lang, shown)
    else:
        with _translations_lock:          # both columns of one meeting must not overwrite each other
            meeting = storage.load(meeting_id)
            if not meeting:
                return
            field = _translation_field(meeting, target)
            for seg in meeting["segments"]:
                if seg["id"] in items:
                    seg[field] = items[seg["id"]]
            storage.save(meeting)
    hub.send({"type": "translation", "meeting_id": meeting_id, "items": items, "field": field})


refiner = translate.Refiner(load_settings,
                            lambda mid, items, target: apply_translations(mid, items, target, refined=True))

translator = translate.LiveTranslator(
    load_settings, apply_translations,
    lambda message: hub.send({"type": "translation_error", "message": message}))


def apply_edits(meeting_id: str, items: dict[str, str]) -> None:
    """The AI line editor fixed some live lines: replace them in place. The wording that was heard stays
    in raw_text (hover to see it), the same way the summary's corrections after the meeting are shown."""
    live = session
    if not live or live.meeting["id"] != meeting_id:
        return
    changed = []
    with live.lock:
        for seg in live.meeting["segments"]:
            new = items.get(seg["id"])
            if new and new != seg["text"]:
                seg.setdefault("raw_text", seg["text"])
                seg["text"], seg["corrected"] = new, True
                changed.append({"id": seg["id"], "text": new, "raw_text": seg["raw_text"], "corrected": True})
        if changed:
            storage.save(live.meeting)
    if not changed:
        return
    hub.send({"type": "segment_update", "meeting_id": meeting_id, "segments": changed})
    cfg = load_settings()                                      # the translation follows the corrected text
    if cfg.get("translate_live"):
        by_id = {s["id"]: s for s in live.meeting["segments"]}
        targets = translation_targets(live.meeting.get("translate_to") or cfg.get("translate_to", "en"),
                                      live.meeting.get("translate_also", ""))
        for c in changed:
            lang = by_id.get(c["id"], {}).get("lang")
            for target, _ in targets:
                if lang != target:
                    translator.submit(meeting_id, c["id"], c["text"], target, lang)


editor = polish.LineEditor(load_settings, apply_edits)


def start_translation(meeting_id: str, target: str | None = None, force: bool = False) -> None:
    """Translate every line of a saved meeting that has no translation yet (after refine, or on demand).
    Without an explicit target both columns are filled: translate_to and translate_also."""
    def work():
        meeting = storage.load(meeting_id)
        if not meeting or not meeting["segments"]:
            return
        settings = load_settings()
        tgt = target or meeting.get("translate_to") or settings.get("translate_to", "az")
        targets = [(tgt, "translation")] if target else translation_targets(tgt, meeting.get("translate_also", ""))
        hub.send({"type": "translation_state", "meeting_id": meeting_id, "state": "running"})
        try:
            for lang, field in targets:
                translate.translate_segments(settings, meeting["segments"], lang, field=field, force=force,
                                             progress=lambda got, lang=lang: apply_translations(meeting_id, got, lang))
        except Exception as exc:
            log.warning("Translation of %s failed: %s", meeting_id, exc)
            hub.send({"type": "translation_error", "message": str(exc)})
        hub.send({"type": "translation_state", "meeting_id": meeting_id, "state": "done"})

    threading.Thread(target=work, daemon=True, name="translate-meeting").start()


# ---------------------------------------------------------------- recognition model downloads
# The speech-recognition models are large and optional: the app runs with whichever are present and
# lets the user download the rest from Settings, with live progress over the WebSocket. Both engines
# can also fetch themselves on first load, but downloading here keeps that off the recording path.

# Rough on-disk size per Whisper model (int8 model.bin + tokenizer/config), for the progress bar.
WHISPER_SIZE = {"large-v3-turbo": 1_620_000_000, "large-v3": 3_100_000_000,
                "medium": 1_530_000_000, "small": 490_000_000}
PARAKEET_SIZE = 680_000_000

_downloads: dict[str, dict] = {}
_dl_lock = threading.Lock()


def _recognition_catalog() -> list[dict]:
    whisper_model = load_settings().get("whisper_model", "large-v3-turbo")
    return [
        {"key": "whisper", "engine": "whisper", "label": f"faster-whisper {whisper_model}",
         "expected": WHISPER_SIZE.get(whisper_model, 1_620_000_000),
         "dir": WHISPER_DIR / whisper_model,
         "present": lambda m=whisper_model: setup_models.whisper_present(m),
         "download": lambda m=whisper_model: setup_models.whisper(m)},
        {"key": "parakeet", "engine": "parakeet", "label": "Parakeet TDT 0.6B v3 (int8)",
         "expected": PARAKEET_SIZE, "dir": PARAKEET_DIR,
         "present": setup_models.parakeet_present, "download": setup_models.parakeet},
    ]


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _set_download(key: str, state: str, percent: int, message: str = "") -> None:
    _downloads[key] = {"state": state, "percent": percent, "message": message}
    hub.send({"type": "model_download", "key": key, **_downloads[key]})


def _run_download(entry: dict) -> None:
    key, dest, expected = entry["key"], entry["dir"], entry["expected"]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            pct = min(99, int(_dir_bytes(dest) / expected * 100)) if expected else 0
            _set_download(key, "downloading", pct)
            stop.wait(1.0)

    mon = threading.Thread(target=monitor, daemon=True, name=f"dl-mon-{key}")
    mon.start()
    try:
        entry["download"]()
        stop.set()
        _set_download(key, "done", 100)
        # If the engine failed earlier only because this model was missing, load it now.
        if engine.state == "error" and load_settings().get("engine") == entry["engine"] and not session:
            engine.load_async(load_settings())
    except Exception as exc:
        stop.set()
        log.exception("Model download failed: %s", key)
        _set_download(key, "error", 0, str(exc))


# ---------------------------------------------------------------- recording session

class Session:
    def __init__(self, title: str, copilot_cfg: dict | None = None):
        self.settings = load_settings()
        self.meeting = storage.new_meeting(title)
        self.copilot = None
        self.language = None   # a copilot session can fix the call language
        if copilot_cfg is not None:
            self.meeting["mode"] = "copilot"
            self.meeting["copilot"] = {k: v for k, v in copilot_cfg.items() if not k.startswith("_")}
            self.meeting["answers"] = []
            lang = copilot_cfg.get("language", "auto")
            self.language = lang if lang in ("az", "de", "en") else None
            storage.save(self.meeting)
        self.folder = storage.meeting_dir(self.meeting["id"])
        self.lock = threading.RLock()
        self.closed = False
        self.stop_event = threading.Event()
        self.t0 = time.monotonic()
        self.errors: dict[str, str] = {}
        self.writers = {ch: sf.SoundFile(self.folder / f"{ch}.wav", "w", SAMPLE_RATE, 1, "PCM_16") for ch in CHANNELS}
        self.hq_writers = {ch: sf.SoundFile(self.folder / f"{ch}_hq.wav", "w", PLAYBACK_RATE, 1, "PCM_16") for ch in CHANNELS}
        self.resamplers = {ch: soxr.ResampleStream(PLAYBACK_RATE, SAMPLE_RATE, 1, dtype="float32") for ch in CHANNELS}
        self.segmenters = {ch: make_segmenter(ch, self._on_utterance) for ch in CHANNELS}
        devices = {"mic": self.settings["mic_device"], "system": self.settings["speaker_device"]}
        self.captures = {
            ch: Capture(ch, devices[ch], self._enqueue, self._on_error, self.stop_event, self.t0) for ch in CHANNELS
        }
        self.interim_busy = {ch: False for ch in CHANNELS}
        self.system_speaking = False
        self.system_inflight = 0     # interviewer sentences sent to speech-to-text, text not back yet
        # LocalAgreement (WhisperLiveKit / whisper_streaming): words that two consecutive
        # hypotheses agree on are committed and never change; only the tail stays tentative
        self.agreement: dict[str, dict] = {}
        if copilot_cfg is not None:
            self.copilot = copilot.Copilot(self, copilot_cfg, self.settings, hub)
        # live echo cancellation: the mic is cleaned with the loopback as reference before it is
        # segmented and transcribed, so speaker sound picked up by the mic never reaches captions
        self.mic = livemic.MicPath()
        # the microphone is you; your voiceprint (learned while the speakers are silent) tells your
        # words apart from the other participants' voices leaking out of the loudspeakers
        self.mic_is_me = self.settings.get("mic_is_me", True) is not False
        self.me = livemic.Voice(DATA_DIR / "my_voice.npy")
        self.me_sim: dict[str, float] = {}       # mic line id -> similarity to your voice
        self.voices = diarize.VoiceWorker() if diarize.available() else None
        if self.voices:
            threading.Thread(target=self.voices.embed, args=(np.zeros(SAMPLE_RATE, np.float32),),
                             daemon=True).start()   # loads the model in the child before the first line
        self.speakers = diarize.LiveSpeakers()
        self.meeting["speakers"] = {ME: self.my_name()} if self.mic_is_me else {}
        # finished sentences are handled one by one on their own thread (voiceprint, echo check,
        # speaker), so speech-to-text never waits for them
        self.lines: "queue.Queue[tuple]" = queue.Queue()
        self.regroup_due = 0
        self.call_heard = False                  # someone spoke on the system channel: an online call
        # you can mute either source while recording; muted audio is written as silence so the
        # timeline (and screen video sync) never shifts
        self.muted = {"mic": False, "system": False}
        # Capture threads only read the device and enqueue; one processing thread does resampling,
        # echo cancellation and segmentation. A slow moment in processing never blocks the device
        # reads, so Windows never drops audio.
        self.chunks: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue()
        threading.Thread(target=self._process_loop, daemon=True, name="audio-processing").start()
        threading.Thread(target=self._line_loop, daemon=True, name="live-lines").start()
        # start the devices last: their first chunk must find every field above
        for c in self.captures.values():
            c.start()
        threading.Thread(target=self._levels, daemon=True).start()
        threading.Thread(target=self._interim_loop, daemon=True).start()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def my_name(self) -> str:
        return str(self.settings.get("my_name") or "").strip() or "Me"

    # Screen clips: the browser records the shared screen and uploads chunks. Each clip is
    # stamped with the meeting second it started at, so it lines up with the audio.
    def start_screen(self) -> dict:
        with self.lock:
            clips = self.meeting.setdefault("screens", [])
            for c in clips:  # a new share closes any clip the browser forgot to stop
                if c["duration"] is None:
                    c["duration"] = round(self.elapsed - c["offset"], 2)
            n = len(clips) + 1
            clip = {"n": n, "file": f"screen-{n}.webm", "offset": round(self.elapsed, 2), "duration": None}
            clips.append(clip)
            storage.save(self.meeting)
            return clip

    def stop_screen(self, n: int) -> None:
        with self.lock:
            for c in self.meeting.get("screens", []):
                if c["n"] == n and c["duration"] is None:
                    c["duration"] = round(self.elapsed - c["offset"], 2)
            storage.save(self.meeting)

    def switch_device(self, channel: str, device_id: str) -> None:
        """Move one channel to another device without stopping the meeting or shifting its timeline."""
        old = self.captures[channel]
        old.halt.set()
        old.join(timeout=2)
        new = Capture(channel, device_id, self._enqueue, self._on_error, self.stop_event, self.t0, written=old.written)
        self.captures[channel] = new
        new.start()
        hub.send({"type": "device_switched", "channel": channel})

    def set_muted(self, channel: str, muted: bool) -> None:
        self.muted[channel] = muted
        if muted and channel == "system":
            self.interim_busy["system"] = False
        hub.send({"type": "muted", "meeting_id": self.meeting["id"], "muted": dict(self.muted)})

    def _enqueue(self, channel: str, samples: np.ndarray) -> None:
        if self.muted.get(channel):
            samples = np.zeros_like(samples)   # muted: silence keeps the timeline intact
        self.chunks.put((channel, samples))

    def _process_loop(self) -> None:
        while True:
            channel, samples = self.chunks.get()
            try:
                if channel == "__stop__":
                    return
                self._on_chunk(channel, samples)
            except Exception:
                log.exception("Audio processing failed for a %s chunk", channel)
            finally:
                self.chunks.task_done()

    def _on_chunk(self, channel: str, samples: np.ndarray) -> None:
        with self.lock:
            if self.closed:
                return
            self.hq_writers[channel].write(samples)
            speech = self.resamplers[channel].resample_chunk(samples)
            if not len(speech):
                return
            self.writers[channel].write(speech)          # raw track: the offline pass cleans it again, better
            if channel == "system":
                self.segmenters["system"].feed(speech)
                # read by Murmur Copilot: is the other side talking right now (VAD, not captions)
                self.system_speaking = self.segmenters["system"].speaking()
                self.mic.add_system(speech)
            else:
                self.mic.add_mic(speech)
            self._drain_mic()

    def _drain_mic(self, force: bool = False) -> None:
        clean = self.mic.drain(force)
        if len(clean):
            self.segmenters["mic"].feed(clean)

    def _on_error(self, channel: str, message: str) -> None:
        self.errors[channel] = message
        hub.send({"type": "capture_error", "channel": channel, "message": message})

    def _on_utterance(self, channel: str, offset: float, audio: np.ndarray) -> None:
        if channel == "mic" and livemic.mostly_echo(*self.mic.echo_stats(offset, offset + len(audio) / SAMPLE_RATE)):
            return
        if channel == "system":
            self.system_inflight += 1
            if self.copilot:
                self.copilot.on_sentence_closed(offset, offset + len(audio) / SAMPLE_RATE)          # Murmur Copilot waits for this sentence's text before answering
        # in a copilot session the live text must be quick (beam 1, no word timings): the answer waits
        # for it. The after-recording quality pass transcribes everything again with full settings.
        engine.submit(Job(channel, offset, audio,
                          lambda ch, off, pieces: self._on_pieces(ch, off, pieces, audio),
                          context={"names": known_names(self.meeting), "language": self.language,
                                   "fast": self.copilot is not None}, group=self.meeting["id"]))

    def _on_pieces(self, channel: str, offset: float, pieces, audio=None) -> None:
        """Speech-to-text result (on the STT worker): hand it to the line thread and return at once."""
        if channel == "system" and audio is not None:
            self.system_inflight = max(0, self.system_inflight - 1)
        if pieces:
            self.lines.put((channel, offset, pieces, audio))

    def _line_loop(self) -> None:
        while True:
            item = self.lines.get()
            if item is None:
                return
            try:
                self._handle_line(*item)
            except Exception:
                log.exception("Live line failed")
            finally:
                self.lines.task_done()

    def _embed(self, audio) -> np.ndarray | None:
        if self.voices is None or audio is None or len(audio) < SAMPLE_RATE * diarize.MIN_PRINT:
            return None
        return self.voices.embed(audio)

    def _mic_verdict(self, offset: float, audio, pieces) -> tuple[list, np.ndarray | None, float | None]:
        """Which mic pieces are really you. Returns (pieces kept, voiceprint, similarity to your voice).

        Tested on a recorded 1:1 call with a Bluetooth speaker: your own lines scored 0.76-0.96
        against your voiceprint, the other person's voice leaking from the speaker 0.46-0.67, and
        Whisper's inventions on that leak ("Gracias", "Thank you") 0.38-0.63."""
        end = offset + len(audio) / SAMPLE_RATE
        on, _ = self.mic.echo_stats(offset, end)
        # while the speakers play, a mic line must be confident speech: residual echo makes
        # Whisper produce low-confidence fragments ("Thank you.", garbled half-sentences)
        if on >= 0.1:
            pieces = [p for p in pieces if not (self.mic.speakers_active(offset + p.start, offset + p.end)
                                                and (p.no_speech > 0.3 or p.logprob < -0.6))]
            if not pieces:
                return [], None, None
        v = self._embed(audio)
        if v is None:                                  # too short to recognise a voice
            if self.call_heard:
                pieces = [p for p in pieces if not livemic.INVENTED.match(p.text.strip())]
            return ([] if on > 0.5 else pieces), None, None
        if not self.me.ready():
            # learn your voice from clear sentences while the speakers are silent
            if on < 0.1 and len(audio) >= SAMPLE_RATE * 2 and all(p.logprob > -0.6 and p.no_speech < 0.3 for p in pieces):
                self.me.add(v)
            return pieces, v, None
        sim = self.me.similarity(v)
        if sim >= livemic.SURELY_ME:
            self.me.add(v)                             # keeps up with a new headset or room
        # In a call, the mic should only carry your voice: anything else is the other side leaking
        # from the speakers, or Whisper inventing text on noise. In a room (nothing on the system
        # channel) other people on the mic are real, so nothing is dropped there.
        if self.call_heard and sim < livemic.NOT_ME:
            log.info("Mic line dropped, not your voice (%.2f, speakers %.0f%%): %s", sim, on * 100,
                     " ".join(p.text for p in pieces)[:80])
            return [], v, sim
        return pieces, v, sim

    def _handle_line(self, channel: str, offset: float, pieces, audio) -> None:
        mid = self.meeting["id"]
        vprint, sim = None, None
        if channel == "mic":
            pieces, vprint, sim = self._mic_verdict(offset, audio, pieces)
            if not pieces:
                return
        candidates = [{
            "id": uuid.uuid4().hex[:8],
            "start": round(offset + p.start, 2),
            "end": round(offset + p.end, 2),
            "channel": channel,
            "text": p.text,
            "lang": p.lang,
            "words": word_times(offset, p.words),
        } for p in pieces]
        removed = []
        with self.lock:
            kept = []
            for seg in candidates:
                if channel == "mic":
                    # not clearly your voice and the same words as the other side said: speaker bleed
                    if (sim is None or sim < livemic.SURELY_ME) and live_echo(self.meeting["segments"], seg) is not None:
                        continue
                else:
                    for copy in mic_copies(self.meeting["segments"], seg):   # system arrived later
                        if self.me_sim.get(copy["id"], 0.0) < livemic.SURELY_ME:
                            self.meeting["segments"].remove(copy)
                            removed.append(copy["id"])
                kept.append(seg)
        for sid in removed:
            hub.send({"type": "segment_removed", "meeting_id": mid, "id": sid})
        if not kept:
            return                                                # nothing real: no speaker gets created
        if channel == "system":
            self.call_heard = True
        if self.copilot:
            for seg in kept:                                      # before the voiceprint: answers start sooner
                self.copilot.on_segment(seg)
        if channel == "mic" and self.mic_is_me:
            speaker = ME
        else:
            if vprint is None and channel == "system":
                vprint = self._embed(audio)
            speaker = self.speakers.assign(channel, vprint)
        renamed = False
        with self.lock:
            names_ = self.meeting["speakers"]
            if speaker not in names_:
                names_[speaker] = self.my_name() if speaker == ME else f"Speaker {speaker[3:]}"
                renamed = True
            for seg in kept:
                seg["speaker"] = speaker  # live guess; the regrouping below corrects it
                self.meeting["segments"].append(seg)
                if channel == "system":
                    self.speakers.remember(seg["id"], vprint)
                elif sim is not None:
                    self.me_sim[seg["id"]] = sim
                if speaker != ME and names.is_default(names_.get(speaker, "")):
                    found = names.find_name(seg["text"])
                    if found and found.lower() not in {v.lower() for v in names_.values()}:
                        names_[speaker] = found
                        renamed = True
            self.meeting["duration"] = round(self.elapsed, 1)
            storage.save(self.meeting)
        if renamed:
            hub.send({"type": "speakers", "meeting_id": mid, "speakers": dict(self.meeting["speakers"])})
        self.agreement.pop(channel, None)
        for seg in kept:
            hub.send({"type": "segment", "meeting_id": mid, "segment": seg})
        self._translate(kept)
        self._polish(channel, offset, audio, kept)
        if channel == "system" and time.monotonic() >= self.regroup_due:
            self._regroup()

    def _translate(self, segs: list[dict]) -> None:
        """Automatic translation: every new line that is not in the target language goes to the translator.
        Settings are read every time, so changes apply mid-meeting."""
        cfg = load_settings()
        if not cfg.get("translate_live"):
            return
        target = cfg.get("translate_to", "en")
        also = cfg.get("translate_also", "")
        mid = self.meeting["id"]
        self.meeting["translate_to"] = target      # saved with the meeting: refine re-translates afterwards
        self.meeting["translate_also"] = also if also != target else ""
        for seg in segs:
            for lang, _ in translation_targets(target, also):
                if seg.get("lang") != lang:
                    translator.submit(mid, seg["id"], seg["text"], lang, seg.get("lang"))

    def _polish(self, channel: str, offset: float, audio, segs: list[dict]) -> None:
        """Hand every new line to the AI line editor (see `live_polish`, polish.py)."""
        if self.copilot or not load_settings().get("live_polish", True):
            return
        for seg in segs:
            editor.submit(self.meeting["id"], seg["id"], seg["text"])

    def _regroup(self) -> None:
        """Cluster every remote line heard so far again (the method used after the meeting) and
        correct the live labels. Voiceprints are already known, so this takes milliseconds."""
        started = time.monotonic()
        with self.lock:
            segs = [dict(s) for s in self.meeting["segments"] if s["channel"] == "system"]
        if len(segs) < 4:
            return
        labels = self.speakers.regroup(segs)
        with self.lock:
            names_ = self.meeting["speakers"]
            assign = {}
            for s in self.meeting["segments"]:
                key = labels.get(s["id"])
                if key and key != s.get("speaker"):
                    s["speaker"] = key
                    assign[s["id"]] = key
                    names_.setdefault(key, f"Speaker {key[3:]}")
            if not assign:
                self.regroup_due = time.monotonic() + 15
                return
            names.auto_name(self.meeting)
            keep = {s["speaker"] for s in self.meeting["segments"]} | ({ME} if self.mic_is_me else set())
            self.meeting["speakers"] = {k: v for k, v in names_.items() if k in keep}
            storage.save(self.meeting)
            speakers_now = dict(self.meeting["speakers"])
        took = time.monotonic() - started
        self.regroup_due = time.monotonic() + max(15.0, 20 * took)
        log.info("Live regroup: %d remote lines, %d relabelled, %.2fs", len(segs), len(assign), took)
        hub.send({"type": "relabel", "meeting_id": self.meeting["id"], "assign": assign, "speakers": speakers_now})

    def _interim_loop(self) -> None:
        """Every 0.8 s, transcribe the sentence still being spoken (fast pass) so captions appear
        while the person talks. Only when the engine has nothing else to do: finals come first."""
        while not self.stop_event.wait(0.8):
            if not engine.idle():
                continue
            for ch in CHANNELS:
                if self.interim_busy[ch]:
                    continue
                with self.lock:
                    pending = self.segmenters[ch].pending()
                if not pending:
                    continue
                start, audio = pending
                self.interim_busy[ch] = True
                engine.submit(Job(ch, start, audio[-SAMPLE_RATE * 15:] if len(audio) > SAMPLE_RATE * 15 else audio,
                                  self._on_interim, context={"interim": True, "language": self.language},
                                  group=self.meeting["id"]))

    def _on_interim(self, channel: str, offset: float, pieces) -> None:
        self.interim_busy[channel] = False
        text = " ".join(p.text for p in pieces).strip()
        if not text or self.stop_event.is_set():
            return
        words = text.split()
        st = self.agreement.get(channel)
        if not st or abs(st["start"] - offset) > 0.05:      # a new sentence started
            st = self.agreement[channel] = {"start": offset, "prev": [], "committed": []}
        prev = st["prev"]
        agree = 0
        while agree < min(len(prev), len(words)) and _norm(prev[agree]) == _norm(words[agree]):
            agree += 1
        committed_n = len(st["committed"])
        keeps_prefix = [_norm(w) for w in words[:committed_n]] == [_norm(w) for w in st["committed"]]
        if keeps_prefix and agree > committed_n:          # committed text only grows, never rewrites
            st["committed"] = words[:agree]
        st["prev"] = words
        committed = " ".join(st["committed"])
        tentative = " ".join(words[len(st["committed"]):])  # the part that may still change
        last = ME if channel == "mic" and self.mic_is_me else self.speakers.last.get(channel)
        label = self.meeting["speakers"].get(last, "") if last else ""
        hub.send({"type": "interim", "meeting_id": self.meeting["id"], "channel": channel,
                  "start": round(offset, 2), "text": text, "committed": committed,
                  "tentative": tentative, "label": label})

    def _levels(self) -> None:
        while not self.stop_event.is_set():
            hub.send({
                "type": "levels",
                "elapsed": round(self.elapsed, 1),
                "mic": round(min(1.0, self.segmenters["mic"].level * 12), 3),
                "system": round(min(1.0, self.segmenters["system"].level * 12), 3),
                "muted": self.muted,
            })
            time.sleep(0.25)

    def stop(self) -> dict:
        duration = self.elapsed
        self.stop_event.set()
        if self.copilot:
            self.copilot.stop()
        for c in self.captures.values():
            c.join(timeout=2)
        total = int(duration * PLAYBACK_RATE)
        for c in self.captures.values():
            c.pad_to(total)                 # enqueues the closing silence
        self.chunks.join()                  # everything captured is processed
        self.chunks.put(("__stop__", None))
        with self.lock:
            for ch, r in self.resamplers.items():  # drain the resampler's tail
                tail = r.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if len(tail):
                    self.writers[ch].write(tail)
                    if ch == "mic":
                        self.mic.add_mic(tail)
                    else:
                        self.segmenters[ch].feed(tail)
            self._drain_mic(force=True)
            for s in self.segmenters.values():
                s.flush()
        self.meeting["duration"] = round(duration, 1)
        self.meeting["status"] = "processing"
        for clip in self.meeting.get("screens", []):
            if clip["duration"] is None:
                clip["duration"] = round(duration - clip["offset"], 2)
        storage.save(self.meeting)
        threading.Thread(target=self._finish, daemon=True).start()
        return self.meeting

    def _finish(self) -> None:
        mid = self.meeting["id"]
        hub.send({"type": "meeting", "meeting_id": mid, "status": "processing", "step": "transcribing"})
        engine.wait_group(mid)             # this meeting's last live sentences (not other meetings' work)
        self.lines.join()                  # ... and their voiceprints and speakers
        self.lines.put(None)
        self.me.save()
        if self.voices:
            self.voices.close()
        with self.lock:
            self.closed = True
            for w in [*self.writers.values(), *self.hq_writers.values()]:
                w.close()
            remove_echo(self.meeting["segments"], keep={k for k, v in self.me_sim.items() if v >= livemic.SURELY_ME})
        postq.submit(mid, self._post_process)   # waits its turn, and while another recording runs

    def _post_process(self) -> None:
        mid = self.meeting["id"]
        try:
            build_audio(self.folder)
            finalize_screens(self.meeting)
            live_spans = live_name_spans(self.meeting)
            postq.checkpoint(mid)
            clean_mic_for_recognition(self.meeting)
            postq.checkpoint(mid)
            if self.settings.get("refine_after_recording", True) and engine.state == "ready":
                refine_transcript(self.meeting)
            postq.checkpoint(mid)
            hub.send({"type": "meeting", "meeting_id": mid, "status": "processing", "step": "diarizing"})
            run_diarization(self.meeting, num_speakers=0)
            carry_live_names(self.meeting, live_spans)
        except Exception as exc:
            log.exception("Post-processing failed")
            self.meeting["error"] = f"Speaker separation failed: {exc}"
        self.meeting["status"] = "ready"
        storage.save(self.meeting, exports=True)
        hub.send({"type": "meeting", "meeting_id": mid, "status": "ready"})
        if self.settings.get("auto_summary") and self.meeting["segments"]:
            start_summary(mid)
        if self.meeting.get("translate_to"):
            start_translation(mid)


FFMPEG_FALLBACK = FFMPEG


def decode_audio(path: Path, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32 at `rate` with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg") or (str(FFMPEG_FALLBACK) if FFMPEG_FALLBACK.exists() else None)
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip()[-300:] or "ffmpeg error")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg") or (str(FFMPEG_FALLBACK) if FFMPEG_FALLBACK.exists() else None)
    if not path:
        raise RuntimeError("ffmpeg not found")
    return path


def finalize_screens(meeting: dict) -> None:
    """Make screen clips seekable: browser-recorded WebM has no seek index (cues); a remux adds it."""
    folder = storage.meeting_dir(meeting["id"])
    kept = []
    for clip in meeting.get("screens", []):
        src = folder / clip["file"]
        if not src.exists() or src.stat().st_size < 1024:
            src.unlink(missing_ok=True)
            continue
        fixed = src.with_name(src.stem + ".fixed.webm")
        proc = subprocess.run(
            [ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", str(src), "-c", "copy", str(fixed)],
            capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0 and fixed.exists():
            fixed.replace(src)
        else:
            log.warning("Remux of %s failed: %s", src.name, proc.stderr.decode(errors="replace")[-300:])
            fixed.unlink(missing_ok=True)
        kept.append(clip)
    meeting["screens"] = kept


def transcribe_tracks(meeting: dict, tracks: dict[str, np.ndarray], quality: bool = False,
                      into: list | None = None) -> None:
    """Offline transcription of whole channels with progress events.

    Used for file import, "Transcribe again" and the after-recording refine pass. With `into`,
    segments are collected there instead of being shown live (the draft stays on screen).
    """
    mid = meeting["id"]
    lock = threading.Lock()
    duration = max(len(a) for a in tracks.values()) / SAMPLE_RATE or 1.0
    live = into is None
    target = meeting["segments"] if live else into
    # 100 ms loopback energy: while the speakers play, mic lines must be confident speech
    # (the echo canceller's leftovers make Whisper invent "Thank you." and half-sentences)
    sys_on = None
    if "mic" in tracks and "system" in tracks:
        x = tracks["system"]
        n = len(x) // (SAMPLE_RATE // 10)
        sys_on = (x[: n * (SAMPLE_RATE // 10)].reshape(n, -1) ** 2).mean(axis=1) > 1e-5

    def doubtful_echo(channel, start, end, p):
        if channel != "mic" or sys_on is None:
            return False
        a, b = int(start * 10), max(int(end * 10), int(start * 10) + 1)
        active = sys_on[a:b]
        return len(active) and active.mean() > 0.5 and (p.no_speech > 0.3 or p.logprob < -0.6)

    def on_pieces(channel, offset, pieces):
        with lock:
            for p in pieces:
                if doubtful_echo(channel, offset + p.start, offset + p.end, p):
                    continue
                seg = {"id": uuid.uuid4().hex[:8], "start": round(offset + p.start, 2),
                       "end": round(offset + p.end, 2), "channel": channel, "speaker": channel,
                       "text": p.text, "lang": p.lang, "words": word_times(offset, p.words)}
                target.append(seg)
                if live:
                    hub.send({"type": "segment", "meeting_id": mid, "segment": seg})
            if live:
                storage.save(meeting)
        hub.send({"type": "progress", "meeting_id": mid, "percent": min(99, int(offset / duration * 100))})

    jobs = []
    context = {"quality": quality, "batch": True}    # batch lane: never competes with a live recording
    lang = (meeting.get("copilot") or {}).get("language")
    if lang in ("az", "de", "en"):
        context["language"] = lang
    for ch, audio in tracks.items():
        segmenter = make_segmenter(ch, lambda c, off, a: jobs.append(
            Job(c, off, a, on_pieces, context={**context, "lang_key": f"{mid}-{c}", "names": known_names(meeting)},
                group=mid)),
            max_segment=28.0 if quality else None)
        step = SAMPLE_RATE * 10
        for i in range(0, len(audio), step):
            segmenter.feed(audio[i:i + step])
        segmenter.flush()
    # Interleave channels in time order so progress moves forward steadily.
    for job in sorted(jobs, key=lambda j: j.offset):
        engine.submit(job)
    engine.wait_group(mid)


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _norm_line(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def mic_copies(segments: list[dict], system_seg: dict) -> list[dict]:
    """Mic lines that repeat what this (later arriving) system line said: speaker bleed."""
    sys_words = {w for s in segments[-40:] if s["channel"] == "system"
                 and abs(s["start"] - system_seg["start"]) < 15 for w in _words(s["text"])}
    sys_words |= set(_words(system_seg["text"]))
    out = []
    for s in segments[-40:]:
        if s["channel"] != "mic" or s["end"] < system_seg["start"] - 4 or s["start"] > system_seg["end"] + 4:
            continue
        w = _words(s["text"])
        if w and sum(x in sys_words for x in w) / len(w) >= 0.6:
            out.append(s)
    return out


def live_echo(segments: list[dict], seg: dict) -> dict | None:
    """A segment on the other channel, close in time, saying the same words (speaker bleed)."""
    words = _words(seg["text"])
    if not words:
        return None
    other = "system" if seg["channel"] == "mic" else "mic"
    window = 3 if len(words) >= 2 else 2           # a single word ("Tuesday.") must be very close in time
    nearby = [s for s in segments[-40:] if s["channel"] == other
              and s["start"] <= seg["end"] + window and s["end"] >= seg["start"] - window]
    theirs = {w for s in nearby for w in _words(s["text"])}   # one mic line can span two remote lines
    if not theirs or sum(w in theirs for w in words) / len(words) < 0.6:
        return None
    best = max(nearby, key=lambda s: len(set(_words(s["text"])) & set(words)))
    return best


def live_name_spans(meeting: dict) -> list[tuple[float, float, str]]:
    names_ = meeting.get("speakers", {})
    return [(s["start"], s["end"], names_[s["speaker"]]) for s in meeting["segments"]
            if not names.is_default(names_.get(s["speaker"], "Speaker 0"))
            and names_.get(s["speaker"]) not in ("Microphone", "Online", "Speaker")]


def carry_live_names(meeting: dict, spans: list[tuple[float, float, str]]) -> None:
    """After the full diarization renumbers speakers, give each one the name it had live
    (by time overlap), unless it already got a name from its own introduction."""
    votes: dict[str, dict[str, float]] = {}
    for seg in meeting["segments"]:
        for a, b, name in spans:
            ov = min(b, seg["end"]) - max(a, seg["start"])
            if ov > 0:
                votes.setdefault(seg["speaker"], {}).setdefault(name, 0.0)
                votes[seg["speaker"]][name] += ov
    taken = {v.lower() for v in meeting["speakers"].values() if not names.is_default(v)}
    for key, tally in votes.items():
        if key in meeting["speakers"] and names.is_default(meeting["speakers"][key]):
            name = max(tally, key=tally.get)
            if name.lower() not in taken:
                meeting["speakers"][key] = name
                taken.add(name.lower())


def known_names(meeting: dict) -> list[str]:
    """Speaker names already set for this meeting, used as recognition hints."""
    return [n for n in meeting.get("speakers", {}).values() if n and not names.is_default(n)
            and n not in ("Microphone", "Online", "Speaker")]


def _frame_db(x: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    frame = rate // 10
    n = len(x) // frame
    if n == 0:
        return np.zeros(0)
    return 20 * np.log10(np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9)


def dyn_level(x: np.ndarray, rate: int) -> np.ndarray:
    """Even out loud and quiet voices on one channel (you vs. a phone on speaker): ffmpeg dynaudnorm."""
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
             "-af", "dynaudnorm=f=250:g=15:p=0.9:m=12:s=8", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-"],
            input=x.astype(np.float32).tobytes(), capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0 and proc.stdout:
            y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
            return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    except Exception:
        log.exception("dynaudnorm failed")
    return x


def clean_mic_for_recognition(meeting: dict) -> None:
    """Prepare the 16 kHz mic track that recognition and speaker separation use.

    1. Echo: when the speakers played meeting audio, DTLN-aec (the neural AEC anarlog ships)
       removes it using the loopback as reference; NLMS is the fallback without the model.
    2. Noise: with no loopback audio (phone call on speaker, people in the room) DTLN runs as a
       noise suppressor, only when the room is actually noisy.
    3. Level: dynaudnorm brings a quiet voice (phone speaker, far side of the table) up to yours.
    Guard: if processing cuts real near-end speech by more than 10 dB, the original is kept.
    The listening mix is not touched by steps 1-2.
    """
    folder = storage.meeting_dir(meeting["id"])
    mic_p, sys_p = folder / "mic.ogg", folder / "system.ogg"
    if not mic_p.exists() or meeting.get("mic_cleaned"):
        return
    try:
        mic, _ = sf.read(mic_p, dtype="float32")
        ref = sf.read(sys_p, dtype="float32")[0] if sys_p.exists() else np.zeros_like(mic)
        if len(ref) < len(mic):
            ref = np.pad(ref, (0, len(mic) - len(ref)))
        has_ref = float(np.mean(ref[: len(mic)] ** 2)) > 1e-7
        delay, corr = aec.estimate_delay(mic, ref[: len(mic)], SAMPLE_RATE) if has_ref else (0, 0.0)
        ref = aec.align(ref, delay)
        mic_db = _frame_db(mic)
        noise_floor = float(np.percentile(mic_db, 10)) if len(mic_db) else -120.0
        noisy = noise_floor > -55.0
        info = {"echo": has_ref, "noise_floor_db": round(noise_floor, 1),
                "echo_delay_ms": delay * 1000 // SAMPLE_RATE, "echo_corr": round(corr, 2)}
        hub.send({"type": "meeting", "meeting_id": meeting["id"], "status": "processing", "step": "echo"})
        out, method = mic, "none"
        if dtln.available() and (has_ref or noisy):
            out = dtln.DTLN(128).process(mic, ref if has_ref else None)
            method = "dtln-aec" if has_ref else "dtln-denoise"
        elif has_ref:
            out, _ = aec.cancel_echo(mic, ref, SAMPLE_RATE)
            method = "nlms"
        if method != "none":
            # near-end speech = loud mic frames while the speakers were quiet
            ref_db = _frame_db(ref[: len(mic)])
            out_db = _frame_db(out)
            k = min(len(mic_db), len(ref_db), len(out_db))
            near = (mic_db[:k] > np.percentile(mic_db[:k], 75)) & (ref_db[:k] < -50)
            loss = float(np.mean(mic_db[:k][near] - out_db[:k][near])) if near.sum() > 10 else 0.0
            info["near_speech_loss_db"] = round(loss, 1)
            if loss > 10:
                log.warning("Mic cleanup removed real speech (%.1f dB); keeping the original", loss)
                out, method = mic, "skipped"
        out = dyn_level(out, SAMPLE_RATE)
        write_ogg(mic_p, out)
        info["method"] = method
        meeting["mic_processing"] = info
        meeting["mic_cleaned"] = True
        log.info("Mic track for recognition: %s", info)
    except Exception:
        log.exception("Mic cleanup failed; keeping the original mic track")


def refine_transcript(meeting: dict) -> None:
    """Second, more accurate pass over the saved audio; replaces the live draft when it succeeds."""
    folder = storage.meeting_dir(meeting["id"])
    tracks = {ch: sf.read(folder / f"{ch}.ogg", dtype="float32")[0]
              for ch in CHANNELS if (folder / f"{ch}.ogg").exists()}
    if not tracks or not meeting["segments"]:
        return
    hub.send({"type": "meeting", "meeting_id": meeting["id"], "status": "processing", "step": "refining"})
    refined: list[dict] = []
    transcribe_tracks(meeting, tracks, quality=True, into=refined)
    if refined:
        remove_echo(refined)
        meeting["draft_segments"] = len(meeting["segments"])
        meeting["segments"] = refined
        meeting["refined"] = True


def retranscribe(meeting_id: str) -> None:
    """Re-run transcription on a meeting's saved audio (new model, word timestamps, other language)."""
    meeting = storage.load(meeting_id)
    folder = storage.meeting_dir(meeting_id)
    try:
        tracks = {ch: sf.read(folder / f"{ch}.ogg", dtype="float32")[0]
                  for ch in CHANNELS if (folder / f"{ch}.ogg").exists()}
        if not tracks:
            raise RuntimeError("No audio")
        meeting.update(segments=[], sentiments=None, error="", status="processing")
        storage.save(meeting)
        postq.checkpoint(meeting_id)
        clean_mic_for_recognition(meeting)
        tracks = {ch: sf.read(folder / f"{ch}.ogg", dtype="float32")[0]
                  for ch in CHANNELS if (folder / f"{ch}.ogg").exists()}
        hub.send({"type": "meeting", "meeting_id": meeting_id, "status": "processing", "step": "transcribing"})
        transcribe_tracks(meeting, tracks, quality=True)
        remove_echo(meeting["segments"])
        postq.checkpoint(meeting_id)
        hub.send({"type": "meeting", "meeting_id": meeting_id, "status": "processing", "step": "diarizing"})
        run_diarization(meeting, num_speakers=0)
    except Exception as exc:
        log.exception("Re-transcription failed")
        meeting["error"] = f"Could not transcribe again: {exc}"
    meeting["status"] = "ready"
    storage.save(meeting, exports=True)
    hub.send({"type": "meeting", "meeting_id": meeting_id, "status": "ready"})


def process_import(meeting: dict, source: Path) -> None:
    """Transcribe an uploaded recording the same way as a live meeting, then diarize all of it."""
    mid = meeting["id"]
    folder = storage.meeting_dir(mid)
    try:
        hq = decode_audio(source, PLAYBACK_RATE)
        audio = soxr.resample(hq, PLAYBACK_RATE, SAMPLE_RATE).astype(np.float32)
        meeting["duration"] = round(len(audio) / SAMPLE_RATE, 1)
        sf.write(folder / "system.wav", audio, SAMPLE_RATE, subtype="PCM_16")
        sf.write(folder / "system_hq.wav", hq, PLAYBACK_RATE, subtype="PCM_16")
        del hq
        transcribe_tracks(meeting, {"system": audio})
        build_audio(folder)
        postq.checkpoint(mid)
        hub.send({"type": "meeting", "meeting_id": mid, "status": "processing", "step": "diarizing"})
        run_diarization(meeting, num_speakers=0)
    except Exception as exc:
        log.exception("Import failed")
        meeting["error"] = f"Could not process the file: {exc}"
    finally:
        source.unlink(missing_ok=True)
    meeting["status"] = "ready"
    storage.save(meeting, exports=True)
    hub.send({"type": "meeting", "meeting_id": mid, "status": "ready"})
    if load_settings().get("auto_summary") and meeting["segments"]:
        start_summary(mid)


def recover_meetings(ids: list[str]) -> None:
    """Finish meetings that a crash or closed window left half-processed."""
    for mid in ids:
        meeting = storage.load(mid)
        folder = storage.meeting_dir(mid)
        source = next(folder.glob("source.*"), None)
        try:
            if source:
                meeting["segments"], meeting["error"] = [], ""
                meeting["status"] = "processing"
                storage.save(meeting)
                process_import(meeting, source)
                continue
            if any((folder / f"{ch}.wav").exists() for ch in CHANNELS):
                remove_echo(meeting["segments"])
                build_audio(folder)
                for clip in meeting.get("screens", []):
                    if clip["duration"] is None:
                        clip["duration"] = round(meeting["duration"] - clip["offset"], 2)
                finalize_screens(meeting)
                run_diarization(meeting, num_speakers=0)
        except Exception as exc:
            log.exception("Recovery of %s failed", mid)
            meeting["error"] = f"Recovery failed: {exc}"
        meeting["status"] = "ready"
        storage.save(meeting, exports=True)
        hub.send({"type": "meeting", "meeting_id": mid, "status": "ready"})


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def remove_echo(segments: list[dict], keep: set[str] | None = None) -> None:
    """Drop mic segments that are just the meeting audio leaking from the speakers into the mic.
    `keep`: lines known to be in your voice (the same words can be said on both sides: "ciao")."""
    system = [s for s in segments if s["channel"] == "system"]
    sure = keep or set()
    keep = []
    for seg in segments:
        if seg["channel"] == "mic" and seg["id"] not in sure:
            words = _words(seg["text"])
            nearby = set()
            for s in system:
                if s["start"] <= seg["end"] + 2 and s["end"] >= seg["start"] - 2:
                    nearby.update(_words(s["text"]))
            if words and sum(w in nearby for w in words) / len(words) >= 0.6:
                continue
        keep.append(seg)
    segments[:] = keep


def word_times(offset: float, words) -> list:
    """Absolute [start, end, word] triples so the UI can seek to any word."""
    return [[round(offset + a, 2), round(offset + b, 2), w] for a, b, w in words]


def write_ogg(path: Path, data: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Write OGG Vorbis in small blocks: one large write overflows libsndfile's stack on Windows."""
    with sf.SoundFile(path, "w", rate, 1, format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(data), rate):
            f.write(data[i:i + rate])


def level_speech(x: np.ndarray, rate: int, target_db: float = -20.0, max_gain_db: float = 18.0) -> np.ndarray:
    """Bring a track's speech to a comfortable level.

    Raw microphones often arrive 10-15 dB quieter than call audio (Zoom and Teams apply their own
    gain; a raw WASAPI capture does not), which makes your own voice sound far away. The level is
    measured on the loudest 30% of 100 ms frames (speech), never on silence, and peaks above 0.9
    are softly limited instead of clipped.
    """
    frame = rate // 10
    n = len(x) // frame
    if n < 5:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    speech = rms[rms >= max(1e-4, float(np.percentile(rms, 70)))]
    if not len(speech):
        return x
    level_db = 20 * np.log10(float(speech.mean()))
    gain_db = float(np.clip(target_db - level_db, -6.0, max_gain_db))
    y = x * np.float32(10 ** (gain_db / 20))
    over = np.abs(y) > 0.9
    y[over] = np.sign(y[over]) * (0.9 + 0.1 * np.tanh((np.abs(y[over]) - 0.9) / 0.1))
    return y.astype(np.float32)


def clean_mic(x: np.ndarray, rate: int) -> np.ndarray:
    """Rumble filter (80 Hz) + FFT noise reduction for the microphone, for listening only.

    Recognition keeps using the untouched 16 kHz track: Whisper does better on raw audio.
    """
    if not len(x):
        return x
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
             "-af", "highpass=f=80,afftdn=nr=12:nf=-45:tn=1", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-"],
            input=x.astype(np.float32).tobytes(), capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0 and proc.stdout:
            y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
            return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
        log.warning("Mic cleanup failed: %s", proc.stderr.decode(errors="replace")[-200:])
    except Exception:
        log.exception("Mic cleanup failed")
    return x


def write_mix(folder: Path, tracks: dict[str, np.ndarray], rate: int) -> None:
    """Level each channel separately, then mix them into the file you listen to."""
    if not tracks:
        return
    n = max(len(t) for t in tracks.values())
    mix = np.zeros(n, dtype=np.float32)
    for t in tracks.values():
        mix[: len(t)] += level_speech(t, rate)
    peak = float(np.max(np.abs(mix))) or 1.0
    write_ogg(folder / "audio.ogg", mix / max(peak / 0.98, 1.0), rate)


def remix_existing(folder: Path) -> bool:
    """Rebuild audio.ogg of an older meeting with levelled channels (from its 16 kHz tracks)."""
    tracks = {ch: sf.read(folder / f"{ch}.ogg", dtype="float32")[0]
              for ch in CHANNELS if (folder / f"{ch}.ogg").exists()}
    if not tracks:
        return False
    write_mix(folder, tracks, SAMPLE_RATE)
    return True


def build_audio(folder: Path) -> None:
    """16 kHz OGG per channel for recognition, plus a levelled 48 kHz mix (audio.ogg) for listening."""
    speech, hq = {}, {}
    for ch in CHANNELS:
        wav = folder / f"{ch}.wav"
        if wav.exists():
            speech[ch], _ = sf.read(wav, dtype="float32")
            write_ogg(folder / f"{ch}.ogg", speech[ch])
        hq_wav = folder / f"{ch}_hq.wav"
        if hq_wav.exists():
            hq[ch], _ = sf.read(hq_wav, dtype="float32")
    if "mic" in hq:
        hq["mic"] = dyn_level(clean_mic(hq["mic"], PLAYBACK_RATE), PLAYBACK_RATE)
    if hq:
        write_mix(folder, hq, PLAYBACK_RATE)
    else:
        write_mix(folder, speech, SAMPLE_RATE)
    for ch in CHANNELS:
        (folder / f"{ch}.wav").unlink(missing_ok=True)
        (folder / f"{ch}_hq.wav").unlink(missing_ok=True)


def run_diarization(meeting: dict, num_speakers: int) -> None:
    """Label everyone on both channels as Speaker 1, 2, 3… (names reset; rename them in the UI)."""
    if not meeting["segments"]:
        return
    if not diarize.available():
        raise RuntimeError("diarization models are missing (run setup.bat)")
    folder = storage.meeting_dir(meeting["id"])
    tracks = {}
    for ch in CHANNELS:
        path = folder / f"{ch}.ogg"
        if path.exists():
            tracks[ch], _ = sf.read(path, dtype="float32")
    settings = load_settings()
    mic_is_me = settings.get("mic_is_me", True) is not False
    voice = livemic.Voice(DATA_DIR / "my_voice.npy")
    me_vec = voice.sum if voice.ready() else None
    # runs in a child process: a recording that starts meanwhile must never wait for it
    result = diarize.label_in_subprocess(tracks, meeting["segments"], num_speakers, mic_is_me, me=me_vec)
    if result["drop"]:
        gone = set(result["drop"])
        log.info("Removed %d mic lines that were not your voice", len(gone))
        meeting["segments"] = [s for s in meeting["segments"] if s["id"] not in gone]
    for seg in meeting["segments"]:
        seg["speaker"] = result["labels"].get(seg["id"], seg.get("speaker", ""))
    if result["me"] is not None and not voice.ready():
        voice.add(result["me"] / (np.linalg.norm(result["me"]) + 1e-9))
        voice.count = max(voice.count, 3)
        voice.save()
    my_name = str(settings.get("my_name") or "").strip() or "Me"
    used = sorted({s["speaker"] for s in meeting["segments"]}, key=lambda k: (not k.startswith("spk"), len(k), k))
    meeting["speakers"] = {k: f"Speaker {k[3:]}" if k.startswith("spk") else my_name if k == ME
                           else meeting["speakers"].get(k, k) for k in used}
    names.auto_name(meeting)  # "my name is Sarah" -> Sarah; speakers who never say their name stay "Speaker N"


def merge_same_names(meeting: dict) -> None:
    """Speakers renamed to the same name become one speaker (fixes over-split clusters)."""
    first: dict[str, str] = {}
    remap = {}
    for key, name in meeting["speakers"].items():
        norm = name.strip().lower()
        if norm in first:
            remap[key] = first[norm]
        else:
            first[norm] = key
    for seg in meeting["segments"]:
        seg["speaker"] = remap.get(seg["speaker"], seg["speaker"])
    for key in remap:
        meeting["speakers"].pop(key, None)


def apply_corrections(meeting: dict, corrections: dict[str, str]) -> None:
    """Replace misheard lines with the AI-corrected text; the original stays in raw_text."""
    for seg in meeting["segments"]:
        new = corrections.get(seg["id"])
        if new and new != seg["text"]:
            seg.setdefault("raw_text", seg["text"])
            seg["text"] = new
            seg["corrected"] = True


def apply_speaker_names(meeting: dict, found: dict[str, str]) -> None:
    """Names the AI found from introductions; labels arrive as shown in the transcript ("Speaker 2")."""
    by_label = {v: k for k, v in meeting.get("speakers", {}).items()}
    taken = {v.lower() for v in meeting["speakers"].values() if not names.is_default(v)}
    for label, name in found.items():
        key = by_label.get(label)
        name = name.strip()
        if key and name and names.is_default(meeting["speakers"][key]) and name.lower() not in taken:
            meeting["speakers"][key] = name
            taken.add(name.lower())


def start_summary(meeting_id: str, language: str | None = None) -> None:
    def work():
        meeting = storage.load(meeting_id)
        if not meeting:
            return
        settings = load_settings()
        lang = summary.resolve_language(meeting, settings, language)
        hub.send({"type": "summary", "meeting_id": meeting_id, "state": "running", "language": lang})
        try:
            result = summary.analyse(meeting, settings, lang)
            meeting = storage.load(meeting_id) or meeting
            text = meeting["summary"] = result["summary"]
            meeting["summary_lang"] = lang
            meeting.setdefault("summaries", {})[lang] = text  # each language is kept, switching back is free
            meeting["sentiments"] = result["sentiments"]
            apply_corrections(meeting, result.get("corrections", {}))
            apply_speaker_names(meeting, result.get("speaker_names", {}))
            first = text.splitlines()[0] if text else ""
            if first.startswith("# ") and meeting["title"].startswith("Meet ") and not storage.user_edited(meeting_id, "title"):
                meeting["title"] = first[2:].strip()
            storage.save(meeting, exports=True)
            hub.send({"type": "summary", "meeting_id": meeting_id, "state": "done"})
        except Exception as exc:
            log.exception("Summary failed")
            hub.send({"type": "summary", "meeting_id": meeting_id, "state": "error", "message": str(exc)})

    threading.Thread(target=work, daemon=True).start()


session: Session | None = None
session_lock = threading.Lock()


# ---------------------------------------------------------------- API

class StartBody(BaseModel):
    title: str = ""


class MeetingPatch(BaseModel):
    title: str | None = None
    archived: bool | None = None
    speakers: dict[str, str] | None = None


class SegmentPatch(BaseModel):
    text: str


class ModelTest(BaseModel):
    model: str


class DiarizeBody(BaseModel):
    num_speakers: int = 0


def _get(meeting_id: str) -> dict:
    meeting = storage.load(meeting_id)
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    return meeting


def _preload_translation() -> None:
    try:
        localmt.load()
    except Exception as exc:
        log.warning("Translation model not loaded: %s", exc)


@app.on_event("startup")
async def on_startup():
    hub.loop = asyncio.get_running_loop()
    engine.load_async(load_settings())
    cfg = load_settings()
    if cfg.get("translate_live") and cfg.get("translate_engine", "local") == "local" and localmt.present():
        threading.Thread(target=_preload_translation, daemon=True, name="nllb-load").start()
    unfinished = [i["id"] for i in storage.list_meetings() if i["status"] != "ready"]
    if unfinished:
        postq.submit(unfinished[0], lambda: recover_meetings(unfinished))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    await ws.send_text(json.dumps({"type": "engine", **engine.status()}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.clients.discard(ws)


@app.get("/api/status")
def api_status():
    return {
        "engine": engine.status(),
        "recording": bool(session),
        "meeting_id": session.meeting["id"] if session else None,
        "muted": session.muted if session else {"mic": False, "system": False},
        "elapsed": session.elapsed if session else 0,
        "diarization": diarize.available(),
    }


@app.post("/api/record/start")
def record_start(body: StartBody):
    global session
    with session_lock:
        if session:
            raise HTTPException(409, "A recording is already running")
        session = Session(body.title)
        postq.recording(True)
        return session.meeting


@app.post("/api/record/stop")
def record_stop():
    global session
    with session_lock:
        if not session:
            raise HTTPException(409, "No recording is running")
        meeting = session.stop()
        session = None
        postq.recording(False)             # waiting meetings continue now, one after the other
        return meeting


@app.post("/api/import")
async def import_file(file: UploadFile, title: str = Form("")):
    meeting = storage.new_meeting(title.strip() or Path(file.filename or "File").stem)
    meeting["status"] = "processing"
    meeting["source"] = file.filename
    meeting["speakers"] = {"system": "Speaker"}
    source = storage.meeting_dir(meeting["id"]) / ("source" + Path(file.filename or "").suffix.lower())
    with source.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    storage.save(meeting)
    postq.submit(meeting["id"], lambda: process_import(meeting, source))
    return meeting


class MuteBody(BaseModel):
    channel: str
    muted: bool


@app.post("/api/record/mute")
def record_mute(body: MuteBody):
    if not session:
        raise HTTPException(409, "No recording is running")
    if body.channel not in CHANNELS:
        raise HTTPException(400, "channel must be mic or system")
    session.set_muted(body.channel, body.muted)
    return {"muted": session.muted}


@app.post("/api/record/screen/start")
def screen_start():
    if not session:
        raise HTTPException(409, "No recording is running")
    return session.start_screen()


@app.post("/api/record/screen/{n}/chunk")
async def screen_chunk(n: int, request: Request):
    """Append one MediaRecorder chunk to the clip file (sent every couple of seconds)."""
    s = session
    if not s:
        raise HTTPException(409, "No recording is running")
    clip = next((c for c in s.meeting.get("screens", []) if c["n"] == n), None)
    if not clip:
        raise HTTPException(404, "Unknown screen clip")
    data = await request.body()
    with (s.folder / clip["file"]).open("ab") as f:
        f.write(data)
    return {"ok": True}


@app.post("/api/record/screen/{n}/stop")
def screen_stop(n: int):
    if session:
        session.stop_screen(n)
    return {"ok": True}


@app.get("/api/meetings/{meeting_id}/screen/{n}")
def meeting_screen(meeting_id: str, n: int):
    meeting = _get(meeting_id)
    clip = next((c for c in meeting.get("screens", []) if c["n"] == n), None)
    path = storage.meeting_dir(meeting_id) / clip["file"] if clip else None
    if not path or not path.exists():
        raise HTTPException(404, "No screen recording")
    return FileResponse(path, media_type="video/webm")


@app.get("/api/meetings")
def meetings_list():
    return storage.list_meetings()


@app.get("/api/meetings/{meeting_id}")
def meeting_get(meeting_id: str):
    meeting = _get(meeting_id)
    meeting["has_audio"] = (storage.meeting_dir(meeting_id) / "audio.ogg").exists()
    return meeting


@app.patch("/api/meetings/{meeting_id}")
def meeting_patch(meeting_id: str, body: MeetingPatch):
    meeting = _get(meeting_id)
    if body.title is not None and body.title.strip():
        storage.set_user_fields(meeting, title=body.title.strip())
    if body.archived is not None:
        storage.set_user_fields(meeting, archived=body.archived)
    if body.speakers:
        meeting["speakers"].update({k: v.strip() for k, v in body.speakers.items() if v.strip()})
        merge_same_names(meeting)
    storage.save(meeting, exports=meeting["status"] == "ready")
    return meeting


@app.delete("/api/meetings/{meeting_id}")
def meeting_delete(meeting_id: str):
    if session and session.meeting["id"] == meeting_id:
        raise HTTPException(409, "A running recording cannot be deleted")
    storage.delete(meeting_id)
    return {"ok": True}


@app.patch("/api/meetings/{meeting_id}/segments/{segment_id}")
def segment_patch(meeting_id: str, segment_id: str, body: SegmentPatch):
    """Edit one transcript line by hand. An empty text removes the line."""
    meeting = _get(meeting_id)
    seg = next((s for s in meeting["segments"] if s["id"] == segment_id), None)
    if not seg:
        raise HTTPException(404, "Line not found")
    text = " ".join(body.text.split())
    if not text:
        meeting["segments"].remove(seg)
    elif text != seg["text"]:
        seg.setdefault("raw_text", seg["text"])
        seg["text"] = text
        seg["edited"] = True
        if text == seg["raw_text"] and not seg.get("corrected"):  # changed back to what was recognised
            seg.pop("edited", None)
            seg.pop("raw_text", None)
    storage.save(meeting, exports=meeting["status"] == "ready")
    return {"segment": None if not text else seg}


@app.post("/api/models/{provider}/test")
def provider_model_test(provider: str, body: ModelTest):
    """Send a tiny request so the user can tell whether a model actually answers."""
    if provider not in summary.OPENAI_COMPATIBLE:
        raise HTTPException(404, "Unknown provider")
    key = load_settings().get(f"{provider}_api_key", "")
    if not key:
        raise HTTPException(400, "No API key")
    started = time.monotonic()
    try:
        summary.ping(provider, key, body.model)
    except Exception as exc:
        raise HTTPException(400, str(exc)[:300])
    return {"seconds": round(time.monotonic() - started, 1)}


class SummaryBody(BaseModel):
    language: str | None = None   # az, de, en; empty = settings / meeting language
    force: bool = False           # regenerate even if this language was already written


@app.post("/api/meetings/{meeting_id}/summary")
def meeting_summary(meeting_id: str, body: SummaryBody | None = None):
    body = body or SummaryBody()
    meeting = _get(meeting_id)
    lang = summary.resolve_language(meeting, load_settings(), body.language)
    cached = meeting.get("summaries", {}).get(lang)
    if cached and not body.force:
        meeting["summary"], meeting["summary_lang"] = cached, lang
        storage.save(meeting, exports=True)
        return {"ok": True, "cached": True, "language": lang}
    start_summary(meeting_id, lang)
    return {"ok": True, "cached": False, "language": lang}


@app.post("/api/meetings/{meeting_id}/diarize")
def meeting_diarize(meeting_id: str, body: DiarizeBody):
    meeting = _get(meeting_id)
    try:
        run_diarization(meeting, body.num_speakers)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    storage.save(meeting, exports=True)
    return meeting


class TranslateBody(BaseModel):
    target: str = ""


@app.post("/api/meetings/{meeting_id}/translate")
def meeting_translate(meeting_id: str, body: TranslateBody | None = None):
    meeting = _get(meeting_id)
    if meeting["status"] != "ready":
        raise HTTPException(409, "The meeting is still being processed")
    settings = load_settings()
    target = (body.target if body else "") or settings.get("translate_to", "az")
    also = settings.get("translate_also", "")
    meeting["translate_to"] = target
    meeting["translate_also"] = also if also != target else ""
    storage.save(meeting)
    start_translation(meeting_id, force=True)   # both columns again, with the current model and rules
    return {"ok": True, "target": target, "also": meeting["translate_also"]}


@app.post("/api/meetings/{meeting_id}/retranscribe")
def meeting_retranscribe(meeting_id: str):
    meeting = _get(meeting_id)
    if meeting["status"] != "ready":
        raise HTTPException(409, "The meeting is still being processed")
    meeting.update(status="processing", error="")
    storage.save(meeting)
    postq.submit(meeting_id, lambda: retranscribe(meeting_id))
    return {"ok": True}


@app.get("/api/meetings/{meeting_id}/audio")
def meeting_audio(meeting_id: str):
    path = storage.meeting_dir(meeting_id) / "audio.ogg"
    if not path.exists():
        raise HTTPException(404, "No audio")
    return FileResponse(path, media_type="audio/ogg")


@app.get("/api/meetings/{meeting_id}/export/{fmt}")
def meeting_export(meeting_id: str, fmt: str):
    meeting = _get(meeting_id)
    content = {"txt": storage.to_txt, "srt": storage.to_srt}.get(fmt)
    if fmt == "md":
        text = meeting.get("summary") or ""
    elif content:
        text = content(meeting)
    else:
        raise HTTPException(400, "Format must be txt, srt or md")
    filename = f"{meeting_id}.{fmt}"
    return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/devices")
def devices():
    return list_devices()


@app.get("/api/models/{provider}")
def provider_models(provider: str):
    if provider not in summary.OPENAI_COMPATIBLE:
        raise HTTPException(404, "Unknown provider")
    key = load_settings().get(f"{provider}_api_key", "")
    if not key:
        return []
    try:
        return summary.list_models(provider, key)
    except Exception as exc:
        raise HTTPException(400, f"{summary.OPENAI_COMPATIBLE[provider]['name']}: {exc}")


@app.get("/api/recognition-models")
def recognition_models():
    """Status of the speech-recognition models: which are installed, their size, and any download
    in progress. The Settings page uses this to show a Download button next to each engine."""
    out = []
    for e in _recognition_catalog():
        try:
            present = e["present"]()
        except Exception:
            present = False
        dl = _downloads.get(e["key"], {})
        out.append({
            "key": e["key"], "label": e["label"], "engine": e["engine"],
            "expected": e["expected"], "present": present, "size": _dir_bytes(e["dir"]),
            "state": dl.get("state") or ("done" if present else "idle"),
            "percent": dl.get("percent", 100 if present else 0),
            "message": dl.get("message", ""),
        })
    return out


@app.post("/api/recognition-models/{key}/download")
def recognition_download(key: str):
    entry = next((e for e in _recognition_catalog() if e["key"] == key), None)
    if not entry:
        raise HTTPException(404, "Unknown model")
    with _dl_lock:
        if (_downloads.get(key) or {}).get("state") == "downloading":
            return {"ok": True, "state": "downloading"}
        _set_download(key, "downloading", 0)
    threading.Thread(target=_run_download, args=(entry,), daemon=True, name=f"dl-{key}").start()
    return {"ok": True, "state": "downloading"}


@app.get("/api/settings")
def settings_get():
    # The app only listens on 127.0.0.1, so the key is shown in full to the local user.
    s = load_settings()
    s["has_api_key"] = bool(s.get("anthropic_api_key"))
    s["available_providers"] = summary.available_providers(s)
    return s


@app.post("/api/settings")
def settings_post(body: dict):
    old = load_settings()
    new = save_settings(body)
    if session:
        for key, channel in (("mic_device", "mic"), ("speaker_device", "system")):
            if old[key] != new[key]:
                session.switch_device(channel, new[key])
    model_keys = ("engine", "whisper_model", "device")
    if any(old[k] != new[k] for k in model_keys) or engine.state == "error":
        if session:
            raise HTTPException(409, "The model cannot be changed while recording")
        engine.load_async(new)
    else:
        engine.settings = dict(new)
    return settings_get()


# ---------------------------------------------------------------- Murmur Copilot

class CopilotStart(BaseModel):
    type: str = "interview"
    company: str = ""
    title: str = ""
    job_description: str = ""
    resume_id: str = ""
    document_ids: list[str] | str = "all"
    language: str = "auto"
    provider: str = ""
    model: str = ""
    prefs: dict = {}
    instructions: str = ""
    auto_generate: bool = True


class AnswerBody(BaseModel):
    question: str = ""
    image: str = ""      # optional screenshot as a data URL (image/jpeg)


class LibraryItem(BaseModel):
    name: str | None = None
    text: str | None = None


class UrlBody(BaseModel):
    url: str
    name: str = ""


@app.get("/api/copilot/defaults")
def copilot_defaults():
    s = load_settings()
    d = s.get("copilot_defaults") or {}
    return {
        "language": d.get("language", "auto"),
        "provider": d.get("provider") or s.get("summary_provider", ""),
        "model": d.get("model") or s.get(f"{s.get('summary_provider', '')}_model", ""),
        "prefs": {**copilot.DEFAULT_PREFS, **(d.get("prefs") or {})},
        "instructions": d.get("instructions", copilot.DEFAULT_INSTRUCTIONS),
        "default_instructions": copilot.DEFAULT_INSTRUCTIONS,
        "auto_generate": d.get("auto_generate", True),
        "resume_id": d.get("resume_id", ""),
        "document_ids": d.get("document_ids", "all"),
        "available_providers": summary.available_providers(s),
        "claude_models": copilot.CLAUDE_MODELS,
    }


@app.post("/api/copilot/start")
def copilot_start(body: CopilotStart):
    global session
    cfg = body.model_dump()
    remember = {k: cfg[k] for k in ("language", "provider", "model", "prefs", "instructions", "auto_generate",
                                    "resume_id", "document_ids")}
    save_settings({"copilot_defaults": remember})
    title = " · ".join(x for x in (body.company.strip(), body.title.strip()) if x) or "Interview"
    with session_lock:
        if session:
            raise HTTPException(409, "A recording is already running")
        session = Session(title, cfg)
        postq.recording(True)
        return session.meeting


@app.post("/api/copilot/answer")
def copilot_answer(body: AnswerBody):
    if not session or not session.copilot:
        raise HTTPException(409, "No copilot session is running")
    image = None
    if body.image:
        import base64

        try:
            image = base64.b64decode(body.image.split(",", 1)[-1])
        except ValueError:
            raise HTTPException(400, "The screenshot could not be read")
    return session.copilot.answer(body.question, image=image)


@app.get("/api/meetings/{meeting_id}/shots/{name}")
def meeting_shot(meeting_id: str, name: str):
    path = storage.meeting_dir(meeting_id) / "shots" / Path(name).name
    if not re.fullmatch(r"\d+\.jpg", path.name) or not path.exists():
        raise HTTPException(404, "No screenshot")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/copilot/auto")
def copilot_auto(body: dict):
    if not session or not session.copilot:
        raise HTTPException(409, "No copilot session is running")
    session.copilot.cfg["auto_generate"] = bool(body.get("auto_generate"))
    session.meeting["copilot"]["auto_generate"] = session.copilot.cfg["auto_generate"]
    return {"auto_generate": session.copilot.cfg["auto_generate"]}


@app.post("/api/copilot/job-from-url")
def copilot_job_from_url(body: UrlBody):
    try:
        return copilot.job_from_page(load_settings(), body.url.strip())
    except Exception as exc:
        raise HTTPException(400, f"Could not read the page: {exc}")


def _kind(kind: str) -> str:
    if kind not in copilot.KINDS:
        raise HTTPException(404, "Unknown library")
    return kind


@app.get("/api/copilot/{kind}")
def library_list(kind: str):
    return copilot.list_items(_kind(kind))


@app.get("/api/copilot/{kind}/{item_id}")
def library_get(kind: str, item_id: str):
    item = copilot.get_item(_kind(kind), item_id)
    if not item:
        raise HTTPException(404, "Not found")
    return item


@app.post("/api/copilot/{kind}/upload")
async def library_upload(kind: str, file: UploadFile):
    _kind(kind)
    name = Path(file.filename or "file").name
    folder = copilot.LIBRARY_DIR / kind
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"upload-{uuid.uuid4().hex[:8]}{Path(name).suffix.lower()}"
    data = await file.read()
    tmp.write_bytes(data)
    try:
        text = await asyncio.to_thread(copilot.extract_text, tmp)
        item = copilot.add_item(kind, Path(name).stem, text, "uploaded", name, len(data))
        tmp.replace(folder / f"{item['id']}{tmp.suffix}")   # keep the original next to its text
        return item
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))


@app.post("/api/copilot/{kind}/url")
def library_from_url(kind: str, body: UrlBody):
    try:
        title, text = copilot.fetch_url(body.url.strip())
        return copilot.add_item(_kind(kind), body.name or title, text, "scraped", body.url.strip(), len(text))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not read the page: {exc}")


@app.post("/api/copilot/{kind}")
def library_create(kind: str, body: LibraryItem):
    try:
        return copilot.add_item(_kind(kind), body.name or "", body.text or "", "manual")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.patch("/api/copilot/{kind}/{item_id}")
def library_update(kind: str, item_id: str, body: LibraryItem):
    try:
        return copilot.update_item(_kind(kind), item_id, body.name, body.text)
    except KeyError:
        raise HTTPException(404, "Not found")


@app.delete("/api/copilot/{kind}/{item_id}")
def library_delete(kind: str, item_id: str):
    copilot.delete_item(_kind(kind), item_id)
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main():
    import uvicorn

    if not os.environ.get("MURMUR_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
