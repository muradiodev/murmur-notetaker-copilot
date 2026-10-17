"""Murmur Notetaker local web server: recording, live transcript over WebSocket, meetings API."""

import asyncio
import json
import logging
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

from . import aec, copilot, diarize, dtln, names, setup_models, storage, summary
from .audio import Capture, list_devices, make_segmenter
import soxr

from .config import (FFMPEG, HOST, PORT, PLAYBACK_RATE, PARAKEET_DIR, SAMPLE_RATE, WHISPER_DIR,
                     load_settings, save_settings)
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
        self.live_aec = None
        if dtln.available():
            try:
                self.live_aec = dtln.StreamingDTLN()
            except Exception:
                log.exception("Live echo cancellation unavailable")
        self.sys16 = np.zeros(0, np.float32)   # recent loopback at 16 kHz
        self.sys16_start = 0                   # absolute sample index of sys16[0]
        self.sys16_end = 0
        self.mic16 = np.zeros(0, np.float32)   # mic waiting for its matching loopback
        self.mic16_start = 0
        self.mic_energy: dict[int, tuple[float, float, float]] = {}  # 100 ms frame -> (raw, clean, loopback)
        self.mic_wait_since = None
        self.mic_raw = np.zeros(0, np.float32)   # recent raw mic (for measuring the echo delay)
        self.mic_raw_start = 0
        self.ref_delay = 0                        # samples the echo trails the loopback
        self.delay_checked_at = 0
        self.speakers = diarize.LiveSpeakers()
        self.meeting["speakers"] = {}
        # 16 kHz audio kept in memory (int16) so the full diarization can run while recording
        self.audio16 = {"mic": [], "system": []}
        # you can mute either source while recording; muted audio is written as silence so the
        # timeline (and screen video sync) never shifts
        self.muted = {"mic": False, "system": False}
        # Capture threads only read the device and enqueue; one processing thread does resampling,
        # echo cancellation and segmentation. A slow moment in processing never blocks the device
        # reads, so Windows never drops audio.
        self.chunks: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue()
        threading.Thread(target=self._process_loop, daemon=True, name="audio-processing").start()
        # start the devices last: their first chunk must find every field above
        for c in self.captures.values():
            c.start()
        threading.Thread(target=self._levels, daemon=True).start()
        threading.Thread(target=self._interim_loop, daemon=True).start()
        threading.Thread(target=self._rediarize_loop, daemon=True).start()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

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
                self.audio16["system"].append(np.clip(speech * 32767, -32768, 32767).astype(np.int16))
                self.sys16 = np.concatenate([self.sys16, speech])
                self.sys16_end += len(speech)
                keep = SAMPLE_RATE * 10
                if len(self.sys16) > keep:
                    self.sys16_start += len(self.sys16) - keep
                    self.sys16 = self.sys16[-keep:]
            else:
                self.mic16 = np.concatenate([self.mic16, speech])
            self._drain_mic()

    def _drain_mic(self, force: bool = False) -> None:
        """Clean pending mic audio once the loopback for the same moment has arrived.
        Loopback delivers nothing while the PC is silent, so after 300 ms the mic goes on alone."""
        if not len(self.mic16):
            return
        if self.live_aec is None:
            self.audio16["mic"].append(np.clip(self.mic16 * 32767, -32768, 32767).astype(np.int16))
            self.segmenters["mic"].feed(self.mic16)
            self.mic16_start += len(self.mic16)
            self.mic16 = np.zeros(0, np.float32)
            return
        end = self.mic16_start + len(self.mic16)
        self._update_echo_delay(end)
        if self.sys16_end < end - self.ref_delay and not force:
            now = time.monotonic()
            self.mic_wait_since = self.mic_wait_since or now
            if now - self.mic_wait_since < 0.3:
                return
        self.mic_wait_since = None
        # the reference for mic sample p is loopback sample p - delay (the echo arrives later)
        ref = np.zeros(len(self.mic16), np.float32)
        d = self.ref_delay
        a, b = max(self.mic16_start, self.sys16_start + d), min(end, self.sys16_end + d)
        if b > a:
            ref[a - self.mic16_start:b - self.mic16_start] = self.sys16[a - d - self.sys16_start:b - d - self.sys16_start]
        try:
            clean = self.live_aec.process(self.mic16, ref)
        except Exception:
            log.exception("Live echo cancellation failed; continuing without it")
            self.live_aec, clean = None, self.mic16
        frame = SAMPLE_RATE // 10
        for i in range(0, len(clean) - frame + 1, frame):
            k = (self.mic16_start + i) // frame
            self.mic_energy[k] = (float(np.mean(self.mic16[i:i + frame] ** 2)), float(np.mean(clean[i:i + frame] ** 2)),
                                  float(np.mean(ref[i:i + frame] ** 2)))
        if len(self.mic_energy) > 3000:                     # keep ~5 minutes
            for k in sorted(self.mic_energy)[:-3000]:
                del self.mic_energy[k]
        self.mic_raw = np.concatenate([self.mic_raw, self.mic16])   # history for the delay estimate
        if len(self.mic_raw) > SAMPLE_RATE * 12:
            self.mic_raw_start += len(self.mic_raw) - SAMPLE_RATE * 12
            self.mic_raw = self.mic_raw[-SAMPLE_RATE * 12:]
        self.mic16_start = end
        self.mic16 = np.zeros(0, np.float32)
        if len(clean):
            self.audio16["mic"].append(np.clip(clean * 32767, -32768, 32767).astype(np.int16))
            self.segmenters["mic"].feed(clean)

    def _on_error(self, channel: str, message: str) -> None:
        self.errors[channel] = message
        hub.send({"type": "capture_error", "channel": channel, "message": message})

    def _update_echo_delay(self, end: int) -> None:
        """Every 5 s, re-measure how far the echo trails the loopback (speakers, Bluetooth and
        drivers add 50-600 ms). Without this the echo canceller compares the wrong moments."""
        if end - self.delay_checked_at < SAMPLE_RATE * 5:
            return
        self.delay_checked_at = end
        a = max(self.mic_raw_start, self.sys16_start)
        b = min(self.mic_raw_start + len(self.mic_raw), self.sys16_end)
        if b - a < SAMPLE_RATE * 6 or float(np.mean(self.sys16[a - self.sys16_start:b - self.sys16_start] ** 2)) < 1e-6:
            return  # not enough loopback audio to measure against
        delay, corr = aec.estimate_delay(self.mic_raw[a - self.mic_raw_start:b - self.mic_raw_start],
                                         self.sys16[a - self.sys16_start:b - self.sys16_start], SAMPLE_RATE)
        if corr > 0.3 and abs(delay - self.ref_delay) > SAMPLE_RATE // 50:
            log.info("Echo delay %d ms (corr %.2f)", delay * 1000 // SAMPLE_RATE, corr)
            self.ref_delay = delay

    def _speakers_active(self, start: float, end: float) -> bool:
        frame = SAMPLE_RATE // 10
        rows = [self.mic_energy.get(k) for k in range(int(start * 10), int(end * 10) + 1)]
        rows = [r for r in rows if r]
        return bool(rows) and sum(r[2] > 1e-5 for r in rows) / len(rows) > 0.5

    def _mostly_echo(self, offset: float, audio: np.ndarray) -> bool:
        """A mic sentence spoken while the speakers played, that the echo canceller cut by more
        than 10 dB, was speaker sound, not you: real speech loses about 1 dB there."""
        frame = SAMPLE_RATE // 10
        k0 = int(offset * SAMPLE_RATE) // frame
        rows = [self.mic_energy.get(k) for k in range(k0, k0 + len(audio) // frame + 1)]
        rows = [r for r in rows if r]
        if len(rows) < 5:
            return False
        raw, clean, ref = (sum(r[i] for r in rows) for i in range(3))
        speakers_on = sum(r[2] > 1e-5 for r in rows) / len(rows) > 0.5
        return speakers_on and 10 * np.log10((raw + 1e-12) / (clean + 1e-12)) > 10

    def _on_utterance(self, channel: str, offset: float, audio: np.ndarray) -> None:
        if channel == "mic" and self._mostly_echo(offset, audio):
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
                                   "fast": self.copilot is not None}))

    def _on_pieces(self, channel: str, offset: float, pieces, audio=None) -> None:
        if channel == "system" and audio is not None:
            self.system_inflight = max(0, self.system_inflight - 1)
        if not pieces:
            return
        mid = self.meeting["id"]
        if channel == "mic":
            # while the speakers play, a mic line must be confident speech: residual echo makes
            # Whisper produce low-confidence fragments ("Thank you.", garbled half-sentences)
            pieces = [p for p in pieces if not (self._speakers_active(offset + p.start, offset + p.end)
                                                and (p.no_speech > 0.3 or p.logprob < -0.6))]
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
                    if live_echo(self.meeting["segments"], seg) is not None:
                        continue                                  # the mic only heard the speakers
                else:
                    for copy in mic_copies(self.meeting["segments"], seg):   # system arrived later
                        self.meeting["segments"].remove(copy)
                        removed.append(copy["id"])
                kept.append(seg)
        for sid in removed:
            hub.send({"type": "segment_removed", "meeting_id": mid, "id": sid})
        if not kept:
            return                                                # nothing real: no speaker gets created
        if self.copilot:
            for seg in kept:                                      # before the voiceprint: answers start sooner
                self.copilot.on_segment(seg)
        vprint = None
        try:
            speaker, vprint = self.speakers.assign(channel, audio) if audio is not None else (channel, None)
        except Exception:
            log.exception("Live speaker id failed")
            speaker = channel
        renamed = False
        with self.lock:
            names_ = self.meeting["speakers"]
            if speaker not in names_:
                names_[speaker] = f"Speaker {speaker[3:]}" if speaker.startswith("spk") else speaker
                renamed = True
            for seg in kept:
                seg["speaker"] = speaker  # live guess; the background diarization corrects it
                self.meeting["segments"].append(seg)
                self.speakers.remember(seg["id"], vprint)
                if names.is_default(names_.get(speaker, "")):
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

    def _rediarize_loop(self) -> None:
        """Run the same full diarization as after the meeting on everything recorded so far, and
        correct the live labels. Interval grows with the meeting (at least 2x the last run time)."""
        interval, seen = 20.0, 0
        window = 180.0   # only the last 3 minutes are re-clustered: cost stays flat in long meetings
        while not self.stop_event.wait(interval):
            with self.lock:
                horizon = self.elapsed - window
                segs = [dict(s) for s in self.meeting["segments"] if s["end"] >= horizon]
                tracks = {ch: np.concatenate(parts).astype(np.float32) / 32767
                          for ch, parts in self.audio16.items() if parts}
            total = len(self.meeting["segments"])
            if len(segs) < 3 or total == seen or not tracks:
                continue
            seen = total
            started = time.monotonic()
            try:
                labels = diarize.label_in_subprocess(tracks, segs, threads=2)
                for s in segs:
                    s["speaker"] = labels.get(s["id"], s.get("speaker", ""))
            except Exception:
                log.exception("Live re-diarization failed")
                continue
            took = time.monotonic() - started
            interval = max(20.0, 2 * took)
            self._apply_relabel(segs)
            log.info("Live re-diarization: %d lines, %.1fs, next in %.0fs", len(segs), took, interval)

    def _apply_relabel(self, segs: list[dict]) -> None:
        """Map the new clusters onto the live speaker keys (largest shared talk time first) so
        names stay attached, then update every line and tell the browser."""
        with self.lock:
            current = {s["id"]: s for s in self.meeting["segments"]}
            overlap: dict[tuple[str, str], float] = {}
            for s in segs:
                old = current.get(s["id"])
                if old and s.get("speaker", "").startswith("spk"):
                    pair = (s["speaker"], old["speaker"])
                    overlap[pair] = overlap.get(pair, 0.0) + (s["end"] - s["start"])
            mapping, used = {}, set()
            for (new, old), _ in sorted(overlap.items(), key=lambda kv: -kv[1]):
                if new not in mapping and old not in used:
                    mapping[new] = old
                    used.add(old)
            names_ = self.meeting["speakers"]
            assign = {}
            for s in segs:
                new = s.get("speaker", "")
                if not new.startswith("spk") or s["id"] not in current:
                    continue
                if new not in mapping:  # a speaker the live guess had merged into someone else
                    key = self.speakers._new_key()
                    mapping[new] = key
                    names_.setdefault(key, f"Speaker {key[3:]}")
                assign[s["id"]] = mapping[new]
            for sid, key in assign.items():
                current[sid]["speaker"] = key
            names.auto_name(self.meeting)
            keep = {s["speaker"] for s in self.meeting["segments"]}
            self.meeting["speakers"] = {k: v for k, v in names_.items() if k in keep}
            self.speakers.rebuild({sid: k for sid, k in assign.items()})
            storage.save(self.meeting)
            speakers_now = dict(self.meeting["speakers"])
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
                                  self._on_interim, context={"interim": True, "language": self.language}))

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
        last = self.speakers.last.get(channel)
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
                        self.mic16 = np.concatenate([self.mic16, tail])
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
        engine.wait_idle()
        with self.lock:
            self.closed = True
            for w in [*self.writers.values(), *self.hq_writers.values()]:
                w.close()
            remove_echo(self.meeting["segments"])
        try:
            build_audio(self.folder)
            finalize_screens(self.meeting)
            live_spans = live_name_spans(self.meeting)
            clean_mic_for_recognition(self.meeting)
            if self.settings.get("refine_after_recording", True) and engine.state == "ready":
                refine_transcript(self.meeting)
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
    context = {"quality": quality}
    lang = (meeting.get("copilot") or {}).get("language")
    if lang in ("az", "de", "en"):
        context["language"] = lang
    for ch, audio in tracks.items():
        segmenter = make_segmenter(ch, lambda c, off, a: jobs.append(
            Job(c, off, a, on_pieces, context={**context, "lang_key": f"{mid}-{c}", "names": known_names(meeting)})),
            max_segment=28.0 if quality else None)
        step = SAMPLE_RATE * 10
        for i in range(0, len(audio), step):
            segmenter.feed(audio[i:i + step])
        segmenter.flush()
    # Interleave channels in time order so progress moves forward steadily.
    for job in sorted(jobs, key=lambda j: j.offset):
        engine.submit(job)
    engine.wait_idle()


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


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
        clean_mic_for_recognition(meeting)
        tracks = {ch: sf.read(folder / f"{ch}.ogg", dtype="float32")[0]
                  for ch in CHANNELS if (folder / f"{ch}.ogg").exists()}
        hub.send({"type": "meeting", "meeting_id": meeting_id, "status": "processing", "step": "transcribing"})
        transcribe_tracks(meeting, tracks, quality=True)
        remove_echo(meeting["segments"])
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


def remove_echo(segments: list[dict]) -> None:
    """Drop mic segments that are just the meeting audio leaking from the speakers into the mic."""
    system = [s for s in segments if s["channel"] == "system"]
    keep = []
    for seg in segments:
        if seg["channel"] == "mic":
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
    diarize.label_meeting(tracks, meeting["segments"], num_speakers)
    used = sorted({s["speaker"] for s in meeting["segments"]}, key=lambda k: (not k.startswith("spk"), len(k), k))
    meeting["speakers"] = {k: f"Speaker {k[3:]}" if k.startswith("spk") else meeting["speakers"].get(k, k) for k in used}
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
            if first.startswith("# ") and meeting["title"].startswith("Meet "):
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


@app.on_event("startup")
async def on_startup():
    hub.loop = asyncio.get_running_loop()
    engine.load_async(load_settings())
    unfinished = [i["id"] for i in storage.list_meetings() if i["status"] != "ready"]
    if unfinished:
        threading.Thread(target=recover_meetings, args=(unfinished,), daemon=True).start()


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
        return session.meeting


@app.post("/api/record/stop")
def record_stop():
    global session
    with session_lock:
        if not session:
            raise HTTPException(409, "No recording is running")
        meeting = session.stop()
        session = None
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
    threading.Thread(target=process_import, args=(meeting, source), daemon=True).start()
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
    if body.title is not None:
        meeting["title"] = body.title.strip() or meeting["title"]
    if body.archived is not None:
        meeting["archived"] = body.archived
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


@app.post("/api/meetings/{meeting_id}/retranscribe")
def meeting_retranscribe(meeting_id: str):
    meeting = _get(meeting_id)
    if meeting["status"] != "ready":
        raise HTTPException(409, "The meeting is still being processed")
    threading.Thread(target=retranscribe, args=(meeting_id,), daemon=True).start()
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

    threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
