"""Microphone + system-audio (WASAPI loopback) capture and a simple speech segmenter."""

import ctypes
import logging
import threading
import time
from collections import deque

import numpy as np
import soundcard as sc

from .config import PLAYBACK_RATE, SAMPLE_RATE, SILERO_VAD

log = logging.getLogger(__name__)

CHUNK = PLAYBACK_RATE // 10  # 100 ms at the capture rate


def list_devices() -> dict:
    mics = [{"id": m.id, "name": m.name} for m in sc.all_microphones(include_loopback=False)]
    speakers = [{"id": s.id, "name": s.name} for s in sc.all_speakers()]
    return {"microphones": mics, "speakers": speakers}


def _get_mic(device_id: str):
    if device_id:
        try:
            return sc.get_microphone(device_id, include_loopback=False)
        except (IndexError, ValueError):
            log.warning("Mic %s not found, using default", device_id)
    return sc.default_microphone()


def _get_loopback(speaker_id: str):
    speaker = None
    if speaker_id:
        try:
            speaker = sc.get_speaker(speaker_id)
        except (IndexError, ValueError):
            log.warning("Speaker %s not found, using default", speaker_id)
    speaker = speaker or sc.default_speaker()
    return sc.get_microphone(str(speaker.id), include_loopback=True)


class Capture(threading.Thread):
    """Records one source and emits 16 kHz mono float32 chunks.

    WASAPI loopback delivers nothing while the system is silent, so gaps are
    filled with zeros against the wall clock to keep both channels aligned.
    """

    def __init__(self, channel: str, device_id: str, on_chunk, on_error, stop_event, t0: float, written: int = 0):
        super().__init__(daemon=True, name=f"capture-{channel}")
        self.channel = channel
        self.device_id = device_id
        self.on_chunk = on_chunk
        self.on_error = on_error
        self.stop_event = stop_event
        self.t0 = t0
        # Samples already in the file: a device switch mid-meeting continues the same timeline.
        self.written = written
        self.halt = threading.Event()

    def _emit(self, samples: np.ndarray) -> None:
        self.written += len(samples)
        self.on_chunk(self.channel, samples)

    def run(self) -> None:
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        except (AttributeError, OSError):
            pass
        try:
            device = _get_mic(self.device_id) if self.channel == "mic" else _get_loopback(self.device_id)
            log.info("%s capture: %s", self.channel, device.name)
            with device.recorder(samplerate=PLAYBACK_RATE, blocksize=CHUNK) as rec:
                while not self.stop_event.is_set() and not self.halt.is_set():
                    data = rec.record(numframes=CHUNK)
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    mono = np.nan_to_num(mono.astype(np.float32))
                    expected = int((time.monotonic() - self.t0) * PLAYBACK_RATE)
                    gap = expected - self.written - len(mono)
                    if gap > PLAYBACK_RATE * 0.3:
                        self._emit(np.zeros(gap, dtype=np.float32))
                    self._emit(mono)
        except Exception as exc:  # device errors must not kill the session
            log.exception("%s capture failed", self.channel)
            self.on_error(self.channel, str(exc))

    def pad_to(self, total_samples: int) -> None:
        """Pad with silence up to the session length (called at stop)."""
        gap = total_samples - self.written
        if gap > 0:
            self._emit(np.zeros(gap, dtype=np.float32))


class Segmenter:
    """Energy-based VAD that cuts the stream into utterances for transcription."""

    FRAME = 480  # 30 ms
    PREROLL_FRAMES = 10
    MIN_SPEECH = 0.4
    END_SILENCE = 0.55
    MAX_SEGMENT = 18.0

    def __init__(self, channel: str, on_segment, max_segment: float | None = None):
        self.channel = channel
        self.on_segment = on_segment
        self.max_segment = max_segment or self.MAX_SEGMENT
        self.pos = 0
        self.rest = np.zeros(0, dtype=np.float32)
        self.preroll = deque(maxlen=self.PREROLL_FRAMES)
        self.buf: list[np.ndarray] = []
        self.start = 0
        self.in_speech = False
        self.silence = 0
        self.speech = 0
        self.noise = 0.002
        self.level = 0.0

    def feed(self, samples: np.ndarray) -> None:
        data = np.concatenate([self.rest, samples])
        n = len(data) // self.FRAME
        self.rest = data[n * self.FRAME:]
        for i in range(n):
            self._frame(data[i * self.FRAME:(i + 1) * self.FRAME])

    def _frame(self, frame: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
        self.level = 0.8 * self.level + 0.2 * rms
        is_speech = rms > max(0.004, self.noise * 3.0)
        if not is_speech:
            self.noise = 0.97 * self.noise + 0.03 * rms

        if is_speech and not self.in_speech:
            self.in_speech = True
            self.buf = list(self.preroll)
            self.start = self.pos - len(self.buf) * self.FRAME
            self.silence = 0
            self.speech = 0

        if self.in_speech:
            self.buf.append(frame)
            if is_speech:
                self.speech += self.FRAME
                self.silence = 0
            else:
                self.silence += self.FRAME
            duration = len(self.buf) * self.FRAME / SAMPLE_RATE
            if self.silence >= self.END_SILENCE * SAMPLE_RATE:
                self._cut(keep_going=False)
            elif duration >= self.max_segment:
                self._cut(keep_going=True)
        else:
            self.preroll.append(frame)
        self.pos += self.FRAME

    def _cut(self, keep_going: bool) -> None:
        if self.speech >= self.MIN_SPEECH * SAMPLE_RATE and self.buf:
            self.on_segment(self.channel, self.start / SAMPLE_RATE, np.concatenate(self.buf))
        self.buf = []
        self.speech = 0
        self.silence = 0
        self.preroll.clear()
        self.in_speech = keep_going
        self.start = self.pos + self.FRAME

    def flush(self) -> None:
        if self.in_speech:
            self._cut(keep_going=False)

    def speaking(self) -> bool:
        return self.in_speech

    def pending(self, min_seconds: float = 0.8):
        """(start_sec, audio) of the utterance still being spoken, for interim captions; else None."""
        if not self.in_speech or self.speech < min_seconds * SAMPLE_RATE or not self.buf:
            return None
        return self.start / SAMPLE_RATE, np.concatenate(self.buf)


class SileroSegmenter:
    """Neural voice-activity segmentation (Silero VAD via sherpa-onnx), same interface as Segmenter.

    Far more robust to keyboard, fan and room noise than an energy threshold, so fewer false cuts
    and fewer Whisper hallucinations on non-speech. Settings follow Meetily's live path: onset 0.5,
    250 ms minimum speech, 500 ms of silence closes a sentence, 300 ms kept before speech.
    """

    WINDOW = 512           # samples per Silero step at 16 kHz (32 ms)
    PRE_PAD = 0.3          # seconds of audio kept before the detected start
    HISTORY = 30           # seconds kept for that padding

    def __init__(self, channel: str, on_segment, max_segment: float | None = None):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(SILERO_VAD)
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.5
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = max_segment or Segmenter.MAX_SEGMENT
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = 1
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
        self.channel = channel
        self.on_segment = on_segment
        self.rest = np.zeros(0, dtype=np.float32)
        self.history = np.zeros(SAMPLE_RATE * self.HISTORY, dtype=np.float32)
        self.pos = 0  # samples fed so far
        self.level = 0.0

    def _remember(self, x: np.ndarray) -> None:
        n = len(x)
        if n >= len(self.history):
            self.history[:] = x[-len(self.history):]
        else:
            self.history = np.roll(self.history, -n)
            self.history[-n:] = x
        self.pos += n

    def _padded(self, start: int, samples: np.ndarray) -> tuple[float, np.ndarray]:
        pad = int(self.PRE_PAD * SAMPLE_RATE)
        first = max(0, start - pad)
        back = self.pos - first          # how far back the padding starts
        have = min(back, len(self.history))
        lead_len = min(start - first, have)
        if lead_len > 0:
            begin = len(self.history) - back
            lead = self.history[max(0, begin):max(0, begin) + lead_len]
            return (start - len(lead)) / SAMPLE_RATE, np.concatenate([lead, samples])
        return start / SAMPLE_RATE, samples

    def _drain(self) -> None:
        while not self.vad.empty():
            seg = self.vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            self.vad.pop()
            if len(samples) >= SAMPLE_RATE * 0.25:
                offset, audio = self._padded(seg.start, samples)
                self.on_segment(self.channel, offset, audio)

    def feed(self, samples: np.ndarray) -> None:
        data = np.concatenate([self.rest, samples])
        n = len(data) // self.WINDOW
        self.rest = data[n * self.WINDOW:]
        for i in range(n):
            w = data[i * self.WINDOW:(i + 1) * self.WINDOW]
            self.level = 0.8 * self.level + 0.2 * float(np.sqrt(np.mean(w * w)))
            self.vad.accept_waveform(w)
            self._remember(w)
        self._drain()

    def flush(self) -> None:
        if len(self.rest):
            w = np.zeros(self.WINDOW, dtype=np.float32)
            w[: len(self.rest)] = self.rest
            self.vad.accept_waveform(w)
            self._remember(w)
            self.rest = np.zeros(0, dtype=np.float32)
        self.vad.flush()
        self._drain()

    def speaking(self) -> bool:
        return self.vad.is_speech_detected()

    def pending(self, min_seconds: float = 0.8):
        if not self.vad.is_speech_detected():
            return None
        seg = self.vad.current_segment
        samples = np.asarray(seg.samples, dtype=np.float32)
        if len(samples) < min_seconds * SAMPLE_RATE:
            return None
        return seg.start / SAMPLE_RATE, samples


def make_segmenter(channel: str, on_segment, max_segment: float | None = None):
    """Silero when its model is present (setup.bat downloads it), otherwise the energy segmenter."""
    if SILERO_VAD.exists():
        try:
            return SileroSegmenter(channel, on_segment, max_segment)
        except Exception:
            log.exception("Silero VAD unavailable, using the energy segmenter")
    return Segmenter(channel, on_segment, max_segment)
