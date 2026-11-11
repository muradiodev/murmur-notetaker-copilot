"""Live microphone path: echo cancellation against the loopback and the "was that me?" decision.

With loudspeakers (a Bluetooth box, laptop speakers) the microphone hears the other participants
too. Their voices already arrive clean on the system channel, so anything of them on the mic
channel is a duplicate. Three layers keep it out of the live transcript:

1. DTLN-aec cleans the mic with the loopback as reference (`MicPath.drain`), aligned by the
   measured echo delay (Bluetooth: 150-700 ms, re-measured every 5 s).
2. A mic sentence that the canceller cut by more than 10 dB while the speakers played was speaker
   sound, not you (`MicPath.echo_stats`).
3. What survives is held until the system channel has transcribed the same moment, then compared
   word by word and with your voiceprint (`EchoJudge`). Only then is it shown, so a duplicate never
   appears and disappears again.
"""

import logging
import re
import time

import numpy as np

from . import aec, dtln
from .config import SAMPLE_RATE

log = logging.getLogger(__name__)

FRAME = SAMPLE_RATE // 10          # 100 ms bookkeeping frames
SPEAKERS_ON = 1e-5                 # loopback power above this: the speakers are playing
# Similarity of a mic line's voiceprint (ERes2NetV2) to yours. Measured by replaying two recorded
# calls (Bluetooth speaker + webcam mic) through this path: your lines 0.76-0.96; the other side
# leaking from the speaker 0.46-0.67; Whisper's inventions on room noise ("Thank you", subtitles
# credits, the vocabulary list) 0.25-0.65.
NOT_ME = 0.68
SURELY_ME = 0.80
# Whisper's favourite inventions on a quiet or echoing mic. A short mic line (too short for a
# voiceprint) that says only this is dropped in a call.
INVENTED = re.compile(r"^(thank you( very much| so much)?|thanks|amen|you|bye[- ]bye)[\s.!,]*$", re.IGNORECASE)


class MicPath:
    def __init__(self, use_aec: bool = True):
        self.live_aec = None
        if use_aec and dtln.available():
            try:
                self.live_aec = dtln.StreamingDTLN()
            except Exception:
                log.exception("Live echo cancellation unavailable")
        self.sys16 = np.zeros(0, np.float32)   # recent loopback at 16 kHz
        self.sys16_start = 0                   # absolute sample index of sys16[0]
        self.sys16_end = 0
        self.mic16 = np.zeros(0, np.float32)   # mic waiting for its matching loopback
        self.mic16_start = 0
        self.energy: dict[int, tuple[float, float, float]] = {}  # 100 ms frame -> (raw, clean, loopback)
        self.wait_since = None
        self.mic_raw = np.zeros(0, np.float32)  # recent raw mic (for measuring the echo delay)
        self.mic_raw_start = 0
        self.ref_delay = 0                      # samples the echo trails the loopback
        self.delay_checked_at = 0

    def add_system(self, speech: np.ndarray) -> None:
        self.sys16 = np.concatenate([self.sys16, speech])
        self.sys16_end += len(speech)
        keep = SAMPLE_RATE * 10
        if len(self.sys16) > keep:
            self.sys16_start += len(self.sys16) - keep
            self.sys16 = self.sys16[-keep:]

    def add_mic(self, speech: np.ndarray) -> None:
        self.mic16 = np.concatenate([self.mic16, speech])

    def drain(self, force: bool = False) -> np.ndarray:
        """Clean pending mic audio once the loopback for the same moment has arrived. Loopback
        delivers nothing while the PC is silent, so after 300 ms the mic goes on alone."""
        if not len(self.mic16):
            return np.zeros(0, np.float32)
        end = self.mic16_start + len(self.mic16)
        if self.live_aec is None:
            out = self.mic16
            self._account(out, out, np.zeros(len(out), np.float32))
            self.mic16_start, self.mic16 = end, np.zeros(0, np.float32)
            return out
        self._update_delay(end)
        if self.sys16_end < end - self.ref_delay and not force:
            now = time.monotonic()
            self.wait_since = self.wait_since or now
            if now - self.wait_since < 0.3:
                return np.zeros(0, np.float32)
        self.wait_since = None
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
        self._account(self.mic16, clean, ref)
        self.mic_raw = np.concatenate([self.mic_raw, self.mic16])   # history for the delay estimate
        if len(self.mic_raw) > SAMPLE_RATE * 12:
            self.mic_raw_start += len(self.mic_raw) - SAMPLE_RATE * 12
            self.mic_raw = self.mic_raw[-SAMPLE_RATE * 12:]
        self.mic16_start = end
        self.mic16 = np.zeros(0, np.float32)
        return clean

    def _account(self, raw: np.ndarray, clean: np.ndarray, ref: np.ndarray) -> None:
        for i in range(0, min(len(clean), len(raw)) - FRAME + 1, FRAME):
            k = (self.mic16_start + i) // FRAME
            self.energy[k] = (float(np.mean(raw[i:i + FRAME] ** 2)), float(np.mean(clean[i:i + FRAME] ** 2)),
                              float(np.mean(ref[i:i + FRAME] ** 2)))
        if len(self.energy) > 3000:                     # keep ~5 minutes
            for k in sorted(self.energy)[:-3000]:
                del self.energy[k]

    def _update_delay(self, end: int) -> None:
        """Every 5 s, re-measure how far the echo trails the loopback (speakers, Bluetooth and
        drivers add 50-700 ms). Without this the echo canceller compares the wrong moments."""
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

    def _rows(self, start: float, end: float) -> list[tuple[float, float, float]]:
        rows = [self.energy.get(k) for k in range(int(start * 10), int(end * 10) + 1)]
        return [r for r in rows if r]

    def speakers_active(self, start: float, end: float) -> bool:
        rows = self._rows(start, end)
        return bool(rows) and sum(r[2] > SPEAKERS_ON for r in rows) / len(rows) > 0.5

    def echo_stats(self, start: float, end: float) -> tuple[float, float]:
        """(share of the time the speakers played, how much the canceller removed in dB)."""
        rows = self._rows(start, end)
        if len(rows) < 3:
            return 0.0, 0.0
        raw, clean = sum(r[0] for r in rows), sum(r[1] for r in rows)
        on = sum(r[2] > SPEAKERS_ON for r in rows) / len(rows)
        return on, float(10 * np.log10((raw + 1e-12) / (clean + 1e-12)))


# --------------------------------------------------------------------------- decisions

def mostly_echo(on: float, drop_db: float) -> bool:
    """Speakers played for most of the sentence and the canceller removed most of it: real speech
    loses about 1 dB there, speaker sound 10 dB or more."""
    return on > 0.5 and drop_db > 10


def words(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text.lower())


def echo_of(mic_text: str, start: float, end: float, system_lines: list[dict], window: float = 3.0) -> bool:
    """The mic line repeats what the system channel said around the same time (speaker bleed)."""
    w = words(mic_text)
    if not w:
        return False
    theirs = {x for s in system_lines if s["start"] <= end + window and s["end"] >= start - window
              for x in words(s["text"])}
    return bool(theirs) and sum(x in theirs for x in w) / len(w) >= 0.6


class Voice:
    """Your voiceprint, learned from what the mic hears while the speakers are silent (that can only
    be you, or someone next to you). Kept between meetings in data/my_voice.npy."""

    def __init__(self, path=None):
        self.path = path
        self.sum = None
        self.count = 0
        if path is not None and path.exists():
            try:
                data = np.load(path)
                self.sum, self.count = data[:-1].astype(np.float32), int(data[-1])
            except Exception:
                log.warning("Could not read %s", path)

    def add(self, v: np.ndarray) -> None:
        self.sum = v.copy() if self.sum is None or self.sum.shape != v.shape else self.sum + v
        self.count += 1
        if self.count > 200:                        # keep adapting: old evidence fades
            self.sum *= 200 / self.count
            self.count = 200

    def ready(self) -> bool:
        return self.sum is not None and self.count >= 3

    def similarity(self, v: np.ndarray) -> float:
        if self.sum is None or self.sum.shape != v.shape:
            return 1.0
        c = self.sum / (np.linalg.norm(self.sum) + 1e-9)
        return float(v @ c)

    def save(self) -> None:
        if self.path is not None and self.sum is not None:
            try:
                np.save(self.path, np.append(self.sum, self.count).astype(np.float32))
            except Exception:
                log.warning("Could not save %s", self.path)
