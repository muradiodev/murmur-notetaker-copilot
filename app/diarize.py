"""Speaker diarization across both channels (sherpa-onnx, no PyTorch, no token)."""

import logging
import os
from bisect import bisect_right

import numpy as np

from .config import DIARIZATION_DIR, SAMPLE_RATE

log = logging.getLogger(__name__)

SEGMENTATION_MODEL = DIARIZATION_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING_MODEL = DIARIZATION_DIR / "wespeaker_en_voxceleb_resnet34.onnx"
THREADS = max(1, (os.cpu_count() or 2) // 2)
PAD = 0.3   # seconds of context kept around each transcribed segment
GAP = 0.5   # silence inserted between stitched pieces
THRESHOLD = 0.35  # clustering distance for WeSpeaker embeddings: lower = more speakers (tuned on real calls)


def available() -> bool:
    return SEGMENTATION_MODEL.exists() and EMBEDDING_MODEL.exists()


def diarize(samples: np.ndarray, num_speakers: int = 0, threads: int | None = None) -> list[tuple[float, float, int]]:
    """Return (start, end, speaker_index) turns. num_speakers=0 means detect automatically."""
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEGMENTATION_MODEL)),
            num_threads=threads or THREADS,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING_MODEL), num_threads=threads or THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1,
            threshold=THRESHOLD,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Diarization config is invalid (models missing?)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    assert sd.sample_rate == SAMPLE_RATE
    result = sd.process(samples.astype(np.float32)).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in result]


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


_extractor = None


def _embedder():
    global _extractor
    if _extractor is None:
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING_MODEL), num_threads=THREADS)
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    return _extractor


import threading as _threading

_embed_lock = _threading.Lock()


def voiceprint(audio: np.ndarray) -> np.ndarray:
    with _embed_lock:  # the live worker and the background re-diarization share one extractor
        return _voiceprint(audio)


def _voiceprint(audio: np.ndarray) -> np.ndarray:
    ex = _embedder()
    stream = ex.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio.astype(np.float32))
    stream.input_finished()
    v = np.asarray(ex.compute(stream), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _reassign_by_voiceprint(tracks, segments, raw: dict[str, int], margin: float = 0.08) -> None:
    prints = {}
    for seg in segments:
        if seg["id"] not in raw or seg["end"] - seg["start"] < 1.2:
            continue
        audio = tracks.get(seg["channel"])
        if audio is None:
            continue
        clip = audio[int(seg["start"] * SAMPLE_RATE):int(seg["end"] * SAMPLE_RATE)]
        if len(clip) >= SAMPLE_RATE:
            prints[seg["id"]] = voiceprint(clip)
    labels = sorted(set(raw.values()))
    if len(labels) < 2 or len(prints) < 3:
        return
    for _ in range(2):
        centroids = {}
        for lab in labels:
            vs = [prints[i] for i, l in raw.items() if l == lab and i in prints]
            if vs:
                c = np.mean(vs, axis=0)
                centroids[lab] = c / (np.linalg.norm(c) + 1e-9)
        moved = 0
        for sid, v in prints.items():
            own = raw[sid]
            if own not in centroids:
                continue
            scores = {lab: float(v @ c) for lab, c in centroids.items()}
            best = max(scores, key=scores.get)
            if best != own and scores[best] - scores[own] > margin:
                raw[sid] = best
                moved += 1
        if not moved:
            break


def label_meeting(tracks: dict[str, np.ndarray], segments: list[dict], num_speakers: int = 0,
                  threads: int | None = None) -> None:
    """Diarize every channel together and set seg["speaker"] to "spk1", "spk2", ...

    Only the transcribed parts of each channel are stitched into one signal, so a
    person is clustered the same way whether they spoke into the mic or remotely,
    and silence is not processed twice.
    """
    pieces: dict[str, list[tuple[float, float]]] = {}  # channel -> [(orig_start, stitched_start)]
    parts, cursor = [], 0
    gap = np.zeros(int(GAP * SAMPLE_RATE), dtype=np.float32)
    for ch, audio in tracks.items():
        spans = _merge([(max(0.0, s["start"] - PAD), s["end"] + PAD) for s in segments if s["channel"] == ch])
        for a, b in spans:
            chunk = audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
            if not len(chunk):
                continue
            pieces.setdefault(ch, []).append((a, cursor / SAMPLE_RATE))
            parts += [chunk, gap]
            cursor += len(chunk) + len(gap)
    if not parts:
        return

    turns = diarize(np.concatenate(parts), num_speakers, threads)

    def stitched(seg: dict) -> tuple[float, float]:
        ch_pieces = pieces.get(seg["channel"], [])
        i = max(0, bisect_right([p[0] for p in ch_pieces], seg["start"]) - 1)
        orig, at = ch_pieces[i] if ch_pieces else (0.0, 0.0)
        return at + seg["start"] - orig, at + seg["end"] - orig

    raw: dict[str, int] = {}
    for seg in segments:
        if seg["channel"] not in pieces:
            continue
        s0, s1 = stitched(seg)
        best, best_overlap = None, 0.0
        for start, end, spk in turns:
            overlap = min(end, s1) - max(start, s0)
            if overlap > best_overlap:
                best, best_overlap = spk, overlap
        if best is None and turns:
            mid = (s0 + s1) / 2
            best = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        raw[seg["id"]] = best if best is not None else 0

    # Second look, line by line: compare each line's own voiceprint with every speaker's average
    # voiceprint and move it if it clearly sounds like someone else. Fixes lines that the
    # window-based clustering glued to the previous speaker.
    try:
        _reassign_by_voiceprint(tracks, segments, raw)
    except Exception:
        log.exception("Voiceprint reassignment skipped")

    # Number speakers in order of first appearance: whoever talks first is Speaker 1.
    order: dict[int, int] = {}
    for seg in sorted(segments, key=lambda s: s["start"]):
        if seg["id"] in raw:
            idx = order.setdefault(raw[seg["id"]], len(order) + 1)
            seg["speaker"] = f"spk{idx}"



class LiveSpeakers:
    """Online speaker labels while recording: each sentence's voiceprint joins the closest known
    speaker, or starts a new one. A background full diarization regularly corrects these guesses
    (see Session._rediarize_loop); `rebuild` then re-seeds the centroids from its result."""

    # Cosine similarity to join a known speaker (same speaker >= 0.93, different <= 0.72 on our
    # 4-voice test; real compressed audio overlaps more, which the background pass fixes).
    def __init__(self, threshold: float = 0.74):
        self.threshold = threshold
        self.centroids: dict[str, np.ndarray] = {}
        self.last: dict[str, str] = {}          # channel -> last speaker key
        self.prints: dict[str, np.ndarray] = {}  # segment id -> voiceprint
        self._next = 1

    def _new_key(self) -> str:
        while f"spk{self._next}" in self.centroids:
            self._next += 1
        key = f"spk{self._next}"
        self._next += 1
        return key

    def assign(self, channel: str, audio: np.ndarray) -> tuple[str, np.ndarray | None]:
        if len(audio) < SAMPLE_RATE * 0.9:          # too short for a reliable voiceprint
            key = self.last.get(channel) or self._new_key()
            self.centroids.setdefault(key, np.zeros(_embedder().dim, dtype=np.float32))
            self.last[channel] = key
            return key, None
        v = voiceprint(audio)
        best, score = None, -1.0
        for key, c in self.centroids.items():
            n = np.linalg.norm(c)
            if n > 0:
                sim = float(v @ (c / n))
                if sim > score:
                    best, score = key, sim
        if best is None or score < self.threshold:
            best = self._new_key()
            self.centroids[best] = np.zeros_like(v)
        self.centroids[best] = self.centroids[best] + v
        self.last[channel] = best
        return best, v

    def remember(self, segment_id: str, v: np.ndarray | None) -> None:
        if v is not None:
            self.prints[segment_id] = v

    def rebuild(self, labels: dict[str, str]) -> None:
        """Re-seed centroids from corrected labels {segment id: speaker key}."""
        cents: dict[str, np.ndarray] = {}
        for sid, key in labels.items():
            v = self.prints.get(sid)
            if v is not None:
                cents[key] = cents.get(key, np.zeros_like(v)) + v
        for key in set(labels.values()):
            self.centroids[key] = cents.get(key, self.centroids.get(key, np.zeros(_embedder().dim, dtype=np.float32)))
        for key in list(self.centroids):
            if key not in set(labels.values()) and not np.linalg.norm(self.centroids[key]):
                del self.centroids[key]


def _label_job(tracks: dict[str, np.ndarray], segments: list[dict], threads: int) -> dict[str, str]:
    """Runs in a child process: returns {segment id: speaker key}."""
    label_meeting(tracks, segments, 0, threads=threads)
    return {s["id"]: s.get("speaker", "") for s in segments}


_pool = None


def label_in_subprocess(tracks: dict[str, np.ndarray], segments: list[dict], threads: int = 2) -> dict[str, str]:
    """Full diarization in a separate process. sherpa-onnx keeps the GIL while it works, which
    starves the audio capture threads (Windows then drops samples: "data discontinuity").
    A child process has its own GIL, so recording never waits for diarization."""
    global _pool
    import concurrent.futures
    import multiprocessing

    if _pool is None:
        _pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    try:
        return _pool.submit(_label_job, tracks, segments, threads).result()
    except concurrent.futures.process.BrokenProcessPool:
        _pool = None
        raise
