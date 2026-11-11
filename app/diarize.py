"""Who said what: speaker labels for a meeting, live and after the recording (sherpa-onnx, CPU).

How a call is split:
- The microphone is you ("me"). In an online meeting the other people only come through the
  system channel; whatever of them leaks into the mic is removed as echo (see livemic.py). When
  the system channel is (almost) silent, it is an in-person meeting: the mic voices are clustered.
- The system channel (Teams, Zoom ...) is split by voice: every transcribed line gets a voiceprint
  (3D-Speaker ERes2NetV2, trained on 200k speakers; on our own calls it separates two remote voices
  with 5% errors, the older WeSpeaker ResNet34 with 16%), and lines are grouped by average-linkage
  clustering. Lines too short for a voiceprint join their neighbour.
Tested on the user's 1:1 calls (the other side always came out as one person; before: 4-5) and
group calls (a 30-minute team meeting: 6 instead of 28 speakers).
"""

import logging
import os
import threading

import numpy as np

from .config import DIARIZATION_DIR, SAMPLE_RATE

log = logging.getLogger(__name__)

# Best model first; the old one stays as a fallback until setup_models fetched the new one.
EMBEDDING_MODELS = [
    (DIARIZATION_DIR / "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx", 0.48, 0.50),
    (DIARIZATION_DIR / "wespeaker_en_voxceleb_resnet34.onnx", 0.72, 0.74),
]
THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))
MIN_PRINT = 1.2        # seconds: shorter lines give unreliable voiceprints
SMALL_SPEAKER = 12.0   # seconds of talk: a cluster this small is usually a stray part of a bigger one
ME = "me"


def _model():
    for path, cluster_th, join_th in EMBEDDING_MODELS:
        if path.exists():
            return path, cluster_th, join_th
    return None, 0.0, 0.0


EMBEDDING_MODEL, CLUSTER_THRESHOLD, JOIN_THRESHOLD = _model()


def available() -> bool:
    return EMBEDDING_MODEL is not None


# --------------------------------------------------------------------------- voiceprints

_extractor = None
_embed_lock = threading.Lock()


def _embedder():
    global _extractor
    if _extractor is None:
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING_MODEL), num_threads=THREADS)
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    return _extractor


def voiceprint(audio: np.ndarray) -> np.ndarray:
    """Unit-length voice embedding of a 16 kHz clip. Holds the GIL while it runs: call it in a child
    process while recording (`VoiceWorker`), never on the audio path."""
    with _embed_lock:
        ex = _embedder()
        stream = ex.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio.astype(np.float32))
        stream.input_finished()
        v = np.asarray(ex.compute(stream), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _embed_job(audio16: np.ndarray) -> np.ndarray:
    return voiceprint(audio16.astype(np.float32) / 32767)


class VoiceWorker:
    """Voiceprints computed in a child process: sherpa-onnx keeps the GIL while it works, which
    starves the audio capture threads (Windows then drops samples). The child has its own GIL."""

    def __init__(self):
        import concurrent.futures
        import multiprocessing

        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=1,
                                                           mp_context=multiprocessing.get_context("spawn"))

    def embed(self, audio: np.ndarray, timeout: float = 10.0) -> np.ndarray | None:
        try:
            clip = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            return self.pool.submit(_embed_job, clip).result(timeout=timeout)
        except Exception:
            log.exception("Voiceprint failed")
            return None

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- clustering

def cluster(V: np.ndarray, weights: np.ndarray, threshold: float, num: int = 0) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine similarity (weighted by talk time).
    Merges while the two closest groups are more similar than `threshold`; with `num`, until
    exactly `num` groups are left. Returns a label per row."""
    n = len(V)
    if n == 0:
        return np.zeros(0, int)
    sim = (V @ V.T).astype(np.float64)
    np.fill_diagonal(sim, -9.0)
    size = weights.astype(np.float64).copy()
    members = {i: [i] for i in range(n)}
    groups = n
    while groups > 1:                      # merged-away rows and columns are set to -9
        k = int(np.argmax(sim))
        a, b = divmod(k, n)
        if num > 0:
            if groups <= num:
                break
        elif sim[a, b] < threshold:
            break
        wa, wb = size[a], size[b]
        row = (sim[a] * wa + sim[b] * wb) / (wa + wb)
        sim[a, :] = row
        sim[:, a] = row
        sim[a, a] = -9.0
        sim[b, :] = -9.0
        sim[:, b] = -9.0
        size[a] = wa + wb
        members[a] += members.pop(b)
        groups -= 1
    labels = np.zeros(n, int)
    for idx, (_, rows) in enumerate(sorted(members.items(), key=lambda kv: min(kv[1]))):
        labels[rows] = idx
    return labels


def _centroids(V: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for c in set(labels.tolist()):
        v = V[labels == c].mean(axis=0)
        out[c] = v / (np.linalg.norm(v) + 1e-9)
    return out


def _absorb_small(V: np.ndarray, w: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """A speaker with only a few seconds of talk is usually a stray part of a bigger one (a cough,
    a laugh, a line said away from the microphone): join the most similar big speaker if close."""
    talk = {c: float(w[labels == c].sum()) for c in set(labels.tolist())}
    big = [c for c, t in talk.items() if t >= SMALL_SPEAKER]
    if not big:
        return labels
    cents = _centroids(V, labels)
    out = labels.copy()
    for c, t in talk.items():
        if t < SMALL_SPEAKER:
            best = max(big, key=lambda b: float(cents[c] @ cents[b]))
            if float(cents[c] @ cents[best]) > threshold - 0.12:
                out[labels == c] = best
    return out


def group_lines(segs: list[dict], prints: dict[str, np.ndarray], num: int = 0,
                threshold: float | None = None) -> dict[str, int]:
    """Lines of one channel -> {segment id: group index}. Lines without a voiceprint (too short)
    follow the line just before them (or the nearest one)."""
    th = CLUSTER_THRESHOLD if threshold is None else threshold
    segs = sorted(segs, key=lambda s: s["start"])
    long_ = [s for s in segs if s["id"] in prints]
    out: dict[str, int] = {}
    if long_:
        V = np.stack([prints[s["id"]] for s in long_])
        w = np.array([max(0.1, s["end"] - s["start"]) for s in long_])
        labels = cluster(V, w, th, num)
        if num <= 0:
            labels = _absorb_small(V, w, labels, th)
        out = {s["id"]: int(lab) for s, lab in zip(long_, labels)}
    prev = None
    for s in segs:
        if s["id"] in out:
            prev = s
            continue
        if prev is not None and s["start"] - prev["end"] < 2.0:
            out[s["id"]] = out[prev["id"]]
        elif long_:
            mid = (s["start"] + s["end"]) / 2
            near = min(long_, key=lambda x: abs((x["start"] + x["end"]) / 2 - mid))
            out[s["id"]] = out[near["id"]]
        else:
            out[s["id"]] = 0
    return out


def _prints_for(tracks: dict[str, np.ndarray], segs: list[dict]) -> dict[str, np.ndarray]:
    prints = {}
    for s in segs:
        if s["end"] - s["start"] < MIN_PRINT or s["channel"] not in tracks:
            continue
        clip = tracks[s["channel"]][int(s["start"] * SAMPLE_RATE):int(s["end"] * SAMPLE_RATE)]
        if len(clip) >= SAMPLE_RATE * MIN_PRINT * 0.9:
            prints[s["id"]] = voiceprint(clip)
    return prints


def online_call(segments: list[dict]) -> bool:
    """Other people came through the system channel (a call), not only through the mic (a room)."""
    talk = {"mic": 0.0, "system": 0.0}
    for s in segments:
        talk[s["channel"]] = talk.get(s["channel"], 0.0) + max(0.0, s["end"] - s["start"])
    total = talk["mic"] + talk["system"]
    return talk["system"] >= 30 or (total > 0 and talk["system"] / total >= 0.1)


def label_meeting(tracks: dict[str, np.ndarray], segments: list[dict], num_speakers: int = 0,
                  mic_is_me: bool = True, prints: dict[str, np.ndarray] | None = None) -> None:
    """Set seg["speaker"] for every line: "me" for your mic, "spk1", "spk2" ... for the others,
    numbered in order of first appearance. num_speakers > 0 forces the number of people."""
    prints = dict(prints or {})
    missing = [s for s in segments if s["id"] not in prints]
    prints.update(_prints_for(tracks, missing))
    mic = [s for s in segments if s["channel"] == "mic"]
    system = [s for s in segments if s["channel"] == "system"]
    me = mic_is_me and online_call(segments) and bool(mic)
    groups: dict[str, str] = {}
    if me:
        for s in mic:
            groups[s["id"]] = ME
    elif mic:
        want = max(0, num_speakers - (1 if system else 0)) if num_speakers else 0
        for sid, g in group_lines(mic, prints, want).items():
            groups[sid] = f"m{g}"
    if system:
        want = max(1, num_speakers - (1 if mic else 0)) if num_speakers else 0
        for sid, g in group_lines(system, prints, want).items():
            groups[sid] = f"s{g}"
    order: dict[str, str] = {}
    for seg in sorted(segments, key=lambda s: s["start"]):
        g = groups.get(seg["id"])
        if g is None:
            continue
        if g == ME:
            seg["speaker"] = ME
            continue
        if g not in order:
            order[g] = f"spk{len(order) + 1}"
        seg["speaker"] = order[g]


class LiveSpeakers:
    """Speaker labels while recording. Each remote line's voiceprint joins the closest known speaker
    or starts a new one; every few lines all voiceprints so far are clustered again
    (`regroup`, the same method as after the meeting) and the labels are corrected."""

    def __init__(self, threshold: float | None = None):
        self.threshold = JOIN_THRESHOLD if threshold is None else threshold
        self.centroids: dict[str, np.ndarray] = {}
        self.prints: dict[str, np.ndarray] = {}   # segment id -> voiceprint
        self.last: dict[str, str] = {}            # channel -> last speaker key
        self._next = 1

    def new_key(self) -> str:
        while f"spk{self._next}" in self.centroids:
            self._next += 1
        key = f"spk{self._next}"
        self._next += 1
        return key

    def assign(self, channel: str, v: np.ndarray | None) -> str:
        if v is None:                                 # too short for a voiceprint: same as before
            key = self.last.get(channel) or self.new_key()
            self.centroids.setdefault(key, np.zeros(0, np.float32))
            self.last[channel] = key
            return key
        best, score = None, -1.0
        for key, c in self.centroids.items():
            n = np.linalg.norm(c) if c.shape == v.shape else 0.0
            if n > 0:
                sim = float(v @ (c / n))
                if sim > score:
                    best, score = key, sim
        if best is None or score < self.threshold:
            # a speaker started by short lines only ("Hallo!", "So") has no voiceprint yet: this
            # first real voiceprint is most likely the same person, not a new one
            empty = [k for k, c in self.centroids.items() if c.shape != v.shape or not np.linalg.norm(c)]
            last = self.last.get(channel)
            best = last if last in empty else empty[-1] if empty else self.new_key()
        c = self.centroids.get(best)
        self.centroids[best] = v.copy() if c is None or c.shape != v.shape else c + v
        self.last[channel] = best
        return best

    def remember(self, segment_id: str, v: np.ndarray | None) -> None:
        if v is not None:
            self.prints[segment_id] = v

    def regroup(self, segs: list[dict]) -> dict[str, str]:
        """Correct the live labels with everything heard so far; returns {segment id: key}.

        Starts from the current speakers instead of from scratch: a clustering from scratch on a
        growing meeting toggles two similar voices between merged and separate every few lines (on a
        recorded 30-minute team meeting: 113 label changes on screen). Here a line only moves when it
        is clearly closer to another speaker, and two speakers only merge when their voices match on
        average; both are lasting decisions (13 changes; the same number of speakers as the full
        labeling after the meeting, which then replaces these labels)."""
        lines = [s for s in segs if s["id"] in self.prints and s.get("speaker", "").startswith("spk")]
        label = {s["id"]: s["speaker"] for s in lines}
        weight = {s["id"]: max(0.1, s["end"] - s["start"]) for s in lines}
        for _ in range(2):                                    # move lines that clearly sound like someone else
            sums: dict[str, np.ndarray] = {}
            for sid, key in label.items():
                sums[key] = sums.get(key, 0) + self.prints[sid]
            unit = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in sums.items()}
            moved = 0
            for sid, key in list(label.items()):
                v = self.prints[sid]
                best = max(unit, key=lambda k: float(v @ unit[k]))
                if best != key and float(v @ unit[best]) - float(v @ unit[key]) >= self.STICKY:
                    label[sid] = best
                    moved += 1
            if not moved:
                break
        while True:                                            # merge speakers whose voices match
            members: dict[str, list[str]] = {}
            for sid, key in label.items():
                members.setdefault(key, []).append(sid)
            keys = list(members)
            if len(keys) < 2:
                break
            sums = {k: np.sum([self.prints[i] for i in members[k]], axis=0) for k in keys}
            best, pair = -1.0, None
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    avg = float(sums[a] @ sums[b]) / (len(members[a]) * len(members[b]))
                    if avg > best:
                        best, pair = avg, (a, b)
            if best < CLUSTER_THRESHOLD:
                break
            a, b = pair
            keep, gone = (a, b) if sum(weight[i] for i in members[a]) >= sum(weight[i] for i in members[b]) else (b, a)
            for sid in members[gone]:
                label[sid] = keep
        labels = dict(label)
        prev = None
        for s in sorted(segs, key=lambda s: s["start"]):     # short lines follow their neighbour
            if s["id"] not in labels:
                if prev is not None and s["start"] - prev["end"] < 2.0 and prev["id"] in labels:
                    labels[s["id"]] = labels[prev["id"]]
                elif s.get("speaker"):
                    labels[s["id"]] = s["speaker"]
            prev = s
        cents: dict[str, np.ndarray] = {}
        for sid, key in label.items():
            cents[key] = cents.get(key, 0) + self.prints[sid]
        self.centroids = cents
        return labels

    STICKY = 0.06


# --------------------------------------------------------------------------- after the meeting

NOT_ME = 0.68     # same rule as live (livemic.NOT_ME)


def foreign_mic_lines(tracks: dict[str, np.ndarray], segments: list[dict], prints: dict[str, np.ndarray],
                      me: np.ndarray | None) -> tuple[list[str], np.ndarray | None]:
    """In a call, mic lines that are not in your voice: the other side leaking from the speakers, or
    text Whisper invented on noise. Your voice comes from earlier meetings (`me`) or, the first
    time, from this meeting's clear mic lines while the speakers were silent.
    Returns (ids to drop, your voiceprint as measured in this meeting)."""
    from .livemic import INVENTED

    mic = [s for s in segments if s["channel"] == "mic" and s["id"] in prints]
    system = tracks.get("system")
    if not online_call(segments):
        return [], None
    invented = [s["id"] for s in segments if s["channel"] == "mic" and s["id"] not in prints
                and INVENTED.match(s["text"].strip())]
    if not mic:
        return invented, None

    def speakers_on(s):
        if system is None:
            return 0.0
        clip = system[int(s["start"] * SAMPLE_RATE):int(s["end"] * SAMPLE_RATE)]
        n = len(clip) // 1600
        return float(((clip[: n * 1600].reshape(n, 1600) ** 2).mean(axis=1) > 1e-5).mean()) if n else 0.0

    quiet = [prints[s["id"]] for s in mic if s["end"] - s["start"] >= 2 and speakers_on(s) < 0.1]
    if me is None or me.shape != next(iter(prints.values())).shape:
        if len(quiet) < 3:
            return invented, None
        me = np.mean(quiet, axis=0)
    me = me / (np.linalg.norm(me) + 1e-9)
    drop = invented + [s["id"] for s in mic if float(prints[s["id"]] @ me) < NOT_ME]
    mine = [prints[s["id"]] for s in mic if float(prints[s["id"]] @ me) >= 0.8]
    return drop, (np.sum(mine, axis=0) if mine else None)


def _label_job(tracks: dict[str, np.ndarray], segments: list[dict], num_speakers: int, mic_is_me: bool,
               prints: dict[str, np.ndarray], me: np.ndarray | None) -> dict:
    """Runs in a child process: {"labels": {segment id: speaker key}, "drop": [ids], "me": [sum, n]}."""
    prints = dict(prints)
    prints.update(_prints_for(tracks, [s for s in segments if s["id"] not in prints]))
    drop, mine = foreign_mic_lines(tracks, segments, prints, me) if mic_is_me else ([], None)
    keep = [s for s in segments if s["id"] not in set(drop)]
    label_meeting(tracks, keep, num_speakers, mic_is_me, prints)
    return {"labels": {s["id"]: s.get("speaker", "") for s in keep}, "drop": drop, "me": mine}


_pool = None


def label_in_subprocess(tracks: dict[str, np.ndarray], segments: list[dict], num_speakers: int = 0,
                        mic_is_me: bool = True, prints: dict[str, np.ndarray] | None = None,
                        me: np.ndarray | None = None) -> dict:
    """Full labeling in a separate process, so a recording that starts meanwhile never waits for it."""
    global _pool
    import concurrent.futures
    import multiprocessing

    if _pool is None:
        _pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    try:
        return _pool.submit(_label_job, tracks, segments, num_speakers, mic_is_me, prints or {}, me).result()
    except concurrent.futures.process.BrokenProcessPool:
        _pool = None
        raise
