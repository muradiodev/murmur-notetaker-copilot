"""Acoustic echo cancellation: remove the meeting audio that the speakers leak into the microphone.

The reference is the system loopback (exactly what the speakers played). A partitioned-block
frequency-domain NLMS filter (overlap-save) models the speaker -> room -> mic path up to ~256 ms,
and a spectral post-filter removes the residual echo the linear filter misses.

Runs once after a recording, on the 16 kHz recognition tracks: the final transcript and the speaker
separation then work on a mic track without the remote voices. With headphones there is no echo;
the filter then removes almost nothing and the original track is kept.
"""

import numpy as np

BLOCK = 256        # 16 ms at 16 kHz
PARTITIONS = 16    # 16 x 16 ms = 256 ms echo path
MU = 0.5           # adaptation step (normalised by power summed over partitions, stable up to ~1)
BETA = 1.6         # residual echo over-subtraction
FLOOR = 0.12       # never silence a bin completely (keeps the near-end voice natural)


WARMUP_SECONDS = 60


def estimate_delay(mic: np.ndarray, ref: np.ndarray, rate: int = 16000, max_ms: int = 1000) -> tuple[int, float]:
    """How many samples the speaker sound reaches the mic after it appears in the loopback.

    Windows captures loopback and microphone through different buffers, so the echo can trail the
    reference by tens to hundreds of milliseconds; echo cancellers need them roughly aligned.
    Cross-correlates 10 ms log-energy envelopes. Returns (delay samples, correlation); delay 0
    when the match is weak (e.g. headphones: the mic never hears the speakers).
    """
    hop = rate // 100
    n = min(len(mic), len(ref)) // hop
    if n < 200:
        return 0, 0.0

    def env(x):
        e = np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)
        return (e - e.mean()) / (e.std() + 1e-9)

    m, r = env(mic), env(ref)
    best, best_c = 0, -1.0
    for lag in range(0, max_ms // 10 + 1):
        c = float(np.mean(m[lag:] * r[: n - lag])) if lag else float(np.mean(m * r))
        if c > best_c:
            best, best_c = lag, c
    return (best * hop, best_c) if best_c > 0.3 else (0, best_c)


def align(ref: np.ndarray, delay: int) -> np.ndarray:
    """Delay the reference by `delay` samples so it lines up with the echo in the mic."""
    if delay <= 0:
        return ref
    return np.concatenate([np.zeros(delay, ref.dtype), ref[:-delay]])


def cancel_echo(mic: np.ndarray, ref: np.ndarray, rate: int = 16000) -> tuple[np.ndarray, float]:
    """Return (mic without echo, echo reduction in dB measured where the reference is active).

    Offline, so it can cheat: a first pass over the opening minute learns the echo path, and the
    real pass starts from that filter. No echo leaks through while the filter is still converging.
    """
    warm = int(WARMUP_SECONDS * rate)
    _, _, W = _run(mic[:warm], ref[:warm], None)
    out, erle, _ = _run(mic, ref, W)
    return out, erle


def _run(mic: np.ndarray, ref: np.ndarray, W0) -> tuple[np.ndarray, float, np.ndarray]:
    n = min(len(mic), len(ref))
    if n < BLOCK * 8:
        return mic, 0.0, W0 if W0 is not None else np.zeros((PARTITIONS, BLOCK + 1), dtype=np.complex128)
    d = mic[:n].astype(np.float64)
    x = ref[:n].astype(np.float64)
    blocks = n // BLOCK
    F = 2 * BLOCK
    W = W0.copy() if W0 is not None else np.zeros((PARTITIONS, BLOCK + 1), dtype=np.complex128)
    X = np.zeros((PARTITIONS, BLOCK + 1), dtype=np.complex128)   # reference spectra history
    xbuf = np.zeros(F)
    power = np.full(BLOCK + 1, 1e-6)
    out = np.zeros(blocks * BLOCK)
    win = np.sqrt(np.hanning(F + 1)[:F])
    prev_tail = np.zeros(BLOCK)
    e_hist = np.zeros(F)
    erle_num = erle_den = 1e-12

    for b in range(blocks):
        s = b * BLOCK
        xbuf = np.concatenate([xbuf[BLOCK:], x[s:s + BLOCK]])
        X = np.roll(X, 1, axis=0)
        X[0] = np.fft.rfft(xbuf)
        Y = (W * X).sum(axis=0)
        y = np.fft.irfft(Y)[BLOCK:]                      # echo estimate for this block
        e = d[s:s + BLOCK] - y
        # NLMS update normalised by the reference power summed over all partitions (per bin)
        power = 0.9 * power + 0.1 * (np.abs(X) ** 2).sum(axis=0)
        E = np.fft.rfft(np.concatenate([np.zeros(BLOCK), e]))
        grad = np.conj(X) * E / (power + 1e-3 * F)
        g = np.fft.irfft(grad, axis=1)
        g[:, BLOCK:] = 0                                  # keep the update causal (overlap-save constraint)
        W += MU * np.fft.rfft(g, axis=1)
        if not np.isfinite(W).all():
            W[:] = 0                                      # numerical blow-up: start over
            e = d[s:s + BLOCK].copy()
            y = np.zeros(BLOCK)

        # residual echo post-filter on a windowed frame of the error signal
        e_hist = np.concatenate([e_hist[BLOCK:], e])
        Ef = np.fft.rfft(e_hist * win)
        Yf = np.fft.rfft(np.concatenate([np.zeros(BLOCK), y]) * win)
        gain = np.maximum(FLOOR, 1 - BETA * (np.abs(Yf) ** 2) / (np.abs(Ef) ** 2 + 1e-9))
        frame = np.fft.irfft(Ef * gain) * win
        if b > 0:  # overlap-add: the half that is now complete belongs to the previous block
            done = slice(s - BLOCK, s)
            out[done] = prev_tail + frame[:BLOCK]
            if float(np.mean(x[done] ** 2)) > 1e-5:      # measure only while the speakers play
                erle_den += float(np.mean(d[done] ** 2))
                erle_num += float(np.mean(out[done] ** 2))
        prev_tail = frame[BLOCK:]

    result = mic.astype(np.float32).copy()
    result[: len(out)] = out.astype(np.float32)
    return result, 10 * np.log10(erle_den / erle_num), W
