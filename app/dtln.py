"""DTLN-aec: neural echo cancellation + noise suppression (same models anarlog/Hyprnote ships).

Dual-signal transformation LSTM network (Westhausen & Meyer, Interspeech 2021; trained for the
Microsoft AEC challenge). 16 kHz, 512-sample blocks, 128-sample hop. Stage 1 estimates a spectral
mask from the mic and loopback magnitudes; stage 2 refines the masked block in a learned basis.
It also removes background noise, which helps recognition in noisy rooms.
Models: models/aec/dtln_aec_{128|256}_{1|2}.onnx (128 = faster, 256 = a bit better).
"""

import numpy as np

from .config import MODELS_DIR

AEC_DIR = MODELS_DIR / "aec"
BLOCK_LEN = 512
BLOCK_SHIFT = 128


def available(size: int = 128) -> bool:
    return all((AEC_DIR / f"dtln_aec_{size}_{i}.onnx").exists() for i in (1, 2))


class DTLN:
    def __init__(self, size: int = 128, threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self.s1 = ort.InferenceSession(str(AEC_DIR / f"dtln_aec_{size}_1.onnx"), opts, providers=prov)
        self.s2 = ort.InferenceSession(str(AEC_DIR / f"dtln_aec_{size}_2.onnx"), opts, providers=prov)
        i1, i2 = self.s1.get_inputs(), self.s2.get_inputs()
        # input order: (mic magnitude, states, loopback magnitude) / (masked block, states, loopback block)
        self.n1 = [i.name for i in i1]
        self.n2 = [i.name for i in i2]
        self.state_shape = i1[1].shape

    def process(self, mic: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
        """mic and ref: 16 kHz float32. Without ref, it works as a noise suppressor."""
        n = len(mic)
        ref = np.zeros(n, np.float32) if ref is None else ref[:n]
        if len(ref) < n:
            ref = np.pad(ref, (0, n - len(ref)))
        pad = BLOCK_LEN - BLOCK_SHIFT
        audio = np.concatenate([np.zeros(pad, np.float32), mic.astype(np.float32), np.zeros(pad, np.float32)])
        lpb = np.concatenate([np.zeros(pad, np.float32), ref.astype(np.float32), np.zeros(pad, np.float32)])
        out = np.zeros(len(audio), np.float32)
        s1 = np.zeros(self.state_shape, np.float32)
        s2 = np.zeros(self.state_shape, np.float32)
        in_buf = np.zeros(BLOCK_LEN, np.float32)
        lpb_buf = np.zeros(BLOCK_LEN, np.float32)
        out_buf = np.zeros(BLOCK_LEN, np.float32)
        blocks = (len(audio) - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
        a1, b1, c1 = self.n1
        a2, b2, c2 = self.n2
        for k in range(blocks):  # mirrors run_aec.py from the DTLN-aec repository
            seg = slice(k * BLOCK_SHIFT, k * BLOCK_SHIFT + BLOCK_SHIFT)
            in_buf[:-BLOCK_SHIFT] = in_buf[BLOCK_SHIFT:]
            in_buf[-BLOCK_SHIFT:] = audio[seg]
            lpb_buf[:-BLOCK_SHIFT] = lpb_buf[BLOCK_SHIFT:]
            lpb_buf[-BLOCK_SHIFT:] = lpb[seg]
            spec = np.fft.rfft(in_buf).astype(np.complex64)
            mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
            lmag = np.abs(np.fft.rfft(lpb_buf)).reshape(1, 1, -1).astype(np.float32)
            mask, s1 = self.s1.run(None, {a1: mag, b1: s1, c1: lmag})
            est = np.fft.irfft(spec * mask.reshape(-1)).astype(np.float32).reshape(1, 1, -1)
            block, s2 = self.s2.run(None, {a2: est, b2: s2, c2: lpb_buf.reshape(1, 1, -1)})
            out_buf[:-BLOCK_SHIFT] = out_buf[BLOCK_SHIFT:]
            out_buf[-BLOCK_SHIFT:] = 0
            out_buf += block.reshape(-1)
            out[seg] = out_buf[:BLOCK_SHIFT]
        return out[pad:pad + n]


class StreamingDTLN:
    """Live version: feed mic + loopback chunks as they arrive, get cleaned mic back.

    Output stays aligned with the input (no added delay). Any chunk length works; samples are processed in
    128-sample hops and the remainder waits for the next call.
    """

    def __init__(self, model: DTLN | None = None):
        self.m = model or DTLN(128, threads=1)
        self.s1 = np.zeros(self.m.state_shape, np.float32)
        self.s2 = np.zeros(self.m.state_shape, np.float32)
        self.in_buf = np.zeros(BLOCK_LEN, np.float32)
        self.lpb_buf = np.zeros(BLOCK_LEN, np.float32)
        self.out_buf = np.zeros(BLOCK_LEN, np.float32)
        self.rest_mic = np.zeros(0, np.float32)
        self.rest_ref = np.zeros(0, np.float32)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        mic = np.concatenate([self.rest_mic, mic.astype(np.float32)])
        ref = np.concatenate([self.rest_ref, ref.astype(np.float32)])
        hops = len(mic) // BLOCK_SHIFT
        self.rest_mic, self.rest_ref = mic[hops * BLOCK_SHIFT:], ref[hops * BLOCK_SHIFT:]
        out = np.zeros(hops * BLOCK_SHIFT, np.float32)
        a1, b1, c1 = self.m.n1
        a2, b2, c2 = self.m.n2
        for k in range(hops):
            seg = slice(k * BLOCK_SHIFT, (k + 1) * BLOCK_SHIFT)
            self.in_buf[:-BLOCK_SHIFT] = self.in_buf[BLOCK_SHIFT:]
            self.in_buf[-BLOCK_SHIFT:] = mic[seg]
            self.lpb_buf[:-BLOCK_SHIFT] = self.lpb_buf[BLOCK_SHIFT:]
            self.lpb_buf[-BLOCK_SHIFT:] = ref[seg]
            spec = np.fft.rfft(self.in_buf).astype(np.complex64)
            mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
            lmag = np.abs(np.fft.rfft(self.lpb_buf)).reshape(1, 1, -1).astype(np.float32)
            mask, self.s1 = self.m.s1.run(None, {a1: mag, b1: self.s1, c1: lmag})
            est = np.fft.irfft(spec * mask.reshape(-1)).astype(np.float32).reshape(1, 1, -1)
            block, self.s2 = self.m.s2.run(None, {a2: est, b2: self.s2, c2: self.lpb_buf.reshape(1, 1, -1)})
            self.out_buf[:-BLOCK_SHIFT] = self.out_buf[BLOCK_SHIFT:]
            self.out_buf[-BLOCK_SHIFT:] = 0
            self.out_buf += block.reshape(-1)
            out[seg] = self.out_buf[:BLOCK_SHIFT]
        return out
