"""Local machine translation with NLLB-200 (distilled 1.3B, CTranslate2 int8). Nothing leaves the machine.

CTranslate2 and `tokenizers` already ship with faster-whisper, so this needs no extra package. The
model (about 1.4 GB) lives in models/nllb-200-distilled-1.3B-ct2-int8 and is fetched by setup_models.
It runs on the GPU next to Whisper (about 1.5 GB of video memory) and falls back to the CPU.
"""

import logging
import threading
import time

from .config import NLLB_DIR, add_nvidia_dll_dirs, load_team_terms

log = logging.getLogger(__name__)

# NLLB language codes for the languages Murmur works with
CODES = {"de": "deu_Latn", "en": "eng_Latn", "az": "azj_Latn"}

_lock = threading.Lock()
_state: dict = {"translator": None, "tokenizer": None, "device": None, "error": ""}


def present() -> bool:
    return (NLLB_DIR / "model.bin").exists() and (NLLB_DIR / "tokenizer.json").exists()


def loaded() -> bool:
    return _state["translator"] is not None


def load() -> None:
    """Load the model once (a few seconds). Raises RuntimeError with a readable message."""
    with _lock:
        if _state["translator"] is not None:
            return
        if not present():
            raise RuntimeError("The translation model is not downloaded yet (run: python -m app.setup_models nllb)")
        add_nvidia_dll_dirs()
        import ctranslate2
        from tokenizers import Tokenizer

        last = None
        for device, compute in (("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")):
            try:
                translator = ctranslate2.Translator(str(NLLB_DIR), device=device, compute_type=compute,
                                                    inter_threads=1, intra_threads=4 if device == "cpu" else 0)
                _state.update(translator=translator, tokenizer=Tokenizer.from_file(str(NLLB_DIR / "tokenizer.json")),
                              device=f"{device}/{compute}", error="")
                log.info("NLLB translation model loaded on %s", _state["device"])
                return
            except Exception as exc:        # no CUDA, not enough memory ...: try the next option
                last = exc
                log.warning("NLLB on %s/%s failed: %s", device, compute, exc)
        _state["error"] = str(last)
        raise RuntimeError(f"Could not load the translation model: {last}")


# Team vocabulary that a general translation model gets wrong (e.g. a word that means "intent" in the
# project, not "complaint"). German source words are swapped for the English term before translating,
# which NLLB keeps as it is. Lives in data/team_terms.json ("glossary"); more in Settings ("translate_glossary").
GLOSSARY = load_team_terms().get("glossary", {})


def _glossary_pattern(extra: dict[str, str] | None):
    import re

    terms = {k: v for k, v in {**GLOSSARY, **(extra or {})}.items() if k.strip()}
    if not terms:
        return None, {}
    keys = sorted(terms, key=len, reverse=True)           # longest first: "Terminverwaltung" before "Termin"
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE)
    lower = {k.lower(): v for k, v in terms.items()}
    return pattern, lower


def parse_glossary(text: str) -> dict[str, str]:
    """"Termin=appointment, Rechnung=invoice" (commas or new lines) -> dict."""
    out = {}
    for part in str(text or "").replace("\n", ",").split(","):
        if "=" in part:
            a, b = part.split("=", 1)
            if a.strip() and b.strip():
                out[a.strip()] = b.strip()
    return out


def _sentences(text: str) -> list[str]:
    """NLLB drops sentences when it gets several at once, so each sentence is translated on its own."""
    import re

    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def translate(texts: list[str], source: str, target: str, glossary: dict[str, str] | None = None) -> list[str]:
    """Translate from `source` to `target` (codes: de, en, az). Same order as the input."""
    if not texts:
        return []
    flat, owner = [], []
    pattern = lower = None
    if source == "de":
        pattern, lower = _glossary_pattern(glossary)
    for i, text in enumerate(texts):
        for sentence in _sentences(text):
            flat.append(pattern.sub(lambda m: lower[m.group(1).lower()], sentence) if pattern else sentence)
            owner.append(i)
    if not flat:
        return [""] * len(texts)
    done = _translate_sentences(flat, source, target)
    out = [""] * len(texts)
    for i, piece in zip(owner, done):
        out[i] = (out[i] + " " + piece).strip()
    return out


def _translate_sentences(texts: list[str], source: str, target: str) -> list[str]:
    load()
    tok, tr = _state["tokenizer"], _state["translator"]
    src, tgt = CODES.get(source, CODES["de"]), CODES.get(target, CODES["en"])
    batch = [[src, *tok.encode(t.strip(), add_special_tokens=False).tokens, "</s>"] for t in texts]
    t0 = time.monotonic()
    with _lock:                                   # one translation at a time keeps GPU memory flat
        results = tr.translate_batch(batch, target_prefix=[[tgt]] * len(batch), beam_size=3,
                                     max_decoding_length=256, repetition_penalty=1.1, no_repeat_ngram_size=4)
    out = []
    for r in results:
        tokens = [t for t in r.hypotheses[0][1:] if t not in (tgt, "</s>")]
        out.append(tok.decode([tok.token_to_id(t) for t in tokens if tok.token_to_id(t) is not None],
                              skip_special_tokens=True).strip())
    log.info("Local translation %s->%s: %d lines in %.2fs", source, target, len(texts), time.monotonic() - t0)
    return out
