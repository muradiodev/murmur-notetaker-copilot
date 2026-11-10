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


def translate(texts: list[str], source: str, target: str, glossary: dict[str, str] | None = None,
              refine: bool = False) -> list[str]:
    """Translate from `source` to `target` (codes: de, en, az). Same order as the input.
    refine: also translate through English and keep, per sentence, the version that fits both the
    original and the English sentence best (see `_refine`). Slower, used after a line was shown."""
    if not texts:
        return []
    from . import azfix

    flat, owner, originals, ach = [], [], [], []
    pattern = lower = None
    if source == "de":
        pattern, lower = _glossary_pattern(glossary)
    for i, text in enumerate(texts):
        for sentence in _sentences(text):
            prepared, had_ach = azfix.prepare(sentence, target) if source == "de" else (sentence, False)
            originals.append(sentence)
            ach.append(had_ach)
            flat.append(pattern.sub(lambda m: lower[m.group(1).lower()], prepared) if pattern else prepared)
            owner.append(i)
    if not flat:
        return [""] * len(texts)
    todo = [k for k, s in enumerate(flat) if s.strip()]          # "Ach so." alone needs no model
    done = [""] * len(flat)
    for k, piece in zip(todo, _translate_sentences([flat[k] for k in todo], source, target) if todo else []):
        done[k] = piece
    if refine and todo and "en" not in (source, target):
        keep_direct = [k for k in todo if source == "de" and azfix.keeps_plural_you(originals[k])]
        pick = [k for k in todo if k not in keep_direct]
        for k, piece in zip(pick, _refine([flat[k] for k in pick], [done[k] for k in pick], source, target)):
            done[k] = piece
    out = [""] * len(texts)
    for k, (i, piece) in enumerate(zip(owner, done)):
        if source == "de":
            piece = azfix.finish(originals[k], piece, target, ach[k])
        out[i] = (out[i] + " " + piece).strip()
    return out


def _score(source: str, sources: list[str], target: str, candidates: list[str]) -> list[float]:
    """Mean log-probability of each candidate translation given its source sentence."""
    tok, tr = _state["tokenizer"], _state["translator"]
    src, tgt = CODES.get(source, CODES["de"]), CODES.get(target, CODES["en"])
    with _lock:
        res = tr.score_batch([[src, *tok.encode(s, add_special_tokens=False).tokens, "</s>"] for s in sources],
                             [[tgt, *tok.encode(c, add_special_tokens=False).tokens, "</s>"] for c in candidates])
    return [sum(r.log_probs) / max(1, len(r.log_probs)) for r in res]


def _refine(sources: list[str], direct: list[str], source: str, target: str) -> list[str]:
    """Second opinion through English: the model knows German->English and English->Azerbaijani far
    better than German->Azerbaijani, so the English route often keeps parts the direct one drops.
    Both versions are scored against the original AND the English sentence; the better one wins
    (on 30 real meeting sentences the English route won 22 times)."""
    if not sources:
        return []
    english = _translate_sentences(sources, source, "en")
    pivot = _translate_sentences(english, "en", target)
    d = [a + b for a, b in zip(_score(source, sources, target, direct), _score("en", english, target, direct))]
    p = [a + b for a, b in zip(_score(source, sources, target, pivot), _score("en", english, target, pivot))]
    return [pv if pv.strip() and ps > ds else dv for dv, pv, ds, ps in zip(direct, pivot, d, p)]


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
