"""Translation of transcript lines (German to Azerbaijani by default) with the user's AI provider.

Only text leaves the machine, the same as for summaries. Live translation runs one small request per
burst of new lines on its own thread, so it never slows speech recognition down.
"""

import json
import logging
import queue
import threading
import time
from collections import deque
from typing import Callable

import anthropic

from . import localmt, summary

log = logging.getLogger(__name__)

FAST_MODEL = "claude-haiku-5-5"      # translation of one sentence must come back in about a second
LANGS = {"az": "Azerbaijani", "de": "German", "en": "English"}

SYSTEM = """You translate lines of a live meeting transcript into {target}. The transcript comes from \
automatic speech recognition: it may contain small recognition errors, so translate what the speaker \
most likely meant. Use natural, clear {target}. Keep names, numbers, product names, ticket keys and \
technical terms. {vocabulary}If a line is already in {target}, return it unchanged. Never add comments.

You get optional <context> (earlier lines, only for understanding, do not translate them) and the \
<lines> to translate, as JSON objects with an id and a text. Return only JSON:
{{"translations": [{{"id": "<same id>", "text": "<translation>"}}]}}"""


def _provider(settings: dict) -> str:
    ready = summary.available_providers(settings)
    if not ready:
        raise RuntimeError("Translation needs an API key. Add one in Settings, API keys")
    choice = settings.get("summary_provider", "claude")
    return choice if choice in ready else ready[0]


def _complete(settings: dict, system: str, user: str) -> str:
    provider = _provider(settings)
    if provider == "claude":
        headers = {"anthropic-workspace-id": settings["anthropic_workspace_id"]} if settings.get("anthropic_workspace_id") else None
        client = anthropic.Anthropic(api_key=settings["anthropic_api_key"], default_headers=headers, timeout=30, max_retries=1)
        response = client.messages.create(model=FAST_MODEL, max_tokens=2000, system=system,
                                          messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in response.content if b.type == "text")
    model = settings.get(f"{provider}_model", "")
    if not model:
        raise RuntimeError(f"Choose a {provider} model in Settings to translate")
    client = summary._client(provider, settings[f"{provider}_api_key"]).with_options(timeout=30, max_retries=1)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return response.choices[0].message.content or ""


def uses_local(settings: dict) -> bool:
    """Local NLLB when chosen and downloaded; otherwise the AI provider if a key exists."""
    if settings.get("translate_engine", "local") == "local" and localmt.present():
        return True
    if settings.get("translate_engine", "local") == "local" and not summary.available_providers(settings):
        raise RuntimeError("The translation model is not downloaded yet. Run: python -m app.setup_models nllb")
    return False


def translate_lines(settings: dict, lines: list[tuple[str, str]], target: str, context: list[str] | None = None,
                    sources: dict[str, str] | None = None, refine: bool = False) -> dict[str, str]:
    """[(id, text)] -> {id: translation}. `sources` maps ids to their language (de, en, az).
    Raises when the engine fails."""
    if uses_local(settings):
        by_lang: dict[str, list[tuple[str, str]]] = {}
        for i, t in lines:
            by_lang.setdefault((sources or {}).get(i) or "de", []).append((i, t))
        out: dict[str, str] = {}
        for lang, part in by_lang.items():
            glossary = localmt.parse_glossary(settings.get("translate_glossary", ""))
            for (i, _), text in zip(part, localmt.translate([t for _, t in part], lang, target, glossary,
                                                            refine=refine)):
                if text:
                    out[i] = text
        return out
    vocabulary = str(settings.get("vocabulary", "")).strip()
    system = SYSTEM.format(target=LANGS.get(target, target),
                           vocabulary=f"Keep these terms untranslated: {vocabulary}. " if vocabulary else "")
    user = ""
    if context:
        user += "<context>\n" + "\n".join(context) + "\n</context>\n"
    user += "<lines>\n" + json.dumps([{"id": i, "text": t} for i, t in lines], ensure_ascii=False) + "\n</lines>"
    data = summary._extract_json(_complete(settings, system, user))
    wanted = {i for i, _ in lines}
    return {str(x["id"]): str(x["text"]).strip() for x in data.get("translations", [])
            if isinstance(x, dict) and str(x.get("id")) in wanted and str(x.get("text", "")).strip()}


def refine_wanted(settings: dict, target: str) -> bool:
    """English second opinion (local engine only, and only for targets other than English)."""
    return uses_local(settings) and target != "en" and settings.get("translate_refine", True) is not False


class Refiner:
    """Live: a line is shown with the fast direct translation first; this worker then translates it
    again through English (`localmt._refine`) and replaces the line when the result differs."""

    def __init__(self, get_settings: Callable[[], dict], on_result: Callable[[str, dict[str, str], str], None]):
        self.get_settings, self.on_result = get_settings, on_result
        self.q: "queue.Queue[tuple[str, str, str, str, str, str]]" = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="live-refine").start()

    def submit(self, meeting_id: str, seg_id: str, text: str, target: str, source: str | None, shown: str) -> None:
        if refine_wanted(self.get_settings(), target):
            self.q.put((meeting_id, seg_id, text, target, source or "de", shown))

    def _run(self) -> None:
        while True:
            batch = [self.q.get()]
            while len(batch) < 8:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            groups: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
            for meeting_id, seg_id, text, target, source, shown in batch:
                groups.setdefault((meeting_id, target, source), []).append((seg_id, text, shown))
            for (meeting_id, target, source), lines in groups.items():
                try:
                    settings = self.get_settings()
                    glossary = localmt.parse_glossary(settings.get("translate_glossary", ""))
                    better = localmt.translate([t for _, t, _ in lines], source, target, glossary, refine=True)
                    changed = {i: b for (i, _, shown), b in zip(lines, better) if b and b != shown}
                    if changed:
                        self.on_result(meeting_id, changed, target)
                except Exception:
                    log.exception("Live refine failed")


def translate_segments(settings: dict, segments: list[dict], target: str, chunk: int = 20,
                       progress: Callable[[dict[str, str]], None] | None = None,
                       field: str = "translation", force: bool = False) -> dict[str, str]:
    """Translate a whole transcript in chunks (after the meeting, or on demand). Lines already in the
    target language are copied as they are. Non-English targets get the English second opinion
    (`refine_wanted`)."""
    done: dict[str, str] = {}
    todo = [s for s in sorted(segments, key=lambda s: s["start"])
            if (force or not s.get(field)) and s.get("lang") != target]
    previous: list[str] = []
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        got = translate_lines(settings, [(s["id"], s["text"]) for s in part], target, previous[-4:],
                              {s["id"]: s.get("lang") for s in part}, refine=refine_wanted(settings, target))
        done.update(got)
        previous = [s["text"] for s in part]
        if progress:
            progress(got)
    return done


class LiveTranslator:
    """Background worker: lines go in as they are recognised, translations come out through `on_result`."""

    def __init__(self, get_settings: Callable[[], dict],
                 on_result: Callable[[str, dict[str, str]], None], on_error: Callable[[str], None]):
        self.get_settings, self.on_result, self.on_error = get_settings, on_result, on_error
        self.q: "queue.Queue[tuple[str, str, str, str, str | None]]" = queue.Queue()
        self.context: dict[str, deque] = {}
        self.last_error = 0.0
        threading.Thread(target=self._run, daemon=True, name="live-translation").start()

    def submit(self, meeting_id: str, seg_id: str, text: str, target: str, source: str | None = None) -> None:
        self.q.put((meeting_id, seg_id, text, target, source))

    def _run(self) -> None:
        while True:
            batch = [self.q.get()]
            while len(batch) < 8:                      # a burst of sentences becomes one request
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
            sources: dict[str, str] = {}
            for meeting_id, seg_id, text, target, source in batch:
                groups.setdefault((meeting_id, target), []).append((seg_id, text))
                if source:
                    sources[seg_id] = source
            for (meeting_id, target), lines in groups.items():
                ctx = self.context.setdefault(meeting_id, deque(maxlen=4))
                try:
                    got = translate_lines(self.get_settings(), lines, target, list(ctx), sources)
                except Exception as exc:
                    log.warning("Live translation failed: %s", exc)
                    if time.monotonic() - self.last_error > 30:      # one message, not one per sentence
                        self.last_error = time.monotonic()
                        self.on_error(str(exc))
                    continue
                for _, text in lines:
                    ctx.append(text)
                if got:
                    try:
                        self.on_result(meeting_id, got, target)
                    except Exception:
                        log.exception("Storing live translation failed")
