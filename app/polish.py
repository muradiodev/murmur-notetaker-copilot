"""AI line editor for the live transcript.

After a meeting, the AI summary fixes words the recognizer misheard ("Koffein" -> "Caffeine",
"Hafke" -> "Kafka"). This does the same while you record: every new line is checked a moment
after it appears, in small batches with the lines before it as context, and a line that needs a fix is
replaced in place. Only text is sent to your AI provider (the one chosen for summaries).
Tests on real meetings showed that a second Whisper pass does not give this: Whisper itself is already
about as accurate live as it is afterwards, what differs is the understanding of context and vocabulary.
"""

import logging
import queue
import re
import threading
from collections import deque
from typing import Callable

from . import summary
from .config import load_team_terms
from .translate import _complete

log = logging.getLogger(__name__)

SYSTEM = """You correct lines of a live meeting transcript produced by automatic speech recognition. The lines are mostly German, sometimes English or Azerbaijani.

Fix a word when it makes no sense in that sentence and a similar-sounding word, a common IT or security term, a name or a term from the vocabulary below clearly fits: wrong technical terms, product and person names, words split or glued wrongly, stuck repetitions. Examples of fixes: "oh was" -> OWASP, "Beckengruppe" -> Backend-Gruppe, "Merch" -> Merge, "Cloud" -> Claude (when the talk is about the AI assistant), "Koffein" -> Caffeine (when it is about a cache), "in this very event" -> History-Event.

Keep the speaker's own wording, sentence structure, language and meaning. Do not rephrase, summarise, translate, polish style, or add or remove content. Punctuation and capital letters alone are not worth a change.

{team}{vocabulary}

You get optional <context> lines (earlier lines, for understanding only) and <lines> as JSON objects with an id and a text. Return only JSON, listing ONLY the lines you changed:
{{"edits": [{{"id": "<same id>", "text": "<corrected line>"}}]}}
If nothing needs fixing return {{"edits": []}}."""


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def acceptable(old: str, new: str) -> bool:
    """Guard against a model that rewrites instead of correcting: the words must stay mostly the same."""
    a, b = _words(old), _words(new)
    if not b or a == b:
        return False
    if not 0.5 <= len(b) / max(1, len(a)) <= 1.6:
        return False
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.5


def edit_lines(settings: dict, lines: list[tuple[str, str]], context: list[str] | None = None) -> dict[str, str]:
    """[(id, text)] -> {id: corrected text} for the lines that really changed."""
    import json

    vocabulary = str(settings.get("vocabulary", "")).strip()
    team = str(load_team_terms().get("polish_context", "")).strip()
    system = SYSTEM.format(team=f"{team} " if team else "", vocabulary=f"Also: {vocabulary}." if vocabulary else "")
    user = ""
    if context:
        user += "<context>\n" + "\n".join(context) + "\n</context>\n"
    user += "<lines>\n" + json.dumps([{"id": i, "text": t} for i, t in lines], ensure_ascii=False) + "\n</lines>"
    data = summary._extract_json(_complete(settings, system, user))
    old = dict(lines)
    out = {}
    for e in data.get("edits", []):
        if isinstance(e, dict) and str(e.get("id")) in old:
            text = str(e.get("text", "")).strip()
            if acceptable(old[str(e["id"])], text):
                out[str(e["id"])] = text
    return out


class LineEditor:
    """Background worker: new lines in, corrections out through `on_result(meeting_id, {id: text})`."""

    def __init__(self, get_settings: Callable[[], dict], on_result: Callable[[str, dict[str, str]], None]):
        self.get_settings, self.on_result = get_settings, on_result
        self.q: "queue.Queue[tuple[str, str, str]]" = queue.Queue()
        self.context: dict[str, deque] = {}
        threading.Thread(target=self._run, daemon=True, name="live-polish").start()

    def submit(self, meeting_id: str, seg_id: str, text: str) -> None:
        self.q.put((meeting_id, seg_id, text))

    def _run(self) -> None:
        while True:
            batch = [self.q.get()]
            while len(batch) < 6:                         # a burst of sentences becomes one request
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            groups: dict[str, list[tuple[str, str]]] = {}
            for meeting_id, seg_id, text in batch:
                groups.setdefault(meeting_id, []).append((seg_id, text))
            for meeting_id, lines in groups.items():
                ctx = self.context.setdefault(meeting_id, deque(maxlen=8))
                try:
                    settings = self.get_settings()
                    if not summary.available_providers(settings):
                        continue                          # no API key: the live text simply stays as it is
                    got = edit_lines(settings, lines, list(ctx))
                except Exception as exc:
                    log.warning("Live polish failed: %s", exc)
                    continue
                for seg_id, text in lines:
                    ctx.append(got.get(seg_id, text))
                if got:
                    try:
                        self.on_result(meeting_id, got)
                    except Exception:
                        log.exception("Storing live line edits failed")
