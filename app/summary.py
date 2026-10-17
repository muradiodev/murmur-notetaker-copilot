"""Meeting summary and per-sentence sentiment with Claude."""

import json

import anthropic

from .storage import fmt_clock

MODEL = "claude-opus-5-5"

LANG_NAMES = {"az": "Azerbaijani", "de": "German", "en": "English"}

# Section headings per summary language, put into the prompt directly (asking the model to
# translate headings was unreliable: it kept the Azerbaijani ones for English meetings).
HEADINGS = {
    "en": {"h_summary": "Summary", "h_topics": "Key topics", "h_decisions": "Decisions",
           "h_tasks": "Action items", "h_questions": "Open questions"},
    "de": {"h_summary": "Zusammenfassung", "h_topics": "Wichtige Themen", "h_decisions": "Entscheidungen",
           "h_tasks": "Aufgaben", "h_questions": "Offene Fragen"},
    "az": {"h_summary": "Qısa xülasə", "h_topics": "Əsas mövzular", "h_decisions": "Qərarlar",
           "h_tasks": "Tapşırıqlar", "h_questions": "Açıq suallar"},
}

SYSTEM_PROMPT = """You analyse a meeting transcript. It was produced by automatic speech recognition, \
may mix Azerbaijani, German and English, may contain recognition errors, and labels speakers generically \
or by the names the user assigned. Infer the intended meaning where the wording is garbled, but never \
invent facts, names, numbers or commitments that the transcript does not support.

Return JSON with these fields.

"summary": meeting notes written entirely in {language} (title, headings and text), in Markdown, with \
exactly this structure and these headings:

# <short meeting title in {language}>

## {h_summary}
2-4 sentences: what the meeting was about and its outcome.

## {h_topics}
Bullet points of the main topics, each with the key points discussed.

## {h_decisions}
Bullet points of decisions made. Write "-" if none.

## {h_tasks}
Checklist items "- [ ] Owner: task (deadline if mentioned)". Write "-" if none.

## {h_questions}
Unresolved questions or follow-ups. Write "-" if none.

Use the speaker labels exactly as given.

"sentiments": the sentiment of each transcript line that is clearly positive (agreement, enthusiasm, \
praise, satisfaction, good news) or clearly negative (disagreement, frustration, concern, complaint, bad \
news), keyed by the line id in square brackets. Omit neutral lines — everything you leave out counts as \
neutral. Judge the speaker's attitude, not the topic.

"corrections": transcript lines where speech recognition clearly misheard words (wrong words that
do not fit the context, broken names, product or company names, numbers, words from the wrong
language), keyed by line id, with the corrected line text. Keep the language the person spoke, keep
their wording and style, do not translate, summarise or polish grammar. Only fix what was misheard.
Omit lines that are already right; an empty list is fine.

"speaker_names": for speakers labelled generically (like "Speaker 2"), the real name only if that
speaker clearly introduces themselves ("my name is…", "I'm…", "ich heiße…", "mənim adım…") or is
unmistakably addressed by name right before they answer. Never guess; leave a speaker out when the
transcript does not say who they are."""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["positive", "negative"]},
                },
                "required": ["id", "sentiment"],
                "additionalProperties": False,
            },
        },
        "speaker_names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"speaker": {"type": "string"}, "name": {"type": "string"}},
                "required": ["speaker", "name"],
                "additionalProperties": False,
            },
        },
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "sentiments", "corrections", "speaker_names"],
    "additionalProperties": False,
}


def transcript_text(meeting: dict) -> str:
    names = meeting.get("speakers", {})
    lines = []
    for seg in sorted(meeting["segments"], key=lambda s: s["start"]):
        who = names.get(seg["speaker"], seg["speaker"])
        lines.append(f"[{seg['id']}] {fmt_clock(seg['start'])} {who}: {seg['text']}")
    return "\n".join(lines)


# Providers that speak the OpenAI chat API. NVIDIA's build.nvidia.com endpoint is OpenAI-compatible.
OPENAI_COMPATIBLE = {
    "openai": {"name": "OpenAI", "base_url": None},
    "nvidia": {"name": "NVIDIA", "base_url": "https://integrate.api.nvidia.com/v1"},
}
PROVIDERS = ("claude", "openai", "nvidia")


def _claude(system: str, user: str, api_key: str, workspace_id: str = "") -> dict:
    if not api_key:
        raise RuntimeError("Add an Anthropic API key in Settings")
    # Keys created outside a workspace must name the workspace on every request.
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(api_key=api_key, default_headers=headers)
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=32000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to process this transcript (refusal)")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The response was too long (max_tokens)")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def _client(provider: str, api_key: str):
    import openai

    return openai.OpenAI(api_key=api_key, base_url=OPENAI_COMPATIBLE[provider]["base_url"], timeout=240, max_retries=1)


def _extract_json(text: str) -> dict:
    """Models without schema support sometimes wrap the JSON in prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("The model did not return JSON")
    return json.loads(text[start:end + 1])


def _openai_compatible(provider: str, system: str, user: str, api_key: str, model: str) -> dict:
    import openai

    name = OPENAI_COMPATIBLE[provider]["name"]
    if not api_key:
        raise RuntimeError(f"Add a {name} API key in Settings")
    if not model:
        raise RuntimeError(f"Choose a {name} model in Settings")
    client = _client(provider, api_key)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    schema_format = {"type": "json_schema", "json_schema": {"name": "meeting_notes", "strict": True, "schema": SCHEMA}}
    try:
        response = client.chat.completions.create(model=model, messages=messages, response_format=schema_format)
    except openai.BadRequestError:
        # Not every hosted model supports JSON schemas; ask for plain JSON instead.
        messages[0]["content"] += "\n\nRespond with only the JSON object, no other text."
        response = client.chat.completions.create(model=model, messages=messages)
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise RuntimeError(f"{name} declined: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise RuntimeError("The response was too long")
    return _extract_json(choice.message.content or "")


CHAT_SKIP = ("audio", "realtime", "transcribe", "tts", "image", "embed", "search", "moderation", "whisper",
             "dall-e", "rerank", "parakeet", "canary", "riva", "guard", "reward", "clip", "flux", "stable-diffusion",
             "nemoretriever", "vila", "neva", "kosmos", "deplot", "paligemma", "fuyu", "safety", "content-safety")


def list_models(provider: str, api_key: str) -> list[str]:
    """Chat models the key can actually use, so the user picks a real name instead of typing one."""
    models = list(_client(provider, api_key).models.list())
    ids = [m for m in models if not any(x in m.id.lower() for x in CHAT_SKIP)]
    if provider == "openai":
        ids = [m for m in ids if m.id.startswith(("gpt-", "o"))]
        return [m.id for m in sorted(ids, key=lambda m: m.created or 0, reverse=True)]
    return sorted(m.id for m in ids)


def ping(provider: str, api_key: str, model: str) -> None:
    # No token limit: newer OpenAI models reject max_tokens (they want max_completion_tokens),
    # NVIDIA's endpoint only knows max_tokens. The summary call sends neither, so neither does this.
    _client(provider, api_key).with_options(timeout=40, max_retries=0).chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Reply with the single word OK."}])


def available_providers(settings: dict) -> list[str]:
    keys = {"claude": "anthropic_api_key", "openai": "openai_api_key", "nvidia": "nvidia_api_key"}
    return [p for p in PROVIDERS if settings.get(keys[p])]


def meeting_language(meeting: dict, fallback: str = "en") -> str:
    """The language most words of the meeting were spoken in (only AZ, DE, EN are offered)."""
    words: dict[str, int] = {}
    for seg in meeting["segments"]:
        if seg.get("lang") in LANG_NAMES:
            words[seg["lang"]] = words.get(seg["lang"], 0) + len(seg["text"].split())
    return max(words, key=words.get) if words else fallback


def resolve_language(meeting: dict, settings: dict, requested: str | None = None) -> str:
    choice = requested or settings.get("summary_language", "auto")
    return meeting_language(meeting) if choice in (None, "", "auto") else choice


def analyse(meeting: dict, settings: dict, language: str | None = None) -> dict:
    """Return {"summary": markdown, "sentiments": {segment_id: "positive" | "negative"}} from the chosen provider."""
    text = transcript_text(meeting)
    if not text.strip():
        raise ValueError("The transcript is empty")
    language = language or resolve_language(meeting, settings)
    system = SYSTEM_PROMPT.format(language=LANG_NAMES.get(language, language), **HEADINGS.get(language, HEADINGS["en"]))
    user = f"<transcript>\n{text}\n</transcript>"
    provider = settings.get("summary_provider", "claude")
    if provider not in available_providers(settings):
        ready = available_providers(settings)
        if not ready:
            raise RuntimeError("No API key yet. Add one in Settings, API keys")
        provider = ready[0]
    if provider == "claude":
        data = _claude(system, user, settings["anthropic_api_key"], settings.get("anthropic_workspace_id", ""))
    else:
        data = _openai_compatible(provider, system, user, settings[f"{provider}_api_key"], settings.get(f"{provider}_model", ""))
    valid = {s["id"] for s in meeting["segments"]}
    return {
        "summary": str(data.get("summary", "")).strip(),
        "sentiments": {s["id"]: s["sentiment"] for s in data.get("sentiments", [])
                       if isinstance(s, dict) and s.get("id") in valid and s.get("sentiment") in ("positive", "negative")},
        "speaker_names": {str(n["speaker"]): str(n["name"]) for n in data.get("speaker_names", [])
                          if isinstance(n, dict) and n.get("speaker") and n.get("name")},
        "corrections": {c["id"]: str(c["text"]).strip() for c in data.get("corrections", [])
                        if isinstance(c, dict) and c.get("id") in valid and str(c.get("text", "")).strip()},
    }
