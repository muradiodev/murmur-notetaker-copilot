"""Find speakers' names from self-introductions ("my name is Sarah", "ich heiße Lena", "mənim adım Leyla")."""

import re

# The lead-in is case-insensitive; the name itself must start with a capital letter.
PATTERNS = [
    r"(?i:\b(?:my name is|my name's|i am|i'm|this is|it's)\s+)([A-Z][a-z]{1,20})\b",
    r"(?i:\b(?:ich bin|ich heiße|ich heisse|mein name ist|hier ist|hier spricht)\s+(?:der\s+|die\s+)?)([A-ZÄÖÜ][a-zäöüß]{1,20})\b",
    r"(?i:\b(?:mənim adım|adım|mən)\s+)([A-ZƏÖÜÇŞĞİ][a-zəöüçşğı]{1,20}?)(?:dır|dir|dur|dür|yam|yəm|am|əm)?\b",
]
NOT_NAMES = {
    "here", "sorry", "glad", "happy", "ready", "sure", "not", "just", "also", "the", "very", "so", "okay", "ok",
    "fine", "good", "great", "back", "late", "new", "going", "done", "right", "really", "thinking", "calling",
    "hier", "da", "sehr", "nicht", "auch", "froh", "bereit", "mit", "der", "die", "das", "ein", "eine", "jetzt",
    "einverstanden", "sicher", "fertig", "zurück", "gleich", "dabei",
    "burada", "hazır", "razı", "gəldim",
}
COMPILED = [re.compile(p) for p in PATTERNS]
DEFAULT_NAME = re.compile(r"^(Speaker|Danışan) \d+$")


def find_name(text: str) -> str | None:
    for rx in COMPILED:
        for m in rx.finditer(text):
            name = m.group(1)
            if name.lower() not in NOT_NAMES:
                return name
    return None


def is_default(name: str) -> bool:
    return bool(DEFAULT_NAME.match(name or ""))


def auto_name(meeting: dict, lines_per_speaker: int = 4) -> dict[str, str]:
    """Name each still-unnamed speaker from their first few lines. Returns what was set."""
    speakers = meeting.get("speakers", {})
    taken = {v.lower() for v in speakers.values() if not is_default(v)}
    seen: dict[str, int] = {}
    found: dict[str, str] = {}
    for seg in sorted(meeting["segments"], key=lambda s: s["start"]):
        key = seg["speaker"]
        if key in found or not is_default(speakers.get(key, "")):
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > lines_per_speaker:
            continue
        name = find_name(seg["text"])
        if name and name.lower() not in taken:
            found[key] = name
            taken.add(name.lower())
    speakers.update(found)
    return found
