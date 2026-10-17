"""Meeting persistence: one folder per meeting with JSON, audio and exports."""

import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from .config import MEETINGS_DIR

_lock = threading.RLock()


def fmt_clock(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def meeting_dir(meeting_id: str) -> Path:
    return MEETINGS_DIR / meeting_id


def new_meeting(title: str = "") -> dict:
    now = datetime.now()
    meeting = {
        "id": now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        "title": title or f"Meeting {now.strftime('%d.%m.%Y %H:%M')}",
        "created_at": now.isoformat(timespec="seconds"),
        "duration": 0.0,
        "status": "recording",
        "segments": [],
        # Live labels by channel; diarization replaces them with Speaker 1, 2, 3…
        "speakers": {"mic": "Microphone", "system": "Online"},
        "summary": "",
        "error": "",
    }
    meeting_dir(meeting["id"]).mkdir(parents=True, exist_ok=True)
    save(meeting)
    return meeting


_deleted: set[str] = set()


def save(meeting: dict, exports: bool = False) -> None:
    with _lock:
        if meeting["id"] in _deleted:
            return  # a background job (summary, transcription) finished after the meeting was deleted
        folder = meeting_dir(meeting["id"])
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "meeting.json.tmp"
        tmp.write_text(json.dumps(meeting, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(folder / "meeting.json")
        if exports:
            write_exports(meeting)


def load(meeting_id: str) -> dict | None:
    path = meeting_dir(meeting_id) / "meeting.json"
    if not path.exists():
        return None
    with _lock:
        return json.loads(path.read_text(encoding="utf-8"))


def list_meetings() -> list[dict]:
    items = []
    if not MEETINGS_DIR.exists():
        return items
    for folder in MEETINGS_DIR.iterdir():
        m = load(folder.name) if folder.is_dir() else None
        if m:
            cp = m.get("copilot") or {}
            items.append({k: m[k] for k in ("id", "title", "created_at", "duration", "status")}
                         | {"has_summary": bool(m.get("summary")), "archived": bool(m.get("archived")),
                            "mode": m.get("mode", "notes"),
                            "copilot": {k: cp.get(k, "") for k in ("type", "company", "title")} if cp else None,
                            "answers": len(m.get("answers", []))})
    return sorted(items, key=lambda x: x["created_at"], reverse=True)


def delete(meeting_id: str) -> None:
    folder = meeting_dir(meeting_id)
    with _lock:
        _deleted.add(meeting_id)
        if folder.exists() and folder.parent == MEETINGS_DIR:
            shutil.rmtree(folder)


def _ordered(meeting: dict) -> list[dict]:
    return sorted(meeting["segments"], key=lambda s: s["start"])


def turns(meeting: dict) -> list[dict]:
    """Merge consecutive segments of the same speaker into one turn (start → end)."""
    result: list[dict] = []
    for seg in _ordered(meeting):
        if result and result[-1]["speaker"] == seg["speaker"]:
            turn = result[-1]
            turn["end"] = max(turn["end"], seg["end"])
            turn["text"] += " " + seg["text"]
        else:
            result.append({"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"], "text": seg["text"]})
    return result


def to_txt(meeting: dict) -> str:
    names = meeting.get("speakers", {})
    lines = [meeting["title"], meeting["created_at"].replace("T", " "), ""]
    for t in turns(meeting):
        who = names.get(t["speaker"], t["speaker"])
        lines += [f"{who}  [{fmt_clock(t['start'])} – {fmt_clock(t['end'])}]", t["text"], ""]
    return "\n".join(lines)


def to_srt(meeting: dict) -> str:
    names = meeting.get("speakers", {})
    blocks = []
    for i, seg in enumerate(_ordered(meeting), 1):
        who = names.get(seg["speaker"], seg["speaker"])
        blocks.append(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{who}: {seg['text']}\n")
    return "\n".join(blocks)


def write_exports(meeting: dict) -> None:
    folder = meeting_dir(meeting["id"])
    (folder / "transcript.txt").write_text(to_txt(meeting), encoding="utf-8")
    (folder / "transcript.srt").write_text(to_srt(meeting), encoding="utf-8")
    if meeting.get("summary"):
        (folder / "summary.md").write_text(meeting["summary"] + "\n", encoding="utf-8")
