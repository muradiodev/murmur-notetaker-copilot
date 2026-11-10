"""Clean the transcripts of saved meetings: collapse recognizer loops ("Intententent...") and fix the team
vocabulary the recognizer mishears (see transcriber.HEARD_AS). Changed lines keep what was heard in
`raw_text` (hover in the UI) and are marked as corrected. A backup of every meeting.json is made first.

    .venv\\Scripts\\python.exe -m tools.clean_meetings          # do it
    .venv\\Scripts\\python.exe -m tools.clean_meetings --dry    # only show what would change
"""

import json
import shutil
import sys
import time

from app.config import MEETINGS_DIR
from app.transcriber import clean_text


def main() -> None:
    dry = "--dry" in sys.argv
    files = sorted(MEETINGS_DIR.glob("*/meeting.json"))
    backup = MEETINGS_DIR.parent / f"backup-meetings-{time.strftime('%Y%m%d-%H%M%S')}"
    changed_total = 0
    for f in files:
        meeting = json.loads(f.read_text(encoding="utf-8"))
        changed = []
        for seg in meeting.get("segments", []):
            new = clean_text(seg["text"])
            if new != seg["text"]:
                changed.append((seg["text"], new))
                if not dry:
                    seg.setdefault("raw_text", seg["text"])
                    seg["text"], seg["corrected"] = new, True
        if not changed:
            continue
        changed_total += len(changed)
        print(f"{f.parent.name}: {len(changed)} line(s)")
        for old, new in changed[:6]:
            print(f"   {old[:90]!r}\n   -> {new[:90]!r}")
        if not dry:
            target = backup / f.parent.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target / "meeting.json")
            f.write_text(json.dumps(meeting, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{changed_total} line(s) {'would change' if dry else 'cleaned'}" + ("" if dry else f", backup in {backup}"))


if __name__ == "__main__":
    main()
