"""Turn past interview transcripts into an interview playbook for Murmur Copilot.

  python tools/build_playbook.py            # all steps
  python tools/build_playbook.py --merge    # only rebuild the playbook from saved extractions

Input:  data/copilot/parakeet/_all_sessions.json  ({id: {title, language, createdAt, text}})
Output: data/copilot/parakeet/sessions/*.txt      (clean transcripts)
        data/copilot/parakeet/extract/*.json      (per interview: questions, what was said, stories, facts)
        data/copilot/parakeet/playbook.md         (merged; also stored as an Murmur Copilot document)

Nothing is invented: the model only extracts what the candidate actually said in the calls.

Set MURMUR_CANDIDATE to your name and MURMUR_OFFERS to a comma-separated list of company names whose
interviews ended with an offer (their answers are treated as the proven versions).
"""

import json
import os
import re
import sys
import concurrent.futures as cf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import copilot, summary  # noqa: E402
from app.config import load_settings  # noqa: E402

BASE = ROOT / "data" / "copilot" / "parakeet"
CANDIDATE = os.environ.get("MURMUR_CANDIDATE", "the candidate")
OFFERS = tuple(o.strip().lower() for o in os.environ.get("MURMUR_OFFERS", "").split(",") if o.strip())  # weighted most
MODEL = "gpt-6-sol"

UI_LINES = {"AI notes", "Transcript", "Ask AI", "Call details", "Rate us on G2", "Give feedback", "Clear transcript",
            "Download", "Report this answer", "Scroll to bottom"}


def clean(title: str, raw: str) -> str:
    """'text\\nSpeaker · 10:59 AM' blocks -> 'Speaker: text'; AI suggestions marked as such."""
    lines = [l.rstrip() for l in raw.splitlines()]
    out, buf = [], []
    for line in lines:
        s = line.strip()
        if not s or s in UI_LINES or s.startswith("Rate "):
            continue
        m = re.match(r"^(You|Interviewer \d+|Interviewee)\s*·\s*\d", s)
        a = re.match(r"^·\s*(Answer|Chat)", s)
        if m:
            who = "Candidate" if m.group(1) in ("You", "Interviewee") else "Interviewer"
            text = " ".join(buf).strip()
            if text:
                out.append(f"{who}: {text}")
            buf = []
        elif a:
            text = " ".join(buf).strip()
            if text:
                out.append(f"[AI suggestion shown to the candidate] {text}")
            buf = []
        else:
            buf.append(s.replace("💬", "").replace("⭐️", "").replace("**", ""))
    if buf:
        out.append(" ".join(buf))
    head = f"{title}\n"
    return head + "\n".join(out)


EXTRACT = """You read the transcript of a real job interview of the candidate {name}. Speech recognition errors are common; the labels "Candidate" and "Interviewer" can occasionally be swapped, judge from content. Lines marked [AI suggestion shown to the candidate] are suggestions from an assistant; the candidate often read them out, so what they said right after usually repeats them.

Extract only what is really in the transcript. Reply with JSON:
{{
 "company": "...", "role": "...", "language": "de|en",
 "stage": "e.g. HR screen, technical, final, offer/salary call",
 "questions": [ {{"question": "the interviewer's question, cleaned up", "topic": "intro|motivation|experience|project|technical|behavioral|salary|availability|company|other",
                 "candidate_answer": "what the candidate actually said, condensed to the concrete content (facts, names, numbers, examples) in 1-5 sentences",
                 "used_ai_suggestion": true/false,
                 "interviewer_reaction": "positive/neutral/negative + short note, only if visible"}} ],
 "stories": [ {{"title": "...", "project": "...", "situation": "...", "what_candidate_did": "...", "result": "...", "details": "names, numbers, tech as stated"}} ],
 "facts": [ "facts about the candidate's background, experience, motivation or preferences that they stated" ],
 "what_worked": [ "moments where the interviewer was clearly convinced or interested, and why" ],
 "improve": [ "answers that were weak, too long, unclear or that the interviewer pushed back on" ]
}}""".format(name=CANDIDATE)

MERGE = """You build an interview playbook for {name} from extractions of their real interviews (transcripts). Interviews marked offer="yes" ended with job offers: treat their answers as the proven versions. The playbook is given to a live interview assistant that writes the candidate's answers, so it must be factual, concrete and easy to reuse. Use only information from the extractions. Write in English, but keep key phrases in the original language where the candidate answered in another language (section 9).

Markdown with these sections:
1. Proven self-introduction: a short (45 s) and a long (2 min) version, built from how the candidate introduced themselves in the most successful interviews.
2. Facts sheet: employers, roles, dates, projects, team sizes, numbers, technologies, as stated. Mark anything that differs between interviews.
3. Story library: every concrete story (project, situation, what the candidate did, result, details), grouped by competency (technical problem, architecture, conflict, failure/lesson, leadership/mentoring, customer/stakeholder, startup/founder, learning a new domain, delivery under pressure). Note which interview it worked in.
4. Questions they were asked, grouped by topic, each with the best answer outline (from what worked), and how often it came up.
5. Motivation answers: why they are looking, why this kind of company, salary and availability lines they used (as they said them).
6. Technical topics they were tested on and how they explained them.
7. What worked in the offer interviews: patterns to repeat.
8. Weak spots seen across interviews and how to answer better.
9. Phrasing in other languages: short natural formulations used successfully.""".format(name=CANDIDATE)


def client(settings):
    return summary._client("openai", settings["openai_api_key"]).with_options(timeout=300)


def extract(settings, name: str, text: str) -> dict:
    r = client(settings).chat.completions.create(
        model=MODEL, reasoning_effort="low", response_format={"type": "json_object"},
        messages=[{"role": "system", "content": EXTRACT}, {"role": "user", "content": text[:180000]}])
    return json.loads(r.choices[0].message.content)


def main():
    settings = load_settings()
    sessions_dir, extract_dir = BASE / "sessions", BASE / "extract"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads((BASE / "_all_sessions.json").read_text(encoding="utf-8"))

    todo = []
    for sid, s in data.items():
        title = s["title"] or "untitled"
        if title.startswith("Murmur Copilot") or len(s["text"]) < 3000:
            continue
        name = f"{s['createdAt'][:10]}_{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}_{sid[:8]}"
        text = clean(title, s["text"])
        (sessions_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        todo.append((name, text))
    print(f"{len(todo)} interviews with content")

    if "--merge" not in sys.argv:
        def job(item):
            name, text = item
            out = extract_dir / f"{name}.json"
            if out.exists():
                return name, "cached"
            try:
                out.write_text(json.dumps(extract(settings, name, text), ensure_ascii=False, indent=1), encoding="utf-8")
                return name, "ok"
            except Exception as exc:
                return name, f"error {exc}"
        with cf.ThreadPoolExecutor(6) as pool:
            for name, state in pool.map(job, todo):
                print(f"  {state:7} {name}", flush=True)

    parts = []
    for path in sorted(extract_dir.glob("*.json")):
        e = json.loads(path.read_text(encoding="utf-8"))
        offer = any(o in path.name.lower() for o in OFFERS)
        parts.append(f'<interview file="{path.stem}" offer="{"yes" if offer else "unknown"}">\n'
                     + json.dumps(e, ensure_ascii=False) + "\n</interview>")
    print("merging", len(parts), "extractions")
    r = client(settings).chat.completions.create(
        model=MODEL, reasoning_effort="medium",
        messages=[{"role": "system", "content": MERGE}, {"role": "user", "content": "\n\n".join(parts)}])
    playbook = r.choices[0].message.content.strip()
    (BASE / "playbook.md").write_text(playbook, encoding="utf-8")
    print("playbook", len(playbook), "chars")

    # store it in Murmur Copilot's document library (replace an older version)
    name = "Interview playbook (from my past interviews)"
    for it in copilot.list_items("documents"):
        if it["name"] == name:
            copilot.delete_item("documents", it["id"])
    copilot.add_item("documents", name, playbook, "manual")
    print("saved as Murmur Copilot document:", name)


if __name__ == "__main__":
    main()
