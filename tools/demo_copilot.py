"""Create (or remove) demo Murmur Copilot sessions, to see how finished sessions look.

  python tools/demo_copilot.py           # create 4 demo sessions
  python tools/demo_copilot.py --remove  # delete them again

Demo sessions have "demo": true in meeting.json and "Demo" in the title. No audio files.
"""

import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import storage  # noqa: E402
from app.config import MEETINGS_DIR  # noqa: E402

SESSIONS = [
    {
        "company": "Northwind Labs", "title": "Java Software Engineer", "type": "interview", "lang": "de", "days": 0,
        "hour": 10, "minutes": 52,
        "turns": [
            ("I", "Hallo Alex, schön, dass du da bist. Stell dich bitte kurz vor."),
            ("Y", "Gerne. Ich bin Alex, Backend-Entwickler mit Schwerpunkt Java und Spring Boot."),
            ("I", "Was verstehst du unter Clean Architecture?"),
            ("Y", "Die Fachlogik bleibt unabhängig von Spring Boot, Datenbank und REST-Controllern."),
            ("I", "Wie habt ihr Idempotenz bei Kafka sichergestellt?"),
            ("Y", "Wir haben die Ereignis-ID gespeichert und vor der Verarbeitung geprüft, ob sie schon erledigt war."),
            ("I", "Super, das klingt gut. Hast du noch Fragen an uns?"),
            ("Y", "Ja, wie ist das Team aufgeteilt und welche Rolle spielt Kafka zwischen den Produktbereichen?"),
        ],
        "answers": [
            ("Kurze Vorstellung", "Stell dich bitte kurz vor.",
             "Gerne. Ich bin Alex und arbeite seit acht Jahren als Backend-Entwickler, vor allem mit Java, Spring Boot und Microservices.\n\n"
             "Bei Contoso habe ich Services für Abrechnung und Freigaben gebaut und zwei jüngere Entwickler bei Code-Reviews begleitet. "
             "Jetzt suche ich eine Rolle, in der ich weiter selbst entwickle und technische Verantwortung übernehme."),
            ("Clean Architecture im Backend", "Was verstehst du unter Clean Architecture?",
             "Für mich heißt Clean Architecture: Die Fachlogik bleibt getrennt von Spring Boot, der Datenbank und den APIs. So kann ich eine Regel "
             "ändern und testen, ohne den Controller anzufassen.\n\n"
             "Bei Contoso nimmt der Controller die Anfrage an, der Service prüft die fachlichen Regeln, und das Repository kümmert sich um die Daten. "
             "So blieben die Verantwortlichkeiten klar, und wir konnten Änderungen mit JUnit und Mockito absichern."),
            ("Idempotenz bei Kafka", "Wie habt ihr Idempotenz bei Kafka sichergestellt?",
             "Idempotenz heißt: Wenn eine Nachricht zweimal ankommt, passiert die Aktion nur einmal.\n\n"
             "Bei Contoso konnte ein externes System dasselbe Ereignis erneut senden. Wir haben die Ereignis-ID gespeichert und vor der "
             "Verarbeitung geprüft, ob sie schon erledigt war. Fehlgeschlagene Nachrichten gingen nach begrenzten Versuchen in ein Dead-Letter-Topic."),
            ("Fragen an das Team", "Hast du noch Fragen an uns?",
             "Ja, zwei Fragen. Wie ist das Team aufgeteilt, und wer entscheidet über die Schnittstellen zwischen den Produktbereichen? "
             "Und welche Rolle spielt Kafka heute zwischen diesen Bereichen?"),
        ],
        "summary": "# Technisches Interview Northwind Labs\n\n## Zusammenfassung\nZweites Gespräch mit zwei Entwicklern. Fokus auf Architektur, "
                   "Microservices und Kafka. Die Stimmung war positiv, das Team will sich bald melden.\n\n## Wichtige Themen\n"
                   "- Clean Architecture: Fachlogik getrennt von Spring Boot\n- Idempotenz bei Kafka\n\n"
                   "## Entscheidungen\n-\n\n## Aufgaben\n- [ ] Alex: Rückmeldung zu den nächsten Schritten abwarten\n\n## Offene Fragen\n- Teamzuordnung",
    },
    {
        "company": "Fabrikam GmbH", "title": "Software Engineer – Java Modernisation", "type": "interview", "lang": "de", "days": 2,
        "hour": 16, "minutes": 38,
        "turns": [
            ("I", "Hast du mit SOAP und bestehenden Integrationen gearbeitet?"),
            ("Y", "Ja, mit älteren Java-Systemen und ihren Schnittstellen."),
            ("I", "Hast du Erfahrung mit Keycloak?"),
            ("Y", "Ja, für Login und Rollen in Spring-Boot-Services."),
        ],
        "answers": [
            ("SOAP und alte Integrationen", "Hast du mit SOAP und bestehenden Integrationen gearbeitet?",
             "Ja. Ich habe mit älteren Java-Webanwendungen, JSP und Hibernate gearbeitet. Dort musste ich zuerst verstehen, wie die "
             "Schnittstellen wirklich genutzt werden.\n\n"
             "Bei einer Modernisierung würde ich die alte Schnittstelle zuerst mit Tests absichern und dann einen Teil nach dem anderen ersetzen."),
            ("Keycloak", "Hast du Erfahrung mit Keycloak?",
             "Ja. Ich habe Keycloak für Login und Rollen in Spring-Boot-Services genutzt. Die Services prüfen das Token, und die Rollen steuern, "
             "welche Aktionen erlaubt sind. Wichtig ist mir, dass die Berechtigung im Backend geprüft wird und nicht nur im Frontend."),
        ],
        "summary": "# Kennenlernen Fabrikam\n\n## Zusammenfassung\nGespräch mit Lead Engineer. Schwerpunkt auf Modernisierung "
                   "älterer Java-Systeme und Integrationen.\n\n## Wichtige Themen\n- SOAP und bestehende Integrationen\n"
                   "- Keycloak und Berechtigungen\n\n## Entscheidungen\n- Nächstes Gespräch mit dem Team\n\n"
                   "## Aufgaben\n- [ ] Alex: Termin für das zweite Gespräch bestätigen\n\n## Offene Fragen\n-",
    },
    {
        "company": "Globex", "title": "Software Engineer", "type": "interview", "lang": "en", "days": 1,
        "hour": 11, "minutes": 46,
        "turns": [
            ("I", "What were the bigger challenges with the RAG platform you built?"),
            ("Y", "Reliable answers with sources, and respecting document permissions."),
            ("I", "How do you get up to speed in a brand new domain?"),
            ("Y", "I start with the business process, talk to experts and follow one workflow end to end."),
        ],
        "answers": [
            ("Challenges of the RAG platform", "What were the bigger challenges with the RAG platform you built?",
             "The hardest part was making answers reliable. Users needed to see where an answer came from, so we kept the source with every "
             "retrieved part.\n\nThe second challenge was security. People could only get answers from documents they were allowed to see, so "
             "we checked permissions before any text went to the model."),
            ("Learning a new domain", "How do you get up to speed in a brand new domain?",
             "I start with the business process before the code. I talk to product people and users and follow one real example from start "
             "to finish.\n\nThen I look at the data model and the tests to see how the rules are built today, before I change anything."),
        ],
        "summary": "# Intro call Globex\n\n## Summary\nFirst call with the hiring manager. Company overview, then Alex's background "
                   "and the RAG project.\n\n## Key topics\n- RAG challenges: sources and permissions\n- Learning a new domain\n\n"
                   "## Decisions\n-\n\n## Action items\n- [ ] Recruiter: share next steps\n\n## Open questions\n- Team placement",
    },
    {
        "company": "Initech", "title": "Recruiter call", "type": "regular", "lang": "en", "days": 4,
        "hour": 15, "minutes": 18,
        "turns": [
            ("I", "Hi Alex, thanks for taking the call. What are your salary expectations?"),
            ("Y", "I'm looking at a range that fits a senior backend role."),
            ("I", "When could you start?"),
            ("Y", "I'm flexible and could start soon."),
        ],
        "answers": [
            ("Salary expectations", "What are your salary expectations?",
             "I'm looking for a package that fits a senior backend role. I'd like to understand the full package first, including "
             "benefits and the level of the role. Could you share the range you have planned for this position?"),
            ("Start date", "When could you start?",
             "I'm flexible and could start soon. If the process moves quickly, I can adapt to your timeline."),
        ],
        "summary": "# Recruiter call Initech\n\n## Summary\nShort recruiter call about expectations and start date.\n\n## Key topics\n"
                   "- Salary range\n- Start date\n\n## Decisions\n-\n\n## Action items\n- [ ] Recruiter: send the role details\n\n## Open questions\n-",
    },
]


def build(spec: dict) -> dict:
    created = (datetime.now() - timedelta(days=spec["days"])).replace(hour=spec["hour"], minute=5, second=0, microsecond=0)
    meeting = storage.new_meeting(f"Demo · {spec['company']} · {spec['title']}")
    meeting.update({
        "created_at": created.isoformat(timespec="seconds"),
        "status": "ready", "mode": "copilot", "demo": True,
        "duration": float(spec["minutes"] * 60),
        "speakers": {"spk1": "Interviewer", "spk2": "Alex"},
        "copilot": {"type": spec["type"], "company": spec["company"], "title": spec["title"], "language": spec["lang"],
                    "provider": "openai", "model": "gpt-6-sol", "auto_generate": True},
        "summary": spec["summary"], "summary_lang": spec["lang"],
    })
    t, segments, starts = 8.0, [], {}
    for who, text in spec["turns"]:
        dur = max(2.5, len(text.split()) * 0.42)
        seg = {"id": uuid.uuid4().hex[:8], "start": round(t, 2), "end": round(t + dur, 2),
               "channel": "system" if who == "I" else "mic", "text": text, "lang": spec["lang"], "words": [],
               "speaker": "spk1" if who == "I" else "spk2"}
        segments.append(seg)
        if who == "I":
            starts[text] = seg["end"]
        t += dur + (18 if who == "Y" else 1.2)
    meeting["segments"] = segments
    answers = []
    for i, (title, question, text) in enumerate(spec["answers"]):
        at = starts.get(question, 10 + i * 40)
        answers.append({"id": uuid.uuid4().hex[:8], "question": question, "title": title, "text": text, "auto": True,
                        "at": round(at + 0.6, 1), "state": "done", "lag": 0.6, "first": round(0.9 + (i % 3) * 0.2, 2),
                        "seconds": round(2.1 + (i % 4) * 0.3, 1)})
    meeting["answers"] = answers
    storage.save(meeting, exports=True)
    return meeting


def remove() -> int:
    n = 0
    for folder in MEETINGS_DIR.iterdir():
        m = storage.load(folder.name) if folder.is_dir() else None
        if m and m.get("demo"):
            storage.delete(m["id"])
            n += 1
    return n


if __name__ == "__main__":
    if "--remove" in sys.argv:
        print("removed", remove(), "demo sessions")
    else:
        remove()
        for spec in SESSIONS:
            m = build(spec)
            print("created", m["id"], m["title"])
