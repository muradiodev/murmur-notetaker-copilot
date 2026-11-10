"""Fixes applied to the local model's output, sentence by sentence, knowing the German source.

- Azerbaijani: German "du" becomes "sən" (NLLB likes the polite "siz"). Only when the speaker said
  "ihr / euch / euer" the plural "siz" stays.
- Terms that must stay as they are (Ticket, product names ...) are swapped for a code like "QX01" before
  translating. The model keeps codes untouched and puts the Azerbaijani suffix after them ("QX01-də"); the
  term is put back and the suffix is fitted to it ("Jira-da", "ticketin", "Docker-də").
- "Ach so" becomes a plain "Hmmm" in Azerbaijani ("I see" in English) instead of a literal translation.

Extra terms come from data/team_terms.json, "keep_terms":
  [{"de": "Jira", "out": "Jira", "harmony": "back", "vowel_end": true, "join": "-"}]
harmony: vowel group of the term's last syllable (front, front-round, back, back-round), vowel_end: the term
ends in a vowel sound, join: "-" for names ("Jira-da"), "" for ordinary words ("ticketin").
"""

import re

from .config import load_team_terms

KEEP_TERMS = [
    {"de": "Ticket", "out": "ticket", "harmony": "front", "vowel_end": False, "join": ""},
    *load_team_terms().get("keep_terms", []),
]
KEEP_TERMS = [t for t in KEEP_TERMS if isinstance(t, dict) and t.get("de")]

_PLURAL_YOU = re.compile(r"\b(ihr|euch|euer|eure[mnrs]?)\b", re.IGNORECASE)
_ACH_SO = re.compile(r"\bach\s?so\b[\s,.!?]*", re.IGNORECASE)


def _code(i: int) -> str:
    return f"QX{i + 1:02d}"


_SOURCE = [re.compile(r"(?<![\wÄÖÜäöüß])" + re.escape(t["de"]) + r"(e?s)?(?![\wÄÖÜäöüß])", re.IGNORECASE)
           for t in KEEP_TERMS]
_HARMONY = {   # vowel in a suffix -> vowel after a term of this group
    "front": str.maketrans("aıuüoö", "əiiiəə"), "front-round": str.maketrans("aıuioə", "əüüüəə"),
    "back": str.maketrans("əiüuöo", "aııııa"), "back-round": str.maketrans("əiüıöo", "auuuua"),
}
_VOWELS = "aıoueəiöü"


def protect(source: str) -> str:
    """German sentence -> same sentence with the kept terms swapped for codes (plural "s" stays)."""
    for i, pattern in enumerate(_SOURCE):
        source = pattern.sub(lambda m, c=_code(i): c + ("s" if m.group(1) else ""), source)
    return source


def _fit_suffix(term: dict, suffix: str) -> str:
    suffix = suffix.translate(_HARMONY.get(term.get("harmony", "front"), _HARMONY["front"]))
    if not term.get("vowel_end") and len(suffix) > 1 and suffix[0] in "nys" and suffix[1] in _VOWELS:
        suffix = suffix[1:]               # the code ends in a vowel ("bir"): QX01-nin -> ticketin, QX01-yə -> ticketə
    return suffix


def restore(text: str, target: str) -> str:
    for i, term in enumerate(KEEP_TERMS):
        out = term.get("out", term["de"])
        if target == "az":
            text = re.sub(_code(i) + r"(?:-?(\w+))?",
                          lambda m, t=term, o=out: o + ((t.get("join", "-") + _fit_suffix(t, m.group(1)))
                                                        if m.group(1) else ""), text)
        else:
            text = re.sub(_code(i) + r"(s?)\b", lambda m, o=out: o + m.group(1), text)
    return re.sub(r"(^|[.!?]\s+)([a-zəıöüç])", lambda m: m.group(1) + m.group(2).upper(), text)


# --------------------------------------------------------------------------- du -> sən

_PRONOUNS = {
    "siz": "sən", "sizə": "sənə", "sizi": "səni", "sizin": "sənin", "sizdə": "səndə", "sizdən": "səndən",
    "sizinlə": "səninlə", "sizcə": "səncə", "özünüz": "özün", "özünüzü": "özünü", "özünüzə": "özünə",
    "özünüzdə": "özündə", "özünüzdən": "özündən",
}
_Q = r"(mı|mi|mu|mü)?"
_VERB = [   # bilərsiniz -> bilərsən, yazırsınız -> yazırsan, gördünüz -> gördün, etdinizmi -> etdinmi
    (re.compile(r"(\w)(sınız)" + _Q + r"\b"), "san"), (re.compile(r"(\w)(siniz)" + _Q + r"\b"), "sən"),
    (re.compile(r"(\w)(sunuz)" + _Q + r"\b"), "san"), (re.compile(r"(\w)(sünüz)" + _Q + r"\b"), "sən"),
    (re.compile(r"(\w)(dınız)" + _Q + r"\b"), "dın"), (re.compile(r"(\w)(diniz)" + _Q + r"\b"), "din"),
    (re.compile(r"(\w)(dunuz)" + _Q + r"\b"), "dun"), (re.compile(r"(\w)(dünüz)" + _Q + r"\b"), "dün"),
]
# your (plural) -> your: anlayışınız -> anlayışın, evinizdə -> evində
_POSSESSIVE = re.compile(r"(\w{2,})(ınız|iniz|unuz|ünüz)(\w*)\b")
_POSS_SINGLE = {"ınız": "ın", "iniz": "in", "unuz": "un", "ünüz": "ün"}
# plural imperative -> singular: "Durun," -> "Dur,", "Gözləyin." -> "Gözlə.", "göstərin!" -> "göstər!".
# A word ending in -ın/-in is often "your ..." (kitabın), so only safe cases change: a one-word clause,
# a word before "!", a -yin ending (only imperatives have it) or a common verb from the list.
_IMPERATIVE = re.compile(r"(^|[,.!?]\s+)(\w{2,}?)y?(ın|in|un|ün)(?=[,!])")
_IMP_VERBS = {"yazın", "baxın", "verin", "edin", "göstərin", "gedin", "olun", "durun", "gəlin", "açın",
              "bağlayın", "deyin", "soruşun", "yoxlayın", "gözləyin", "dinləyin", "göndərin", "basın", "seçin"}
_IMP_END = re.compile(r"\b(\w{2,}?)y?(?:ın|in|un|ün)(?=\s*!)|\b(\w{2,}?)y(?:ın|in|un|ün)(?=[.?]|$)")


def _imp_stem(word: str) -> str:
    stem = word[:-2]
    return stem[:-1] if stem.endswith("y") else stem


def _match_case(word: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if word[:1].isupper() else repl


def to_informal(text: str) -> str:
    text = re.sub(r"\w+", lambda m: _match_case(m.group(0), _PRONOUNS.get(m.group(0).lower(), m.group(0)))
                  if m.group(0).lower() in _PRONOUNS else m.group(0), text)
    for pattern, ending in _VERB:
        text = pattern.sub(lambda m, e=ending: m.group(1) + e + (m.group(3) or ""), text)
    text = _POSSESSIVE.sub(lambda m: m.group(1) + _POSS_SINGLE[m.group(2)] + m.group(3), text)
    text = _IMPERATIVE.sub(lambda m: m.group(1) + m.group(2), text)
    text = _IMP_END.sub(lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"\w+", lambda m: _match_case(m.group(0), _imp_stem(m.group(0).lower()))
                  if m.group(0).lower() in _IMP_VERBS else m.group(0), text)
    return text


# --------------------------------------------------------------------------- entry points

def prepare(source: str, target: str) -> tuple[str, bool]:
    """German sentence before translation -> (sentence to translate, had "ach so")."""
    source = protect(source)
    if target not in ("az", "en") or not _ACH_SO.search(source):
        return source, False
    return _ACH_SO.sub("", source).strip(), True


def finish(source: str, text: str, target: str, ach_so: bool = False) -> str:
    """Model output for one sentence -> fixed output."""
    text = restore(text, target)
    if target == "az":
        if not _PLURAL_YOU.search(source):
            text = to_informal(text)
        if ach_so:
            text = ("Hmmm. " + text) if text else "Hmmm."
    elif target == "en" and ach_so:
        text = ("I see. " + text) if text else "I see."
    return text
