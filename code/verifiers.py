import json
import re

EN_STOP = {"the", "and", "is", "of", "to", "in", "that", "it", "for", "with",
           "as", "are", "this", "on", "be", "by", "an", "or", "from", "which"}
FR_STOP = {"le", "la", "les", "des", "une", "un", "est", "et", "de", "du",
           "dans", "que", "qui", "pour", "avec", "sur", "ce", "cette", "aux",
           "nous", "vous", "sont", "plus", "au", "en", "par", "il", "elle"}
FR_DIACRITICS = set("éèêëàâäçùûüîïôö")


def _words(text):
    return re.findall(r"[A-Za-zÀ-ſ'’\-]+", text)


def _sentences(text):
    parts = re.split(r"(?<=[.!?])[\s\"')\]]*\s+", text.strip())
    return [p for p in parts if p.strip()]


def _strip_fences(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def _is_english(text):
    w = [x.lower() for x in _words(text)]
    if not w:
        return False
    en = sum(x in EN_STOP for x in w)
    fr = sum(x in FR_STOP for x in w)
    dia = sum(c in FR_DIACRITICS for c in text.lower())
    return en >= fr and dia <= 2


def _is_french(text):
    w = [x.lower() for x in _words(text)]
    if not w:
        return False
    en = sum(x in EN_STOP for x in w)
    fr = sum(x in FR_STOP for x in w)
    dia = sum(c in FR_DIACRITICS for c in text.lower())
    return fr > en or dia >= 3


def _json_answer_reasoning(text):
    try:
        obj = json.loads(_strip_fences(text))
    except Exception:
        return False
    return isinstance(obj, dict) and set(obj.keys()) == {"answer", "reasoning"}


def _plain_prose(text):
    s = _strip_fences(text).strip()
    if not s:
        return False
    if s.startswith("{") or s.startswith("["):
        return False
    if "```" in text:
        return False
    try:
        json.loads(s)
        return False
    except Exception:
        return True


def _numbered_five(text):
    items = re.findall(r"^\s*([1-9])[.)]\s+\S", text, re.M)
    return [int(x) for x in items] == [1, 2, 3, 4, 5]


def _single_paragraph(text):
    s = text.strip()
    if not s:
        return False
    if re.search(r"\n\s*\n", s) or "\n" in s:
        return False
    return not re.match(r"^\s*[-*•]|^\s*\d[.)]", s)


def _all_caps(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _all_lower(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _quoted_phrase(text):
    return bool(re.search(r'"[^"\n]{2,}"', text) or re.search(r"“[^”\n]{2,}”", text))


def _no_quotes(text):
    return not any(ch in text for ch in '"“”')


RULES = [
    ("all capital letters",                _all_caps),
    ("all lowercase letters",              _all_lower),
    ("in English, no other language",      _is_english),
    ("in French, no other language",       _is_french),
    ("must not contain any digits",        lambda t: len(re.findall(r"\d", t)) == 0),
    ("must include at least three digits", lambda t: len(re.findall(r"\d", t)) >= 3),
    ("Respond in JSON format with keys",   _json_answer_reasoning),
    ("plain text prose, no JSON",          _plain_prose),
    ("numbered list with exactly five",    _numbered_five),
    ("single paragraph without any list",  _single_paragraph),
    ("Do not use any quotation marks",     _no_quotes),
    ("at least one quoted phrase",         _quoted_phrase),
    ("exactly 10 sentences",               lambda t: len(_sentences(t)) == 10),
    ("at least five sentences",            lambda t: len(_sentences(t)) >= 5),
    ("at least 300 words",                 lambda t: len(_words(t)) >= 300),
    ("less than 50 words",                 lambda t: len(_words(t)) < 50),
]
def get_checker(system_message):
    for key, fn in RULES:
        if key.lower() in system_message.lower():
            return key, fn
    raise KeyError(f"no verifier for system message: {system_message!r}")


def check(system_message, output):
    _, fn = get_checker(system_message)
    try:
        return bool(fn(output))
    except Exception:
        return False


if __name__ == "__main__":
    import glob
    import os
    root = os.environ.get("FL_DATA", ".")
    seen = set()
    for f in sorted(glob.glob(os.path.join(root, "*_instruction.json"))):
        for s in json.load(open(f)):
            seen.add(s["system_message"])
    print(f"{len(seen)} distinct system constraints")
    for s in sorted(seen):
        key, _ = get_checker(s)
        print(f"  [{key}] <- {s[:70]}")
