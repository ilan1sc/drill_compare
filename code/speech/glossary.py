"""Living aviation dictionary / glossary (PRD 12.2).

- SEED below is the starting glossary: maneuvers, systems, abbreviations,
  callouts, academy terms. Each entry carries English/Hebrew forms,
  spelling variants and a category, so it can be used both to correct/tag
  the transcript and to feed downstream semantic classification (the
  biomarkers).
- The glossary is versioned on disk under exports/audio/glossary/: each
  approve_terms() call writes a new glossary_vN.json and repoints
  glossary_current.json at it -- nothing is ever edited in place.
- After each sortie, review_queue() surfaces words that matched no
  glossary entry and are either frequent or low-ASR-confidence, for a
  human to accept into the dictionary via approve_terms().

    python3 code/speech/glossary.py tag   exports/audio/<sortie>/transcript.json
    python3 code/speech/glossary.py queue exports/audio/<sortie>/transcript.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Transcript  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GLOSSARY_DIR = ROOT / "exports" / "audio" / "glossary"
CURRENT = GLOSSARY_DIR / "glossary_current.json"

# id -> {category, en, he, variants[], notes}
SEED = {
    # maneuvers
    "AUTOROTATION": {"category": "maneuver", "en": "autorotation", "he": "אוטורוטציה",
                      "variants": ["auto rotation", "auto-rotation", "auto rotate"]},
    "QUICK_STOP": {"category": "maneuver", "en": "quick stop", "he": "עצירה מהירה",
                   "variants": ["quickstop", "quick-stop"]},
    "STEEP_TURN": {"category": "maneuver", "en": "steep turn", "he": "פנייה חדה", "variants": []},
    "HOVER": {"category": "maneuver", "en": "hover", "he": "ריחוף", "variants": ["hovering"]},
    "CONFINED_AREA": {"category": "maneuver", "en": "confined area", "he": "שטח מוקף",
                       "variants": []},
    "TRAFFIC_PATTERN": {"category": "maneuver", "en": "traffic pattern", "he": "מסלול תעבורה",
                         "variants": ["pattern", "circuit"]},
    "GO_AROUND": {"category": "maneuver", "en": "go around", "he": "המראת סירוב",
                  "variants": ["go-around", "going around"]},
    "APPROACH": {"category": "maneuver", "en": "approach", "he": "גישה", "variants": []},

    # systems / controls
    "COLLECTIVE": {"category": "system", "en": "collective", "he": "קולקטיב", "variants": []},
    "CYCLIC": {"category": "system", "en": "cyclic", "he": "ציקלי", "variants": []},
    "PEDAL": {"category": "system", "en": "pedal", "he": "דוושה", "variants": ["pedals"]},
    "TORQUE": {"category": "system", "en": "torque", "he": "מומנט", "variants": []},
    "ROTOR": {"category": "system", "en": "rotor", "he": "רוטור", "variants": []},
    "TAIL_ROTOR": {"category": "system", "en": "tail rotor", "he": "רוטור זנב", "variants": []},
    "ENGINE_OIL": {"category": "system", "en": "engine oil", "he": "שמן מנוע", "variants": []},
    "XMSN_OIL": {"category": "system", "en": "transmission oil", "he": "שמן תמסורת",
                 "variants": ["xmsn oil"]},
    "HYDRAULIC": {"category": "system", "en": "hydraulic", "he": "הידראולי", "variants": []},
    "FUEL_QTY": {"category": "system", "en": "fuel quantity", "he": "כמות דלק",
                 "variants": ["fuel qty"]},

    # abbreviations (mostly straight off the PFD/panel)
    "HTAWS": {"category": "abbrev", "en": "helicopter terrain awareness and warning system",
              "he": "", "variants": []},
    "XPDR": {"category": "abbrev", "en": "transponder", "he": "משיב", "variants": ["squawk"]},
    "RA": {"category": "abbrev", "en": "radar altitude", "he": "גובה רדאר", "variants": ["radalt"]},
    "PI": {"category": "abbrev", "en": "power index", "he": "מדד הספק", "variants": []},
    "DA": {"category": "abbrev", "en": "density altitude", "he": "", "variants": []},
    "OAT": {"category": "abbrev", "en": "outside air temperature", "he": "", "variants": []},
    "N1": {"category": "abbrev", "en": "gas generator speed", "he": "", "variants": []},
    "N2": {"category": "abbrev", "en": "power turbine speed", "he": "", "variants": []},
    "NR": {"category": "abbrev", "en": "rotor rpm", "he": "", "variants": []},
    "GS": {"category": "abbrev", "en": "ground speed", "he": "", "variants": []},
    "TAS": {"category": "abbrev", "en": "true airspeed", "he": "", "variants": []},
    "MSL": {"category": "abbrev", "en": "mean sea level", "he": "", "variants": []},
    "AGL": {"category": "abbrev", "en": "above ground level", "he": "", "variants": []},
    "CRM": {"category": "abbrev", "en": "crew resource management", "he": "", "variants": []},

    # callouts
    "CHECK": {"category": "callout", "en": "check", "he": "בדוק", "variants": ["checking"]},
    "CLEAR": {"category": "callout", "en": "clear", "he": "פנוי", "variants": ["clear left",
              "clear right"]},
    "CONFIRM": {"category": "callout", "en": "confirm", "he": "אשר", "variants": ["confirmed"]},
    "STANDBY": {"category": "callout", "en": "standby", "he": "המתן", "variants": ["stand by"]},
    "COPY": {"category": "callout", "en": "copy", "he": "קיבלתי", "variants": ["copy that"]},
    "ROGER": {"category": "callout", "en": "roger", "he": "", "variants": []},

    # academy / training terminology
    "SORTIE": {"category": "academy", "en": "sortie", "he": "גיחה", "variants": ["flight"]},
    "CADET": {"category": "academy", "en": "cadet", "he": "צוער", "variants": []},
    "INSTRUCTOR": {"category": "academy", "en": "instructor", "he": "מדריך", "variants": []},
    "DEBRIEF": {"category": "academy", "en": "debrief", "he": "תחקיר", "variants": []},
    "CHECKLIST": {"category": "academy", "en": "checklist", "he": "רשימת בדיקה", "variants": []},
}


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s, flags=re.UNICODE).strip().lower()


def _index(glossary: dict) -> dict[str, str]:
    """surface-form (normalised) -> glossary id"""
    idx = {}
    for gid, e in glossary.items():
        forms = [e["en"], *e.get("variants", [])]
        if e.get("he"):
            forms.append(e["he"])
        for f in forms:
            if f:
                idx[_norm(f)] = gid
    return idx


def load_glossary() -> dict:
    if CURRENT.exists():
        return json.loads(Path(CURRENT.resolve()).read_text())
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    path = GLOSSARY_DIR / "glossary_v1.json"
    path.write_text(json.dumps(SEED, indent=2, ensure_ascii=False))
    CURRENT.write_text(json.dumps(SEED, indent=2, ensure_ascii=False))
    _log_version(1, "seed", list(SEED))
    return SEED


def _next_version() -> int:
    vs = [int(p.stem.split("_v")[1]) for p in GLOSSARY_DIR.glob("glossary_v*.json")]
    return (max(vs) + 1) if vs else 1


def _log_version(v: int, note: str, added: list[str]) -> None:
    log = GLOSSARY_DIR / "version_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps({"version": v, "note": note, "added": added}) + "\n")


def approve_terms(new_entries: dict, note: str = "reviewed additions") -> dict:
    """Merge new_entries into the glossary and write a new version."""
    glossary = load_glossary()
    glossary = {**glossary, **new_entries}
    v = _next_version()
    (GLOSSARY_DIR / f"glossary_v{v}.json").write_text(
        json.dumps(glossary, indent=2, ensure_ascii=False))
    CURRENT.write_text(json.dumps(glossary, indent=2, ensure_ascii=False))
    _log_version(v, note, list(new_entries))
    return glossary


def match_transcript(transcript: Transcript, glossary: dict | None = None) -> Transcript:
    """Tag each Word.glossary_term with the canonical id it matches
    (single word or, for multi-word terms, the first word of the phrase)."""
    glossary = glossary or load_glossary()
    idx = _index(glossary)
    # multi-word phrases, longest first, so "traffic pattern" beats "pattern"
    phrases = sorted((f for f in idx if " " in f), key=len, reverse=True)

    for seg in transcript.segments:
        n = len(seg.words)
        i = 0
        while i < n:
            matched = False
            for ph in phrases:
                k = len(ph.split())
                if i + k <= n:
                    window = _norm(" ".join(w.text for w in seg.words[i:i + k]))
                    if window == ph:
                        for w in seg.words[i:i + k]:
                            w.glossary_term = idx[ph]
                        i += k
                        matched = True
                        break
            if matched:
                continue
            gid = idx.get(_norm(seg.words[i].text))
            if gid:
                seg.words[i].glossary_term = gid
            i += 1
    return transcript


def unknown_terms(transcript: Transcript, min_count: int = 2, max_conf: float = 0.75):
    """Words with no glossary match that are either frequent or ASR-shaky --
    candidates for the review queue. Returns [(text, count, avg_prob, example_ctx)]."""
    counts, confs, ctx = Counter(), {}, {}
    for seg in transcript.segments:
        for i, w in enumerate(seg.words):
            if w.glossary_term or not w.text.strip():
                continue
            key = _norm(w.text)
            if not key:
                continue
            counts[key] += 1
            confs.setdefault(key, []).append(w.prob)
            ctx.setdefault(key, seg.text[:120])
    out = []
    for term, n in counts.items():
        avg = sum(confs[term]) / len(confs[term])
        if n >= min_count or avg < max_conf:
            out.append((term, n, round(avg, 2), ctx[term]))
    return sorted(out, key=lambda r: (-r[1], r[2]))


def review_queue(transcript: Transcript, out_csv: str | Path) -> Path:
    import csv
    rows = unknown_terms(transcript)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["term", "count", "avg_confidence", "example_context"])
        w.writerows(rows)
    return out_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["tag", "queue"])
    ap.add_argument("transcript_json")
    args = ap.parse_args()

    t = Transcript.load(args.transcript_json)
    if args.action == "tag":
        match_transcript(t)
        t.save(args.transcript_json)
        n = sum(1 for w in t.words() if w.glossary_term)
        print(f"tagged {n} words -> {args.transcript_json}")
    else:
        out = Path(args.transcript_json).parent / "glossary_review_queue.csv"
        review_queue(t, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
