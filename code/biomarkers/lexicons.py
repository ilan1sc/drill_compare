"""Seed cue-phrase lexicons for the audio biomarkers (PRD 12.3).

These are first-pass, keyword/pattern heuristics -- not a trained
classifier. Every biomarker documents that limitation. Extend these lists
(and pull confirmed vocabulary from the glossary, code/speech/glossary.py)
as real transcripts come in; that is the intended feedback loop.

Hebrew entries were expanded once from real cockpit transcripts (three
sorties, ~9000 words: s1 c2/c4/c6 -- see
exports/audio/<sortie>/glossary_review_queue.csv for the frequency data
they were pulled from) but still need review by a Hebrew-speaking SME
before being trusted for scoring -- flight-training register is narrow
and three sorties is a small sample.
"""

CUES = {
    "observation": {
        "en": ["i see", "i have", "i'm at", "im at", "we're at", "were at", "we have",
               "showing", "reading", "indicating", "looks like", "it's at", "its at",
               "altitude is", "airspeed is", "we are at", "coming up on"],
        "he": ["אני רואה", "יש לי", "אנחנו על", "מראה", "קורא", "נראה", "אנחנו ב",
               # readout call-outs: "height is low", "speed's good" -- very common
               # cockpit shorthand where the reading is just stated bare, e.g.
               # "גובה 960" (altitude 960), "מהירות קצת נמוכה" (speed's a bit low)
               "גובה", "מהירות", "מרגיש לך", "נראה לי", "אני שומע"],
    },
    "intention": {
        "en": ["i will", "i'll", "i'm going to", "im going to", "let's", "lets",
               "i plan to", "going to", "i'm gonna", "gonna", "i intend to", "we will"],
        "he": ["אני אעשה", "אני הולך", "בוא נ", "בואו נ", "אני מתכוון", "נעשה",
               "אני אקרב", "אני אזוז", "בא לי ל"],
    },
    "action": {
        "en": ["lowering", "raising", "applying", "reducing", "increasing", "adding power",
               "pulling", "pushing", "rolling", "leveling", "levelling", "trimming",
               "coming down", "coming up", "initiating", "entering", "rolling in",
               "rolling out", "cutting", "chopping throttle"],
        "he": ["מוריד", "מעלה", "מפעיל", "מקטין", "מגביר", "מושך", "דוחף", "מתחיל",
               # confirmed from real transcripts: "מרים קולקטיב" (raising collective),
               # "מכניס רגל שמאל" (putting in left pedal), "פתח פי-איי" (opened PI)
               "מרים", "מכניס", "פותח", "פתח", "סוגר", "עושה פנייה", "נכנס ל"],
    },
    "verification": {
        "en": ["confirmed", "confirm", "verified", "as expected", "that's correct",
               "thats correct", "good", "correct", "check", "checks out", "roger",
               "copy", "affirm", "affirmative", "matches"],
        "he": ["מאושר", "מאומת", "כצפוי", "נכון", "בסדר", "מקבל", "מאשר",
               "טוב", "אוקיי", "אוקי", "כן"],
    },
    "error_correction": {
        "en": ["sorry", "correction", "i meant", "let me redo", "mistake", "wrong",
               "actually", "scratch that", "disregard", "my bad", "that's not right",
               "thats not right", "let me try again"],
        "he": ["סליחה", "תיקון", "התכוונתי", "טעות", "לא נכון", "בעצם", "בואו ננסה שוב",
               "לא הבנתי", "טעיתי"],
    },
    "prediction": {
        "en": ["should", "expect", "going to be", "about to", "will be", "we'll see",
               "well see", "anticipate", "likely", "probably going to"],
        "he": ["צריך להיות", "צפוי", "עומד ל", "יהיה", "כנראה"],
    },
    "prioritization": {
        "en": ["first", "priority", "most important", "before anything", "need to first",
               "top priority", "main thing", "focus on", "let's focus"],
        "he": ["קודם", "עדיפות", "הכי חשוב", "לפני הכל", "להתמקד", "לפני ש"],
    },
    "info_sharing": {
        "en": ["you have controls", "i have controls", "be advised", "heads up",
               "watch out", "traffic", "obstacle", "fyi", "for your information",
               "just so you know", "note that"],
        "he": ["אתה שולט", "אני שולט", "שים לב", "תשומת לב", "תראה", "מטרד",
               "תראה לי", "תבהיר לי", "איזה ציר אתה רואה"],
    },
    "disfluency_fillers": {
        "en": ["um", "uh", "er", "erm", "hmm", "like", "you know"],
        "he": ["אה", "אמ", "אהה", "כאילו"],
    },
}


def has_cue(text: str, category: str) -> bool:
    t = f" {text.lower()} "
    cues = CUES[category]
    return any(f" {c} " in t or t.startswith(f"{c} ") or t.endswith(f" {c} ")
              for c in (*cues["en"], *cues["he"]))


def matched_cues(text: str, category: str) -> list[str]:
    t = f" {text.lower()} "
    cues = CUES[category]
    return [c for c in (*cues["en"], *cues["he"]) if f" {c} " in t]
