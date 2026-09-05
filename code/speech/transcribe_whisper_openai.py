"""ASR backend: OpenAI Whisper large-v3 (the openai-whisper package, not
transformers) -- the standard multilingual baseline, no Hebrew fine-tune.
This is what code/speech/pipeline.py (solution 1) uses.

Install: pip install openai-whisper (present). Ungated, ~3 GB.
Language is auto-detected unless --language is given; word timestamps
via word_timestamps=True (per-word `probability` -> Word.prob, so this is
the one backend whose confidence numbers in the report are real).

    transcribe(video, language=None) -> schema.Transcript
    python3 code/speech/transcribe_whisper_openai.py "/path/s1 c2.mp4"
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_common import build_transcript, load_wav, run_cli  # noqa: E402
from schema import Segment, Word  # noqa: E402

MODEL_NAME = "large-v3"
AVIATION_PROMPT = ("autorotation, quick stop, steep turn, hover, collective, cyclic, "
                   "pedal, torque, HTAWS, radar altitude, power index, checklist, "
                   "go around, נחיתה, המראה, מסוק")
_MODEL = None


def transcribe(video, language: str | None = None):
    import torch
    import whisper
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model(MODEL_NAME,
                                    device="cuda" if torch.cuda.is_available() else "cpu")
    y, sr, wav = load_wav(video)
    res = _MODEL.transcribe(y, language=language, word_timestamps=True,
                            initial_prompt=AVIATION_PROMPT,
                            condition_on_previous_text=False, verbose=False)
    segs = []
    for s in res["segments"]:
        text = s["text"].strip()
        if not re.search(r"\w", text, re.UNICODE):
            continue
        ws = [Word(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"]),
                  prob=float(w.get("probability", 1.0)))
             for w in s.get("words", []) if re.search(r"\w", w["word"], re.UNICODE)]
        segs.append(Segment(id=len(segs), start=float(s["start"]), end=float(s["end"]),
                            text=text, lang=res.get("language", "?"), speaker=None, words=ws))
    return build_transcript(Path(video).stem, video, [w for s in segs for w in s.words], segs)


if __name__ == "__main__":
    run_cli("whisper_openai", transcribe)
