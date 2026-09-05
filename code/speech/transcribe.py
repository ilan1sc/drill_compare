"""Speech-to-text with word-level timestamps (openai-whisper, large-v3).

Hebrew/English: whisper auto-detects language and code-switches
reasonably well at the segment level; no language is forced. Aviation
terms get a boost via --prompt (a short list of expected vocabulary
fed as Whisper's initial_prompt) and are re-tagged more precisely later
by glossary.match_transcript().

Low-confidence words: whisper's word_timestamps give a per-word
`probability`; anything below --min-conf is flagged (Word.prob kept as-is,
callers filter/report it -- see glossary.review_queue()).

    python3 code/speech/transcribe.py "/path/s1 c2.mp4"
    python3 code/speech/transcribe.py "/path/s1 c2.mp4" --model medium
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_audio import extract  # noqa: E402
from schema import Segment, Transcript, Word  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = ROOT / "exports" / "audio"

AVIATION_PROMPT = (
    "autorotation, quick stop, steep turn, hover, collective, cyclic, pedal, "
    "torque, rotor RPM, HTAWS, transponder, radar altitude, power index, "
    "checklist, go around, confirm, standby, נחיתה, המראה, מסוק, טייס"
)

_MODEL_CACHE = {}


def _load_model(name: str):
    import torch
    import whisper
    if name not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE[name] = whisper.load_model(name, device=device)
    return _MODEL_CACHE[name]


def transcribe(video: str | Path, model_name: str = "large-v3",
              prompt: str = AVIATION_PROMPT) -> Transcript:
    video = Path(video)
    wav = extract(video)
    model = _load_model(model_name)
    result = model.transcribe(
        str(wav), word_timestamps=True, initial_prompt=prompt,
        condition_on_previous_text=False,   # avoid drift across a 45min sortie
        verbose=False,
    )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not re.search(r"\w", text, flags=re.UNICODE):
            continue    # whisper sometimes emits empty/punctuation-only
                        # segments over engine noise / silence -- drop them
        words = [
            Word(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"]),
                prob=float(w.get("probability", 1.0)))
            for w in seg.get("words", []) if re.search(r"\w", w["word"], flags=re.UNICODE)
        ]
        segments.append(Segment(
            id=len(segments), start=float(seg["start"]), end=float(seg["end"]),
            text=text, lang=result.get("language", "?"), speaker=None, words=words,
        ))

    return Transcript(sortie=video.stem, source_video=str(video),
                      duration=segments[-1].end if segments else 0.0,
                      segments=segments)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--out", help="transcript JSON path (default exports/audio/<sortie>/transcript.json)")
    args = ap.parse_args()

    t = transcribe(args.video, args.model)
    out = Path(args.out) if args.out else AUDIO_DIR / t.sortie / "transcript.json"
    t.save(out)
    print(f"{len(t.segments)} segments, {sum(len(s.words) for s in t.segments)} words, "
          f"{t.duration:.0f}s -> {out}")


if __name__ == "__main__":
    main()
