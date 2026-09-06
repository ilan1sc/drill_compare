# Audio pipeline — results & engine verdicts

Summary of the speech/audio analysis pipeline, the engine bake-offs, and
the full-data run. Generated 2026-09-06.

## Pipeline

```
sortie.mp4
  -> 16 kHz mono wav                    (extract_audio.py)
  -> TRANSCRIBE  ivrit-ai/whisper-large-v3   (transcribe_ivrit_v3.py; forced language=he, word timestamps)
  -> DIARIZE     3D-Speaker / CAM++          (diarize_3dspeaker.py; 2 speakers: cadet + instructor)
  -> GLOSSARY    aviation glossary v2        (glossary.py; maneuver / system / callout tagging)
  -> BIOMARKERS  b01..b15                    (run_biomarkers.py; per-utterance, evidence timestamps)
  -> MOVIE       audio_vis_biomarkers.py     (speaker banner + RTL subtitle + live biomarker panel)
```

Driver: `code/speech/pipeline_best.py` -> writes `exports/audio_best/<sortie>/`
(`transcript.json`, `glossary_review_queue.csv`, `biomarkers/b01_*.json ... _summary.json`).
Batch over all data: `.scratch/run_best_all.sh`.

## Transcription — winner: `ivrit-ai/whisper-large-v3`

s1 c6, 180 s window (`exports/audio/s1 c6/asr_compare/report.md`):

| engine | coverage | trigram-repeat (hallucination) | Hebrew % | timestamps | RTF | verdict |
|---|---|---|---|---|---|---|
| **ivrit-ai/whisper-large-v3** | **0.74** | **0.023** | 97 | word | 2.7x | **best — high coverage, no looping** |
| ivrit-ai/whisper-large-v3-turbo | 0.61 | 0.019 | 97 | word | 15x | good fast fallback; loops on some clips |
| whisper large-v3 (vanilla) | 0.42 | 0.019 | 97 | word | 3.3x | weak Hebrew coverage, 12% low-conf words |
| ivrit-ai/yi-whisper-large-v3 | 0.65 | **0.197** | 100 | word | 1.5x | severe looping — unusable |
| facebook/mms-1b-all | 0.71 | 0.0 | 99 | segment | 6.9x | never hallucinates but sparse, no confidence |
| SeamlessM4T v2 large | 1.0 | 0.008 | 97 | **none** | 2.8x | no timestamps -> can't align to diarization / biomarkers |

- **coverage** = transcribed speech seconds / window seconds.
- **trigram-repeat** = 1 - unique_trigrams / total_trigrams; high = looped / hallucinated output.
- SeamlessM4T's 1.0 coverage is window-level segments with no word timing, so it cannot feed diarization or biomarkers.

## Diarization — winner: 3D-Speaker (CAM++)

s1 c6, 300 s crew window (ground truth = 2 speakers, ~50/50)
(`exports/audio/s1 c6/diarize_compare/_report.txt`):

| engine | speakers found | cadet / instr split | verdict |
|---|---|---|---|
| **3D-Speaker** | **2** | 50.5 / 49.5 | **best — clean, consistent across sorties, handles >2 via its own clustering** |
| NeMo Sortformer | 2 (here) | 50 / 50 | ties here but over-split on s1 c2 (invented 2 extra speakers) |
| WeSpeaker | 2 | 50 / 50 | ties here; collapsed to 1 speaker on other audio |
| MFCC (`diarize.py`, original) | 5 | instructor split 3 ways | over-splits |
| SpeechBrain ECAPA | 5 | cadet only 32% | worst over-split |
| pyannote / WhisperX / Kaldi | — | — | couldn't run (HF-gated / no pip / needs compiled Kaldi) |

Chosen because it is the most consistently clean across sorties and, when the
speaker count is not forced, picks it itself (crew + sim instructor + HE/EN
sim callouts, up to ~5).

## Full-data run (all sorties, best pipeline)

| sortie | words | cadet / instr words | obs/min | act/min | hesit/min | comm-relevance |
|---|---|---|---|---|---|---|
| s1 c2 | 2861 | 1788 / 1068 | 0.31 | 0.09 | 1.11 | 0.21 |
| s1 c3 | 4007 | 2774 / 1233 | 2.06 | 2.23 | 1.62 | 0.50 |
| s1 c4 | 4117 | 2916 / 1200 | 4.28 | 2.39 | 2.14 | 0.55 |
| s1 c5 | 3266 | 3129 / **137** | 2.27 | 1.30 | 1.79 | 0.44 |
| s1 c6 | 3735 | 2080 / 1638 | 1.75 | 1.69 | 1.12 | 0.51 |
| s1 c8 | 3775 | 2242 / 1527 | 0.48 | 0.20 | 1.16 | 0.27 |

Per-sortie 3-min biomarker movies: `exports/audio_best/<sortie>/biomarker_vis_600_180.mp4`.

## Known issues

- **s1 c5** diarization collapsed nearly to one speaker (3129 vs 137 words) — the
  instructor is barely separated. Re-run with `--speakers auto`.
- **s1 c2** statement rates (obs 0.31/min, act 0.09/min, relevance 0.21) look
  suspiciously low vs c3/c4/c6 — check transcript/diarization quality.
- The other four (c3, c4, c6, c8) look solid.

## How to re-run

```
# one sortie, best engines end-to-end
python3 code/speech/pipeline_best.py "/home/ilan/Documents/flight school/data/s1 c6.mp4"

# with 3D-Speaker choosing the speaker count itself
python3 code/speech/pipeline_best.py "...s1 c5.mp4" --speakers auto

# biomarker movie for a window
python3 code/speech/audio_vis_biomarkers.py "...s1 c6.mp4" \
    --dir "exports/audio_best/s1 c6" --start 600 --duration 180

# re-run the engine bake-offs
python3 code/speech/transcribe_compare.py "...s1 c6.mp4" --transcript "exports/audio/s1 c6/transcript.json"
python3 code/speech/diarize_compare.py    "...s1 c6.mp4" --transcript "exports/audio/s1 c6/transcript.json"
```
