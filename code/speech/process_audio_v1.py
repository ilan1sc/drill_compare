"""Package the 'best' audio pipeline (winners of the engine bake-offs --
see audio_res.md) into one command: trim a window out of a source video,
transcribe + diarize + glossary-tag + run every biomarker, then render the
biomarker visualization movie for that same window.

    ASR          : ivrit-ai/whisper-large-v3      (transcribe_ivrit_v3.py)
    diarization  : 3D-Speaker / CAM++             (diarize_3dspeaker.py)
    glossary     : aviation glossary v2           (glossary.py)
    biomarkers   : b01..b15                       (run_biomarkers.py)
    movie        : speaker banner + RTL subtitles + live biomarker panel
                   (audio_vis_biomarkers.py)

Writes everything into <out-dir>/<sample-name>/:
    sample.wav, sample_video.mp4, transcript.json, glossary_review_queue.csv,
    biomarkers/b01_*.json ... _summary.json, biomarker_vis_<start>_<dur>.mp4

Video source defaults to the copilot-cam cockpit camera
(06_F2-COPILOT CAM COCKPIT...) -- pass either that file directly, or the
cadet's source dir and it's auto-selected.

--duration defaults to the REST OF THE FILE (probed via ffprobe) when
omitted, so leaving it out runs the whole sortie; pass it to take a short
sample instead.

Usage:
    # whole sortie, no --duration needed
    python3 code/speech/process_audio_v1.py \\
        "/path/to/26.1.s1.c6.sorek_bigeyes_exp_v2" \\
        --out-dir "/path/to/26.1.s1.audio/26.1.s1.c6.sorek_bigeyes"

    # a short sample instead
    python3 code/speech/process_audio_v1.py \\
        "/path/to/26.1.s1.c6.sorek_bigeyes_exp_v2" \\
        --out-dir "/path/to/26.1.s1.audio/26.1.s1.c6.sorek_bigeyes" \\
        --start 600 --duration 60

    # separate mic/audio feed instead of the video's own audio track
    python3 code/speech/process_audio_v1.py "...COCKPIT.mp4" \\
        --audio-source "...00_F2-AUDIO(...).m4a" \\
        --out-dir "...26.1.s1.c2.ilan_oria" --start 600 --duration 300

    # skip the movie (transcript + biomarkers only, much faster)
    python3 code/speech/process_audio_v1.py "...mp4" --out-dir ... --skip-movie
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "biomarkers"))

import diarize_3dspeaker   # noqa: E402
import glossary             # noqa: E402
import transcribe_ivrit_v3  # noqa: E402
import run_biomarkers       # noqa: E402

ENGINES = {"asr": "ivrit-ai/whisper-large-v3", "diarization": "3D-Speaker (CAM++)",
           "glossary": "aviation glossary (glossary.py)"}

DEFAULT_CAM_GLOB = "06_F2-COPILOT CAM COCKPIT*"


def _resolve_video(video: str, cam_glob: str = DEFAULT_CAM_GLOB) -> Path:
    """video may be an exact file, or a cadet source dir (e.g.
    .../26.1.s1.c6.sorek_bigeyes_exp_v2) -- in which case the copilot-cam
    cockpit video (06_F2-COPILOT CAM COCKPIT...) is picked by default."""
    p = Path(video)
    if p.is_dir():
        hits = sorted(p.glob(cam_glob))
        if not hits:
            sys.exit(f"no file matching {cam_glob!r} in {p}")
        return hits[0]
    return p


def _probe_duration(src: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def _ffmpeg_trim_audio(src: Path, start: float, duration: float, out_wav: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(duration),
         "-i", str(src), "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out_wav)],
        check=True)


def _ffmpeg_trim_video(src: Path, start: float, duration: float, out_mp4: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(duration),
         "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", str(out_mp4)],
        check=True)


def run(video: str, out_dir: str, start: float = 0.0, duration: float | None = None,
       sample_name: str | None = None, audio_source: str | None = None,
       speakers: int | None = 2, skip_movie: bool = False,
       skip_biomarkers: bool = False, cadet_speaker: str | None = None,
       cam_glob: str = DEFAULT_CAM_GLOB) -> Path:
    """duration=None (the default) means "to the end of the file" -- probed
    via ffprobe -- so a plain run with no --duration processes the whole
    sortie; pass a duration to take a short sample instead."""
    t0 = time.time()
    video = _resolve_video(video, cam_glob)
    print(f"      video source: {video.name}")
    if duration is None:
        duration = max(0.0, _probe_duration(video) - start)
        print(f"      --duration not given -> using rest of file: {duration:.0f}s")
    name = sample_name or (f"sample_{int(duration)}s" if start or duration else "sample_full")
    sample_dir = Path(out_dir) / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    wav = sample_dir / f"{name}.wav"
    # trimmed video is a scratch file only, used to render the movie below --
    # the original/trimmed source video is never kept in the library.
    sample_video = Path(tempfile.gettempdir()) / f"process_audio_v1_{name}_{start:.0f}.mp4"
    audio_src = Path(audio_source) if audio_source else video

    print(f"[0/5] trimming {start:.0f}s + {duration:.0f}s ...")
    _ffmpeg_trim_audio(audio_src, start, duration, wav)
    if not skip_movie:
        _ffmpeg_trim_video(video, start, duration, sample_video)

    print("[1/5] transcribe  ivrit-ai/whisper-large-v3 ...")
    t = transcribe_ivrit_v3.transcribe(str(wav))
    print(f"      {len(t.segments)} segments, {sum(len(s.words) for s in t.segments)} words, "
          f"{t.duration:.0f}s  ({time.time() - t0:.0f}s)")

    print("[2/5] diarize  3D-Speaker ...")
    t = diarize_3dspeaker.diarize(t, wav, n_speakers=speakers)

    print("[3/5] glossary tag ...")
    glossary.match_transcript(t, glossary.load_glossary())

    t.save(sample_dir / "transcript.json")
    q = glossary.review_queue(t, sample_dir / "glossary_review_queue.csv")
    print(f"      transcript -> {sample_dir / 'transcript.json'}")
    print(f"      review queue -> {q}")

    if not skip_biomarkers:
        print("[4/5] biomarkers ...")
        run_biomarkers.run_all(
            sample_dir / "transcript.json", cadet_speaker=cadet_speaker,
            wav_path=str(wav), audio_dir=Path(out_dir),
            extra_summary={"engines": ENGINES, "source_video": str(video),
                           "sample_window": f"{start:.0f}-{start + duration:.0f}s"})
    else:
        print("[4/5] skipped")

    if not skip_movie:
        print("[5/5] biomarker visualization movie ...")
        try:
            subprocess.run(
                [sys.executable, str(HERE / "audio_vis_biomarkers.py"), str(sample_video),
                 "--dir", str(sample_dir), "--start", "0", "--duration", str(duration)],
                check=True)
        finally:
            sample_video.unlink(missing_ok=True)
    else:
        print("[5/5] skipped")

    print(f"done {name} in {time.time() - t0:.0f}s -> {sample_dir}")
    return sample_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="source cockpit video, or a cadet source dir (in which case the "
                    "copilot-cam cockpit video, 06_F2-COPILOT CAM COCKPIT*, is auto-selected)")
    ap.add_argument("--cam-glob", default=DEFAULT_CAM_GLOB,
                    help="which file to pick when `video` is a directory (default: copilot cam)")
    ap.add_argument("--out-dir", required=True, help="cadet library dir, e.g. .../26.1.s1.c6.sorek_bigeyes")
    ap.add_argument("--start", type=float, default=0.0, help="window start, seconds into the source")
    ap.add_argument("--duration", type=float, default=None,
                    help="window length, seconds (default: rest of the file -- whole sortie)")
    ap.add_argument("--sample-name", default=None, help="subfolder name (default sample_<duration>s)")
    ap.add_argument("--audio-source", default=None,
                    help="separate audio file (e.g. dedicated mic feed) instead of the video's own track")
    ap.add_argument("--speakers", default="2", help="int, or 'auto' to let 3D-Speaker decide")
    ap.add_argument("--cadet-speaker", default=None)
    ap.add_argument("--skip-biomarkers", action="store_true")
    ap.add_argument("--skip-movie", action="store_true", help="skip trimming video + rendering the movie")
    args = ap.parse_args()
    spk = None if args.speakers == "auto" else int(args.speakers)
    run(args.video, args.out_dir, args.start, args.duration, args.sample_name,
        args.audio_source, spk, args.skip_movie, args.skip_biomarkers, args.cadet_speaker,
        args.cam_glob)


if __name__ == "__main__":
    main()
