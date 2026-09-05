"""12.3 Observation -> Intention -> Action -> Verification loop
completeness and timing.

Method: take every segment tagged observation/intention/action/
verification (b01-b04's lexicons), sort by time, and split into episodes
wherever the gap between consecutive tagged segments exceeds
--episode-gap seconds (default 25s -- roughly "a new topic"). Within each
episode:
  - completeness: which of the 4 stages appeared at least once
  - order: whether first-occurrence times are O <= I <= A <= V
  - timing: latency between each pair of consecutive present stages

Limitation: episode boundaries and stage order are both heuristics over
noisy cue-phrase tags (b01-b04); treat the loop-completion rate as a
rough proxy, not a certified count. A stage silently skipped in speech
(action performed without verbalizing it) will show as "missing" here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base import BiomarkerResult, evidence_row, run_cli, tag_segments  # noqa: E402

STAGES = ["observation", "intention", "action", "verification"]


def _episodes(events, gap):
    episodes, cur = [], []
    for e in events:
        if cur and e["t"] - cur[-1]["t"] > gap:
            episodes.append(cur)
            cur = []
        cur.append(e)
    if cur:
        episodes.append(cur)
    return episodes


def compute(transcript, cadet_speaker=None, episode_gap: float = 25.0) -> BiomarkerResult:
    tags = tag_segments(transcript, cadet_speaker)
    events = sorted(
        ({"t": seg.start, "stage": stage, "text": seg.text, "speaker": seg.speaker}
         for stage in STAGES for seg in tags[stage]),
        key=lambda e: e["t"],
    )
    episodes = _episodes(events, episode_gap)

    complete, ordered, latencies = 0, 0, {"O->I": [], "I->A": [], "A->V": [], "O->V": []}
    evidence = []
    for ep in episodes:
        first = {}
        for e in ep:
            first.setdefault(e["stage"], e["t"])
        present = [s for s in STAGES if s in first]
        is_complete = len(present) == 4
        complete += is_complete
        if is_complete:
            times = [first[s] for s in STAGES]
            if times == sorted(times):
                ordered += 1
                latencies["O->I"].append(first["intention"] - first["observation"])
                latencies["I->A"].append(first["action"] - first["intention"])
                latencies["A->V"].append(first["verification"] - first["action"])
                latencies["O->V"].append(first["verification"] - first["observation"])
        evidence.append({
            "episode_start": round(ep[0]["t"], 2), "episode_end": round(ep[-1]["t"], 2),
            "stages_present": present,
            "events": [{"t": round(e["t"], 2), "stage": e["stage"], "text": e["text"]}
                      for e in ep],
        })

    n = len(episodes) or 1
    metrics = {
        "episodes": len(episodes),
        "loop_complete_rate": round(complete / n, 3),
        "loop_in_order_rate": round(ordered / n, 3),
        "avg_latency_s": {k: round(sum(v) / len(v), 2) if v else None
                          for k, v in latencies.items()},
    }
    return BiomarkerResult("b05_oiav_loop", metrics, evidence,
                           limitations="episode segmentation + stage order over noisy "
                                       "cue-phrase tags; see module docstring")


if __name__ == "__main__":
    run_cli("b05_oiav_loop", compute)
