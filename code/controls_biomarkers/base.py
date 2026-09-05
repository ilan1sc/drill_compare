"""Shared plumbing for the flight-control-analytics biomarkers (PRD 10).

Flight analytics (code/pfd_compare_csv.py's drills) describes WHAT the
aircraft did. These modules describe HOW the cadet produced that result,
from the raw control-input channels in the sim data CSV -- collective,
plus the CYCLIC's two axes (lateral = roll, longitudinal = pitch):

    CollectivePosition           0..1     -> paired state: vertical speed
    LateralStickPosition        -1..1     -> paired state: bank (roll)      | cyclic
    LongitudinalStickPosition   -1..1     -> paired state: pitch attitude   | cyclic

Each also has a *Force channel (CollectiveForce, LateralStickForce,
LongitudinalStickForce) -- how hard the axis was pushed, not just how far.
The ICD marks these "not used in-app"; they ARE loaded here
(flight[f"{name}_force"]) for any module that wants force vs. position
(e.g. a tense/heavy grip that doesn't show up in position at all), even
though none of the 9 controls_*.py modules currently key off force alone.

(there is no logged pedal channel in this ICD -- only these three HOCAS
axes -- so yaw/pedal control quality is out of scope here.)

Every controls_*.py module exposes `compute(flight, **kwargs) -> Result`
and a CLI: `python3 code/controls_biomarkers/controls_precision.py
data.csv`. Segmentation (hold vs. maneuvering) and "what counts as a
correction" are both derived from each sortie's OWN control-rate
distribution (median + k*MAD), the same scenario-relative-threshold
convention used elsewhere in this project (see b14_pause_hesitation.py)
rather than a fixed universal cutoff.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CONTROLS_DIR = ROOT / "exports" / "controls"

FM = "FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data."
HOCAS = "FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message."
HOT = "FFS1_Ownship.ESL_LHD_Landing_detection.ESL_LHD_Landing_detection.Point_1_HOT"
M2FT = 3.28084
RATE_HZ = 10.0
DT = 1.0 / RATE_HZ

# control channel -> (data column, the state it primarily drives, that state's units).
# lateral + longitudinal together are the cyclic's two axes; collective is
# the third HOCAS axis. Each also has a *Force column (see force_col) loaded
# alongside position -- how hard the axis was pushed, not just how far.
CHANNELS = {
    "collective":   {"col": HOCAS + "CollectivePosition", "force_col": HOCAS + "CollectiveForce",
                     "state": "vs_fpm", "state_label": "vertical speed",
                     "state_unit": "ft/min", "ctrl_unit": "0..1"},
    "lateral":      {"col": HOCAS + "LateralStickPosition", "force_col": HOCAS + "LateralStickForce",
                     "state": "roll_deg", "state_label": "bank angle (cyclic, lateral axis)",
                     "state_unit": "deg", "ctrl_unit": "-1..1"},
    "longitudinal": {"col": HOCAS + "LongitudinalStickPosition",
                     "force_col": HOCAS + "LongitudinalStickForce",
                     "state": "pitch_deg", "state_label": "pitch attitude (cyclic, longitudinal axis)",
                     "state_unit": "deg", "ctrl_unit": "-1..1"},
}


@dataclass
class Result:
    name: str
    metrics: dict
    evidence: list[dict] = field(default_factory=list)
    limitations: str = ""

    def save(self, sortie: str, root: Path = CONTROLS_DIR) -> Path:
        out = root / sortie / f"{self.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        return out


def _smooth(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


def load_flight(data_csv: str | Path, rate_hz: float = RATE_HZ) -> dict:
    """Uniform-grid dict of controls + resulting states, for one sortie."""
    df = pd.read_csv(data_csv, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    t_all = pd.to_numeric(df["Recording Time"], errors="coerce") / 1000.0

    cols = [FM + "AltitudeMSL", FM + "Heading", FM + "TrueAirSpeed", FM + "Roll", FM + "Pitch",
            *[CHANNELS[c]["col"] for c in CHANNELS],
            *[CHANNELS[c]["force_col"] for c in CHANNELS]]
    hot = pd.to_numeric(df[HOT], errors="coerce").interpolate(limit_direction="both")

    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    sub["t"] = t_all
    sub = sub[sub["t"].notna()].sort_values("t")
    sub[cols] = sub[cols].interpolate(limit_direction="both")
    sub["agl"] = ((sub[FM + "AltitudeMSL"] - hot.reindex(sub.index)) * M2FT).interpolate(limit_direction="both")
    sub = sub.dropna().reset_index(drop=True)

    t = sub["t"].to_numpy()
    dt = 1.0 / rate_hz
    g = np.arange(t[0], t[-1], dt)
    R = lambda c: np.interp(g, t, sub[c])  # noqa: E731

    alt_ft = R(FM + "AltitudeMSL") * M2FT
    flight = {
        "t": g, "dt": dt,
        "alt_ft": alt_ft,
        "agl_ft": R("agl"),
        "vs_fpm": _smooth(np.gradient(alt_ft, dt), int(rate_hz)) * 60.0,
        "roll_deg": np.degrees(R(FM + "Roll")),
        "pitch_deg": np.degrees(R(FM + "Pitch")),
        "heading_deg": np.degrees(np.unwrap(np.radians(R(FM + "Heading")))) % 360.0,
        "tas_kt": R(FM + "TrueAirSpeed") * 1.94384,
    }
    for name, spec in CHANNELS.items():
        flight[name] = R(spec["col"])
        flight[f"{name}_rate"] = np.gradient(flight[name], dt)
        flight[f"{name}_force"] = R(spec["force_col"])
    return flight


def robust_threshold(x: np.ndarray, k: float = 3.0) -> float:
    """median + k*MAD -- this sortie's own noise floor, not a fixed cutoff."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-9
    return med + k * mad


def activity_segments(rate: np.ndarray, t: np.ndarray, thresh: float, gap_s: float = 0.5,
                      min_len_s: float = 0.3):
    """Split into (kind, start_idx, end_idx) where kind is 'move' (|rate|
    above thresh, i.e. the pilot is actively working this control) or
    'hold' (below it, i.e. holding it steady)."""
    active = np.abs(rate) > thresh
    segs = []
    i = 0
    n = len(active)
    while i < n:
        j = i
        cur = active[i]
        while j < n and active[j] == cur:
            j += 1
        # merge tiny gaps of the opposite kind
        segs.append([cur, i, j])
        i = j
    # merge short interruptions into neighbours
    merged = []
    for seg in segs:
        if merged and seg[2] - seg[1] < gap_s / (t[1] - t[0]) and merged[-1][0] == seg[0]:
            merged[-1][2] = seg[2]
        else:
            merged.append(seg)
    out = []
    min_len = max(1, int(min_len_s / (t[1] - t[0])))
    for cur, i, j in merged:
        if j - i >= min_len:
            out.append(("move" if cur else "hold", i, j))
    return out


def run_cli(module_name: str, compute_fn):
    import argparse
    ap = argparse.ArgumentParser(description=f"Controls biomarker: {module_name}")
    ap.add_argument("data_csv")
    ap.add_argument("--sortie", default=None, help="output folder name (default: csv filename stem)")
    args = ap.parse_args()

    flight = load_flight(args.data_csv)
    sortie = args.sortie or Path(args.data_csv).stem
    result = compute_fn(flight)
    out = result.save(sortie)
    print(f"{module_name}: {json.dumps(result.metrics, indent=2, ensure_ascii=False)}")
    print(f"Wrote {out}")
