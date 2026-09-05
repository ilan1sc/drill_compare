# ICD — Source simulator CSV (upload input)

Interface Control Document for the **raw flight-log CSV the operator uploads** on the open
page (`.csv`). This is the *input* side; for the file the dashboard *exports* see
[`ICD_CSV_Export.md`](ICD_CSV_Export.md).

**Version 1.2 — 4.9.26**

| Ver | Date | Changes |
|---|---|---|
| 1.0 | 4.9.26 | Initial issue (from code only). |
| 1.1 | 4.9.26 | Reconciled with a real header row: full 16-column list; schema confirmed to match `dataParser.js`. |
| 1.2 | 4.9.26 | Added §4 observed value ranges from the `CSV_3.6.26_check1` recordings. |

- **File name / delimiter:** the recordings seen are named `Elements_0.csv` (or `Elements.csv`)
  and are **TAB-delimited**. `Entities.csv` / `EntitiesEvents*.csv` in the same folder are
  separate feeds, not covered here.

- **Consumer:** `parseFile()` in `src/utils/dataParser.js` (Papaparse, `header:true`,
  `skipEmptyLines:false`).
- **Format:** tab/comma-delimited CSV, first row = header, one row per simulator sample.
  `.xlsx` / `.xls` are rejected by the parser.
- **Log rate** ≈ 20 Hz (~0.05 s/row); not resampled by the app.
- **Every header string except `Recording Time` begins with a leading space**
  (` FFS1_Ownship.…`). The parser keys on the exact space-prefixed names.
- Some channels are **sparsely populated** (a value every few rows, blanks between). The
  parser forward-fills `CollectivePosition`, `LateralStickPosition`,
  `LongitudinalStickPosition` and `Heading` by one sample.

---

## 1. Column list (16 columns, in file order)

`Cons.` = consumed by the app.

| # | Column header (leading space not shown) | Cons. | Quantity | Unit in the file |
|---|---|:--:|---|---|
| 1 | `Recording Time` | ✔ | time since start of recording | **milliseconds** |
| 2 | `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.CollectiveForce` | — | force on the collective lever | force units per sim ICD (not used in-app) |
| 3 | `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.CollectivePosition` | ✔ | collective lever position | **normalised**, ≈ 0…1 |
| 4 | `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.LateralStickForce` | — | force on the cyclic, lateral axis | force units per sim ICD (not used in-app) |
| 5 | `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.LateralStickPosition` | ✔ | lateral (roll) cyclic position | **normalised**, ≈ −1…+1 |
| 6 | `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.LongitudinalStickForce` | — | force on the cyclic, longitudinal axis | force units per sim ICD (not used in-app) |
| 7 | `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.LongitudinalStickPosition` | ✔ | longitudinal (pitch) cyclic position | **normalised**, ≈ −1…+1 |
| 8 | `FFS1_Ownship.ESL_LHD_Landing_detection.ESL_LHD_Landing_detection.Point_1_HOT` | ✔ | terrain elevation under landing-detection point 1 (ground height); internal field `BHAE` | **metres** |
| 9 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.AltitudeAGL` | — | height above ground from the flight model | **metres** (not used; app derives its own `height` from `AltitudeMSL − Point_1_HOT`) |
| 10 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.AltitudeMSL` | ✔ | altitude above mean sea level; internal field `ELHE` | **metres** |
| 11 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Heading` | ✔ | aircraft heading | **radians** |
| 12 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Latitude` | ✔ | latitude (WGS84 geodetic) | **radians** |
| 13 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Longitude` | ✔ | longitude (WGS84 geodetic) | **radians** |
| 14 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Pitch` | ✔ | pitch attitude; internal field `pitchModel` | **radians** |
| 15 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Roll` | ✔ | roll attitude; internal field `rollModel` | **radians** |
| 16 | `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.TrueAirSpeed` | ✔ | true airspeed | **metres / second** |

The header supplied matches what `dataParser.js` expects — **no schema mismatch**. The four
un-consumed columns (`*Force` ×3, `AltitudeAGL`) are read past and ignored.

---

## 2. Transforms the parser applies on read

| Output field | From | Formula | Result unit |
|---|---|---|---|
| `time` | `Recording Time` | `÷ 1000` | seconds |
| `IAS` | `TrueAirSpeed` | `× 1.94384` | knots (labelled "IAS" although it is TAS) |
| `height` | `AltitudeMSL`, `Point_1_HOT` | `(ELHE − BHAE) × 3.28084`; if `BHAE` is `NaN`, `ELHE × 3.28084` | feet AGL |
| `lat`, `lon` | `Latitude`, `Longitude` | `parseFloat` only | radians (unchanged) |
| `heading` | `Heading` | `parseFloat` only, forward-filled 1 sample | radians (unchanged) |
| `rollModel`, `pitchModel` | `Roll`, `Pitch` | `parseFloat` only | radians (unchanged) |
| `RollStick`, `PitchStick`, `COLL` | positions #5, #7, #3 | `parseFloat`, forward-filled 1 sample | normalised (unchanged) |

Post-parse rules:
- Blank / non-numeric cells → `NaN`.
- Any row whose `IAS` is `NaN` after parsing is **dropped**.
- `Recording Time` is expected in ms; a row with a blank time gives `time = NaN`.

---

## 3. Units summary

| Quantity | Column(s) | File unit | Convert to… |
|---|---|---|---|
| Time | `Recording Time` | milliseconds | seconds: `÷ 1000` |
| Latitude / Longitude | `Latitude`, `Longitude` | radians | degrees: `× 180/π` |
| Heading, Pitch, Roll | `Heading`, `Pitch`, `Roll` | radians | degrees: `× 180/π` |
| Altitude MSL / AGL, ground height | `AltitudeMSL`, `AltitudeAGL`, `Point_1_HOT` | metres | feet: `× 3.28084` |
| Airspeed | `TrueAirSpeed` | metres / second | knots: `× 1.94384` |
| Stick / collective positions | `*Position` | dimensionless, normalised (cyclic ≈ −1…+1, collective ≈ 0…1) | — |
| Stick / collective forces | `*Force` | per simulator ICD (not documented in this app) | — |

---

## 4. Observed value ranges

Aggregated min/max over the 8 recordings in
`~/Documents/flight school/data/csv_data/1_26/CSV_3.6.26_check1/{001..008}/Elements*.csv`
(≈ 2.30 M data rows total). These are empirical bounds from this data set, **not** protocol
limits.

| Column | Min | Max | Min–max in familiar units |
|---|---|---|---|
| `Recording Time` | 0 | 3 780 220 ms | 0 – ~63 min per recording |
| `…CollectiveForce` | −117.69 | 85.27 | (force units) |
| `…CollectivePosition` | −0.093 | 0.848 | ≈ 0…1, slightly negative at the stop |
| `…LateralStickForce` | −21.50 | 22.83 | (force units) |
| `…LateralStickPosition` | −0.914 | 0.233 | within ±1 |
| `…LongitudinalStickForce` | −26.27 | 24.37 | (force units) |
| `…LongitudinalStickPosition` | −0.497 | 0.583 | within ±1 |
| `…Point_1_HOT` (ground height) | 53.999 | 196.79 m | 177 – 646 ft |
| `…AltitudeAGL` | 0 | 0 | **dead channel — always 0 in this data** |
| `…AltitudeMSL` | 84.40 | 651.76 m | 277 – 2139 ft MSL |
| `…Heading` | −3.14158 | 3.14154 rad | −180° … +180° (full range) |
| `…Latitude` | 0.545723 | 0.548228 rad | 31.267° – 31.410° N |
| `…Longitude` | 0.602248 | 0.605306 rad | 34.506° – 34.681° E |
| `…Pitch` | −0.3498 | 0.4812 rad | −20.0° … +27.6° |
| `…Roll` | −1.1891 | 1.1101 rad | −68.1° … +63.6° |
| `…TrueAirSpeed` | 0 | 64.113 m/s | 0 – 124.6 kt |

Notes:
- `AltitudeAGL` is 0 for every row — confirms the app is right to derive its own AGL from
  `AltitudeMSL − Point_1_HOT`.
- `Heading` spans the full −π…π; it wraps, so treat it as circular.
- Position channels stay inside ±1 as expected; the small negative `CollectivePosition`
  (−0.09) is just travel past the calibrated zero.
- Per-recording row counts: 001 = 377 591, 002 = 272 327, 003 = 271 671, 004 = 289 197,
  005 = 233 011, 006 = 267 935, 007 = 322 950, 008 = 263 795.
