# ICD — "צ'ק עופר" Exercise CSV / XLSX Export

Interface Control Document for the file produced by the dashboard **"ייצא נתונים"** button.

- **Producer:** `exportExerciseDataToExcel()` in `src/utils/exportToExcel.js`, called from
  `handleExportData()` in `src/hooks/useDashboardData.js`.
- **Input to the exporter:** `currentData` — the array of per-sample point objects built by
  `src/hooks/useCurrentData.js` for the currently selected drill tab (`activeButton`).
- **Format:** `.xlsx`, single sheet `נתוני תרגיל`, one row per flight-log sample.
- **File name:** `<trainee>_<flightNumber>_<activeButton>.xlsx` (entered in two prompts).
- **Sample rate:** the simulator log rate, ~20 Hz (≈0.05 s between rows). Not resampled.

---

## 1. How columns are decided

The exporter has **no fixed column list**. For every point object it iterates
`Object.keys(point)` and emits one column per property found
(`src/utils/exportToExcel.js:19-33`). Therefore the column set =

```
base fields (from the raw-log parser)
  + fields added by the per-drill hook
  + fields added by useCurrentData (deltas, smoothing, drill, segment)
```

Seven keys are renamed to Hebrew headers via `COLUMN_LABELS`
(`src/utils/exportToExcel.js:3-11`); every other key keeps its raw JavaScript name.

| JS key | Header written |
|---|---|
| `time` | `זמן` |
| `altitude` | `גובה` |
| `IAS` | `מהירות (IAS)` |
| `heading` | `כיוון אף` |
| `RollStick_DELTA_SMOOTHED` | `גלגול` |
| `PitchStick_DELTA_SMOOTHED` | `עלרוד` |
| `COLL_DELTA_SMOOTHED` | `כוח` |

`NaN` / `null` / `undefined` values are written as an empty cell.

---

## 2. Source-log columns consumed

The raw simulator CSV is parsed by `parseFile()` in `src/utils/dataParser.js`
(Papaparse, `header:true`). Only the columns below are read; **every source key has a
leading space** in the header. XLSX/XLS input is rejected.

| Source column (leading space omitted here) | Used for |
|---|---|
| `Recording Time` | `time` |
| `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Latitude` | `lat` |
| `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Longitude` | `lon` |
| `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.AltitudeMSL` | `ELHE` |
| `FFS1_Ownship.ESL_LHD_Landing_detection.ESL_LHD_Landing_detection.Point_1_HOT` | `BHAE` |
| `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Heading` | `heading` |
| `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.TrueAirSpeed` | `IAS` |
| `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Roll` | `rollModel` |
| `FFS1_Ownship.LHD_ESL_Flight_Model_Data.LHD_ESL_Flight_Model_Data.Pitch` | `pitchModel` |
| `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.LateralStickPosition` | `RollStick` |
| `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.LongitudinalStickPosition` | `PitchStick` |
| `FFS1_Ownship.ESL_LHD_HOCAS_Message.ESL_LHD_HOCAS_Message.CollectivePosition` | `COLL` |

Post-parse rules:
- `RollStick`, `PitchStick`, `COLL`, `heading` are **forward-filled by one sample** when the
  next sample is blank.
- Any row whose `IAS` is `NaN` after conversion is **dropped**.

---

## 3. Common columns (present for every drill)

### 3.1 From the parser (`dataParser.js:95-114`)

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `__rawRow` | The entire original CSV row object, kept for debugging. | — | raw row. **Serializes as `[object Object]` — exclude from analysis.** |
| `index` | Row ordinal in the parsed file (before drop/segmentation). | integer | array position |
| `זמן` (`time`) | Time from start of recording. | seconds | `Recording Time` (ms) ÷ 1000 |
| `lat` | Ownship latitude. | **radians** | `…Latitude` verbatim |
| `lon` | Ownship longitude. | **radians** | `…Longitude` verbatim |
| `ELHE` | Altitude above mean sea level. | metres | `…AltitudeMSL` verbatim |
| `BHAE` | Terrain elevation under landing-detection point 1 (ground height). | metres | `…Point_1_HOT` verbatim |
| `height` | Height above ground (AGL). | **feet** | `(ELHE − BHAE) × 3.28084`; if `BHAE` is `NaN`, `ELHE × 3.28084` |
| `כיוון אף` (`heading`) | Aircraft heading. | **radians in the raw parser** — converted to **degrees** by the Eight / Pattern family only (see §5). | `…Heading` verbatim, forward-filled |
| `מהירות (IAS)` (`IAS`) | Airspeed. Labelled IAS but derived from **true** airspeed. | **knots** | `…TrueAirSpeed` (m/s) × 1.94384 |
| `RollStick` | Lateral cyclic stick position. | normalised, ≈ −1…+1 | `…LateralStickPosition`, forward-filled |
| `PitchStick` | Longitudinal cyclic stick position. | normalised, ≈ −1…+1 | `…LongitudinalStickPosition`, forward-filled |
| `COLL` | Collective lever position. | normalised, ≈ 0…1 | `…CollectivePosition`, forward-filled |
| `rollModel` | Actual roll attitude of the aircraft. | **radians** | `…Roll` verbatim |
| `pitchModel` | Actual pitch attitude of the aircraft. | **radians** | `…Pitch` verbatim |

### 3.2 From `useCurrentData.js`

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `RollStick_DELTA` | Per-sample **prediction error** of the lateral stick — how far the stick departed from where its recent trajectory predicted it would be. A roughness / over-controlling measure. | arbitrary (stick units × 57.2958); always ≥ 0 | `abs(pred − actual) × 180/π`, where `pred = 3·vₙ₋₁ − 2.5·vₙ₋₂ + 0.5·vₙ₋₃` (`calculatePredictionDelta`). Needs ≥ 4 samples; else `NaN`. |
| `PitchStick_DELTA` | Same, longitudinal stick. | arbitrary | as above on `PitchStick` |
| `COLL_DELTA` | Same, collective. | arbitrary | as above on `COLL` |
| `גלגול` (`RollStick_DELTA_SMOOTHED`) | Smoothed lateral-stick roughness — the trace the dashboard plots as *גלגול*. | arbitrary | 50-sample **centred moving average** of `RollStick_DELTA` (`applyMovingAverage`, window 50) |
| `עלרוד` (`PitchStick_DELTA_SMOOTHED`) | Smoothed longitudinal-stick roughness (*עלרוד*). | arbitrary | 50-sample MA of `PitchStick_DELTA` |
| `כוח` (`COLL_DELTA_SMOOTHED`) | Smoothed collective roughness (*כוח*). | arbitrary | 50-sample MA of `COLL_DELTA` |
| `drill` | Drill identifier in English. | text | map from `activeButton` (`DRILL_NAME_EN` in `useCurrentData.js`) |
| `segment` | 1-based index of the drill phase this sample belongs to (matches the *סגמנט N* toggles in the SwitchPanel). Empty when the drill has no per-sample segmentation or for gap points. | integer or empty | `(point.clusterIndex ?? point.segmentIndex) + 1` |

> **`heading` unit warning.** The raw parser leaves `heading` in radians. `useEightData`,
> `useDoubleEightData` and `useTavnitData` overwrite it with **degrees**; `useTrack4Data`,
> `useMission5Data`, `useMission6Data`, `useFinalData`, `useDoubleYnpData` and
> `useOptimalTurnData` leave it in **radians**. See the per-drill sections.

---

## 4. `drill` values

| `activeButton` | `drill` |
|---|---|
| שמינייה | `Eight` |
| 2 X שמינייה | `DoubleEight` |
| ציר נלמד | `LearnedTrack` |
| ציר משימה | `MissionTrack` |
| תבנית | `Pattern` |
| 2 X תבנית | `DoublePattern` |
| פיינל | `Final` |
| ינ"פ | `YNP` |
| 2 X ינ"פ | `DoubleYNP` |
| ינ"פ 2 נקודות | `YNP2Points` |
| 2 X ינ"פ 2 נקודות | `DoubleYNP2Points` |
| פנייה מיטבית | `OptimalTurn` |
| 2 X פנייה מיטבית | `DoubleOptimalTurn` |

---

## 5. Per-drill columns

Each drill's hook adds fields to the point object on top of §3.1, then `useCurrentData`
adds §3.2. The tables below list the **drill-specific extra columns** and note where the
common columns change meaning.

### 5.1 שמינייה — `Eight` (`useEightData.js`)

Segmentation: each circle of the eight is split in half → `clusterIndex` 0…3 → `segment` 1…4.

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `x` | Longitude, for the location plot. | **radians** | set to `origX` (`= lon`) |
| `y` | Latitude, for the location plot. | **radians** | set to `origY` (`= lat`) |
| `origX` | Original longitude. | radians | `= lon` |
| `origY` | Original latitude. | radians | `= lat` |
| `גובה` (`altitude`) | Height above ground. | feet | `= height` |
| `heading` | **Overwritten to degrees.** | degrees (0…360 style) | `heading_rad × 180/π` |
| `groundSlope` | Terrain slope beneath the track over a 20-sample window. | degrees | `atan(|ΔBHAE| / horizontalΔ) × 180/π`, `horizontalΔ` in the local metre frame |
| `clusterIndex` | Half-circle index. | integer 0…3 | assigned after circle pairing/trimming |

### 5.2 2 X שמינייה — `DoubleEight` (`useDoubleEightData.js`)

Segmentation: `clusterIndex` 0…7 (two eights × four halves) → `segment` 1…8.

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `x` | Longitude, for the plot. | **radians** | `= lon` |
| `y` | Latitude, for the plot. | **radians** | `= lat` |
| `heading` | **Overwritten to degrees.** | degrees | `heading_rad × 180/π` |
| `groundSlope` | Terrain slope over a 20-sample window (0 until enough history). | degrees | as in §5.1 |
| `clusterIndex` | Half-circle index across both eights. | integer 0…7 | assigned after pairing |

### 5.3 תבנית / 2 X תבנית — `Pattern` / `DoublePattern` (`useTavnitData.js` + `utils/pattarnLogic.js`)

Segmentation: every half-circle of every pattern circle becomes a segment;
`segmentIndex = patternIdx × 7 + halfIdx` → `segment = segmentIndex + 1`.
A break point `{x:NaN, y:NaN, segmentIndex:null}` is inserted between patterns (row is
mostly empty).

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `latDeg` | Latitude in degrees. | degrees | `lat_rad × 180/π` |
| `lonDeg` | Longitude in degrees. | degrees | `lon_rad × 180/π` |
| `x` | Longitude, for the plot. | **degrees** | `= lonDeg` |
| `y` | Latitude, for the plot. | **degrees** | `= latDeg` |
| `גובה` (`altitude`) | Height above ground. | feet | `= height` (`|| 0`) |
| `heading` | **Overwritten to degrees.** | degrees | `heading_rad × 180/π` |
| `segmentIndex` | Global half-circle index. | integer | `patternIdx × 7 + halfIdx` |

### 5.4 ציר נלמד — `LearnedTrack` / track 4 (`useTrack4Data.js`)

The nearest of the `TRACK_4_OPTIONS` axes is chosen; the track is sliced from the first to
the last waypoint. Segmentation: chronological leg between consecutive waypoints →
`segmentIndex` 0…3 → `segment` 1…4.

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `x` | Longitude offset from waypoint 0, **scaled for display** (not metres). | scaled units | `(lon − ref.lon) × 1e6 × 10` |
| `y` | Latitude offset from waypoint 0, scaled for display. | scaled units | `(lat − ref.lat) × 1e6 × 10` |
| `lon_rad` | Original longitude. | radians | preserved by `convertToRelativeRadians` |
| `lat_rad` | Original latitude. | radians | preserved |
| `distanceToAxis` | Perpendicular distance from this sample to the planned axis. | **metres** | `calculateDistanceToAxisPerPoint` (flat-earth projection) |
| `segmentIndex` | Leg index (waypoint *i* → *i*+1). | integer 0…3 | `calculateSequentialSegments` |
| `heading` | Not converted. | **radians** | `= heading` from parser |

### 5.5 ציר משימה — `MissionTrack`

Dispatch by flight: flight 6 → `useMission6Data`, otherwise → `useMission5Data`.

#### 5.5a Flight 5 (`useMission5Data.js`)

Track sliced from `MISSION_5_PLAN[0]` to the nearest `MISSION_5_LANDING_SPOTS` spot.

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `x` | East offset from plan origin. | **metres** | `(lon − origin.lon) × R·cos(lat) …` (`convertToMeters`) |
| `y` | North offset from plan origin. | **metres** | `(lat − origin.lat) × R …` |
| `distanceToAxis` | Perpendicular distance to the planned route. | metres | `calculateDistanceToAxisPerPoint` |
| `segmentIndex` | Index of the nearest planned leg. | integer 0…(waypoints−2) | `calculateClosestAxisIndex` |
| `groundSlope` | Terrain slope over a 20-sample window. | degrees | `atan(|ΔBHAE| / horizontalΔ) × 180/π` |
| `isPathPoint` | Sample lies on the planned path portion (before the end point). | boolean | `index ≤ endPointIndex` |
| `isEndPoint` | Sample is the detected end point. | boolean | `index === endPointIndex` |
| `heading` | Not converted. | **radians** | from parser |

#### 5.5b Flight 6 (`useMission6Data.js`)

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `x` | East offset from `MISSION_6_PLAN[0]`. | **metres** | `(lonRad − refLonRad) × cos(refLat) × R` |
| `y` | North offset from `MISSION_6_PLAN[0]`. | **metres** | `(latRad − refLatRad) × R` |
| `distanceToAxis` | Perpendicular distance to the planned route. | metres | `calculateDistanceToAxisPerPoint` |
| `segmentIndex` | `0` before the mid-waypoint, `1` after. | integer 0/1 | `i < midIndex ? 0 : 1` |
| `lat_rad`, `lon_rad` | Original lat/lon if present upstream. | radians | preserved |
| `heading` | Not converted. | **radians** | from parser |

### 5.6 פיינל — `Final` (`useFinalData.js`)

One final-approach segment is detected and clipped to roughly −8000 ft … +1640 ft along
the approach centreline. **No `segmentIndex` / `clusterIndex`** → `segment` is always empty.
`altitude` (`גובה`) is not set; use `y` / `groundHeight` instead.

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `idx` | Index within the lat/lon-valid subset of the log. | integer | running counter |
| `utmX` | East offset from the flight's target point. | **feet** | `(lon − tgt.lon) × cos(tgt.lat) × R × 3.28084` |
| `utmY` | North offset from the target point. | **feet** | `(lat − tgt.lat) × R × 3.28084` |
| `dist` | Straight-line distance to the target point. | feet | `√(utmX² + utmY²)` |
| `finalPointIndex` | 1-based sample index within the final segment. | integer | re-numbered after clipping |
| `relD_meters` | Signed distance **along the approach centreline** (negative = before the target). Name says metres; **value is feet** (built from `utmX/utmY`). | feet | projection of `(utmX,utmY)` onto the start→target unit vector |
| `x` | Same as `relD_meters` — X axis of the final-profile plot. | feet | `= relD_meters` |
| `relativeDist` | Duplicate of `relD_meters`. | feet | `= relD_meters` |
| `y` | Altitude MSL — Y axis of the final-profile plot. | **feet MSL** | `(AltitudeMSL or ELHE) × 3.28084` |
| `groundHeight` | Terrain height at this sample. | feet | `BHAE × 3.28084` |
| `angle` | Momentary flight-path angle over a ~1 s (20-sample) look-ahead. | degrees (string, 1 dp) | `atan2(Δy, |Δx|) × 180/π` |
| `optimalAngle` | Angle from this sample straight to the target ground altitude. | degrees (string, 1 dp) | `atan2(y − targetGroundAlt, |x|) × 180/π` |
| `heading` | Not converted. | **radians** | from parser |

### 5.7 ינ"פ family — `YNP` / `DoubleYNP` / `YNP2Points` / `DoubleYNP2Points` (`useDoubleYnpData.js`)

Judgement-point maneuvers are detected; the log is clipped to a window around the first and
last maneuver. **Per-sample points carry no `segmentIndex` / `clusterIndex`** → `segment`
is empty. Maneuver-level results (score, hit height/speed, distance) live in
`maneuverDetails`, which is **not** part of this export.

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `x` | Longitude, for the plot. | **radians** | `= lon` |
| `y` | Latitude, for the plot. | **radians** | `= lat` |
| `origX` | Original longitude. | radians | `= lon` |
| `origY` | Original latitude. | radians | `= lat` |
| `altitude` (`גובה`) | Height above ground. | feet | `= height` |
| `heading` | Not converted. | **radians** | from parser |

> **Limitation.** `useCurrentData` only builds `currentData` for `activeButton === '2 X ינ"פ'`.
> For the `ינ"פ`, `ינ"פ 2 נקודות` and `2 X ינ"פ 2 נקודות` tabs `currentData` is empty and the
> export button reports "no data" (`exportToExcel.js:14-17`).

### 5.8 פנייה מיטבית / 2 X פנייה מיטבית — `OptimalTurn` / `DoubleOptimalTurn` (`useOptimalTurnData.js`)

Up to two 180° turns near `OPTIMAL_TURN_POINTS` are detected. Segmentation: one segment per
turn → `clusterIndex` 0…1 → `segment` 1…2.

| Column | Meaning | Unit | Source / formula |
|---|---|---|---|
| `x` | Longitude, for the plot. | **radians** | `p.lon || p.lonDeg || 0` |
| `y` | Latitude, for the plot. | **radians** | `p.lat || p.latDeg || 0` |
| `clusterIndex` | Turn index. | integer 0/1 | assigned per detected turn |
| `heading` | Not converted. | **radians** | from parser |

---

## 6. Quick reference — column meaning by drill

Legend: **rad** = radians, **deg** = degrees, **m** = metres, **ft** = feet, `—` = column
absent / empty.

| Column | Eight / DoubleEight | Pattern / DoublePattern | LearnedTrack | MissionTrack f5 | MissionTrack f6 | Final | YNP family | OptimalTurn |
|---|---|---|---|---|---|---|---|---|
| `time` | s | s | s | s | s | s | s | s |
| `lat` / `lon` | rad | rad | rad | rad | rad | rad | rad | rad |
| `ELHE` / `BHAE` | m | m | m | m | m | m | m | m |
| `height` | ft AGL | ft AGL | ft AGL | ft AGL | ft AGL | ft AGL | ft AGL | ft AGL |
| `heading` (`כיוון אף`) | **deg** | **deg** | rad | rad | rad | rad | rad | rad |
| `IAS` (`מהירות`) | kt | kt | kt | kt | kt | kt | kt | kt |
| `RollStick`/`PitchStick`/`COLL` | norm | norm | norm | norm | norm | norm | norm | norm |
| `rollModel` / `pitchModel` | rad | rad | rad | rad | rad | rad | rad | rad |
| `*_DELTA`, `גלגול`/`עלרוד`/`כוח` | arb. roughness | arb. | arb. | arb. | arb. | arb. | arb. | arb. |
| `altitude` (`גובה`) | ft | ft | — | — | — | — | ft | — |
| `x` / `y` | rad | deg | scaled (×1e7) | m | m | ft / ft-MSL | rad | rad |
| `distanceToAxis` | — | — | m | m | m | — | — | — |
| `segmentIndex` / `clusterIndex` | clusterIndex 0–3 / 0–7 | segmentIndex | segmentIndex 0–3 | segmentIndex | 0/1 | — | — | clusterIndex 0/1 |
| `segment` | 1–4 / 1–8 | 1…N | 1–4 | 1…N | 1–2 | — | — | 1–2 |
| `groundSlope` | deg | — | — | deg | — | — | — | — |
| drill-specific | `origX/origY` | `latDeg/lonDeg` | `lon_rad/lat_rad` | `isPathPoint/isEndPoint` | `lat_rad/lon_rad` | `idx,utmX,utmY,dist,finalPointIndex,relD_meters,relativeDist,groundHeight,angle,optimalAngle` | `origX/origY` | — |

---

## 7. Notes & caveats for consumers

1. **`__rawRow`** is a nested object; in the sheet it renders as `[object Object]`. Ignore it.
2. **`heading` units are not uniform** — degrees for Eight/Pattern, radians for everything
   else. Convert on read using the `drill` column.
3. **`x` / `y` have a different meaning and unit per drill** (see §5 / §6). They are plot
   coordinates, not a common projected frame.
4. **`relD_meters` in Final is in feet**, despite the name.
5. **`*_DELTA*` columns are not physical units** — they are `|prediction − actual|` of the
   stick position × 57.2958, i.e. a relative smoothness index. Higher = rougher input.
   `NaN` for the first 3 samples of each drill and wherever the source stick channel is
   blank.
6. **`segment` is blank** for Final and the YNP family (no per-sample phase), and for the
   inserted gap rows in Pattern.
7. **Row filtering:** samples with `NaN` IAS are removed by the parser; each drill hook
   further clips the log to the relevant maneuver window, so the exported row count is far
   smaller than the source log and `index` is not contiguous.
8. **Column order** in the sheet follows first-seen key order: parser fields, then
   hook fields, then `*_DELTA`, `*_DELTA_SMOOTHED`, then `drill`, `segment`.
