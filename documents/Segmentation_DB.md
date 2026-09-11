# `sortie1_segmentation.sqlite`

A standalone SQLite database that segments raw telemetry CSVs into drill
attempts and per-maneuver segments, and picks which segment(s) of each
attempt are the "best" (the ones scoring/comparison should use).

It lives at the repo root, committed as a binary (47 MB). It is **not**
the same database as `code/flight_school.db` (`imports` /
`telemetry_samples`) — see [Relationship to the main DB](#relationship-to-the-main-db)
below.

## Provenance

No script that produces this file is checked into the repo. It was
generated externally and dropped in; the `runs.run_at` timestamps
(`2026-09-09T10:12:xx`) are the only record of when. Treat the database
as a snapshot, not a reproducible build artifact, until the generating
code is found or rewritten.

## Schema

### `runs` — one row per source CSV processed

| column | meaning |
|---|---|
| `run_id` | primary key |
| `source_file` | path to the CSV that was segmented, relative to the data root |
| `sortie` | **always `1`** for every row — a placeholder, not the real sortie number. Use `source_file` to distinguish sorties. |
| `n_samples` | telemetry rows in the source CSV |
| `n_attempts` | drill attempts detected |
| `n_selected` | attempts that ended up with a selected segment |
| `run_at` | ISO timestamp of the run |

16 rows: the 8 folders under `CSV_3.6.26_check1/001`–`008`, plus 8 more
under a second, earlier collection (`data/csv_data/2_25/[14.12.25 גיחה 1
zip]/.../001`–`008`).

### `segments` — one row per detected segment within an attempt

| column | meaning |
|---|---|
| `run_id`, `source_file`, `sortie` | links back to `runs` |
| `drill` | `DoubleEight`, `DoubleOptimalTurn`, `DoublePattern`, or `Final` |
| `attempt` | 1-based attempt number within the drill |
| `segment` | 1-based segment number within the attempt |
| `is_selected` | 1 if this segment is the one scoring/comparison should use |
| `selection_rank` | currently always `NULL` — not populated |
| `reason` | free-text explanation of the segmentation/selection decision (see below) |
| `start_index`, `end_index` | row range into the source CSV |
| `start_time_s`, `end_time_s` | time range, seconds |
| `n_samples` | `end_index - start_index` |

576 rows total, 422 with `is_selected = 1`.

`reason` encodes the actual selection heuristic per drill, e.g.:

- **DoublePattern**: `"heading reversal 158.1° (|180-x|=21.9°), rank 1"` —
  candidate 180°-turn points are scored by closeness to a true reversal
  and ranked; low-rank ones are selected.
- **DoubleEight**: `"pattern of 3 circles (selected: last 2 of >2
  patterns, per sortie-1 rule)"` — when more circle patterns are found
  than expected, a sortie-1-specific rule picks the last two.
- **Final**: `"selected: first detected final approach (JS always keeps
  centerIndices[0])"` — ported directly from the front-end's JS
  selection logic (see `documents/ICD_CSV_Export.md`), first candidate
  wins.
- Rejected candidates are kept as rows too, e.g. `"extra circle pair
  found, not part of the selected double-eight"`, `"extra
  final-approach-shaped pass, not selected"` — so the table is an audit
  trail, not just the winners.

### `tagged_samples` — every telemetry sample, tagged

| column | meaning |
|---|---|
| `run_id`, `source_file`, `row_index`, `time` | identifies the sample |
| `lat`, `lon`, `height`, `IAS`, `heading` | telemetry at that sample |
| `drill`, `attempt`, `segment` | which segment this sample falls in |
| `is_selected` | whether that segment was the selected one |

215,379 rows — a per-sample copy of a few telemetry fields plus the
segment tag, rather than a foreign key into a telemetry table.

## Coverage

| drill | sorties with it |
|---|---|
| `DoublePattern` | all 16 runs |
| `DoubleEight` | 11 of 16 (only the `1_26` collection plus one `2_25` run) |
| `DoubleOptimalTurn` | 7 of 16 |
| `Final` | 5 of 16 (`1_26` collection only) |

## Known issues

- **`sortie` is a constant `1`.** Every run row and every segment row
  carries `sortie = 1`; the real distinguishing key is `source_file`.
  Don't join or group on `sortie` expecting per-sortie semantics.
- **No cadet identity.** Nothing here ties a run to a cadet — that
  mapping would have to come from the filename/folder convention used
  elsewhere in the project (see `code/import_flight_data.py`).
- **`selection_rank` is unused** (always `NULL`) despite `reason`
  strings that clearly carry a rank (`rank 1`, `rank 2`, …). It should
  probably be populated by parsing/re-deriving from `reason`, or the
  generator should be fixed to write it directly.
- **Two unrelated data collections are mixed** in one `runs` table
  (`1_26`/`CSV_3.6.26_check1` and `2_25`/`14.12.25`) with no explicit
  "collection" column — only distinguishable via the `source_file`
  path.

## Relationship to the main DB

`code/flight_school.db` (built by `code/import_flight_data.py`) has only
`imports` and `telemetry_samples`, with a nullable `segment_number`
column carried straight from the CSV — there is no segments table, no
drill/attempt entity, and no selection logic (see the PRD's `Cadet →
Sortie → Drill attempt → Segment` model in
`documents/Flight_Simulator_Data_Platform_PRD.md`, which this database
partially fulfills but doesn't integrate with).

`sortie1_segmentation.sqlite` is the more developed of the two on
segmentation specifically, but it is a separate, disconnected database:
different file, different schema, no shared IDs, no cadet identity, and
no code in this repo that reads or writes it. Before building on it,
decide whether to fold `segments`/`tagged_samples` into
`flight_school.db` (adding cadet/sortie identity and an `import_id`
foreign key) or keep it standalone and have downstream tools
(`segment_comparison.py`, `segment_timeline.py`, the controls-biomarkers
pipeline) read both.
