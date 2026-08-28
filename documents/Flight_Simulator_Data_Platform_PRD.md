# Flight Simulator Drill Data and Scoring Platform — V1

## 1. Purpose

Create a local system that imports simulator drill data as CSV, enriches every telemetry sample with cadet, sortie, drill, and segment identity, stores it in a local database, calculates a strict and explainable score for each drill attempt, and compares each cadet with the relevant drill cohort.

V1 is designed for one computer and one analyst. It uses a local SQLite database and a Python bulk-import program.

## 2. Goals

1. Ingest CSV drill exports and correctly build a local database by cadet, sortie, drill, attempt, and optional segment.
2. Provide a small Python program that imports multiple CSV files from a folder.
3. Calculate a strict score for every complete drill attempt.
4. Compare a cadet with other cadets who completed the same drill under the same scoring rules.
5. Show visuals explaining how far the cadet is from the cohort average, measured in standard deviations.
6. Produce an import-quality report for every uploaded file.

## 3. Scope

### In scope

- Local SQLite database.
- Bulk CSV import from a selected folder.
- Filename-based cadet, sortie, and drill identification.
- Telemetry validation and missing-value interpolation.
- Drill and segment storage.
- Strict per-sample point deductions.
- Drill-score status: scored or incomplete.
- Cadet-versus-cohort comparison for comparable complete attempts.
- Import and scoring audit records.

### Out of scope

- Multi-user access.
- Web dashboard.
- Instructor score overrides.
- Machine-learning scoring.
- V2 behavioral features as score inputs: time outside tolerance, time to corrective action, overshoot, symmetry, and control roughness.

## 4. CSV input contract

CSV is the canonical V1 input format. The existing XLSX export must either be changed to export CSV directly or be converted to CSV before import. The original source file is preserved for audit purposes.

Each row represents one simulator telemetry sample, normally at approximately 20 Hz.

Each imported row contains, or receives during import:

```text
cadet_number
sortie_number
drill_name
drill_code
segment_number
timestamp
```

Common telemetry fields include time, position, height above ground, airspeed, heading, controls, roll, pitch, and drill-specific fields such as distance from axis and ground slope.

The `__rawRow` export field is ignored.

### Filename convention

Every file must use this format:

```text
c<cadet_number>_<sortie_number>_<Hebrew drill name>.csv
```

Example:

```text
c1_1_2 X שמינייה.csv
```

This identifies:

```text
cadet_number = 1
sortie_number = 1
drill_name = 2 X שמינייה
drill_code = DoubleEight
```

The importer maps approved Hebrew drill names to stable internal drill codes. Files that do not match this format are rejected and listed in the import-quality report. Legacy names, such as files using `c8seg`, require an explicit mapping rule before import.

## 5. Data model

V1 uses a simple structure centered on one enriched telemetry table.

### `imports`

One record per import operation.

- Import identifier
- Original filename
- File hash
- Imported time
- Import status
- Cadet number
- Sortie number
- Drill code
- Row count
- Replaced-attempt indicator
- Quality-report summary

### `telemetry_samples`

One record per simulator timestamp.

- Cadet number
- Sortie number
- Drill code
- Attempt identifier
- Segment number, nullable
- Source row index
- Timestamp
- Common normalized telemetry fields
- Drill-specific metric fields
- Interpolation flag by field
- Original imported-file reference

Every row is traceable as:

```text
Cadet → Sortie → Drill attempt → Segment, when applicable → Timestamped sample
```

### `scores`

One record per calculated drill attempt.

- Attempt identifier
- Scoring-rules version
- Status: `scored` or `incomplete`
- Raw score
- Display score
- Total deductions
- Calculation time

### `score_deductions`

One record for every scored sample outside tolerance.

- Attempt identifier
- Timestamp
- Segment
- Scoring phase
- Metric
- Actual value
- Target value
- Tolerance
- Excess deviation
- Metric weight
- Importance weight
- Deducted points
- Scoring-rules version

### `import_quality_issues`

One record for each quality issue.

- Attempt identifier
- Field
- Start and end timestamp
- Missing-sample count
- Action taken: interpolated, incomplete, rejected, or warning
- Details

### `cohort_metric_statistics`

- Drill code
- Scoring-rules version
- Metric and phase
- Number of complete attempts
- Mean
- Standard deviation
- Calculation timestamp

### `cadet_comparisons`

- Cadet
- Sortie
- Drill attempt
- Metric and phase
- Cadet value
- Cohort mean
- Cohort standard deviation
- Standard-deviation distance
- Performance direction

## 6. Import workflow

1. The Python importer scans a chosen folder for CSV files.
2. It validates the filename and required columns.
3. It extracts cadet number, sortie number, and Hebrew drill name.
4. It maps the Hebrew drill name to an internal drill code.
5. It validates required telemetry fields and normalizes units.
6. It fills short missing-value runs where allowed.
7. It replaces any existing attempt with the same cadet, sortie, and drill identity.
8. It stores enriched telemetry rows and the import-quality report.
9. It calculates the drill score only if the imported attempt is complete.

The importer must be idempotent: importing the same file again must not create duplicate telemetry rows.

## 7. Missing-data policy

The normal data rate is approximately 20 Hz.

For a missing run of 1–10 samples:

- Linearly interpolate between the preceding valid value and the following valid value.
- Mark the interpolated samples and field.
- Record the run length and affected timestamps in the quality report.

A missing run longer than 10 samples, or a missing run at the beginning or end of a drill where interpolation cannot use valid values on both sides, makes the entire drill attempt `incomplete`.

An incomplete attempt remains stored with its telemetry and quality report, but receives no final score and is excluded from future rankings and cohort statistics.

## 8. Scoring model

### V1 principle

V1 uses strict rubric scoring, not peer comparison.

Every drill starts with **1,000 raw points**.

For every telemetry sample in a scored phase:

```text
excess deviation = max(0, absolute(actual value − target value) − tolerance)

deduction = excess deviation
          × metric weight
          × importance weight
```

No deduction is applied to values within tolerance.

```text
raw score = max(0, 1,000 − total deductions)
display score = raw score ÷ 10
```

The display score is 0–100.

### Weights

- Metric weights and importance weights are common to all cadets.
- Weights are stored in a versioned scoring-rules configuration.
- Weights will be calibrated against instructor-provided overall drill scores.
- Historical results retain the ruleset version used when they were calculated.
- No manual score override is supported in V1.

### Scoring phases

Each drill has defined phases such as entry, turn, judgment, crossing, and exit.

A phase may be detected without being scored. Segments help organize the maneuver but do not replace the finer scoring phases. The scoring engine uses drill geometry and configuration data to identify the applicable phase at each sample.

## 9. Initial exercise support

V1 begins with Eight and Double Eight drills.

The existing application configuration provides fixed Eight and Double Eight crossing points and automatic Double Eight segmentation into eight half-circle segments.

Initial scoring targets are based on the exercise specification:

- Axis heading: tolerance ±3 degrees.
- Height: target 60 feet AGL; tolerance ±5 feet.
- Airspeed: target 90 knots; tolerance ±3 knots.
- Left or right bank: target 30 degrees; tolerance ±5 degrees.
- Transition-point distance: target 0 metres; tolerance ±2 metres.

Heading and roll must be converted to degrees before scoring, since source units are not uniform across drills.

## 10. Cadet comparison

Comparisons are only made between attempts that have:

- The same drill code.
- The same scoring-rules version.
- Complete score status.
- The same relevant phase or segment when comparing a phase-specific metric.

The system calculates cohort statistics for the final drill score and each scored metric, including heading, height, airspeed, bank, and transition-point accuracy.

```text
standard-deviation distance = (cadet value − cohort mean) / cohort standard deviation
```

The system normalizes direction so that a positive result always means better performance and a negative result always means worse performance.

## 11. Comparison visuals

The cadet drill view must show:

- Final score compared with cohort average.
- Cohort average and standard deviation for each scored metric.
- Standard-deviation distance for the cadet on each metric.
- A performance profile across heading, height, speed, bank, and transition accuracy.
- Drill timeline showing deviations and deductions.
- Segment and phase breakdown showing where the cadet gained or lost points.

The first visual set includes:

- A score distribution with the cadet’s position and cohort average.
- A metric profile showing distance from cohort average in standard deviations.
- A phase-by-phase deduction view.
- A drill timeline with target range, actual value, and out-of-tolerance periods.

If the cohort has too few complete attempts for meaningful standard deviation, the system displays the strict drill score but marks comparison statistics as unavailable.

## 12. Import-quality report

Each import generates a report showing:

- Imported, replaced, skipped, or rejected file.
- Cadet, sortie, and drill identified.
- Source and loaded row counts.
- Missing values by field.
- Every interpolation run and its length.
- Invalid or dropped rows.
- Data-quality warnings.
- Final scoring eligibility.
- Score status and ruleset version, when scored.

## 13. Success criteria

- Valid CSV files import without duplicate attempts.
- Every row is linked to its cadet, sortie, drill, and optional segment.
- Re-uploading an attempt replaces the prior imported data cleanly.
- Short missing telemetry gaps are interpolated and reported.
- Long or non-interpolable gaps mark the whole drill incomplete.
- Every score can be reproduced from telemetry, phase definition, weights, and deductions.
- A cadet can be compared only to the appropriate completed drill cohort.
- The system clearly shows whether the cadet is above or below the cohort average for every scored metric.

## 14. Spreadsheet export

The system exports spreadsheet-friendly CSV files grouped by drill type rather than one oversized telemetry file.

Each exported drill file contains cadet, sortie, drill, source-row, and interpolation fields, plus only the telemetry fields that occur in that drill type. For example, `DoubleEight.csv` contains Double Eight telemetry, while `Final.csv` includes Final-specific approach fields.

This preserves all imported telemetry while avoiding mostly empty drill-specific columns in Excel.

## 15. Command-line operation

All commands should normally be run from the project folder:

```bash
cd "/home/ilan/Documents/flight school/drill_compare"
```

The local SQLite database is stored at:

```text
/home/ilan/Documents/flight school/drill_compare/code/flight_school.db
```

### Import drill files

Import a directory of CSV or XLSX drill exports with:

```bash
python3 code/import_flight_data.py "/home/ilan/Documents/flight school/data/drills_separate/1_26/s1"
```

The importer scans the folder for files ending in `.csv` or `.xlsx`.

Each input filename must follow this pattern:

```text
c<cadet_number>_<sortie_number>_<Hebrew drill name>.xlsx
c<cadet_number>_<sortie_number>_<Hebrew drill name>.csv
```

Example:

```text
c3_1_2 X שמינייה.xlsx
```

This becomes:

```text
cadet = 3
sortie = 1
drill = DoubleEight
```

Re-importing the same cadet, sortie, and drill replaces the existing attempt in the database.

### Export database to CSV

Export the local database into drill-specific, Excel-friendly CSV files with:

```bash
python3 code/import_flight_data.py --export-csv
```

The default export folder is:

```text
/home/ilan/Documents/flight school/drill_compare/exports/csv
```

The export folder contains one file for each drill type, such as `DoubleEight.csv`, and an `imports.csv` file containing import history.

A custom CSV export folder can also be provided:

```bash
python3 code/import_flight_data.py --export-csv "/home/ilan/Desktop/drill_csv_export"
```

### Generate a segment comparison graph

Generate a Plotly HTML graph comparing one cadet against the cohort mean and standard deviation per segment:

```bash
python3 code/segment_comparison.py 3 --drill DoubleEight
```

This creates:

```text
/home/ilan/Documents/flight school/drill_compare/exports/segments/cadet_3_sortie_1_DoubleEight_segments.html
```

Optional arguments:

```bash
python3 code/segment_comparison.py 3 --sortie 1 --drill DoubleEight
```

The segment comparison includes height, IAS, heading, roll, PitchStick, PitchStick_DELTA, RollStick, RollStick_DELTA, and collective_roughness when those fields exist in the imported data.

### Generate the all-cadets score dashboard

Generate the current score dashboard with:

```bash
python3 code/score_dashboard.py
```

By default, the score dashboard uses the `DoubleEight` drill because it is the first drill with V1 target and tolerance rules.

This creates:

```text
/home/ilan/Documents/flight school/drill_compare/exports/dashboards/score_dashboard_DoubleEight.html
/home/ilan/Documents/flight school/drill_compare/exports/dashboards/score_dashboard_DoubleEight.csv
```

To score a different drill after its scoring rules are defined:

```bash
python3 code/score_dashboard.py --drill DoublePattern
```

### Open generated HTML files

Generated dashboards and graphs are regular HTML files. They can be opened directly in a browser from the file manager, or from the command line:

```bash
xdg-open "/home/ilan/Documents/flight school/drill_compare/exports/dashboards/score_dashboard_DoubleEight.html"
xdg-open "/home/ilan/Documents/flight school/drill_compare/exports/segments/cadet_3_sortie_1_DoubleEight_segments.html"
```
