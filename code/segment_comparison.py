#!/usr/bin/env python3
"""Create segment comparison graphs for one cadet and drill."""
import argparse
import sqlite3
import statistics
import math
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports" / "segments"

METRICS = [
    ("height_ft", "height_ft", "Height AGL (ft)", "Height AGL (ft)"),
    ("ias_knots", "ias_knots", "Airspeed (kt)", "Airspeed (kt)"),
    ("relative_heading_degrees", "relative_heading_degrees", "Heading Change From Start", "Relative heading (degrees)"),
    ("roll_degrees", "roll_degrees", "Roll (degrees)", "Roll (degrees)"),
    ("pitch_stick", "pitch_stick", "Pitch Stick", "PitchStick"),
    ("pitch_stick_delta", "json_extract(t.payload_json, '$.PitchStick_DELTA')", "Pitch Stick Delta", "PitchStick_DELTA"),
    ("roll_stick", "roll_stick", "Roll Stick", "RollStick"),
    ("roll_stick_delta", "json_extract(t.payload_json, '$.RollStick_DELTA')", "Roll Stick Delta", "RollStick_DELTA"),
    ("collective_roughness", "json_extract(t.payload_json, '$.collective_roughness')", "Collective Roughness", "collective_roughness"),
]

def signed_angle_delta(value, reference):
    return ((value - reference + 180.0) % 360.0) - 180.0

def segment_stats(connection, cadet, sortie, drill, metric_key, metric_expression):
    if metric_key == "relative_heading_degrees":
        return relative_heading_segment_stats(connection, cadet, sortie, drill)
    query = f"""SELECT i.cadet_number, i.sortie_number, t.segment_number, AVG(CAST({metric_expression} AS REAL))
    FROM telemetry_samples t JOIN imports i ON i.id=t.import_id
    WHERE i.drill_code=? AND i.complete=1 AND t.segment_number IS NOT NULL AND {metric_expression} IS NOT NULL
    GROUP BY i.cadet_number, i.sortie_number, t.segment_number ORDER BY t.segment_number"""
    grouped = {}
    for other_cadet, other_sortie, segment, value in connection.execute(query, (drill,)):
        if value is None:
            continue
        grouped.setdefault(segment, []).append((other_cadet, other_sortie, value))
    output = []
    for segment, values in grouped.items():
        cohort = [value for c, s, value in values if (c, s) != (cadet, sortie)]
        target = next((value for c, s, value in values if (c, s) == (cadet, sortie)), None)
        if cohort:
            output.append((segment, statistics.mean(cohort), statistics.stdev(cohort) if len(cohort) > 1 else 0, target))
    return output

def relative_heading_segment_stats(connection, cadet, sortie, drill):
    query = """SELECT i.cadet_number, i.sortie_number, t.segment_number, t.heading_degrees
    FROM telemetry_samples t JOIN imports i ON i.id=t.import_id
    WHERE i.drill_code=? AND i.complete=1 AND t.segment_number IS NOT NULL AND t.heading_degrees IS NOT NULL
    ORDER BY i.cadet_number, i.sortie_number, t.source_row"""
    start_heading = {}
    values_by_segment = {}
    for other_cadet, other_sortie, segment, heading in connection.execute(query, (drill,)):
        attempt = (other_cadet, other_sortie)
        start_heading.setdefault(attempt, heading)
        relative_heading = signed_angle_delta(heading, start_heading[attempt])
        values_by_segment.setdefault((attempt, segment), []).append(relative_heading)

    grouped = {}
    for (attempt, segment), values in values_by_segment.items():
        grouped.setdefault(segment, []).append((*attempt, statistics.mean(values)))

    output = []
    for segment, values in sorted(grouped.items()):
        cohort = [value for c, s, value in values if (c, s) != (cadet, sortie)]
        target = next((value for c, s, value in values if (c, s) == (cadet, sortie)), None)
        if cohort:
            output.append((segment, statistics.mean(cohort), statistics.stdev(cohort) if len(cohort) > 1 else 0, target))
    return output

def main():
    parser = argparse.ArgumentParser(description="Plot one cadet against cohort segment averages.")
    parser.add_argument("cadet", type=int)
    parser.add_argument("--sortie", type=int, default=1)
    parser.add_argument("--drill", default="DoubleEight")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("flight_school.db"))
    parser.add_argument("--output", type=Path, help="HTML output path")
    args = parser.parse_args()
    if args.output is None:
        args.output = EXPORTS_DIR / f"cadet_{args.cadet}_sortie_{args.sortie}_{args.drill}_segments.html"

    connection = sqlite3.connect(args.database)
    cols = 3
    rows_count = math.ceil(len(METRICS) / cols)
    fig = make_subplots(
        rows=rows_count,
        cols=cols,
        subplot_titles=[title for _, _, title, _ in METRICS],
    )
    available = 0
    for index, (metric_key, metric_expression, title, ylabel) in enumerate(METRICS):
        row, col = divmod(index, cols)
        row += 1; col += 1
        rows = segment_stats(connection, args.cadet, args.sortie, args.drill, metric_key, metric_expression)
        rows = [row for row in rows if row[3] is not None]
        if not rows:
            continue
        segments = [row[0] for row in rows]
        mean = [row[1] for row in rows]
        sd = [row[2] or 0 for row in rows]
        value = [row[3] for row in rows]
        lower, upper = [m-s for m, s in zip(mean, sd)], [m+s for m, s in zip(mean, sd)]
        fig.add_trace(go.Scatter(x=segments, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(x=segments, y=lower, fill="tonexty", fillcolor="rgba(76,120,168,.18)", line=dict(width=0), name="Cohort ±1 SD", showlegend=index == 0), row=row, col=col)
        fig.add_trace(go.Scatter(x=segments, y=mean, mode="lines+markers", name="Cohort mean", showlegend=index == 0), row=row, col=col)
        fig.add_trace(go.Scatter(x=segments, y=value, mode="lines+markers", name=f"Cadet {args.cadet}", showlegend=index == 0, line=dict(color="#e45756", width=3)), row=row, col=col)
        fig.update_xaxes(title_text="Segment", tickmode="array", tickvals=segments, row=row, col=col)
        fig.update_yaxes(title_text=ylabel, row=row, col=col)
        available += 1
    if not available:
        raise SystemExit("No complete comparison data found for this cadet, sortie, and drill.")
    fig.update_layout(title=f"Cadet {args.cadet} · sortie {args.sortie} · {args.drill}", height=1120, template="plotly_white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True)
    print(f"Saved graph: {args.output}")

if __name__ == "__main__":
    main()
