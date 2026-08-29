#!/usr/bin/env python3
"""Create sample-by-sample segment timeline graphs for one cadet and drill."""
import argparse
import json
import math
import sqlite3
import statistics
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports" / "segments"

METRICS = [
    ("height_ft", "Height AGL (ft)", "Height AGL (ft)"),
    ("ias_knots", "Airspeed (kt)", "Airspeed (kt)"),
    ("relative_heading_degrees", "Heading Change From Start", "Relative heading (degrees)"),
    ("roll_degrees", "Roll (degrees)", "Roll (degrees)"),
    ("pitch_stick", "Pitch Stick", "PitchStick"),
    ("pitch_stick_delta", "Pitch Stick Delta", "PitchStick_DELTA"),
    ("roll_stick", "Roll Stick", "RollStick"),
    ("roll_stick_delta", "Roll Stick Delta", "RollStick_DELTA"),
    ("collective_roughness", "Collective Roughness", "collective_roughness"),
]


def numeric(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def unwrap_heading(previous_heading, offset, heading):
    if previous_heading is None:
        return offset
    delta = heading - previous_heading
    if delta < -300.0:
        return offset + 360.0
    if delta > 300.0:
        return offset - 360.0
    return offset


def add_display_time(samples):
    offset = 0.0
    previous_segment = None
    previous_original_time = None
    segment_start_time = None
    for sample in samples:
        original_time = sample["time_original"]
        segment = sample["segment"]
        if original_time is None:
            sample["time_display"] = None
            continue
        if previous_segment is None:
            previous_segment = segment
            segment_start_time = original_time
        elif segment != previous_segment:
            if previous_original_time is not None and segment_start_time is not None:
                offset += previous_original_time - segment_start_time
            previous_segment = segment
            segment_start_time = original_time
        sample["time_display"] = offset + (original_time - segment_start_time)
        previous_original_time = original_time
    return samples


def sample_at_progress(samples, metric_key, progress):
    values = [sample[metric_key] for sample in samples if sample[metric_key] is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = progress * (len(values) - 1)
    left_index = int(math.floor(position))
    right_index = min(left_index + 1, len(values) - 1)
    fraction = position - left_index
    return values[left_index] + (values[right_index] - values[left_index]) * fraction


def segment_progress_time(samples, segment, progress):
    segment_times = [sample["plot_time"] for sample in samples if sample["segment"] == segment and sample["plot_time"] is not None]
    if not segment_times:
        return None
    if len(segment_times) == 1:
        return segment_times[0]
    position = progress * (len(segment_times) - 1)
    left_index = int(math.floor(position))
    right_index = min(left_index + 1, len(segment_times) - 1)
    fraction = position - left_index
    return segment_times[left_index] + (segment_times[right_index] - segment_times[left_index]) * fraction


def load_samples(connection, cadet, sortie, drill):
    rows = connection.execute(
        """SELECT t.source_row, t.timestamp, t.segment_number, t.height_ft, t.ias_knots,
                  t.heading_degrees, t.roll_degrees, t.pitch_stick, t.roll_stick, t.payload_json
           FROM telemetry_samples t
           JOIN imports i ON i.id=t.import_id
           WHERE i.cadet_number=? AND i.sortie_number=? AND i.drill_code=? AND i.complete=1
           ORDER BY t.source_row""",
        (cadet, sortie, drill),
    ).fetchall()
    if not rows:
        raise SystemExit("No complete telemetry found for this cadet, sortie, and drill.")

    output = []
    previous_heading = None
    heading_offset = 0.0
    start_unwrapped_heading = None
    for row in rows:
        payload = json.loads(row[9] or "{}")
        relative_heading = None
        if row[5] is not None:
            heading_offset = unwrap_heading(previous_heading, heading_offset, row[5])
            unwrapped_heading = row[5] + heading_offset
            if start_unwrapped_heading is None:
                start_unwrapped_heading = unwrapped_heading
            relative_heading = unwrapped_heading - start_unwrapped_heading
            previous_heading = row[5]
        output.append(
            {
                "source_row": row[0],
                "time_original": row[1],
                "segment": int(row[2]) if row[2] is not None else None,
                "height_ft": row[3],
                "ias_knots": row[4],
                "relative_heading_degrees": relative_heading,
                "roll_degrees": row[6],
                "pitch_stick": row[7],
                "roll_stick": row[8],
                "pitch_stick_delta": numeric(payload.get("PitchStick_DELTA")),
                "roll_stick_delta": numeric(payload.get("RollStick_DELTA")),
                "collective_roughness": numeric(payload.get("collective_roughness")),
            }
        )
    return add_display_time(output)


def load_cohort_samples(connection, target_cadet, target_sortie, drill):
    attempts = connection.execute(
        """SELECT cadet_number, sortie_number
           FROM imports
           WHERE drill_code=? AND complete=1
           ORDER BY cadet_number, sortie_number""",
        (drill,),
    ).fetchall()
    cohort = []
    for cadet, sortie in attempts:
        if cadet == target_cadet and sortie == target_sortie:
            continue
        try:
            cohort.append(load_samples(connection, cadet, sortie, drill))
        except SystemExit:
            continue
    return cohort


def choose_plot_time(samples, time_mode):
    key = "time_original" if time_mode == "original" else "time_display"
    for sample in samples:
        sample["plot_time"] = sample[key]
    return key


def cohort_series_for_metric(target_samples, cohort_samples, metric_key, samples_per_segment):
    segments = sorted({sample["segment"] for sample in target_samples if sample["segment"] is not None})
    output = []
    for segment in segments:
        target_segment = [sample for sample in target_samples if sample["segment"] == segment]
        cohort_segments = [
            [sample for sample in attempt if sample["segment"] == segment]
            for attempt in cohort_samples
        ]
        cohort_segments = [segment_samples for segment_samples in cohort_segments if segment_samples]
        if len(cohort_segments) < 2:
            continue
        for sample_index in range(samples_per_segment):
            progress = sample_index / (samples_per_segment - 1)
            time = segment_progress_time(target_segment, segment, progress)
            values = [
                sample_at_progress(segment_samples, metric_key, progress)
                for segment_samples in cohort_segments
            ]
            values = [value for value in values if value is not None]
            if time is None or len(values) < 2:
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            output.append(
                {
                    "time": time,
                    "segment": segment,
                    "mean": mean,
                    "upper": mean + sd,
                    "lower": mean - sd,
                    "attempt_count": len(values),
                }
            )
    return output


def segment_shapes(samples, row_count, col_count):
    shapes = []
    annotations = []
    previous_segment = samples[0]["segment"]
    boundaries = []
    segment_starts = {previous_segment: samples[0]["plot_time"]}
    for sample in samples[1:]:
        if sample["segment"] != previous_segment:
            boundaries.append(sample["plot_time"])
            segment_starts[sample["segment"]] = sample["plot_time"]
            previous_segment = sample["segment"]
    for boundary in boundaries:
        for row in range(1, row_count + 1):
            for col in range(1, col_count + 1):
                shapes.append(
                    dict(
                        type="line",
                        x0=boundary,
                        x1=boundary,
                        y0=0,
                        y1=1,
                        xref=f"x{(row - 1) * col_count + col if (row, col) != (1, 1) else ''}",
                        yref=f"y{(row - 1) * col_count + col if (row, col) != (1, 1) else ''} domain",
                        line=dict(color="rgba(80,80,80,0.28)", width=1, dash="dot"),
                    )
                )
    for segment, start_time in segment_starts.items():
        if segment is None:
            continue
        annotations.append(
            dict(
                x=start_time,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"S{segment}",
                showarrow=False,
                yshift=12,
                font=dict(size=10, color="rgba(80,80,80,0.75)"),
            )
        )
    return shapes, annotations


def build_timeline(samples, cohort_samples, cadet, sortie, drill, output, samples_per_segment, time_mode):
    time_key = choose_plot_time(samples, time_mode)
    for attempt_samples in cohort_samples:
        choose_plot_time(attempt_samples, time_mode)
    cols = 3
    rows_count = math.ceil(len(METRICS) / cols)
    fig = make_subplots(
        rows=rows_count,
        cols=cols,
        shared_xaxes=True,
        subplot_titles=[title for _, title, _ in METRICS],
    )

    available = 0
    for index, (metric_key, title, ylabel) in enumerate(METRICS):
        row, col = divmod(index, cols)
        row += 1
        col += 1
        metric_samples = [sample for sample in samples if sample[metric_key] is not None]
        if not metric_samples:
            continue
        cohort_series = cohort_series_for_metric(samples, cohort_samples, metric_key, samples_per_segment)
        if cohort_series:
            fig.add_trace(
                go.Scatter(
                    x=[point["time"] for point in cohort_series],
                    y=[point["upper"] for point in cohort_series],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[point["time"] for point in cohort_series],
                    y=[point["lower"] for point in cohort_series],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(120,120,120,0.18)",
                    line=dict(width=0),
                    name="Other cadets ±1 SD",
                    hoverinfo="skip",
                    showlegend=index == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[point["time"] for point in cohort_series],
                    y=[point["mean"] for point in cohort_series],
                    mode="lines",
                    name="Other cadets mean",
                    line=dict(width=2, color="rgba(90,90,90,0.75)"),
                    customdata=[
                        [point["segment"], point["attempt_count"]]
                        for point in cohort_series
                    ],
                    hovertemplate=(
                        "Display time %{x:.2f}s<br>"
                        "Mean %{y:.3f}<br>"
                        "Segment %{customdata[0]}<br>"
                        "Attempts %{customdata[1]}<extra></extra>"
                    ),
                    showlegend=index == 0,
                ),
                row=row,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=[sample["plot_time"] for sample in metric_samples],
                y=[sample[metric_key] for sample in metric_samples],
                mode="lines",
                name=f"Cadet {cadet}",
                line=dict(width=2.2, color="#e45756"),
                customdata=[
                    [sample["segment"], sample["source_row"], sample["time_original"]]
                    for sample in metric_samples
                ],
                hovertemplate=(
                    "Display time %{x:.2f}s<br>"
                    "Original time %{customdata[2]:.3f}s<br>"
                    "Value %{y:.3f}<br>"
                    "Segment %{customdata[0]}<br>"
                    "Source row %{customdata[1]}<extra></extra>"
                ),
                showlegend=index == 0,
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text=ylabel, row=row, col=col)
        available += 1

    if not available:
        raise SystemExit("No plottable telemetry fields found for this cadet, sortie, and drill.")

    for col in range(1, cols + 1):
        axis_title = "Original simulator time (s)" if time_key == "time_original" else "Compressed drill time (s)"
        fig.update_xaxes(title_text=axis_title, row=rows_count, col=col)

    shapes, annotations = segment_shapes(samples, rows_count, cols)
    fig.update_layout(
        title=f"Cadet {cadet} - sortie {sortie} - {drill} - segment timelines ({time_mode} time)",
        height=1120,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        shapes=shapes,
        annotations=list(fig.layout.annotations) + annotations,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description="Plot sample-by-sample telemetry with segment boundaries.")
    parser.add_argument("cadet", type=int)
    parser.add_argument("--sortie", type=int, default=1)
    parser.add_argument("--drill", default="DoubleEight")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("flight_school.db"))
    parser.add_argument("--output", type=Path, help="HTML output path")
    parser.add_argument("--cohort-samples", type=int, default=120, help="points used to resample each segment")
    parser.add_argument("--time-mode", choices=["compressed", "original"], default="compressed")
    args = parser.parse_args()
    if args.output is None:
        args.output = EXPORTS_DIR / f"cadet_{args.cadet}_sortie_{args.sortie}_{args.drill}_segments2.html"

    connection = sqlite3.connect(args.database)
    samples = load_samples(connection, args.cadet, args.sortie, args.drill)
    cohort_samples = load_cohort_samples(connection, args.cadet, args.sortie, args.drill)
    build_timeline(samples, cohort_samples, args.cadet, args.sortie, args.drill, args.output, args.cohort_samples, args.time_mode)
    print(f"Saved segment timeline: {args.output}")
    print(f'Open HTML: xdg-open "{args.output.resolve()}"')


if __name__ == "__main__":
    main()
