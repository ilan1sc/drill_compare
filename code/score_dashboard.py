#!/usr/bin/env python3
"""Create an all-cadets score dashboard from the local flight-school database."""
import argparse
import csv
import json
import sqlite3
import statistics
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


RULESET_VERSION = "v0.2-provisional-duration-normalized"
STARTING_DISPLAY_SCORE = 100.0
EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports" / "dashboards"

SCORING_METRICS = [
    {
        "key": "height_ft",
        "label": "Height",
        "target": 60.0,
        "tolerance": 5.0,
        "weight": 1.50,
        "source": "height_ft",
        "absolute": False,
    },
    {
        "key": "ias_knots",
        "label": "IAS",
        "target": 90.0,
        "tolerance": 3.0,
        "weight": 2.00,
        "source": "ias_knots",
        "absolute": False,
    },
    {
        "key": "bank_degrees",
        "label": "Bank",
        "target": 30.0,
        "tolerance": 5.0,
        "weight": 1.50,
        "source": "roll_degrees",
        "absolute": True,
    },
    {
        "key": "pitch_stick",
        "label": "PitchStick",
        "target": 0.0,
        "tolerance": 1.0,
        "weight": 3.00,
        "source": "pitch_stick",
        "absolute": False,
    },
    {
        "key": "roll_stick",
        "label": "RollStick",
        "target": 0.0,
        "tolerance": 1.0,
        "weight": 3.00,
        "source": "roll_stick",
        "absolute": False,
    },
    {
        "key": "collective_roughness",
        "label": "Collective roughness",
        "target": 0.0,
        "tolerance": 0.5,
        "weight": 5.00,
        "source": "collective_roughness",
        "absolute": False,
    },
]


def get_payload_value(payload, key):
    value = payload.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_sample_value(sample, payload, metric):
    value = sample.get(metric["source"])
    if value is None and metric["source"] == "collective_roughness":
        value = get_payload_value(payload, "collective_roughness")
    if value is None:
        return None
    value = abs(value) if metric["absolute"] else value
    return value


def score_attempt(connection, import_row):
    import_id = import_row["id"]
    samples = connection.execute(
        """SELECT height_ft, ias_knots, roll_degrees, pitch_stick, roll_stick, payload_json
           FROM telemetry_samples WHERE import_id=? ORDER BY source_row""",
        (import_id,),
    ).fetchall()
    deductions_by_metric = {metric["key"]: 0.0 for metric in SCORING_METRICS}
    samples_by_metric = {metric["key"]: 0 for metric in SCORING_METRICS}

    for sample_row in samples:
        sample = dict(sample_row)
        payload = json.loads(sample.pop("payload_json") or "{}")
        for metric in SCORING_METRICS:
            value = get_sample_value(sample, payload, metric)
            if value is None:
                continue
            excess = max(0.0, abs(value - metric["target"]) - metric["tolerance"])
            deductions_by_metric[metric["key"]] += excess * metric["weight"]
            samples_by_metric[metric["key"]] += 1

    normalized_deductions = {}
    for metric in SCORING_METRICS:
        key = metric["key"]
        if samples_by_metric[key]:
            normalized_deductions[key] = deductions_by_metric[key] / samples_by_metric[key]
        else:
            normalized_deductions[key] = 0.0

    total_deduction = sum(normalized_deductions.values())
    display_score = max(0.0, STARTING_DISPLAY_SCORE - total_deduction)
    raw_score = display_score * 10.0
    used_metrics = [metric["label"] for metric in SCORING_METRICS if samples_by_metric[metric["key"]]]

    return {
        "cadet": import_row["cadet_number"],
        "sortie": import_row["sortie_number"],
        "drill": import_row["drill_code"],
        "filename": import_row["filename"],
        "status": "scored" if import_row["complete"] else "incomplete",
        "row_count": import_row["row_count"],
        "score": display_score if import_row["complete"] else None,
        "raw_score": raw_score if import_row["complete"] else None,
        "total_deduction": total_deduction if import_row["complete"] else None,
        "used_metrics": ", ".join(used_metrics),
        **{f"deduction_{key}": value for key, value in normalized_deductions.items()},
    }


def load_scores(connection, drill_filter):
    connection.row_factory = sqlite3.Row
    query = "SELECT * FROM imports"
    params = []
    if drill_filter:
        query += " WHERE drill_code=?"
        params.append(drill_filter)
    query += " ORDER BY cadet_number, sortie_number, drill_code"
    return [score_attempt(connection, row) for row in connection.execute(query, params)]


def write_summary_csv(scores, output_path):
    csv_path = output_path.with_suffix(".csv")
    fieldnames = [
        "cadet",
        "sortie",
        "drill",
        "status",
        "score",
        "raw_score",
        "total_deduction",
        "row_count",
        "used_metrics",
        "filename",
        *[f"deduction_{metric['key']}" for metric in SCORING_METRICS],
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            writer.writerow({
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in score.items()
                if key in fieldnames
            })
    return csv_path


def mean_score_by_cadet(scores):
    grouped = {}
    for score in scores:
        if score["score"] is None:
            continue
        grouped.setdefault(score["cadet"], []).append(score["score"])
    return sorted((cadet, statistics.mean(values)) for cadet, values in grouped.items())


def mean_score_by_drill(scores):
    grouped = {}
    for score in scores:
        if score["score"] is None:
            continue
        grouped.setdefault(score["drill"], []).append(score["score"])
    return sorted((drill, statistics.mean(values), len(values)) for drill, values in grouped.items())


def top_attempts(scores, limit=30):
    scored = [score for score in scores if score["score"] is not None]
    return sorted(scored, key=lambda row: (row["cadet"], row["sortie"], row["drill"]))[:limit]


def build_dashboard(scores, output_path):
    scored = [score for score in scores if score["score"] is not None]
    if not scored:
        raise SystemExit("No complete attempts found to score.")

    cadet_rows = mean_score_by_cadet(scores)
    drill_rows = mean_score_by_drill(scores)
    attempts = top_attempts(scores)
    metric_deductions = [
        (
            metric["label"],
            sum(score[f"deduction_{metric['key']}"] or 0 for score in scored),
        )
        for metric in SCORING_METRICS
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "box"}], [{"type": "bar"}, {"type": "table"}]],
        subplot_titles=[
            "Average Score by Cadet",
            "Score Distribution by Drill",
            "Total Deductions by Metric",
            "Attempt Scores",
        ],
    )

    fig.add_trace(
        go.Bar(
            x=[f"Cadet {cadet}" for cadet, _ in cadet_rows],
            y=[score for _, score in cadet_rows],
            name="Average score",
            marker_color="#4c78a8",
        ),
        row=1,
        col=1,
    )

    for drill, _, _ in drill_rows:
        fig.add_trace(
            go.Box(
                y=[score["score"] for score in scored if score["drill"] == drill],
                name=drill,
                boxmean=True,
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Bar(
            x=[label for label, _ in metric_deductions],
            y=[deduction for _, deduction in metric_deductions],
            name="Total deductions",
            marker_color="#f58518",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(values=["Cadet", "Sortie", "Drill", "Score", "Rows", "Status"]),
            cells=dict(
                values=[
                    [row["cadet"] for row in attempts],
                    [row["sortie"] for row in attempts],
                    [row["drill"] for row in attempts],
                    [round(row["score"], 2) for row in attempts],
                    [row["row_count"] for row in attempts],
                    [row["status"] for row in attempts],
                ]
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(title_text="Score (0-100)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Score (0-100)", range=[0, 100], row=1, col=2)
    fig.update_yaxes(title_text="Deducted raw points", row=2, col=1)
    fig.update_layout(
        title=f"All Cadets Score Dashboard ({RULESET_VERSION})",
        height=950,
        template="plotly_white",
        showlegend=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description="Create a Plotly score dashboard for all cadets.")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("flight_school.db"))
    parser.add_argument("--drill", default="DoubleEight", help="drill code to score, for example DoubleEight")
    parser.add_argument("--output", type=Path, help="HTML output path")
    args = parser.parse_args()
    if args.output is None:
        args.output = EXPORTS_DIR / f"score_dashboard_{args.drill}.html"

    connection = sqlite3.connect(args.database)
    scores = load_scores(connection, args.drill)
    build_dashboard(scores, args.output)
    csv_path = write_summary_csv(scores, args.output)
    print(f"Saved dashboard: {args.output}")
    print(f'Open HTML: xdg-open "{args.output.resolve()}"')
    print(f"Saved score summary: {csv_path}")
    print(f"Ruleset: {RULESET_VERSION}")


if __name__ == "__main__":
    main()
