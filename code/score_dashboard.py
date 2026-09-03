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


RULESET_VERSION = "v0.3-provisional-per-drill"
STARTING_DISPLAY_SCORE = 100.0
EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports" / "dashboards"

COMMON_CONTROL_METRICS = [
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

def target_metric(key, label, source, target, tolerance, weight, absolute=False):
    return {
        "key": key,
        "label": label,
        "source": source,
        "mode": "target",
        "target": target,
        "tolerance": tolerance,
        "weight": weight,
        "absolute": absolute,
    }


def range_metric(key, label, source, minimum, maximum, weight, absolute=False):
    return {
        "key": key,
        "label": label,
        "source": source,
        "mode": "range",
        "minimum": minimum,
        "maximum": maximum,
        "weight": weight,
        "absolute": absolute,
    }


CIRCLE_DRILL_METRICS = [
    target_metric("height_ft", "Height", "height_ft", 60.0, 5.0, 1.50),
    target_metric("ias_knots", "IAS", "ias_knots", 90.0, 3.0, 2.00),
    target_metric("bank_degrees", "Bank", "roll_degrees", 30.0, 5.0, 1.50, absolute=True),
]

PATTERN_DRILL_METRICS = [
    range_metric("ias_70_90_knots", "IAS 70-90", "ias_knots", 70.0, 90.0, 1.40),
    target_metric("pattern_bank_20_degrees", "Pattern bank", "roll_degrees", 20.0, 5.0, 1.20, absolute=True),
    range_metric("collective_40_65_pi", "Power 40-65 PI", "collective", 0.40, 0.65, 60.0),
]

TRACK_DRILL_METRICS = [
    target_metric("height_ft", "Height", "height_ft", 60.0, 10.0, 1.00),
    target_metric("ias_knots", "IAS", "ias_knots", 90.0, 5.0, 1.40),
    target_metric("distance_to_axis_m", "Distance to axis", "distanceToAxis", 0.0, 20.0, 1.20),
]

FINAL_DRILL_METRICS = [
    target_metric("final_distance_ft", "Distance to target", "dist", 0.0, 250.0, 0.02),
    target_metric("final_path_angle", "Final path angle", "angle", 0.0, 5.0, 1.00),
    target_metric("ias_knots", "IAS", "ias_knots", 90.0, 5.0, 1.40),
]

YNP_DRILL_METRICS = [
    target_metric("height_ft", "Height", "height_ft", 60.0, 10.0, 1.20),
    target_metric("ias_knots", "IAS", "ias_knots", 90.0, 5.0, 1.40),
]

OPTIMAL_TURN_METRICS = [
    target_metric("height_ft", "Height", "height_ft", 60.0, 8.0, 1.20),
    target_metric("ias_knots", "IAS", "ias_knots", 90.0, 5.0, 1.40),
    target_metric("bank_degrees", "Bank", "roll_degrees", 30.0, 7.0, 1.20, absolute=True),
]

DRILL_SCORING_RULES = {
    "Eight": CIRCLE_DRILL_METRICS,
    "DoubleEight": CIRCLE_DRILL_METRICS,
    "Pattern": PATTERN_DRILL_METRICS,
    "DoublePattern": PATTERN_DRILL_METRICS,
    "LearnedTrack": TRACK_DRILL_METRICS,
    "MissionTrack": TRACK_DRILL_METRICS,
    "Final": FINAL_DRILL_METRICS,
    "YNP": YNP_DRILL_METRICS,
    "DoubleYNP": YNP_DRILL_METRICS,
    "YNP2Points": YNP_DRILL_METRICS,
    "DoubleYNP2Points": YNP_DRILL_METRICS,
    "OptimalTurn": OPTIMAL_TURN_METRICS,
    "DoubleOptimalTurn": OPTIMAL_TURN_METRICS,
}

RULE_COVERAGE = {
    "Eight": "Hebrew rules doc: partial implementation; heading-axis and transition-point distance need phase geometry.",
    "DoubleEight": "Hebrew rules doc: partial implementation; heading-axis and transition-point distance need phase geometry.",
    "Pattern": "Hebrew rules doc: partial implementation; phase detection and distance-to-axis need geometry.",
    "DoublePattern": "Hebrew rules doc: partial implementation; phase detection and distance-to-axis need geometry.",
    "LearnedTrack": "Provisional: Hebrew rules doc has heading only in extracted text.",
    "MissionTrack": "Provisional: Hebrew rules doc has heading only in extracted text.",
    "Final": "Provisional: no final scoring rubric found in extracted Hebrew rules doc.",
    "YNP": "Provisional: Hebrew rules doc has heading only in extracted text.",
    "DoubleYNP": "Provisional: Hebrew rules doc has heading only in extracted text.",
    "YNP2Points": "Provisional: Hebrew rules doc has heading only in extracted text.",
    "DoubleYNP2Points": "Provisional: Hebrew rules doc has heading only in extracted text.",
    "OptimalTurn": "Provisional: no optimal-turn scoring rubric found in extracted Hebrew rules doc.",
    "DoubleOptimalTurn": "Provisional: no optimal-turn scoring rubric found in extracted Hebrew rules doc.",
}

ALL_METRICS = []
for metrics in DRILL_SCORING_RULES.values():
    for metric in metrics:
        if metric["key"] not in {existing["key"] for existing in ALL_METRICS}:
            ALL_METRICS.append(metric)
SCORING_METRICS = ALL_METRICS


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
    if value is None:
        value = get_payload_value(payload, metric["source"])
    if value is None:
        return None
    value = abs(value) if metric.get("absolute") else value
    return value


def metric_excess(value, metric):
    if metric.get("mode") == "range":
        if value < metric["minimum"]:
            return metric["minimum"] - value
        if value > metric["maximum"]:
            return value - metric["maximum"]
        return 0.0
    return max(0.0, abs(value - metric["target"]) - metric["tolerance"])


def score_attempt(connection, import_row):
    import_id = import_row["id"]
    metrics = DRILL_SCORING_RULES.get(import_row["drill_code"], YNP_DRILL_METRICS)
    samples = connection.execute(
        """SELECT height_ft, ias_knots, roll_degrees, pitch_stick, roll_stick, payload_json
           FROM telemetry_samples WHERE import_id=? ORDER BY source_row""",
        (import_id,),
    ).fetchall()
    deductions_by_metric = {metric["key"]: 0.0 for metric in metrics}
    samples_by_metric = {metric["key"]: 0 for metric in metrics}

    for sample_row in samples:
        sample = dict(sample_row)
        payload = json.loads(sample.pop("payload_json") or "{}")
        for metric in metrics:
            value = get_sample_value(sample, payload, metric)
            if value is None:
                continue
            excess = metric_excess(value, metric)
            deductions_by_metric[metric["key"]] += excess * metric["weight"]
            samples_by_metric[metric["key"]] += 1

    normalized_deductions = {}
    for metric in metrics:
        key = metric["key"]
        if samples_by_metric[key]:
            normalized_deductions[key] = deductions_by_metric[key] / samples_by_metric[key]
        else:
            normalized_deductions[key] = 0.0

    total_deduction = sum(normalized_deductions.values())
    display_score = max(0.0, STARTING_DISPLAY_SCORE - total_deduction)
    raw_score = display_score * 10.0
    used_metrics = [metric["label"] for metric in metrics if samples_by_metric[metric["key"]]]

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
        "rule_coverage": RULE_COVERAGE.get(import_row["drill_code"], "Provisional: no drill-specific rule coverage note."),
        **{f"deduction_{metric['key']}": normalized_deductions.get(metric["key"], 0.0) for metric in SCORING_METRICS},
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


def available_drills(connection):
    return [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT drill_code FROM imports ORDER BY drill_code"
        )
    ]


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
        "rule_coverage",
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
    parser.add_argument("--drill", help="optional drill code to score, for example DoubleEight")
    parser.add_argument("--output", type=Path, help="HTML output path")
    parser.add_argument("--write-drill-dashboards", action="store_true", help="also write one dashboard per drill")
    args = parser.parse_args()
    if args.output is None:
        suffix = args.drill if args.drill else "all_drills"
        args.output = EXPORTS_DIR / f"score_dashboard_{suffix}.html"

    connection = sqlite3.connect(args.database)
    scores = load_scores(connection, args.drill)
    build_dashboard(scores, args.output)
    csv_path = write_summary_csv(scores, args.output)
    print(f"Saved dashboard: {args.output}")
    print(f'Open HTML: xdg-open "{args.output.resolve()}"')
    print(f"Saved score summary: {csv_path}")
    print(f"Ruleset: {RULESET_VERSION}")
    if args.write_drill_dashboards:
        for drill in available_drills(connection):
            drill_output = EXPORTS_DIR / f"score_dashboard_{drill}.html"
            drill_scores = load_scores(connection, drill)
            build_dashboard(drill_scores, drill_output)
            drill_csv_path = write_summary_csv(drill_scores, drill_output)
            print(f"Saved {drill} dashboard: {drill_output}")
            print(f'Open HTML: xdg-open "{drill_output.resolve()}"')
            print(f"Saved {drill} score summary: {drill_csv_path}")


if __name__ == "__main__":
    main()
