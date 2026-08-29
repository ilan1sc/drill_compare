#!/usr/bin/env python3
"""Create a top-down route map with an orthophoto background."""
import argparse
import json
import math
import sqlite3
from pathlib import Path

import plotly.graph_objects as go


EXPORTS_DIR = Path(__file__).resolve().parent.parent / "exports" / "maps"
ORTHOPHOTO_TILES = [
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
]
ISRAEL_BOUNDS = {"lat_min": 29.0, "lat_max": 34.0, "lon_min": 34.0, "lon_max": 36.5}
RADIAN_LAT_LON_DRILLS = {"Eight", "DoubleEight", "YNP", "DoubleYNP", "OptimalTurn", "DoubleOptimalTurn", "LearnedTrack"}
DEGREE_LAT_LON_DRILLS = {"Pattern", "DoublePattern"}
LAT_LON_REQUIRED_DRILLS = {"MissionTrack", "Final"}


def numeric(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def in_israel(lat, lon):
    return (
        ISRAEL_BOUNDS["lat_min"] <= lat <= ISRAEL_BOUNDS["lat_max"]
        and ISRAEL_BOUNDS["lon_min"] <= lon <= ISRAEL_BOUNDS["lon_max"]
    )


def normalize_coordinate(raw_lat, raw_lon, lat_deg=None, lon_deg=None, drill=None):
    if lat_deg is not None and lon_deg is not None:
        return lat_deg, lon_deg, "latDeg/lonDeg"

    if raw_lat is None or raw_lon is None:
        return None, None, None

    if drill in RADIAN_LAT_LON_DRILLS:
        return math.degrees(raw_lat), math.degrees(raw_lon), "lat/lon radians"
    if drill in DEGREE_LAT_LON_DRILLS or drill in LAT_LON_REQUIRED_DRILLS:
        return raw_lat, raw_lon, "lat/lon degrees"

    lat, lon = math.degrees(raw_lat), math.degrees(raw_lon)
    if in_israel(lat, lon):
        return lat, lon, "lat/lon radians fallback"
    if in_israel(raw_lat, raw_lon):
        return raw_lat, raw_lon, "lat/lon degrees fallback"
    return None, None, None


def load_attempt_route(connection, cadet, sortie, drill):
    rows = connection.execute(
        """SELECT t.source_row, t.segment_number, t.heading_degrees, t.payload_json
           FROM telemetry_samples t
           JOIN imports i ON i.id = t.import_id
           WHERE i.cadet_number=? AND i.sortie_number=? AND i.drill_code=? AND i.complete=1
           ORDER BY t.source_row""",
        (cadet, sortie, drill),
    ).fetchall()

    route = []
    coordinate_modes = set()
    for source_row, segment, heading, payload_json in rows:
        payload = json.loads(payload_json or "{}")
        raw_lat = numeric(payload.get("lat"))
        raw_lon = numeric(payload.get("lon"))
        lat_deg = numeric(payload.get("latDeg"))
        lon_deg = numeric(payload.get("lonDeg"))
        lat, lon, coordinate_mode = normalize_coordinate(raw_lat, raw_lon, lat_deg, lon_deg, drill)
        if lat is None or lon is None:
            continue
        coordinate_modes.add(coordinate_mode)
        route.append(
            {
                "source_row": source_row,
                "segment": int(segment) if segment is not None else None,
                "lat": lat,
                "lon": lon,
                "heading": heading,
            }
        )
    return route, sorted(coordinate_modes)


def load_cohort_routes(connection, target_cadet, target_sortie, drill):
    attempts = connection.execute(
        """SELECT cadet_number, sortie_number
           FROM imports
           WHERE drill_code=? AND complete=1
           ORDER BY cadet_number, sortie_number""",
        (drill,),
    ).fetchall()
    routes = []
    coordinate_modes = set()
    for cadet, sortie in attempts:
        if cadet == target_cadet and sortie == target_sortie:
            continue
        route, modes = load_attempt_route(connection, cadet, sortie, drill)
        if route:
            routes.append(route)
            coordinate_modes.update(modes)
    return routes, sorted(coordinate_modes)


def by_segment(route):
    grouped = {}
    for point in route:
        grouped.setdefault(point["segment"], []).append(point)
    return sorted(grouped.items(), key=lambda item: (-1 if item[0] is None else item[0]))


def add_other_cadets_layer(fig, cohort_routes):
    for index, route in enumerate(cohort_routes):
        fig.add_trace(
            go.Scattermap(
                lat=[point["lat"] for point in route],
                lon=[point["lon"] for point in route],
                mode="lines",
                name="Other cadets",
                line=dict(width=2, color="rgba(90,90,90,0.45)"),
                hovertemplate="Other cadet route<br>Lat %{lat:.6f}<br>Lon %{lon:.6f}<extra></extra>",
                showlegend=index == 0,
            )
        )


def build_map(route, cohort_routes, coordinate_modes, cadet, sortie, drill, output):
    if len(route) < 2:
        raise SystemExit("Not enough geographic points found for this cadet, sortie, and drill.")

    fig = go.Figure()
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]

    add_other_cadets_layer(fig, cohort_routes)

    for index, (segment, points) in enumerate(by_segment(route)):
        name = f"Segment {segment}" if segment is not None else "No segment"
        fig.add_trace(
            go.Scattermap(
                lat=[point["lat"] for point in points],
                lon=[point["lon"] for point in points],
                mode="lines",
                name=name,
                line=dict(width=4, color=palette[index % len(palette)]),
                hovertemplate=name + "<br>Lat %{lat:.6f}<br>Lon %{lon:.6f}<extra></extra>",
            )
        )

    start = route[0]
    end = route[-1]
    for label, point, color, textposition in [
        ("START", start, "#00cc66", "top right"),
        ("END", end, "#cc0033", "bottom right"),
    ]:
        fig.add_trace(
            go.Scattermap(
                lat=[point["lat"]],
                lon=[point["lon"]],
                mode="markers+text",
                text=[label],
                textposition=textposition,
                name=label,
                marker=dict(size=14, color=color),
                hovertemplate="Row %{customdata}<br>Lat %{lat:.6f}<br>Lon %{lon:.6f}<extra></extra>",
                customdata=[point["source_row"]],
            )
        )

    center_lat = sum(point["lat"] for point in route) / len(route)
    center_lon = sum(point["lon"] for point in route) / len(route)
    fig.update_layout(
        title=(
            f"God View Orthophoto - Cadet {cadet} - Sortie {sortie} - {drill}"
            f"<br><sup>Coordinate conversion: {', '.join(coordinate_modes)}</sup>"
        ),
        height=850,
        margin=dict(l=10, r=10, t=78, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        map=dict(
            style="white-bg",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=16,
            layers=[
                dict(
                    sourcetype="raster",
                    source=ORTHOPHOTO_TILES,
                    below="traces",
                    opacity=1.0,
                )
            ],
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description="Create an orthophoto top-down route map.")
    parser.add_argument("cadet", type=int)
    parser.add_argument("--sortie", type=int, default=1)
    parser.add_argument("--drill", default="DoubleEight")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("flight_school.db"))
    parser.add_argument("--output", type=Path, help="HTML output path")
    args = parser.parse_args()
    if args.output is None:
        args.output = EXPORTS_DIR / f"cadet_{args.cadet}_sortie_{args.sortie}_{args.drill}_god_view.html"

    connection = sqlite3.connect(args.database)
    route, coordinate_modes = load_attempt_route(connection, args.cadet, args.sortie, args.drill)
    cohort_routes, cohort_modes = load_cohort_routes(connection, args.cadet, args.sortie, args.drill)
    build_map(route, cohort_routes, sorted(set(coordinate_modes + cohort_modes)), args.cadet, args.sortie, args.drill, args.output)
    print(f"Saved map: {args.output}")
    print(f'Open HTML: xdg-open "{args.output.resolve()}"')


if __name__ == "__main__":
    main()
