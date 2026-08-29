#!/usr/bin/env python3
"""Create a Cesium 3D debrief view with tiled orthophoto imagery."""
import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

from terrain_3d_plot import (
    DEM_PATH,
    EXPORTS_DIR,
    FEET_TO_METERS,
    bilinear_sample,
    download_dem,
    latlon_to_pixel,
    load_route,
    read_geotiff,
)


CESIUM_VERSION = "1.121"
DEFAULT_CESIUM_ION_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6Ik5OZWdObXp4VGVWMHZuSEsiLCJqdGkiOiJmY2E0OTEzZS0xMTI5LTRhOWItYmVkMS1kZGY3OTdlYzQ2NTciLCJpZCI6NDY5NDE3LCJpc3MiOiJodHRwczovL2FwaS5jZXNpdW0uY29tIiwiYXVkIjoidW5kZWZpbmVkX2RlZmF1bHQiLCJpYXQiOjE3ODcwODQyODN9.8XK9LxC1aO38Ylr0PFcRcmj3hWrV-vlhPeGuej-VXQA"
ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SEGMENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def route_with_altitude(route, dem_path):
    elevation, lon_origin, lat_origin, lon_step, lat_step = read_geotiff(dem_path)
    output = []
    for point in route:
        row, col = latlon_to_pixel(point["lat"], point["lon"], lon_origin, lat_origin, lon_step, lat_step)
        ground_m = bilinear_sample(elevation, row, col)
        if ground_m is None:
            continue
        height_ft = point["height_ft"] if point["height_ft"] is not None else 0.0
        output.append(
            {
                "source_row": point["source_row"],
                "segment": point["segment"],
                "lat": round(point["lat"], 8),
                "lon": round(point["lon"], 8),
                "ground_m": round(float(ground_m), 2),
                "aircraft_msl_m": round(float(ground_m + height_ft * FEET_TO_METERS), 2),
                "height_ft": round(float(height_ft), 2),
            }
        )
    if len(output) < 2:
        raise SystemExit("Route does not overlap the DEM tile.")
    return output


def split_by_segment(route):
    groups = []
    current = None
    current_segment = None
    for point in route:
        if current is None or point["segment"] != current_segment:
            current_segment = point["segment"]
            current = {"segment": current_segment, "points": []}
            groups.append(current)
        current["points"].append(point)
    return groups


def html_escape_json(data):
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def build_html(route, coordinate_modes, cadet, sortie, drill, token):
    center_lat = sum(point["lat"] for point in route) / len(route)
    center_lon = sum(point["lon"] for point in route) / len(route)
    max_alt = max(point["aircraft_msl_m"] for point in route)
    segment_groups = split_by_segment(route)
    payload = {
        "cadet": cadet,
        "sortie": sortie,
        "drill": drill,
        "coordinateModes": coordinate_modes,
        "center": {"lat": center_lat, "lon": center_lon, "height": max_alt + 1400},
        "segments": [
            {
                "segment": group["segment"],
                "color": SEGMENT_COLORS[index % len(SEGMENT_COLORS)],
                "points": group["points"],
            }
            for index, group in enumerate(segment_groups)
            if len(group["points"]) >= 2
        ],
    }
    token_json = json.dumps(token or "")
    data_json = html_escape_json(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cesium 3D Terrain - Cadet {cadet} - {drill}</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/{CESIUM_VERSION}/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/{CESIUM_VERSION}/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <style>
    html, body, #cesiumContainer {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      font-family: Arial, sans-serif;
    }}
    .panel {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1;
      background: rgba(20, 24, 30, 0.78);
      color: #fff;
      padding: 12px 14px;
      border-radius: 6px;
      max-width: 360px;
      box-shadow: 0 2px 10px rgba(0,0,0,.25);
    }}
    .panel h1 {{
      font-size: 16px;
      margin: 0 0 8px;
      font-weight: 600;
    }}
    .panel .meta {{
      font-size: 12px;
      opacity: .82;
      margin-bottom: 8px;
    }}
    .segment-row {{
      display: grid;
      grid-template-columns: 16px 1fr;
      gap: 8px;
      align-items: center;
      margin: 4px 0;
      font-size: 13px;
    }}
    .segment-row input {{
      margin: 0;
    }}
    .terrain-note {{
      margin-top: 8px;
      font-size: 11px;
      opacity: .72;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>
  <div class="panel">
    <h1>Cadet {cadet} · Sortie {sortie} · {drill}</h1>
    <div class="meta">Esri orthophoto imagery · DEM-derived aircraft MSL · {len(payload["segments"])} segments</div>
    <div id="segmentControls"></div>
    <div class="terrain-note" id="terrainNote"></div>
  </div>
  <script>
    const data = {data_json};
    const ionToken = {token_json};
    Cesium.Ion.defaultAccessToken = ionToken;

    async function init() {{
      let terrainProvider = new Cesium.EllipsoidTerrainProvider();
      let terrainNote = "Using ellipsoid globe surface. Flight altitude is still computed from local Copernicus 30 m DEM.";
      if (ionToken) {{
        try {{
          terrainProvider = await Cesium.createWorldTerrainAsync();
          terrainNote = "Using Cesium Ion world terrain. Flight altitude is computed from local Copernicus 30 m DEM.";
        }} catch (error) {{
          console.warn("Could not load Cesium Ion terrain:", error);
        }}
      }}

      const viewer = new Cesium.Viewer("cesiumContainer", {{
        terrainProvider,
        baseLayerPicker: false,
        geocoder: false,
        homeButton: true,
        sceneModePicker: true,
        navigationHelpButton: false,
        animation: false,
        timeline: false,
        fullscreenButton: true,
        infoBox: true,
        selectionIndicator: true
      }});

      viewer.imageryLayers.removeAll();
      viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({{
        url: "{ESRI_WORLD_IMAGERY}",
        maximumLevel: 19,
        credit: "Esri World Imagery"
      }}));

      viewer.scene.globe.depthTestAgainstTerrain = false;
      viewer.scene.skyAtmosphere.show = true;
      viewer.scene.globe.enableLighting = false;
      document.getElementById("terrainNote").textContent = terrainNote;

      const controls = document.getElementById("segmentControls");
      const entitiesBySegment = new Map();

      for (const segment of data.segments) {{
        const values = [];
        for (const point of segment.points) {{
          values.push(point.lon, point.lat, point.aircraft_msl_m);
        }}
        const label = segment.segment === null ? "No segment" : `Segment ${{segment.segment}}`;
        const entity = viewer.entities.add({{
          name: label,
          description: `Cadet ${{data.cadet}}<br>Sortie ${{data.sortie}}<br>Drill ${{data.drill}}`,
          polyline: {{
            positions: Cesium.Cartesian3.fromDegreesArrayHeights(values),
            width: 5,
            material: Cesium.Color.fromCssColorString(segment.color),
            clampToGround: false
          }}
        }});
        entitiesBySegment.set(label, entity);

        const row = document.createElement("label");
        row.className = "segment-row";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.addEventListener("change", () => {{
          entity.show = checkbox.checked;
        }});
        const text = document.createElement("span");
        text.textContent = label;
        text.style.color = segment.color;
        row.appendChild(checkbox);
        row.appendChild(text);
        controls.appendChild(row);
      }}

      const first = data.segments[0].points[0];
      const lastSegment = data.segments[data.segments.length - 1];
      const last = lastSegment.points[lastSegment.points.length - 1];
      viewer.entities.add({{
        name: "START",
        position: Cesium.Cartesian3.fromDegrees(first.lon, first.lat, first.aircraft_msl_m),
        point: {{ pixelSize: 12, color: Cesium.Color.LIME }},
        label: {{ text: "START", font: "14px sans-serif", fillColor: Cesium.Color.WHITE, pixelOffset: new Cesium.Cartesian2(0, -20) }}
      }});
      viewer.entities.add({{
        name: "END",
        position: Cesium.Cartesian3.fromDegrees(last.lon, last.lat, last.aircraft_msl_m),
        point: {{ pixelSize: 12, color: Cesium.Color.RED }},
        label: {{ text: "END", font: "14px sans-serif", fillColor: Cesium.Color.WHITE, pixelOffset: new Cesium.Cartesian2(0, -20) }}
      }});

      await viewer.zoomTo(viewer.entities);
      viewer.camera.flyTo({{
        destination: Cesium.Cartesian3.fromDegrees(data.center.lon, data.center.lat, data.center.height),
        orientation: {{
          heading: Cesium.Math.toRadians(0),
          pitch: Cesium.Math.toRadians(-55),
          roll: 0
        }},
        duration: 0.6
      }});
    }}

    init();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Create a Cesium 3D route view with crisp tiled orthophoto imagery.")
    parser.add_argument("cadet", type=int)
    parser.add_argument("--sortie", type=int, default=1)
    parser.add_argument("--drill", default="DoubleEight")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("flight_school.db"))
    parser.add_argument("--dem", type=Path, default=DEM_PATH)
    parser.add_argument("--output", type=Path, help="HTML output path")
    parser.add_argument(
        "--cesium-ion-token",
        default=os.environ.get("CESIUM_ION_TOKEN", DEFAULT_CESIUM_ION_TOKEN),
        help="optional Cesium Ion token for live 3D terrain",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = EXPORTS_DIR / f"cadet_{args.cadet}_sortie_{args.sortie}_{args.drill}_terrain_3d_cesium.html"

    download_dem(args.dem)
    connection = sqlite3.connect(args.database)
    route, coordinate_modes = load_route(connection, args.cadet, args.sortie, args.drill)
    route = route_with_altitude(route, args.dem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_html(route, coordinate_modes, args.cadet, args.sortie, args.drill, args.cesium_ion_token),
        encoding="utf-8",
    )
    print(f"Saved Cesium 3D terrain view: {args.output}")
    print(f'Open HTML: xdg-open "{args.output.resolve()}"')


if __name__ == "__main__":
    main()
