#!/usr/bin/env python3
"""Create a 3D flight path over 30 m Copernicus DEM terrain."""
import argparse
import json
import math
import sqlite3
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from god_view_map import normalize_coordinate, numeric


PROJECT_DIR = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_DIR / "exports" / "terrain_3d"
TERRAIN_DIR = PROJECT_DIR / "terrain"
ORTHOPHOTO_CACHE_DIR = TERRAIN_DIR / "orthophoto_tiles"
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N31_00_E034_00_DEM/Copernicus_DSM_COG_10_N31_00_E034_00_DEM.tif"
DEM_PATH = TERRAIN_DIR / "Copernicus_DSM_COG_10_N31_00_E034_00_DEM.tif"
ORTHOPHOTO_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
EARTH_RADIUS_M = 6371000.0
FEET_TO_METERS = 0.3048
TILE_SIZE = 256


def download_dem(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    print(f"Downloading 30 m DEM: {DEM_URL}")
    urllib.request.urlretrieve(DEM_URL, path)


def load_route(connection, cadet, sortie, drill):
    rows = connection.execute(
        """SELECT t.source_row, t.segment_number, t.height_ft, t.payload_json
           FROM telemetry_samples t
           JOIN imports i ON i.id = t.import_id
           WHERE i.cadet_number=? AND i.sortie_number=? AND i.drill_code=? AND i.complete=1
           ORDER BY t.source_row""",
        (cadet, sortie, drill),
    ).fetchall()
    route = []
    coordinate_modes = set()
    for source_row, segment, height_ft, payload_json in rows:
        payload = json.loads(payload_json or "{}")
        lat, lon, coordinate_mode = normalize_coordinate(
            numeric(payload.get("lat")),
            numeric(payload.get("lon")),
            numeric(payload.get("latDeg")),
            numeric(payload.get("lonDeg")),
            drill,
        )
        if lat is None or lon is None:
            continue
        coordinate_modes.add(coordinate_mode)
        route.append(
            {
                "source_row": source_row,
                "segment": int(segment) if segment is not None else None,
                "lat": lat,
                "lon": lon,
                "height_ft": height_ft,
            }
        )
    if len(route) < 2:
        raise SystemExit("Not enough geographic route points found.")
    return route, sorted(coordinate_modes)


def read_geotiff(path):
    image = Image.open(path)
    tags = image.tag_v2
    pixel_scale = tags.get(33550)
    tiepoint = tags.get(33922)
    if not pixel_scale or not tiepoint:
        raise SystemExit("DEM GeoTIFF is missing georeference tags.")
    lon_origin = float(tiepoint[3])
    lat_origin = float(tiepoint[4])
    lon_step = float(pixel_scale[0])
    lat_step = -float(pixel_scale[1])
    elevation = np.array(image, dtype=float)
    elevation[elevation < -1000] = np.nan
    return elevation, lon_origin, lat_origin, lon_step, lat_step


def latlon_to_pixel(lat, lon, lon_origin, lat_origin, lon_step, lat_step):
    col = (lon - lon_origin) / lon_step
    row = (lat - lat_origin) / lat_step
    return row, col


def bilinear_sample(elevation, row, col):
    row0 = int(math.floor(row))
    col0 = int(math.floor(col))
    row1 = min(row0 + 1, elevation.shape[0] - 1)
    col1 = min(col0 + 1, elevation.shape[1] - 1)
    if row0 < 0 or col0 < 0 or row0 >= elevation.shape[0] or col0 >= elevation.shape[1]:
        return None
    row_fraction = row - row0
    col_fraction = col - col0
    q00 = elevation[row0, col0]
    q10 = elevation[row1, col0]
    q01 = elevation[row0, col1]
    q11 = elevation[row1, col1]
    if np.isnan([q00, q10, q01, q11]).any():
        return None
    return (
        q00 * (1 - row_fraction) * (1 - col_fraction)
        + q10 * row_fraction * (1 - col_fraction)
        + q01 * (1 - row_fraction) * col_fraction
        + q11 * row_fraction * col_fraction
    )


def crop_dem(elevation, route, lon_origin, lat_origin, lon_step, lat_step, padding_pixels, max_grid):
    pixels = [
        latlon_to_pixel(point["lat"], point["lon"], lon_origin, lat_origin, lon_step, lat_step)
        for point in route
    ]
    min_row = max(0, int(math.floor(min(row for row, _ in pixels))) - padding_pixels)
    max_row = min(elevation.shape[0] - 1, int(math.ceil(max(row for row, _ in pixels))) + padding_pixels)
    min_col = max(0, int(math.floor(min(col for _, col in pixels))) - padding_pixels)
    max_col = min(elevation.shape[1] - 1, int(math.ceil(max(col for _, col in pixels))) + padding_pixels)
    row_step = max(1, math.ceil((max_row - min_row + 1) / max_grid))
    col_step = max(1, math.ceil((max_col - min_col + 1) / max_grid))
    rows = np.arange(min_row, max_row + 1, row_step)
    cols = np.arange(min_col, max_col + 1, col_step)
    terrain = elevation[np.ix_(rows, cols)]
    lats = lat_origin + rows * lat_step
    lons = lon_origin + cols * lon_step
    return terrain, lats, lons


def to_local_meters(lat, lon, center_lat, center_lon):
    center_lat_rad = math.radians(center_lat)
    x = math.radians(lon - center_lon) * EARTH_RADIUS_M * math.cos(center_lat_rad)
    y = math.radians(lat - center_lat) * EARTH_RADIUS_M
    return x, y


def build_terrain_grid(terrain, lats, lons, center_lat, center_lon):
    x_values = []
    y_values = []
    for lat in lats:
        y_values.append(to_local_meters(lat, lons[0], center_lat, center_lon)[1])
    for lon in lons:
        x_values.append(to_local_meters(lats[0], lon, center_lat, center_lon)[0])
    x_grid, y_grid = np.meshgrid(np.array(x_values), np.array(y_values))
    return x_grid, y_grid


def axis_range(values):
    return float(np.nanmax(values) - np.nanmin(values))


def scene_aspect(x_grid, y_grid, terrain, flight_z, vertical_exaggeration):
    x_range = max(axis_range(x_grid), 1.0)
    y_range = max(axis_range(y_grid), 1.0)
    terrain_min = float(np.nanmin(terrain))
    z_max = max(float(np.nanmax(terrain)), max(flight_z))
    z_range = max(z_max - terrain_min, 1.0)
    base = max(x_range, y_range)
    return dict(
        x=x_range / base,
        y=y_range / base,
        z=max((z_range / base) * vertical_exaggeration, 0.02),
    )


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def load_tile(x, y, zoom):
    tile_path = ORTHOPHOTO_CACHE_DIR / str(zoom) / str(x) / f"{y}.jpg"
    if tile_path.exists():
        return Image.open(tile_path).convert("RGB")
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    url = ORTHOPHOTO_URL.format(z=zoom, x=x, y=y)
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read()
    tile_path.write_bytes(data)
    return Image.open(BytesIO(data)).convert("RGB")


def orthophoto_color(lat, lon, zoom, tile_cache):
    tile_x, tile_y = latlon_to_tile(lat, lon, zoom)
    x_index = int(math.floor(tile_x))
    y_index = int(math.floor(tile_y))
    pixel_x = int((tile_x - x_index) * TILE_SIZE)
    pixel_y = int((tile_y - y_index) * TILE_SIZE)
    key = (zoom, x_index, y_index)
    if key not in tile_cache:
        try:
            tile_cache[key] = load_tile(x_index, y_index, zoom)
        except Exception:
            tile_cache[key] = None
    tile = tile_cache[key]
    if tile is None:
        return "rgb(120,120,120)"
    pixel_x = max(0, min(TILE_SIZE - 1, pixel_x))
    pixel_y = max(0, min(TILE_SIZE - 1, pixel_y))
    r, g, b = tile.getpixel((pixel_x, pixel_y))
    return f"rgb({r},{g},{b})"


def add_dem_surface(fig, x_grid, y_grid, terrain):
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=terrain,
            colorscale="Earth",
            opacity=0.92,
            showscale=True,
            colorbar=dict(title="Terrain MSL (m)"),
            name="30 m DEM terrain",
            hovertemplate="East %{x:.1f} m<br>North %{y:.1f} m<br>Terrain %{z:.1f} m<extra></extra>",
        )
    )


def add_orthophoto_mesh(fig, x_grid, y_grid, terrain, lats, lons, zoom):
    print(f"Sampling orthophoto tiles at zoom {zoom}...")
    vertex_x = []
    vertex_y = []
    vertex_z = []
    vertex_color = []
    tile_cache = {}
    for row_index, lat in enumerate(lats):
        for col_index, lon in enumerate(lons):
            vertex_x.append(float(x_grid[row_index, col_index]))
            vertex_y.append(float(y_grid[row_index, col_index]))
            vertex_z.append(float(terrain[row_index, col_index]))
            vertex_color.append(orthophoto_color(float(lat), float(lon), zoom, tile_cache))

    i_values = []
    j_values = []
    k_values = []
    row_count, col_count = terrain.shape
    for row in range(row_count - 1):
        for col in range(col_count - 1):
            p00 = row * col_count + col
            p01 = p00 + 1
            p10 = (row + 1) * col_count + col
            p11 = p10 + 1
            i_values.extend([p00, p01])
            j_values.extend([p10, p10])
            k_values.extend([p01, p11])

    fig.add_trace(
        go.Mesh3d(
            x=vertex_x,
            y=vertex_y,
            z=vertex_z,
            i=i_values,
            j=j_values,
            k=k_values,
            vertexcolor=vertex_color,
            name="30 m DEM with orthophoto",
            opacity=1.0,
            flatshading=False,
            hovertemplate="East %{x:.1f} m<br>North %{y:.1f} m<br>Terrain %{z:.1f} m<extra></extra>",
        )
    )
    print(f"Orthophoto tiles used: {sum(1 for tile in tile_cache.values() if tile is not None)}")


def route_3d(route, elevation, lon_origin, lat_origin, lon_step, lat_step, center_lat, center_lon):
    x_values = []
    y_values = []
    terrain_z = []
    flight_z = []
    segments = []
    source_rows = []
    for point in route:
        row, col = latlon_to_pixel(point["lat"], point["lon"], lon_origin, lat_origin, lon_step, lat_step)
        ground_m = bilinear_sample(elevation, row, col)
        if ground_m is None:
            continue
        height_ft = point["height_ft"] if point["height_ft"] is not None else 0.0
        x, y = to_local_meters(point["lat"], point["lon"], center_lat, center_lon)
        x_values.append(x)
        y_values.append(y)
        terrain_z.append(ground_m)
        flight_z.append(ground_m + height_ft * FEET_TO_METERS)
        segments.append(point["segment"])
        source_rows.append(point["source_row"])
    return x_values, y_values, terrain_z, flight_z, segments, source_rows


def split_route_by_segment(x, y, ground_z, flight_z, segments, source_rows):
    grouped = []
    current_segment = None
    current_points = None
    for point_x, point_y, point_ground_z, point_flight_z, segment, source_row in zip(
        x, y, ground_z, flight_z, segments, source_rows
    ):
        if current_points is None or segment != current_segment:
            current_segment = segment
            current_points = {"segment": segment, "x": [], "y": [], "ground_z": [], "flight_z": [], "source_rows": []}
            grouped.append(current_points)
        current_points["x"].append(point_x)
        current_points["y"].append(point_y)
        current_points["ground_z"].append(point_ground_z)
        current_points["flight_z"].append(point_flight_z)
        current_points["source_rows"].append(source_row)
    return grouped


def build_plot(route, coordinate_modes, dem_path, output, cadet, sortie, drill, padding_pixels, max_grid, orthophoto_texture, ortho_zoom, vertical_exaggeration):
    elevation, lon_origin, lat_origin, lon_step, lat_step = read_geotiff(dem_path)
    center_lat = sum(point["lat"] for point in route) / len(route)
    center_lon = sum(point["lon"] for point in route) / len(route)
    terrain, lats, lons = crop_dem(elevation, route, lon_origin, lat_origin, lon_step, lat_step, padding_pixels, max_grid)
    x_grid, y_grid = build_terrain_grid(terrain, lats, lons, center_lat, center_lon)
    x, y, ground_z, flight_z, segments, source_rows = route_3d(
        route, elevation, lon_origin, lat_origin, lon_step, lat_step, center_lat, center_lon
    )
    if len(x) < 2:
        raise SystemExit("Route does not overlap the DEM tile.")
    aspect = scene_aspect(x_grid, y_grid, terrain, flight_z, vertical_exaggeration)

    fig = go.Figure()
    if orthophoto_texture:
        add_orthophoto_mesh(fig, x_grid, y_grid, terrain, lats, lons, ortho_zoom)
    else:
        add_dem_surface(fig, x_grid, y_grid, terrain)
    segment_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for index, segment_route in enumerate(split_route_by_segment(x, y, ground_z, flight_z, segments, source_rows)):
        segment = segment_route["segment"]
        name = f"Segment {segment}" if segment is not None else "No segment"
        fig.add_trace(
            go.Scatter3d(
                x=segment_route["x"],
                y=segment_route["y"],
                z=segment_route["flight_z"],
                mode="lines",
                name=name,
                line=dict(width=8, color=segment_colors[index % len(segment_colors)]),
                customdata=list(zip(segment_route["source_rows"], segment_route["ground_z"])),
                hovertemplate=(
                    "East %{x:.1f} m<br>"
                    "North %{y:.1f} m<br>"
                    "Aircraft MSL %{z:.1f} m<br>"
                    "Ground MSL %{customdata[1]:.1f} m<br>"
                    f"{name}<br>"
                    "Source row %{customdata[0]}<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[x[0], x[-1]],
            y=[y[0], y[-1]],
            z=[flight_z[0], flight_z[-1]],
            mode="markers+text",
            text=["START", "END"],
            textposition="top center",
            name="Start / End",
            marker=dict(size=5, color=["#00aa55", "#cc0033"]),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=(
            f"30 m Terrain 3D - Cadet {cadet} - Sortie {sortie} - {drill}"
            f"<br><sup>DEM: Copernicus N31E034; coordinates: {', '.join(coordinate_modes)}</sup>"
        ),
        height=900,
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(title="Toggle segments", orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0.0, groupclick="toggleitem"),
        scene=dict(
            xaxis_title="East from route center (m)",
            yaxis_title="North from route center (m)",
            zaxis_title="MSL altitude (m)",
            aspectmode="manual",
            aspectratio=aspect,
        ),
        template="plotly_white",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description="Create a Plotly 3D route-over-terrain HTML file.")
    parser.add_argument("cadet", type=int)
    parser.add_argument("--sortie", type=int, default=1)
    parser.add_argument("--drill", default="DoubleEight")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("flight_school.db"))
    parser.add_argument("--dem", type=Path, default=DEM_PATH)
    parser.add_argument("--output", type=Path, help="HTML output path")
    parser.add_argument("--padding-pixels", type=int, default=80)
    parser.add_argument("--max-grid", type=int, default=180, help="maximum terrain samples per grid side")
    parser.add_argument("--orthophoto-texture", action="store_true", help="drape satellite imagery over the DEM mesh")
    parser.add_argument("--ortho-zoom", type=int, default=15, help="orthophoto tile zoom used with --orthophoto-texture")
    parser.add_argument("--vertical-exaggeration", type=float, default=1.0, help="visual multiplier for z scale; 1.0 means true metre-to-metre scale")
    args = parser.parse_args()
    if args.output is None:
        suffix = "terrain_3d_orthophoto" if args.orthophoto_texture else "terrain_3d"
        args.output = EXPORTS_DIR / f"cadet_{args.cadet}_sortie_{args.sortie}_{args.drill}_{suffix}.html"

    download_dem(args.dem)
    connection = sqlite3.connect(args.database)
    route, coordinate_modes = load_route(connection, args.cadet, args.sortie, args.drill)
    build_plot(
        route,
        coordinate_modes,
        args.dem,
        args.output,
        args.cadet,
        args.sortie,
        args.drill,
        args.padding_pixels,
        args.max_grid,
        args.orthophoto_texture,
        args.ortho_zoom,
        args.vertical_exaggeration,
    )
    print(f"Saved 3D terrain plot: {args.output}")
    print(f'Open HTML: xdg-open "{args.output.resolve()}"')


if __name__ == "__main__":
    main()
