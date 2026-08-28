#!/usr/bin/env python3
"""Import flight-simulator CSV exports into a local SQLite database."""
import argparse, csv, hashlib, json, math, re, sqlite3, zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

DRILLS = {"שמינייה": "Eight", "2 X שמינייה": "DoubleEight", "תבנית": "Pattern",
          "2 X תבנית": "DoublePattern", "ציר נלמד": "LearnedTrack",
          "ציר משימה": "MissionTrack", "פיינל": "Final", 'ינ"פ': "YNP",
          '2 X ינ"פ': "DoubleYNP", "פנייה מיטבית": "OptimalTurn",
          "2 X פנייה מיטבית": "DoubleOptimalTurn"}
HEADER_MAP = {"זמן": "time", "מהירות (IAS)": "ias_knots", "כיוון אף": "heading",
              "גלגול": "roll_roughness", "עלרוד": "pitch_roughness", "כוח": "collective_roughness"}
REQUIRED = ("time", "height", "ias_knots", "heading", "rollModel", "pitchModel", "RollStick", "PitchStick", "COLL")
FILENAME = re.compile(r"^c(?P<cadet>\d+)_(?P<sortie>\d+)_(?P<drill>.+)\.(?:csv|xlsx)$", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS imports (
 id INTEGER PRIMARY KEY, filename TEXT NOT NULL, file_hash TEXT NOT NULL,
 cadet_number INTEGER NOT NULL, sortie_number INTEGER NOT NULL, drill_code TEXT NOT NULL,
 imported_at TEXT NOT NULL, status TEXT NOT NULL, row_count INTEGER NOT NULL,
 complete INTEGER NOT NULL, report_json TEXT NOT NULL,
 UNIQUE(cadet_number, sortie_number, drill_code)
);
CREATE TABLE IF NOT EXISTS telemetry_samples (
 id INTEGER PRIMARY KEY, import_id INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
 source_row INTEGER NOT NULL, timestamp REAL NOT NULL, segment_number INTEGER,
 height_ft REAL, ias_knots REAL, heading_degrees REAL, roll_degrees REAL, pitch_degrees REAL,
 roll_stick REAL, pitch_stick REAL, collective REAL, interpolated_fields TEXT NOT NULL,
 payload_json TEXT NOT NULL, UNIQUE(import_id, source_row)
);
"""

def blank(value):
    return value is None or str(value).strip() == "" or str(value).strip().lower() == "nan"

def numeric(value):
    if blank(value): return None
    try: return float(value)
    except ValueError: return None

def parse_filename(path):
    match = FILENAME.match(path.name)
    if not match: raise ValueError("filename must be c<cadet>_<sortie>_<Hebrew drill name>.csv")
    drill_name = match.group("drill").strip()
    if drill_name not in DRILLS: raise ValueError(f"unsupported drill name: {drill_name}")
    return int(match.group("cadet")), int(match.group("sortie")), drill_name, DRILLS[drill_name]

def normalize(row):
    out = {(HEADER_MAP.get(k, k)): v for k, v in row.items()}
    for key in list(out):
        value = numeric(out[key])
        if value is not None: out[key] = value
    return out

def interpolate(rows, field, report):
    values = [numeric(row.get(field)) for row in rows]
    if all(v is None for v in values):
        report["incomplete"].append({"field": field, "reason": "missing column or all values"})
        return
    i = 0
    while i < len(values):
        if values[i] is not None: i += 1; continue
        start = i
        while i < len(values) and values[i] is None: i += 1
        end, count = i, i - start
        if count <= 10 and start > 0 and end < len(values) and values[start-1] is not None and values[end] is not None:
            left, right = values[start-1], values[end]
            for offset in range(count):
                value = left + (right-left) * (offset+1)/(count+1)
                rows[start+offset][field] = value
                rows[start+offset].setdefault("_interpolated", []).append(field)
            report["interpolated"].append({"field": field, "start_row": start+2, "end_row": end+1, "count": count})
        else:
            report["incomplete"].append({"field": field, "start_row": start+2, "end_row": end+1, "count": count})

def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [normalize(row) for row in csv.DictReader(handle)]
    if not rows: raise ValueError("CSV has no data rows")
    return rows

def load_xlsx(path):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.findall(".//m:t", ns))
                      for item in shared_root.findall("m:si", ns)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    parsed = []
    for row in root.findall(".//m:sheetData/m:row", ns):
        values = {}
        for cell in row.findall("m:c", ns):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group(0)
            value_node = cell.find("m:v", ns)
            value = "" if value_node is None else value_node.text
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[column] = value
        parsed.append(values)
    if len(parsed) < 2: raise ValueError("XLSX has no data rows")
    headers = parsed[0]
    return [normalize({headers.get(column, column): value for column, value in row.items()})
            for row in parsed[1:]]

def load_data(path):
    if path.suffix.lower() == ".csv": return load_csv(path)
    if path.suffix.lower() == ".xlsx": return load_xlsx(path)
    raise ValueError("supported formats are CSV and XLSX")

def import_file(conn, path):
    cadet, sortie, drill_name, drill_code = parse_filename(path)
    rows, report = load_data(path), {"interpolated": [], "incomplete": [], "errors": []}
    for field in REQUIRED: interpolate(rows, field, report)
    complete = not report["incomplete"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM imports WHERE cadet_number=? AND sortie_number=? AND drill_code=?", (cadet, sortie, drill_code))
        cur = conn.execute("""INSERT INTO imports
          (filename,file_hash,cadet_number,sortie_number,drill_code,imported_at,status,row_count,complete,report_json)
          VALUES (?,?,?,?,?,?,?,?,?,?)""",
          (path.name,digest,cadet,sortie,drill_code,datetime.now(timezone.utc).isoformat(),
           "complete" if complete else "incomplete",len(rows),int(complete),json.dumps(report, ensure_ascii=False)))
        import_id = cur.lastrowid
        for source_row, row in enumerate(rows, start=2):
            heading = numeric(row.get("heading"))
            if heading is not None and drill_code in {"Eight","DoubleEight","Pattern","DoublePattern"}: heading_degrees = heading
            elif heading is not None: heading_degrees = math.degrees(heading)
            else: heading_degrees = None
            roll = numeric(row.get("rollModel"))
            pitch = numeric(row.get("pitchModel"))
            conn.execute("""INSERT INTO telemetry_samples
              (import_id,source_row,timestamp,segment_number,height_ft,ias_knots,heading_degrees,roll_degrees,pitch_degrees,roll_stick,pitch_stick,collective,interpolated_fields,payload_json)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (import_id, source_row, numeric(row.get("time")), numeric(row.get("segment")), numeric(row.get("height")),
               numeric(row.get("ias_knots")), heading_degrees, math.degrees(roll) if roll is not None else None,
               math.degrees(pitch) if pitch is not None else None, numeric(row.get("RollStick")), numeric(row.get("PitchStick")),
               numeric(row.get("COLL")), json.dumps(row.get("_interpolated", [])), json.dumps(row, ensure_ascii=False)))
        conn.commit()
    except Exception:
        conn.rollback(); raise
    return {"file": path.name, "rows": len(rows), "status": "complete" if complete else "incomplete", "report": report}

def export_csv_tables(conn, destination):
    destination.mkdir(parents=True, exist_ok=True)
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for table in tables:
        if table == "telemetry_samples":
            continue
        columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
        with (destination / f"{table}.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(conn.execute(f'SELECT * FROM "{table}"'))
    base_columns = ["Cadet", "Sortie", "Drill", "Source row", "Interpolated fields"]
    records_by_drill = {}
    query = "SELECT i.cadet_number, i.sortie_number, i.drill_code, t.source_row, t.interpolated_fields, t.payload_json FROM telemetry_samples t JOIN imports i ON i.id=t.import_id ORDER BY i.cadet_number, i.sortie_number, i.drill_code, t.source_row"
    for cadet, sortie, drill, source_row, interpolated, payload_json in conn.execute(query):
        payload = json.loads(payload_json)
        records_by_drill.setdefault(drill, []).append((cadet, sortie, source_row, interpolated, payload))
    for drill, records in records_by_drill.items():
        payload_fields = set().union(*(payload.keys() for *_, payload in records))
        payload_columns = sorted(field for field in payload_fields if field not in {"_interpolated", "drill"})
        with (destination / f"{drill}.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(base_columns + payload_columns)
            for cadet, sortie, source_row, interpolated, payload in records:
                values = [cadet, sortie, drill, source_row, interpolated]
                values += [payload.get(field) for field in payload_columns]
                writer.writerow([round(value, 4) if isinstance(value, float) else value for value in values])
    return tables

def main():
    parser = argparse.ArgumentParser(description="Import flight-simulator CSV files into SQLite.")
    parser.add_argument("folder", type=Path, nargs="?", help="folder containing CSV exports")
    parser.add_argument("--database", type=Path, default=Path(__file__).with_name("flight_school.db"))
    parser.add_argument(
        "--export-csv",
        type=Path,
        nargs="?",
        const=Path(__file__).resolve().parent.parent / "exports" / "csv",
        metavar="FOLDER",
        help="export database tables as CSV files; defaults to exports/csv",
    )
    args = parser.parse_args()
    conn = sqlite3.connect(args.database)
    conn.executescript("PRAGMA foreign_keys=ON;" + SCHEMA)
    if args.export_csv:
        tables = export_csv_tables(conn, args.export_csv.expanduser())
        print(f"Exported {len(tables)} tables to: {args.export_csv.expanduser()}")
        return
    if args.folder is None:
        print("Flight Simulator CSV Importer")
        entered_path = input("Folder containing CSV files: ").strip()
        if not entered_path:
            parser.error("a folder is required")
        args.folder = Path(entered_path).expanduser()
    if not args.folder.is_dir():
        parser.error(f"folder not found: {args.folder}")
    files = sorted([*args.folder.glob("*.csv"), *args.folder.glob("*.xlsx")])
    if not files: parser.error("no CSV or XLSX files found")
    results = []
    for file in files:
        try: results.append(import_file(conn, file))
        except Exception as error: results.append({"file": file.name, "status": "rejected", "error": str(error)})
    for result in results:
        if result["status"] == "rejected":
            print(f"Rejected: {result['file']} — {result['error']}")
        else:
            print(f"{result['status'].title()}: {result['file']} ({result['rows']} rows)")
    print(f"\nDatabase: {args.database}")

if __name__ == "__main__":
    main()
