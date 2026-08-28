#!/usr/bin/env python3
"""Desktop launcher for the flight simulator importer."""
import sys
import subprocess
import threading
from pathlib import Path
try:
    from tkinter import BOTH, END, LEFT, RIGHT, Button, Label, StringVar, Text, Tk, filedialog, messagebox
    from tkinter import ttk
except ModuleNotFoundError:
    print("The desktop UI requires tkinter. Install it with: sudo apt install python3-tk")
    print("Or use the command-line importer: python3 code/import_flight_data.py <folder>")
    raise SystemExit(1)

CODE_DIR = Path(__file__).parent / "code"
sys.path.insert(0, str(CODE_DIR))
from import_flight_data import SCHEMA, import_file
import sqlite3


class ImporterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flight Simulator Data Importer")
        self.root.geometry("720x460")
        self.folder = StringVar()
        self.status = StringVar(value="Choose a folder containing CSV or XLSX drill exports.")

        frame = ttk.Frame(root, padding=18)
        frame.pack(fill=BOTH, expand=True)

        ttk.Label(frame, text="Flight Simulator Data Importer", font=("", 18, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Select the folder containing drill export files.").pack(anchor="w", pady=(4, 16))

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.folder, state="readonly").pack(side=LEFT, fill="x", expand=True)
        ttk.Button(row, text="Choose folder", command=self.choose_folder).pack(side=RIGHT, padx=(8, 0))

        self.import_button = ttk.Button(frame, text="Import files", command=self.start_import, state="disabled")
        self.import_button.pack(anchor="w", pady=14)

        ttk.Label(frame, textvariable=self.status).pack(anchor="w")
        self.report = Text(frame, height=16, wrap="word", state="disabled")
        self.report.pack(fill=BOTH, expand=True, pady=(8, 0))

    def choose_folder(self):
        selected = filedialog.askdirectory(title="Choose drill export folder")
        if selected:
            self.folder.set(selected)
            self.import_button.configure(state="normal")
            self.status.set("Ready to import.")

    def write(self, message):
        self.report.configure(state="normal")
        self.report.insert(END, message + "\n")
        self.report.see(END)
        self.report.configure(state="disabled")

    def start_import(self):
        folder = Path(self.folder.get())
        if not folder.is_dir():
            messagebox.showerror("Folder unavailable", "Choose an existing folder.")
            return
        self.import_button.configure(state="disabled")
        self.status.set("Importing files…")
        self.report.configure(state="normal")
        self.report.delete("1.0", END)
        self.report.configure(state="disabled")
        threading.Thread(target=self.import_files, args=(folder,), daemon=True).start()

    def import_files(self, folder):
        database = CODE_DIR / "flight_school.db"
        files = sorted([*folder.glob("*.csv"), *folder.glob("*.xlsx")])
        if not files:
            self.root.after(0, lambda: self.finished("No CSV or XLSX files were found.", error=True))
            return
        conn = sqlite3.connect(database)
        conn.executescript("PRAGMA foreign_keys=ON;" + SCHEMA)
        complete = incomplete = rejected = 0
        for file in files:
            try:
                result = import_file(conn, file)
                if result["status"] == "complete":
                    complete += 1
                else:
                    incomplete += 1
                self.root.after(0, lambda r=result: self.write(
                    f"{r['status'].title()}: {r['file']} ({r['rows']} rows)"))
            except Exception as error:
                rejected += 1
                self.root.after(0, lambda f=file, e=error: self.write(f"Rejected: {f.name} — {e}"))
        conn.close()
        summary = f"Finished: {complete} complete, {incomplete} incomplete, {rejected} rejected. Database: {database}"
        self.root.after(0, lambda: self.finished(summary))

    def finished(self, message, error=False):
        self.status.set(message)
        self.import_button.configure(state="normal")
        if error:
            messagebox.showerror("Import not started", message)
        else:
            self.write("")
            self.write(message)


if __name__ == "__main__":
    root = Tk()
    ImporterApp(root)
    root.mainloop()
