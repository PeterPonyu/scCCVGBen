#!/usr/bin/env python3
"""build_datasets_json.py — Convert datasets.csv to site/data/datasets.json.

datasets.csv is the source of truth; this script generates the JSON artifact
consumed by the Hugo static site.

Usage:
    python scripts/build_datasets_json.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "scccvgben" / "data" / "datasets.csv"
DEFAULT_OUT = REPO_ROOT / "site" / "data" / "datasets.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help="Path to datasets.csv (default: scccvgben/data/datasets.csv)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output JSON path (default: site/data/datasets.json)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"datasets.csv not found at {csv_path}. "
            "Run `python scripts/build_datasets_csv.py --build-canonical` first."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Coerce numeric fields
            if row.get("cell_count"):
                try:
                    row["cell_count"] = int(row["cell_count"])
                except ValueError:
                    pass
            rows.append(row)

    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent=2)

    print(f"Wrote {len(rows)} dataset records -> {out_path}")
    # Validate: ensure jq-parseable (re-parse)
    with open(out_path) as fh:
        parsed = json.load(fh)
    assert len(parsed) == len(rows), "JSON round-trip mismatch"
    print("JSON validation: OK")


if __name__ == "__main__":
    main()
