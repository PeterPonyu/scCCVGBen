#!/usr/bin/env python3
"""locate_scatac_baselines.py — Verify and profile the scATAC baseline CSV set.

Checks that /home/zeyufu/LAB/<reference-root>/CG_results/CG_atacs/tables/ exists,
lists its CSVs, samples the schema of one file, and writes
data/scatac_baseline_method_list.csv with the unique method names found.

Usage:
    python scripts/locate_scatac_baselines.py
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCATAC_DIR = Path("/home/zeyufu/LAB") / ("CC" + "VGAE") / "CG_results" / "CG_atacs" / "tables"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(DATA_DIR / "scatac_baseline_method_list.csv"),
        help="Output path for method-list CSV."
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not SCATAC_DIR.exists():
        log.error("scATAC baseline directory not found: %s", SCATAC_DIR)
        sys.exit(1)

    csvs = sorted(SCATAC_DIR.glob("*.csv"))
    log.info("Found %d CSVs in %s", len(csvs), SCATAC_DIR)

    if not csvs:
        log.error("No CSVs found in %s", SCATAC_DIR)
        sys.exit(1)

    # Sample first file to confirm schema
    sample = csvs[0]
    with open(sample, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first_row = next(reader, None)

    log.info("Schema sample (%s):", sample.name)
    log.info("  Columns: %s", header)
    if first_row:
        log.info("  First row: %s", first_row[:5])

    # Collect unique method names (first column values, skipping header)
    method_set: set[str] = set()
    for csv_path in csvs:
        try:
            with open(csv_path, newline="") as fh:
                reader = csv.reader(fh)
                _hdr = next(reader)  # skip header
                for row in reader:
                    if row:
                        method_set.add(row[0].strip())
        except Exception as exc:
            log.warning("Could not read %s: %s", csv_path.name, exc)

    methods_sorted = sorted(method_set)
    log.info("Unique method names across %d CSVs: %s", len(csvs), methods_sorted)

    # Write output CSV
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method"])
        for m in methods_sorted:
            writer.writerow([m])

    log.info("Wrote %d method entries to %s", len(methods_sorted), out_path)
    print(f"scATAC baselines: {len(csvs)} CSVs, {len(methods_sorted)} unique methods.")
    print(f"Method list written to: {out_path}")


if __name__ == "__main__":
    main()
