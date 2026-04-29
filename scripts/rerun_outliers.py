"""US-602: Consume rerun-shortlist.txt and launch (or print) encoder-sweep reruns.

Usage
-----
Dry-run (default) — prints each command, archives nothing:
    python scripts/rerun_outliers.py

Execute mode — archives old rows and actually runs each command:
    python scripts/rerun_outliers.py --execute

The shortlist is read from .omc/research/rerun-shortlist.txt by default.
Override with --shortlist <path>.

Only rows whose method starts with ``scCCVGBen_`` are handled; others are
printed as "skip — not handled by this launcher".
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHORTLIST = ROOT / ".omc" / "research" / "rerun-shortlist.txt"
SWEEP_DIR = ROOT / "results" / "encoder_sweep"
ARCHIVE_BASE = SWEEP_DIR / ".archive"
SWEEP_SCRIPT = ROOT / "scripts" / "run_encoder_sweep.py"
DATA_SCRNA = ROOT / "workspace" / "data" / "scrna"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Consume rerun-shortlist.txt and prepare/run encoder-sweep reruns.\n\n"
            "Dry-run (default): prints each command to stdout; nothing is archived "
            "or executed.\n"
            "--execute: archives existing CSV rows for each (dataset, method) pair, "
            "then invokes each sweep command via subprocess.\n\n"
            "Only scCCVGBen_* methods are handled; others are skipped with a message."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--shortlist",
        default=str(DEFAULT_SHORTLIST),
        help="Path to rerun-shortlist.txt (default: %(default)s)",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually run commands; default is dry-run (print only).",
    )
    return p.parse_args()


def load_shortlist(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (dataset, method, metric) triples."""
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                print(f"[warn] malformed shortlist line: {line!r}", file=sys.stderr)
                continue
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def group_by_dataset_method(
    rows: list[tuple[str, str, str]]
) -> dict[tuple[str, str], list[str]]:
    """Group metrics by (dataset, method) pair."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for dataset, method, metric in rows:
        groups[(dataset, method)].append(metric)
    return dict(groups)


def archive_csv_rows(dataset_key: str, method: str, stamp: str) -> None:
    """Move existing rows for (dataset, method) from encoder_sweep CSVs to archive."""
    archive_dir = ARCHIVE_BASE / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Find matching CSV files (dataset_key appears in stem)
    candidates = list(SWEEP_DIR.glob("*.csv"))
    matched = [f for f in candidates if dataset_key in f.stem]
    if not matched:
        print(f"  [archive] No CSV found for dataset_key={dataset_key!r}; skipping archive.")
        return

    for csv_path in matched:
        try:
            df_lines = csv_path.read_text().splitlines()
            if not df_lines:
                continue
            header = df_lines[0]
            method_col_idx = None
            for i, col in enumerate(header.split(",")):
                if col.strip() == "method":
                    method_col_idx = i
                    break
            if method_col_idx is None:
                continue

            keep_lines = [header]
            remove_lines = []
            for line in df_lines[1:]:
                cols = line.split(",")
                if len(cols) > method_col_idx and cols[method_col_idx].strip() == method:
                    remove_lines.append(line)
                else:
                    keep_lines.append(line)

            if remove_lines:
                archive_path = archive_dir / f"{dataset_key}_{method}.csv"
                archive_path.write_text("\n".join([header] + remove_lines) + "\n")
                csv_path.write_text("\n".join(keep_lines) + "\n")
                print(f"  [archive] {len(remove_lines)} rows -> {archive_path}")
        except Exception as exc:
            print(f"  [archive] Error processing {csv_path}: {exc}", file=sys.stderr)


def build_command(dataset_key: str, method: str) -> str:
    """Build the sweep CLI command string for a (dataset, method) pair."""
    # method is like scCCVGBen_GAT -> encoder is GAT
    encoder = method.split("_", 1)[1] if "_" in method else method
    glob_pattern = str(DATA_SCRNA / f"{dataset_key}.h5ad")
    return (
        f"python {SWEEP_SCRIPT} "
        f'--datasets-glob "{glob_pattern}" '
        f"--encoders {encoder}"
    )


def main() -> None:
    args = parse_args()
    shortlist_path = Path(args.shortlist)
    if not shortlist_path.exists():
        print(f"[error] shortlist not found: {shortlist_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_shortlist(shortlist_path)
    if not rows:
        print("Shortlist is empty — nothing to do.")
        return

    groups = group_by_dataset_method(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] {len(groups)} (dataset, method) pairs from shortlist")
    print()

    for (dataset, method), metrics in sorted(groups.items()):
        print(f"--- {dataset} / {method} ---")
        print(f"    flagged metrics: {', '.join(metrics)}")

        if not method.startswith("scCCVGBen_"):
            print(f"    skip — not handled by this launcher (method={method!r})")
            print()
            continue

        cmd = build_command(dataset, method)

        if args.execute:
            archive_csv_rows(dataset, method, stamp)
            print(f"    running: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=ROOT)
            if result.returncode != 0:
                print(f"    [warn] command exited {result.returncode}", file=sys.stderr)
        else:
            print(f"    command: {cmd}")
        print()

    if not args.execute:
        print(f"[DRY-RUN complete] Re-run with --execute to archive and launch.")


if __name__ == "__main__":
    main()
