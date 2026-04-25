"""In-process registry orchestrator for scCCVGBen figures.

Imports each figure module in process, calls its main(args) with
--partial-ok injected, isolates state via mpl.rcdefaults() + plt.close('all')
between renders, and prints a per-figure status table:

    figure_id | script | status | output_path | duration_s | data_completeness_pct

Status enum:
    rendered    -> exit 0, full-data output exists.
    preliminary -> exit 0, output filename carries .PRELIMINARY. infix.
    skipped     -> exit 0, no output produced (e.g. data unavailable).
    failed      -> module raised or exit != 0.

Exits 0 on partial success when --partial-ok is set; otherwise any non-rendered
row is fatal.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import logging
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    script: str
    output_glob: str
    target_n: int
    count_kind: str


REGISTRY: tuple[FigureSpec, ...] = (
    FigureSpec("fig01", "scripts.make_figure1_site",
               "fig1_scCCVGBen_site*.pdf", 0, "static"),
    FigureSpec("fig02", "scripts.make_figure2_model_architecture",
               "fig2_scCCVGBen_model_architecture*.pdf", 0, "static"),
    FigureSpec("axisA", "scripts.make_axisA_figure",
               "fig_axisA_encoder_ranking*.pdf", 100, "encoder_sweep"),
    FigureSpec("axisB", "scripts.make_axisB_figure",
               "fig_axisB_graph_robustness*.pdf", 100, "graph_sweep"),
    FigureSpec("axisC", "scripts.make_axisC_figure",
               "fig_axisC_baselines*.pdf", 200, "reconciled_all"),
    FigureSpec("fig08", "scripts.make_fig08_scrna_benchmark",
               "fig08_scrna_benchmark*.pdf", 100, "reconciled_scrna"),
    FigureSpec("fig09", "scripts.make_fig09_scatac_benchmark",
               "fig09_scatac_benchmark*.pdf", 100, "reconciled_scatac"),
    FigureSpec("fig10_12", "scripts.make_fig10_12_case_studies",
               "fig1[012]*.pdf", 3, "case_candidates"),
    FigureSpec("fig14", "scripts.make_fig14_runtime",
               "fig14_runtime*.pdf", 1, "runtime_logs"),
)


def _reset_state() -> None:
    plt.close("all")
    mpl.rcdefaults()
    sns.set_palette("Spectral")


def _module_basename(module_path: str) -> str:
    return module_path.split(".")[-1] + ".py"


def _classify(out_paths: list[Path]) -> str:
    if not out_paths:
        return "skipped"
    if any(".PRELIMINARY." in p.name for p in out_paths):
        return "preliminary"
    return "rendered"


def _normalise_result_stem(stem: str) -> str:
    from scccvgben.figures import dataset_key_from_result_stem

    return dataset_key_from_result_stem(stem)


def _manifest_keys(modality: str | None = None) -> set[str]:
    import pandas as pd

    manifest = REPO_ROOT / "data" / "benchmark_manifest.csv"
    if not manifest.exists():
        return set()
    df = pd.read_csv(manifest, usecols=["filename_key", "modality"])
    if modality is not None:
        df = df[df["modality"].astype(str).str.lower() == modality.lower()]
    return set(df["filename_key"].astype(str))


def _count_csvs(path: Path, *, modality: str | None = None) -> int:
    if not path.is_dir():
        return 0
    files = {_normalise_result_stem(p.stem) for p in path.glob("*.csv")}
    keep = _manifest_keys(modality)
    return len(files & keep) if keep else len(files)


def _observed_n(spec: FigureSpec) -> int:
    if spec.count_kind == "static":
        return spec.target_n
    if spec.count_kind == "manifest_all":
        return len(_manifest_keys())
    if spec.count_kind == "encoder_sweep":
        return _count_csvs(REPO_ROOT / "results" / "encoder_sweep", modality="scrna")
    if spec.count_kind == "graph_sweep":
        return _count_csvs(REPO_ROOT / "results" / "graph_sweep", modality="scrna")
    if spec.count_kind == "reconciled_scrna":
        return _count_csvs(REPO_ROOT / "results" / "reconciled" / "scrna", modality="scrna")
    if spec.count_kind == "reconciled_scatac":
        return _count_csvs(REPO_ROOT / "results" / "reconciled" / "scatac", modality="scatac")
    if spec.count_kind == "reconciled_all":
        return (
            _count_csvs(REPO_ROOT / "results" / "reconciled" / "scrna", modality="scrna")
            + _count_csvs(REPO_ROOT / "results" / "reconciled" / "scatac", modality="scatac")
        )
    if spec.count_kind == "case_candidates":
        return min(3, _observed_n(FigureSpec("fig08", "", "", 100, "reconciled_scrna"))
                   + _observed_n(FigureSpec("fig09", "", "", 100, "reconciled_scatac")))
    if spec.count_kind == "runtime_logs":
        log_dir = REPO_ROOT / "workspace" / "logs"
        if not log_dir.is_dir():
            return 0
        from scripts.make_fig14_runtime import _parse_logs

        return 1 if not _parse_logs(log_dir).empty else 0
    return 0


def _completeness_pct(spec: FigureSpec) -> float:
    if spec.target_n <= 0:
        return 100.0
    observed = _observed_n(spec)
    return min(100.0, round((observed / spec.target_n) * 100.0, 1))


def _run_one(
    spec: FigureSpec, out_dir: Path, partial_ok: bool, isolate: bool,
) -> dict:
    start = time.perf_counter()
    if isolate:
        return _run_subprocess(spec, out_dir, partial_ok, start)
    try:
        _reset_state()
        module = importlib.import_module(spec.script)
        argv: list[str] = ["--out-dir", str(out_dir)]
        if partial_ok:
            argv.append("--partial-ok")
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            rc = module.main(argv)
        duration = time.perf_counter() - start
        out_paths = sorted(out_dir.glob(spec.output_glob))
        if rc != 0:
            return {
                "figure_id": spec.figure_id,
                "script": _module_basename(spec.script),
                "status": "failed",
                "output_path": "",
                "duration_s": round(duration, 2),
                "data_completeness_pct": 0.0,
            }
        status = _classify(out_paths)
        return {
            "figure_id": spec.figure_id,
            "script": _module_basename(spec.script),
            "status": status,
            "output_path": ";".join(p.name for p in out_paths),
            "duration_s": round(duration, 2),
            "data_completeness_pct": round(_completeness_pct(spec), 1),
        }
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - start
        log.warning("fig %s failed in-process: %s", spec.figure_id, exc)
        return {
            "figure_id": spec.figure_id,
            "script": _module_basename(spec.script),
            "status": "failed",
            "output_path": "",
            "duration_s": round(duration, 2),
            "data_completeness_pct": 0.0,
        }
    finally:
        _reset_state()


def _run_subprocess(
    spec: FigureSpec, out_dir: Path, partial_ok: bool, start: float,
) -> dict:
    import subprocess
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / _module_basename(spec.script)),
           "--out-dir", str(out_dir)]
    if partial_ok:
        cmd.append("--partial-ok")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        duration = time.perf_counter() - start
        out_paths = sorted(out_dir.glob(spec.output_glob))
        if proc.returncode != 0:
            return {
                "figure_id": spec.figure_id,
                "script": _module_basename(spec.script),
                "status": "failed",
                "output_path": "",
                "duration_s": round(duration, 2),
                "data_completeness_pct": 0.0,
            }
        status = _classify(out_paths)
        return {
            "figure_id": spec.figure_id,
            "script": _module_basename(spec.script),
            "status": status,
            "output_path": ";".join(p.name for p in out_paths),
            "duration_s": round(duration, 2),
            "data_completeness_pct": round(_completeness_pct(spec), 1),
        }
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - start
        log.warning("fig %s subprocess failed: %s", spec.figure_id, exc)
        return {
            "figure_id": spec.figure_id,
            "script": _module_basename(spec.script),
            "status": "failed",
            "output_path": "",
            "duration_s": round(duration, 2),
            "data_completeness_pct": 0.0,
        }


def _print_table(rows: list[dict]) -> None:
    cols = ("figure_id", "script", "status", "output_path",
            "duration_s", "data_completeness_pct")
    widths = {c: max(len(c), max((len(str(r[c])) for r in rows), default=0)) for c in cols}
    header = " | ".join(f"{c:<{widths[c]}}" for c in cols)
    bar = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(bar)
    for r in rows:
        print(" | ".join(f"{str(r[c]):<{widths[c]}}" for c in cols))


def _write_status_csv(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "_status.csv"
    cols = ("figure_id", "script", "status", "output_path",
            "duration_s", "data_completeness_pct")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--partial-ok", action="store_true")
    parser.add_argument("--only", help="Comma-separated figure_ids to run.")
    parser.add_argument("--isolate", help="Comma-separated figure_ids to subprocess-isolate.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    isolate = {s.strip() for s in args.isolate.split(",")} if args.isolate else set()

    rows: list[dict] = []
    for spec in REGISTRY:
        if only is not None and spec.figure_id not in only:
            continue
        rows.append(_run_one(spec, args.out_dir, args.partial_ok,
                             isolate=spec.figure_id in isolate))
    _print_table(rows)
    _write_status_csv(rows, args.out_dir)

    fail = [r for r in rows if r["status"] == "failed"]
    nonrendered = [r for r in rows if r["status"] != "rendered"]

    if fail:
        log.error("%d figure(s) failed.", len(fail))
        return 1
    if nonrendered and not args.partial_ok:
        log.error("%d non-rendered figure(s); pass --partial-ok to accept.",
                  len(nonrendered))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
