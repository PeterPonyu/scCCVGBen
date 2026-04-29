#!/usr/bin/env python3
"""verify_raw_counts.py — verify every benchmark h5ad is readable + raw counts.

Checks:
  1. File readable (no corruption).
  2. X is non-negative integer matrix (raw counts, not log/normalized).
  3. Shape is reasonable (n_obs <= 5M, ratio n_obs/n_vars <= 1000).
  4. Has a source GSE traceable via filename_key.

Writes per-file report to data/data_validation.csv.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def verify_one(path: Path, modality: str) -> dict[str, object]:
    name = path.name
    stem = name.replace(".h5ad", "").replace(".h5", "")
    out: dict[str, object] = {"filename_key": stem, "modality": modality, "path": str(path)}
    # GSE traceability
    m = re.search(r"GSE\d+", stem, re.IGNORECASE) or re.search(
        r"GSM\d+",
        stem,
        re.IGNORECASE,
    )
    out["gse_or_gsm"] = m.group(0).upper() if m else ""
    try:
        if path.suffix == ".h5" and modality == "scatac":
            import scanpy as sc

            a = sc.read_10x_h5(str(path))
        else:
            a = ad.read_h5ad(path, backed="r")
        out["n_obs"] = a.n_obs
        out["n_vars"] = a.n_vars
        # Sample the X to check raw-count property
        sample = a.X[: min(100, a.n_obs)]
        if hasattr(sample, "toarray"):
            sample = sample.toarray()
        sample = np.asarray(sample)
        # Raw count heuristics:
        #   - non-negative
        #   - integer-valued (or very close to it, allowing for tiny fp noise)
        #   - max value reasonable (5000-500000 for raw counts; higher if no max cap)
        if sample.size == 0:
            out.update({"is_raw_counts": False, "reason": "empty_X"})
        else:
            smin = float(sample.min())
            smax = float(sample.max())
            # integer check: |value - round(value)| < 1e-6
            resid = np.abs(sample - np.round(sample))
            int_frac = float((resid < 1e-6).mean())
            is_raw_counts = bool(smin >= 0 and int_frac > 0.99 and smax >= 1)
            out["x_dtype"] = str(sample.dtype)
            out["x_min"] = smin
            out["x_max"] = smax
            out["x_int_fraction"] = round(int_frac, 4)
            out["is_raw_counts"] = is_raw_counts
            if not is_raw_counts:
                reasons: list[str] = []
                if smin < 0:
                    reasons.append(f"min<0={smin}")
                if int_frac <= 0.99:
                    reasons.append(f"non-integer ({int_frac:.2%})")
                if smax < 1:
                    reasons.append(f"max<1={smax}")
                out["reason"] = ",".join(reasons)
            else:
                out["reason"] = ""
        # Shape sanity
        out["shape_ok"] = (
            (a.n_obs < 5_000_000)
            and (a.n_vars > 0)
            and (a.n_obs / max(1, a.n_vars) < 1000)
        )
        if hasattr(a, "file"):
            a.file.close()
    except Exception as e:
        out.update(
            {
                "n_obs": 0,
                "n_vars": 0,
                "is_raw_counts": False,
                "shape_ok": False,
                "reason": f"read_err: {str(e)[:80]}",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrna", default=str(ROOT / "workspace" / "data" / "scrna"))
    parser.add_argument("--scatac", default=str(ROOT / "workspace" / "data" / "scatac"))
    parser.add_argument("--out", default=str(ROOT / "data" / "data_validation.csv"))
    parser.add_argument("--datasets", default=str(ROOT / "scccvgben" / "data" / "datasets.csv"))
    args = parser.parse_args()

    # Filter to kept (not dropped) datasets when the manifest provides a
    # drop-status column.  Some callers pass benchmark_manifest.csv, which does
    # not carry that field; in that case we validate every listed dataset.
    ds = pd.read_csv(args.datasets)
    if "drop_status" in ds.columns:
        drop_status = ds["drop_status"].astype(str).str.lower()
        keep = ds[~drop_status.isin({"dropped_smallest", "dropped"})]
    else:
        keep = ds
    keep_keys = set(keep["filename_key"].astype(str))

    rows: list[dict[str, object]] = []
    for p in sorted(Path(args.scrna).glob("*.h5ad")):
        stem = p.stem
        if stem not in keep_keys:
            continue
        rows.append(verify_one(p, "scrna"))
    for p in sorted(Path(args.scatac).glob("*.h5*")):
        stem = p.stem.replace(".h5", "")
        if stem not in keep_keys:
            continue
        rows.append(verify_one(p, "scatac"))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"total files verified: {len(df)}")
    if len(df):
        by_mod = df.groupby("modality").agg(
            n=("filename_key", "count"),
            raw_ok=("is_raw_counts", "sum"),
            shape_ok=("shape_ok", "sum"),
            gse_ok=("gse_or_gsm", lambda x: (x.astype(bool)).sum()),
        )
        print(by_mod)
        print()
        bad = df[~df["is_raw_counts"] | ~df["shape_ok"]]
        if len(bad):
            print(f"NON-COMPLIANT ({len(bad)}):")
            print(
                bad[
                    ["filename_key", "modality", "is_raw_counts", "shape_ok", "reason"]
                ].to_string(index=False)
            )
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
