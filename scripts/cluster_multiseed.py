"""D6 STUB: Leiden multi-seed stability (5 seeds) on raw latents.

THIS SCRIPT IS BLOCKED until latents are persisted by the training pipeline.
See .omc/research/D5_D6_blocker-2026-04-28.md for the 2-line fix.

When latents exist under workspace/latents/, this script will:
  - For each (dataset, method) latent: run Leiden 5x with seeds [0,1,2,3,4]
    at resolution 1.0
  - Report median ARI, IQR, n_clusters per (dataset, method)
  - Write results/cluster_multiseed_2026-04-28.csv
  - Parallelise with multiprocessing.Pool(8)

Output columns: dataset, method, seed, ARI, NMI, n_clusters,
                median_ARI, iqr_ARI, median_NMI, iqr_NMI
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATENT_DIR = ROOT / "workspace" / "latents"
SEEDS = [0, 1, 2, 3, 4]


def _check_latents_exist(latent_dir: Path) -> None:
    npy_files = list(latent_dir.glob("**/*.npy"))
    if not npy_files:
        print(
            f"[cluster_multiseed.py] BLOCKED: no .npy files found under {latent_dir}\n"
            f"See .omc/research/D5_D6_blocker-2026-04-28.md for the 2-line fix to\n"
            f"add latent persistence to the training pipeline."
        )
        sys.exit(1)
    print(f"Found {len(npy_files)} latent files. D6 multi-seed run starting...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=LATENT_DIR,
        help="Directory containing *_latent.npy files (default: workspace/latents/)",
    )
    args = parser.parse_args()
    _check_latents_exist(args.latent_dir)

    import time
    import numpy as np
    import pandas as pd
    from multiprocessing import Pool
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        print("scanpy not installed; cannot run Leiden.")
        sys.exit(1)

    OUT_CSV = ROOT / "results" / "cluster_multiseed_2026-04-28.csv"
    npy_files = sorted(args.latent_dir.glob("**/*.npy"))

    def _run_one(npy_path: Path) -> list[dict]:
        stem = npy_path.stem.replace("_latent", "")
        latent = np.load(str(npy_path))
        labels = None  # fill from manifest when available

        per_seed_rows = []
        for seed in SEEDS:
            try:
                adata_tmp = ad.AnnData(X=latent)
                sc.pp.neighbors(adata_tmp, use_rep="X", n_neighbors=15, random_state=seed)
                sc.tl.leiden(adata_tmp, resolution=1.0, random_state=seed)
                pred = adata_tmp.obs["leiden"].astype(int).values
                n_clusters = len(set(pred))
                ari = adjusted_rand_score(labels, pred) if labels is not None else float("nan")
                nmi = normalized_mutual_info_score(labels, pred) if labels is not None else float("nan")
                per_seed_rows.append({
                    "dataset": stem,
                    "method": stem,
                    "seed": seed,
                    "ARI": ari,
                    "NMI": nmi,
                    "n_clusters": n_clusters,
                })
            except Exception as exc:
                per_seed_rows.append({
                    "dataset": stem,
                    "method": stem,
                    "seed": seed,
                    "ARI": float("nan"),
                    "NMI": float("nan"),
                    "n_clusters": -1,
                    "error": str(exc),
                })

        # Compute median and IQR
        aris = [r["ARI"] for r in per_seed_rows]
        nmis = [r["NMI"] for r in per_seed_rows]
        median_ari = float(np.nanmedian(aris))
        iqr_ari = float(np.nanpercentile(aris, 75) - np.nanpercentile(aris, 25))
        median_nmi = float(np.nanmedian(nmis))
        iqr_nmi = float(np.nanpercentile(nmis, 75) - np.nanpercentile(nmis, 25))

        for r in per_seed_rows:
            r["median_ARI"] = median_ari
            r["iqr_ARI"] = iqr_ari
            r["median_NMI"] = median_nmi
            r["iqr_NMI"] = iqr_nmi

        return per_seed_rows

    t0 = time.time()
    with Pool(processes=8) as pool:
        all_results = pool.map(_run_one, npy_files)

    flat = [r for sublist in all_results for r in sublist]
    out_df = pd.DataFrame(flat)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    elapsed = time.time() - t0
    print(f"Wrote {len(out_df)} rows -> {OUT_CSV}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
