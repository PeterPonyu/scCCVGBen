"""D5 STUB: Leiden / k-means / Louvain clustering sensitivity on raw latents.

THIS SCRIPT IS BLOCKED until latents are persisted by the training pipeline.
See .omc/research/D5_D6_blocker-2026-04-28.md for the 2-line fix.

When latents exist under workspace/latents/ (or the path configured below),
this script will:
  - Walk all *.npy latent files (shape: n_cells x latent_dim)
  - For each (dataset, method): run Leiden (res=1.0, seed=42), k-means (k from
    manifest or default 10), Louvain (res=1.0)
  - Compute ARI and NMI vs ground-truth labels (from original h5ad via manifest)
  - Write results/cluster_sensitivity_2026-04-28.csv
  - Parallelise with multiprocessing.Pool(8)

Usage (once latents exist):
    python scripts/cluster_sensitivity.py --latent-dir workspace/latents

Output columns: dataset, method, algo, seed, ARI, NMI, n_clusters
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATENT_DIR = ROOT / "workspace" / "latents"


def _check_latents_exist(latent_dir: Path) -> None:
    npy_files = list(latent_dir.glob("**/*.npy"))
    if not npy_files:
        print(
            f"[cluster_sensitivity.py] BLOCKED: no .npy files found under {latent_dir}\n"
            f"See .omc/research/D5_D6_blocker-2026-04-28.md for the 2-line fix to\n"
            f"add latent persistence to the training pipeline.\n"
            f"Fallback: run scripts/cluster_sensitivity_partial.py instead."
        )
        sys.exit(1)
    print(f"Found {len(npy_files)} latent files. Full D5 run starting...")


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

    # --- Implementation below activates once latents exist ---
    # (imports deferred so the script exits cleanly at the check above)
    import time
    import numpy as np
    import pandas as pd
    from multiprocessing import Pool
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    try:
        import scanpy as sc
    except ImportError:
        print("scanpy not installed; cannot run Leiden/Louvain.")
        sys.exit(1)

    OUT_CSV = ROOT / "results" / "cluster_sensitivity_2026-04-28.csv"
    MANIFEST = ROOT / "data" / "benchmark_manifest.csv"
    manifest = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame()

    npy_files = sorted(args.latent_dir.glob("**/*.npy"))

    def _run_one(npy_path: Path) -> list[dict]:
        rows = []
        stem = npy_path.stem  # e.g. "Dev_hemato_df_scCCVGBen_GAT_latent"
        # Parse dataset and method from stem
        parts = stem.replace("_latent", "")
        latent = np.load(str(npy_path))

        # Try to get ground-truth labels from manifest
        labels = None
        # (manifest lookup by dataset key — fill in when manifest schema is known)

        algos = [
            ("leiden", 42),
            ("kmeans", 42),
            ("louvain", 42),
        ]

        for algo, seed in algos:
            try:
                import anndata as ad
                adata_tmp = ad.AnnData(X=latent)
                sc.pp.neighbors(adata_tmp, use_rep="X", n_neighbors=15, random_state=seed)

                if algo == "leiden":
                    sc.tl.leiden(adata_tmp, resolution=1.0, random_state=seed)
                    pred = adata_tmp.obs["leiden"].astype(int).values
                elif algo == "louvain":
                    sc.tl.louvain(adata_tmp, resolution=1.0, random_state=seed)
                    pred = adata_tmp.obs["louvain"].astype(int).values
                elif algo == "kmeans":
                    k = 10  # default; override from manifest when available
                    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
                    pred = km.fit_predict(latent)
                else:
                    continue

                n_clusters = len(set(pred))
                ari = adjusted_rand_score(labels, pred) if labels is not None else float("nan")
                nmi = normalized_mutual_info_score(labels, pred) if labels is not None else float("nan")

                rows.append({
                    "dataset": parts,
                    "method": parts,
                    "algo": algo,
                    "seed": seed,
                    "ARI": ari,
                    "NMI": nmi,
                    "n_clusters": n_clusters,
                })
            except Exception as exc:
                rows.append({
                    "dataset": parts,
                    "method": parts,
                    "algo": algo,
                    "seed": seed,
                    "ARI": float("nan"),
                    "NMI": float("nan"),
                    "n_clusters": -1,
                    "error": str(exc),
                })
        return rows

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
