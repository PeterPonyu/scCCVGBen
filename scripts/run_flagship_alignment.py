#!/usr/bin/env python3
"""run_flagship_alignment.py — Retrain the CCVGAE flagship row at legacy config.

Background
----------
fig08/fig09/axisC plot a column called "CCVGAE" (in fig08) or "scCCVGBen_GAT"
(in fig09/axisC) as the flagship method. The data source is
``results/encoder_sweep/{ds}.csv`` row ``method=scCCVGBen_GAT``, which the
encoder_sweep runner trains at scccvgben's defaults: epochs=100, i_dim=5.

But the OTHER columns in fig08 (VAE / CenVAE / CouVAE / GAT-VAE) come from
``results/pair_sweep/`` which trains at epochs=200, i_dim=10 (now aligned to
CCVGAE-revised CGVAE_agent defaults). Mixing 100ep flagship vs 200ep ablation
columns is unfair.

This script trains JUST the flagship configuration at the matching legacy
config — `encoder_type='graph'`, `graph_type='GAT'`, `latent_type='q_z'`,
`w_irecon=1.0`, `i_dim=10`, `epochs=200`, `subgraph_size=300` — and writes
to ``results/encoder_sweep_flagship/{ds}.csv`` with method=`scCCVGBen_GAT`.

The reconciler is updated to read this dir LAST so its rows win on
dedup-by-method, replacing the 100ep encoder_sweep rows for the GAT entry.

Usage::

    python scripts/run_flagship_alignment.py                   # all 55 NEW datasets
    python scripts/run_flagship_alignment.py --datasets-glob 'workspace/data/scrna/GSE12*.h5ad'

Resume: skips a dataset whose output CSV already exists.
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Vendored CentroidVAE module location.
_LEGACY_REPO_NAME = "CC" + "VGAE"
_LEGACY_ROOT = Path("/home/zeyufu/LAB") / _LEGACY_REPO_NAME
if str(_LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Flagship configuration — matches CGVAE1.ipynb's CCVGAE training exactly.
# All other CentroidVAEAgent params keep their CGVAE_agent defaults (which is
# what CCVGAE-revised used).
FLAGSHIP_CFG = dict(
    encoder_type="graph",
    graph_type="GAT",
    latent_type="q_z",
    w_irecon=1.0,
    i_dim=10,
    subgraph_size=300,
    hidden_dim=128,
    latent_dim=10,
    lr=1e-4,
    w_recon=1.0,
    w_kl=1.0,
    w_adj=1.0,
    num_subgraphs_per_epoch=10,
)
EPOCHS = 200


def _train_flagship(adata, silent: bool = True):
    """Train the flagship CCVGAE configuration. Returns the agent."""
    from CentroidVAE import CentroidVAEAgent
    agent = CentroidVAEAgent(adata=adata, layer="counts", **FLAGSHIP_CFG)
    agent.fit(epochs=EPOCHS, silent=silent)
    return agent


def _preprocess(h5ad_path: Path):
    """Match the encoder_sweep preprocessing path so feature/sample spaces are
    consistent between flagship and the rest of axis A."""
    from scccvgben.training.scccvgben_runner import preprocess_scrna_scccvgben
    adata = sc.read_h5ad(h5ad_path)
    return preprocess_scrna_scccvgben(adata)


def _evaluate(agent, adata) -> dict:
    """Compute the canonical 24-metric row using scccvgben's metric pipeline."""
    from scccvgben.training.metrics import compute_metrics
    z = agent.get_latent()
    labels = agent.labels
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    return compute_metrics(Z=z, X_orig=X, labels=labels, method_name="scCCVGBen_GAT")


def _reuse_legacy(out_root: Path, manifest_keys: set) -> int:
    """Import the 45 CCVGAE-overlap legacy `scCCVGBen` rows into flagship dir.

    Source: ``workspace/reused_results/axisA_GAT_scrna/{prefix}_{key}_df.csv``
    (symlinks to /home/zeyufu/LAB/CCVGAE/CG_results/CG_dl_merged/).

    The legacy file contains all 13 baseline methods. We extract only the
    `scCCVGBen` row (= GAT-encoder, w_irecon=1, epochs=200, i_dim=10), rename
    to `scCCVGBen_GAT`, and drop the deprecated NMI/ARI/COR columns. Output
    filename matches the encoder_sweep convention (`{key}.csv`, no prefix)
    so reconciler picks it up cleanly.
    """
    legacy_dir = REPO_ROOT / "workspace" / "reused_results" / "axisA_GAT_scrna"
    if not legacy_dir.is_dir():
        log.warning("legacy dir missing: %s", legacy_dir)
        return 0
    n = 0
    for legacy in sorted(legacy_dir.glob("*_df.csv")):
        from scccvgben.figures import dataset_key_from_result_stem
        key = dataset_key_from_result_stem(legacy.stem)
        if key not in manifest_keys:
            continue
        out_csv = out_root / f"{key}.csv"
        if out_csv.exists():
            continue
        df = pd.read_csv(legacy)
        df = df.drop(columns=["NMI", "ARI", "COR"], errors="ignore")
        sub = df[df["method"] == "scCCVGBen"].copy()
        if sub.empty:
            continue
        sub["method"] = "scCCVGBen_GAT"
        sub.to_csv(out_csv, index=False)
        n += 1
    log.info("reused %d legacy `scCCVGBen` rows → %s as `scCCVGBen_GAT`", n, out_root)
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets-glob", default="workspace/data/scrna/*.h5ad",
                   help="Glob for h5ad inputs (default: all scrna).")
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "data" / "benchmark_manifest.csv",
                   help="Filter datasets to manifest scrna entries.")
    p.add_argument("--out", default="results/encoder_sweep_flagship",
                   help="Output dir relative to repo root.")
    p.add_argument("--no-manifest-filter", action="store_true",
                   help="Skip manifest filter — train on every glob match.")
    p.add_argument("--reuse-only", action="store_true",
                   help="Only import legacy rows; skip fresh training.")
    args = p.parse_args(argv)

    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve dataset paths
    paths = sorted(glob.glob(str(REPO_ROOT / args.datasets_glob)))
    manifest_keys = set()
    if not args.no_manifest_filter and args.manifest.exists():
        mf = pd.read_csv(args.manifest)
        manifest_keys = set(mf[mf.modality == "scrna"].filename_key.astype(str))
        paths = [p for p in paths if Path(p).stem in manifest_keys]

    # Step 1: import legacy rows for the 45 overlap datasets (instant)
    _reuse_legacy(out_root, manifest_keys or {Path(p).stem for p in paths})
    if args.reuse_only:
        return 0

    log.info("flagship retrain: %d datasets, %d epochs, i_dim=%d → %s",
             len(paths), EPOCHS, FLAGSHIP_CFG["i_dim"], out_root)

    n_done = n_skip = n_err = 0
    for h5_path in paths:
        h5_path = Path(h5_path)
        out_csv = out_root / f"{h5_path.stem}.csv"
        if out_csv.exists():
            n_skip += 1
            continue
        t0 = time.time()
        try:
            adata = _preprocess(h5_path)
            agent = _train_flagship(adata)
            row = _evaluate(agent, adata)
            df = pd.DataFrame([row])
            df.to_csv(out_csv, index=False)
            n_done += 1
            log.info("  ✓ %s (%.0fs)", h5_path.stem, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            log.warning("  ✗ %s failed: %s: %s", h5_path.stem,
                        type(exc).__name__, exc)

    log.info("flagship complete: %d new, %d skipped, %d errors",
             n_done, n_skip, n_err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
