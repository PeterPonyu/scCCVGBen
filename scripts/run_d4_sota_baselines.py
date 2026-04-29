"""Runner for 2024-era SOTA scRNA baselines (Reviewer R2.5 response).

METHODS
-------
1. scGPT (Cui et al., Nature Methods 2024)
   Repo   : https://github.com/bowang-lab/scGPT
   Install: pip install scgpt
   Weights: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
            -> place whole-human checkpoint at workspace/sota_weights/scgpt/

2. scFoundation (Hao et al., Nature Methods 2024)
   Repo   : https://github.com/biomap-research/scFoundation
   Install: pip install scfoundation
   Weights: huggingface-cli download biomap-research/scFoundation
              --local-dir workspace/sota_weights/scfoundation

MANUAL PREREQS (before --execute)
----------------------------------
  pip install scgpt scfoundation
  mkdir -p workspace/sota_weights/scgpt workspace/sota_weights/scfoundation
  # scGPT  : download from Google Drive link above
  # scFoundation: huggingface-cli download biomap-research/scFoundation
  #           --local-dir workspace/sota_weights/scfoundation

TIMING
------
  ~5 min/dataset/method on GPU.  110 datasets x 2 methods = ~1100 min seq.
  With --workers 4 shards: ~275 min (~4 h) total D4 wall-clock.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import sys
from pathlib import Path

WEIGHT_ROOTS = {
    "scgpt":        Path("workspace/sota_weights/scgpt"),
    "scfoundation": Path("workspace/sota_weights/scfoundation"),
}
INSTALL_CMDS = {
    "scgpt":        "pip install scgpt",
    "scfoundation": "pip install scfoundation",
}
WEIGHT_URLS = {
    "scgpt": (
        "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y\n"
        "  -> download whole-human checkpoint -> workspace/sota_weights/scgpt/"
    ),
    "scfoundation": (
        "huggingface-cli download biomap-research/scFoundation "
        "--local-dir workspace/sota_weights/scfoundation"
    ),
}
OUTPUT_ROOT = Path("results/sota_baselines")
CSV_COLUMNS = [
    "method", "ASW", "DAV", "CAL", "COR",
    "distance_correlation_umap", "Q_local_umap", "Q_global_umap",
    "K_max_umap", "overall_quality_umap",
    "distance_correlation_tsne", "Q_local_tsne", "Q_global_tsne",
    "K_max_tsne", "overall_quality_tsne",
    "manifold_dimensionality_intrin", "spectral_decay_rate_intrin",
    "participation_ratio_intrin", "anisotropy_score_intrin",
    "trajectory_directionality_intrin", "noise_resilience_intrin",
    "core_quality_intrin", "overall_quality_intrin",
    "data_type_intrin", "interpretation_intrin", "NMI", "ARI",
]


# ---------------------------------------------------------------------------
# Preflight (hard-exit versions used only when --execute is set)
# ---------------------------------------------------------------------------

def preflight_hard(method: str) -> None:
    """Import-check + weight-check; exit 1 on any failure."""
    try:
        importlib.import_module(method)
    except ImportError:
        print(f"[ERROR] {method}: not installed.\n  Run: {INSTALL_CMDS[method]}")
        sys.exit(1)
    wd = WEIGHT_ROOTS[method]
    if not (wd.exists() and any(wd.iterdir())):
        print(f"[ERROR] {method}: weights not found at {wd}\n  {WEIGHT_URLS[method]}")
        sys.exit(1)


def preflight_dry(method: str) -> None:
    """Report weight status without hard-exiting (dry-run safe)."""
    wd = WEIGHT_ROOTS[method]
    if wd.exists() and any(wd.iterdir()):
        print(f"  weights : found at {wd}")
    else:
        print(f"  weights : NOT FOUND at {wd}\n  -> {WEIGHT_URLS[method]}")


# ---------------------------------------------------------------------------
# Embedding stubs
# ---------------------------------------------------------------------------

def _embed_scgpt(adata, wd: Path):
    from scgpt.tasks import embed_data  # noqa: PLC0415
    return embed_data(adata, model_dir=str(wd), gene_col="gene_name",
                      max_length=1200, batch_size=64,
                      use_fast_transformer=True, return_new_adata=False)


def _embed_scfoundation(adata, wd: Path):
    import scfoundation  # noqa: PLC0415
    model = scfoundation.load_model(checkpoint_dir=str(wd))
    return scfoundation.get_embedding(adata, model=model, batch_size=64)


EMBED_FNS = {"scgpt": _embed_scgpt, "scfoundation": _embed_scfoundation}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_pair(method: str, h5ad: Path, execute: bool) -> None:
    name = h5ad.stem
    out = OUTPUT_ROOT / f"{method}__{name}.csv"
    if not execute:
        print(f"  [DRY-RUN] would embed {name} with {method} -> {out}")
        return

    import scanpy as sc          # noqa: PLC0415
    import pandas as pd          # noqa: PLC0415
    from scccvgben.figures.metrics import compute_metrics  # noqa: PLC0415

    print(f"  [RUN] loading {h5ad} ...")
    adata = sc.read_h5ad(h5ad)
    print(f"  [RUN] embedding with {method} ...")
    embeddings = EMBED_FNS[method](adata, WEIGHT_ROOTS[method])
    print(f"  [RUN] computing metrics ...")
    row = {"method": method, **compute_metrics(embeddings, adata)}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(out, index=False)
    print(f"  [DONE] wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run 2024-era SOTA scRNA baselines (scGPT, scFoundation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--methods", default="scgpt,scfoundation",
                   help="Comma-separated methods (default: scgpt,scfoundation).")
    p.add_argument("--datasets-glob", default="workspace/data/scrna/*.h5ad",
                   help="Glob for input h5ad files.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel shards hint (informational; default: 4).")
    p.add_argument("--execute", action="store_true",
                   help="Actually run inference (dry-run by default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    datasets = sorted(glob.glob(args.datasets_glob))

    print(f"Methods  : {methods}")
    print(f"Datasets : {len(datasets)} file(s) matching '{args.datasets_glob}'")
    print(f"Workers  : {args.workers}  |  Execute: {args.execute}")
    print()

    if not args.execute:
        print("*** DRY-RUN — pass --execute to run real inference. ***\n")

    for method in methods:
        print(f"[{method}] preflight ...")
        if args.execute:
            preflight_hard(method)
        else:
            preflight_dry(method)

        if not datasets:
            print(f"  [WARN] no datasets found for glob '{args.datasets_glob}'")
            continue

        for h5ad in map(Path, datasets):
            run_pair(method, h5ad, execute=args.execute)

    print()
    if args.execute:
        print("All pairs complete.")
    else:
        n = len(methods) * max(len(datasets), 1)
        seq_min = n * 5
        print(
            f"Est. wall-clock: ~{seq_min} min sequential, "
            f"~{seq_min // args.workers} min with {args.workers} shards "
            f"(~{max(1, seq_min // args.workers // 60)} h).\n"
            "Re-run with --execute after prerequisites are satisfied."
        )


if __name__ == "__main__":
    main()
