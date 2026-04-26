#!/usr/bin/env python3
"""run_biovalidation.py — Phase 2 case-study sweep + figure composition.

For each case in ``scccvgben.biovalidation.case_definition.CASES``:

  1. ``compute.run_case()`` — train encoder, compute latent + top-K genes
     + UMAP + latent self-corr, fall-back leiden labels if needed.
  2. ``compose.compose_case_figure()`` — emit one 16×12-inch PDF + PNG with
     panels A through G in a fixed GridSpec layout. The composer is fault
     tolerant: a missing payload key produces a placeholder, not a crash.

Resume: a case is skipped when its PDF already exists. Pass ``--force`` to
re-render.

Usage:
  python scripts/run_biovalidation.py                           # all 6 cases
  python scripts/run_biovalidation.py --cases SD                # one case
  python scripts/run_biovalidation.py --epochs 50               # quicker train
  python scripts/run_biovalidation.py --smoke                   # 30 epochs SD only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("biovalidation")

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="all",
                        help="'all' or comma list (SD,UCB,IR,GASTRIC,HSC_AGE,COVID).")
    parser.add_argument("--out", default="figures/biovalidation/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--top-k-genes", type=int, default=5,
                        help="Top-K correlated genes per latent dim (gene grid columns).")
    parser.add_argument("--subsample-cells", type=int, default=3000)
    parser.add_argument("--force", action="store_true",
                        help="Re-render even when output PDF already exists.")
    parser.add_argument("--smoke", action="store_true",
                        help="SD case, 30 epochs only.")
    args = parser.parse_args()

    from scccvgben.biovalidation import CASES
    from scccvgben.biovalidation.case_definition import order
    from scccvgben.biovalidation.compute import run_case
    from scccvgben.biovalidation.compose import compose_case_figure

    if args.smoke:
        case_ids = ["SD"]
        epochs = 30
    else:
        if args.cases == "all":
            case_ids = order()
        else:
            case_ids = [c.strip() for c in args.cases.split(",")]
        epochs = args.epochs

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("biovalidation: %d cases (epochs=%d, k=%d) → %s",
             len(case_ids), epochs, args.top_k_genes, out_dir)

    n_done = n_skip = n_err = 0
    for case_id in case_ids:
        if case_id not in CASES:
            log.warning("skipping unknown case %r (choices: %s)", case_id, list(CASES))
            continue
        case = CASES[case_id]
        pdf_path = out_dir / f"fig_biovalidation_case_{case.case_id}.pdf"
        if pdf_path.exists() and not args.force:
            log.info("[%s] resume — skipping (exists: %s)", case_id, pdf_path.name)
            n_skip += 1
            continue
        t0 = time.time()
        try:
            payload = run_case(
                case,
                epochs=epochs,
                subsample_cells=args.subsample_cells,
                top_k_genes=args.top_k_genes,
                silent=True,
            )
            compose_case_figure(payload, out_dir)
            n_done += 1
            log.info("  ✓ [%s] %.0fs", case_id, time.time() - t0)
        except Exception as exc:
            n_err += 1
            log.exception("  ✗ [%s] failed: %s", case_id, exc)

    log.info("biovalidation complete: %d new, %d resumed, %d errors", n_done, n_skip, n_err)
    sys.exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    main()
