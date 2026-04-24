#!/usr/bin/env python3
"""preprocess_scatac.py — CLI wrapper around scccvgben.data.preprocessing.preprocess_scatac.

Processes each input h5ad in place (overwrites) or writes to a sibling
*_pp.h5ad file when --out-suffix is specified.

Usage:
    python scripts/preprocess_scatac.py file1.h5ad file2.h5ad ...
    python scripts/preprocess_scatac.py workspace/data/scatac/*.h5ad --out-suffix _pp
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad_paths", nargs="+", help="Input .h5ad file paths.")
    parser.add_argument(
        "--out-suffix", default="",
        help="If set, write output to <stem><suffix>.h5ad instead of overwriting."
    )
    args = parser.parse_args()

    try:
        from scccvgben.data.preprocessing import preprocess_scatac
    except ImportError as exc:
        log.error("preprocessing module not available: %s", exc)
        raise SystemExit(1) from exc

    import anndata as ad

    for path_str in args.h5ad_paths:
        path = Path(path_str)
        if not path.exists():
            log.warning("File not found, skipping: %s", path)
            continue

        log.info("Preprocessing scATAC: %s", path)
        try:
            adata = ad.read_h5ad(path)
            adata = preprocess_scatac(adata)

            if args.out_suffix:
                out_path = path.parent / (path.stem + args.out_suffix + ".h5ad")
            else:
                out_path = path

            adata.write_h5ad(out_path)
            log.info("Written -> %s  (%d cells x %d peaks)", out_path, adata.n_obs, adata.n_vars)
        except Exception as exc:
            log.error("Failed to preprocess %s: %s", path, exc)


if __name__ == "__main__":
    main()
