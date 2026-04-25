"""Compatibility entrypoint for the publication-ready scCCVGBen architecture diagram.

The canonical implementation lives in scripts/make_figure2_model_architecture.py.
This wrapper keeps the historical fig02_architecture.{png,pdf} filenames usable
for older manuscript-building commands while rendering the same dense model
architecture design.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_figure2_model_architecture import make_figure  # noqa: E402

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--partial-ok", action="store_true")
    parser.add_argument("--target-n", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outputs = make_figure(args.out_dir, site_static=None, stem="fig02_architecture")
    for path in outputs:
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
