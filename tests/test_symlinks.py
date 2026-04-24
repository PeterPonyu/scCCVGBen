"""Verify workspace/reused_results symlinks resolve to the expected source."""

from __future__ import annotations

import os
import pytest

_REUSED_ROOT = "/home/zeyufu/LAB/scCCVGBen/workspace/reused_results"

# Each subdir may symlink into a different upstream directory
_SUBDIR_SOURCE_ROOT = {
    "scrna_baselines": "/home/zeyufu/LAB/CCVGAE/CG_results/CG_dl_merged",
    "scatac_baselines": "/home/zeyufu/LAB/CCVGAE/CG_results/CG_atacs/tables",
}


def _check_subdir(subdir: str):
    target = os.path.join(_REUSED_ROOT, subdir)
    if not os.path.isdir(target):
        pytest.skip(f"Symlink dir not yet created: {target}")

    entries = list(os.scandir(target))
    if not entries:
        pytest.skip(f"No entries in {target} — symlinks not set up yet")

    source_root = _SUBDIR_SOURCE_ROOT[subdir]
    for entry in entries:
        assert entry.is_symlink(), f"{entry.path} is not a symlink"
        resolved = os.path.realpath(entry.path)
        assert os.path.exists(resolved), (
            f"Symlink {entry.path} -> {resolved} is broken"
        )
        assert resolved.startswith(source_root), (
            f"Symlink {entry.path} resolves to {resolved}, "
            f"expected under {source_root}"
        )


def test_scrna_baselines_symlinks():
    _check_subdir("scrna_baselines")


def test_scatac_baselines_symlinks():
    _check_subdir("scatac_baselines")
