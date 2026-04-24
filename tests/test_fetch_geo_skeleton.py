"""Tests for scripts/fetch_geo_scrna.py — no real network calls, no real downloads.

Verifies:
  - --dry-run mode exits without NotImplementedError
  - --help returns expected argument names
  - main() can parse a candidate CSV and log expected entries
  - _pick_target_file() correctly selects filtered_feature_bc_matrix.h5
  - Idempotency guard is respected (existing non-empty output is skipped)
"""

from __future__ import annotations

import csv
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Script location
SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "fetch_geo_scrna.py")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, SCRIPT] + list(args),
        capture_output=True,
        text=True,
    )


def _make_candidate_csv(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal candidate CSV to tmp_path and return its path."""
    if rows is None:
        rows = [
            {
                "gse": "GSE999001",
                "description": "Test PBMC dataset",
                "priority": "1",
                "tissue": "pbmc",
                "organism": "human",
                "estimated_cells": "3000",
            },
            {
                "gse": "GSE999002",
                "description": "Test lung dataset",
                "priority": "2",
                "tissue": "lung",
                "organism": "mouse",
                "estimated_cells": "5000",
            },
        ]
    csv_path = tmp_path / "candidates.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# --help test
# ---------------------------------------------------------------------------

def test_help_contains_expected_args():
    """--help output must list all required arguments."""
    result = _run_script("--help")
    assert result.returncode == 0, f"--help returned {result.returncode}: {result.stderr}"
    output = result.stdout
    assert "--target" in output
    assert "--candidate-pool" in output
    assert "--candidate-csv" in output
    assert "--out" in output
    assert "--dry-run" in output


# ---------------------------------------------------------------------------
# --dry-run test (no NotImplementedError, no real HTTP)
# ---------------------------------------------------------------------------

def test_dry_run_no_error(tmp_path):
    """--dry-run must exit 0 and not raise NotImplementedError."""
    csv_path = _make_candidate_csv(tmp_path)
    out_dir = tmp_path / "out"
    result = _run_script(
        "--dry-run",
        "--candidate-csv", str(csv_path),
        "--out", str(out_dir),
        "--target", "2",
    )
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert "NotImplementedError" not in result.stdout
    assert "NotImplementedError" not in result.stderr


def test_dry_run_lists_gse_accessions(tmp_path):
    """--dry-run output must contain the GSE accession numbers."""
    csv_path = _make_candidate_csv(tmp_path)
    out_dir = tmp_path / "out"
    result = _run_script(
        "--dry-run",
        "--candidate-csv", str(csv_path),
        "--out", str(out_dir),
        "--target", "2",
    )
    assert "GSE999001" in result.stdout
    assert "GSE999002" in result.stdout


# ---------------------------------------------------------------------------
# main() import-level tests (no subprocess)
# ---------------------------------------------------------------------------

def test_read_candidate_pool_loads_rows(tmp_path):
    """_read_candidate_pool() should return the correct number of dicts."""
    # Import under a clean argv so argparse doesn't choke
    import importlib
    import sys as _sys

    csv_path = _make_candidate_csv(tmp_path)

    # Dynamically import the script as a module
    spec = importlib.util.spec_from_file_location("fetch_geo_scrna", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = mod._read_candidate_pool(csv_path)
    assert len(rows) == 2
    gses = [r.get("gse", r.get("GSE")) for r in rows]
    assert "GSE999001" in gses
    assert "GSE999002" in gses


def test_pick_target_file_prefers_filtered_h5():
    """_pick_target_file() should prefer filtered_feature_bc_matrix.h5."""
    import importlib

    spec = importlib.util.spec_from_file_location("fetch_geo_scrna", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    files = [
        {"url": "https://example.com/barcodes.tsv.gz", "filename": "barcodes.tsv.gz"},
        {"url": "https://example.com/filtered_feature_bc_matrix.h5", "filename": "filtered_feature_bc_matrix.h5"},
        {"url": "https://example.com/raw_feature_bc_matrix.h5", "filename": "raw_feature_bc_matrix.h5"},
        {"url": "https://example.com/something.h5ad", "filename": "something.h5ad"},
    ]
    chosen = mod._pick_target_file(files)
    assert chosen is not None
    assert chosen["filename"] == "filtered_feature_bc_matrix.h5"


def test_pick_target_file_rejects_h5ad_and_non_cellranger():
    """_pick_target_file() must reject .h5ad, .mtx, tarballs, raw matrices.

    Per user directive, only cellranger-standard filtered_feature_bc_matrix.h5
    (or its .h5.gz variant) is accepted to conserve bandwidth.
    """
    import importlib

    spec = importlib.util.spec_from_file_location("fetch_geo_scrna", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # No cellranger standard filtered .h5 present → None
    for files in [
        [{"url": "https://example.com/counts.h5ad", "filename": "counts.h5ad"}],
        [{"url": "https://example.com/counts.mtx.gz", "filename": "counts.mtx.gz"}],
        [{"url": "https://example.com/raw_feature_bc_matrix.h5", "filename": "raw_feature_bc_matrix.h5"}],
        [{"url": "https://example.com/GSE123_RAW.tar", "filename": "GSE123_RAW.tar"}],
    ]:
        chosen = mod._pick_target_file(files)
        assert chosen is None, f"should reject {files[0]['filename']}, got {chosen}"

    # Accept .h5.gz cellranger standard
    files_gz = [{"url": "https://example.com/f.h5.gz", "filename": "filtered_feature_bc_matrix.h5.gz"}]
    chosen = mod._pick_target_file(files_gz)
    assert chosen is not None
    assert chosen["filename"] == "filtered_feature_bc_matrix.h5.gz"


def test_pick_target_file_returns_none_when_no_match():
    """_pick_target_file() returns None when no suitable file found."""
    import importlib

    spec = importlib.util.spec_from_file_location("fetch_geo_scrna", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    files = [
        {"url": "https://example.com/barcodes.tsv.gz", "filename": "barcodes.tsv.gz"},
        {"url": "https://example.com/metadata.txt", "filename": "metadata.txt"},
    ]
    chosen = mod._pick_target_file(files)
    assert chosen is None


# ---------------------------------------------------------------------------
# Idempotency test
# ---------------------------------------------------------------------------

def test_download_gse_idempotent_skip(tmp_path, caplog):
    """_download_gse() must skip silently when output h5ad already exists and is non-empty."""
    import importlib

    spec = importlib.util.spec_from_file_location("fetch_geo_scrna", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache_dir = tmp_path / "cache"

    # Pre-create a non-empty stub
    existing = out_dir / "GSE999001_new.h5ad"
    existing.write_bytes(b"stub content that is non-empty")

    with caplog.at_level(logging.INFO, logger="fetch_geo_scrna"):
        result = mod._download_gse(
            "GSE999001",
            {"gse": "GSE999001", "tissue": "pbmc", "organism": "human", "cell_type": ""},
            out_dir,
            cache_dir,
            dry_run=False,
        )

    assert result == existing
    # Should have logged the idempotency skip message
    assert any("idempotent" in msg.lower() or "already exists" in msg.lower()
               for msg in caplog.messages)


# ---------------------------------------------------------------------------
# main() logging test — mock network, verify log entries
# ---------------------------------------------------------------------------

def test_main_dry_run_logs_candidate_count(tmp_path, caplog):
    """main() in dry-run mode logs the candidate count."""
    import importlib

    csv_path = _make_candidate_csv(tmp_path)
    out_dir = tmp_path / "out"

    spec = importlib.util.spec_from_file_location("fetch_geo_scrna", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Patch sys.argv for argparse
    test_argv = [
        SCRIPT,
        "--dry-run",
        "--candidate-csv", str(csv_path),
        "--out", str(out_dir),
        "--target", "2",
    ]
    with patch("sys.argv", test_argv), caplog.at_level(logging.INFO):
        mod.main()

    # Must log "Loaded 2 candidate accessions"
    assert any("2" in msg and "candidate" in msg.lower() for msg in caplog.messages), (
        f"Expected 'Loaded 2 candidate accessions' log. Got: {caplog.messages}"
    )
