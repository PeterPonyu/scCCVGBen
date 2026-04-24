"""Integration smoke tests for benchmark pipeline scripts.

Skipped by default. Run with: pytest -m integration
"""

from __future__ import annotations

import subprocess
import sys
import pytest


pytestmark = pytest.mark.integration


def _run_script(script: str, extra_args: list[str]) -> int:
    result = subprocess.run(
        [sys.executable, script] + extra_args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
    return result.returncode


@pytest.fixture(scope="module")
def synth_glob(synthetic_h5ad_dir):
    return str(synthetic_h5ad_dir / "*.h5ad")


def test_encoder_sweep_smoke(synth_glob):
    rc = _run_script(
        "scripts/run_encoder_sweep.py",
        ["--smoke", f"--datasets-glob={synth_glob}"],
    )
    assert rc == 0, "run_encoder_sweep.py --smoke returned non-zero"


def test_graph_sweep_smoke(synth_glob):
    rc = _run_script(
        "scripts/run_graph_sweep.py",
        ["--smoke", f"--datasets-glob={synth_glob}"],
    )
    assert rc == 0, "run_graph_sweep.py --smoke returned non-zero"


def test_baseline_backfill_smoke(synth_glob):
    rc = _run_script(
        "scripts/run_baseline_backfill.py",
        ["--smoke", f"--scrna-glob={synth_glob}"],
    )
    assert rc == 0, "run_baseline_backfill.py --smoke returned non-zero"
