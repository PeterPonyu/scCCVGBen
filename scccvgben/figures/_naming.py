"""Filename grammar for figure outputs.

Always emits the canonical ``figXX_<scope><suffix>`` name. The legacy
``.PRELIMINARY.`` infix has been retired — partial-data renders now reuse
the canonical filename and the orchestrator's status table is the single
source of truth for completeness.
"""

from __future__ import annotations

from pathlib import Path

PRELIMINARY_INFIX = "PRELIMINARY"  # retained for back-compat detectors only


def preliminary_path(stem: str, n_obs: int, target: int, *, suffix: str = ".pdf") -> Path:
    # n_obs / target accepted for signature stability; canonical name always.
    del n_obs, target
    return Path(f"{stem}{suffix}")


def is_preliminary(path: Path) -> bool:
    return f".{PRELIMINARY_INFIX}." in path.name
