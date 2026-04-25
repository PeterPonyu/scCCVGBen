"""Filename grammar for figure outputs.

`figXX_<scope>.pdf` for full-data outputs, `figXX_<scope>.PRELIMINARY.pdf`
when n_obs < target. The infix is uppercase, dot-separated, before the
extension. Stable across data refreshes — only the infix drops.
"""

from __future__ import annotations

from pathlib import Path

PRELIMINARY_INFIX = "PRELIMINARY"


def preliminary_path(stem: str, n_obs: int, target: int, *, suffix: str = ".pdf") -> Path:
    if n_obs < target:
        return Path(f"{stem}.{PRELIMINARY_INFIX}{suffix}")
    return Path(f"{stem}{suffix}")


def is_preliminary(path: Path) -> bool:
    return f".{PRELIMINARY_INFIX}." in path.name
