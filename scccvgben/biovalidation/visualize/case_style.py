"""Case-specific visual style for biological-validation figures."""
from __future__ import annotations

from typing import Final


_DEFAULT_CASE_THEME: Final[dict[str, str]] = {
    "accent": "#0F172A",
    "cmap": "viridis",
}

CASE_THEMES: Final[dict[str, dict[str, str]]] = {
    "SD": {"accent": "#2563EB", "cmap": "Blues"},
    "GASTRIC": {"accent": "#BE123C", "cmap": "Reds"},
    "UCB": {"accent": "#B45309", "cmap": "YlOrBr"},
    "HSC_AGE": {"accent": "#0F766E", "cmap": "Greens"},
    "IR": {"accent": "#7C3AED", "cmap": "Purples"},
    "COVID": {"accent": "#DC2626", "cmap": "OrRd"},
}


def case_theme(case_id: str) -> dict[str, str]:
    """Return a copy of the style mapping for a biological-validation case."""
    return dict(CASE_THEMES.get(case_id, _DEFAULT_CASE_THEME))


def case_accent(case_id: str) -> str:
    """Return the accent color for a biological-validation case."""
    return case_theme(case_id)["accent"]


def case_cmap(case_id: str) -> str:
    """Return the continuous colormap for a biological-validation case."""
    return case_theme(case_id)["cmap"]
