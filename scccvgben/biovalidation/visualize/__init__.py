"""Standardized visualization primitives for the bio-validation framework.

All panel plotters export a ``render_*`` function with the signature::

    render_<name>(ax: plt.Axes, ...payload) -> None

so the composer can drop them into a fixed-grid layout without juggling
figure sizes or DPI.

Panel size defaults use the :class:`PanelSpec` constants (``PANEL_W_INCH``,
``PANEL_H_INCH``, ``PANEL_DPI``) — the composer enforces these uniformly to
avoid the aspect-ratio drift that plagued the legacy reference compose pipeline.
"""
from .panel import PANEL_W_INCH, PANEL_H_INCH, PANEL_DPI, PanelSpec, render_placeholder

__all__ = [
    "PANEL_W_INCH", "PANEL_H_INCH", "PANEL_DPI",
    "PanelSpec", "render_placeholder",
]
