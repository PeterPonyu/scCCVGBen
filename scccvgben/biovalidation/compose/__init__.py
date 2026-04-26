"""Figure composition layer.

Reads a payload from ``compute.case_run.run_case`` and emits one PDF + PNG
per case using a fixed ``GridSpec`` layout. The composer never raises on a
missing panel — it falls back to ``visualize.panel.render_placeholder`` so a
single broken compute step still yields a usable figure with the other
panels intact.
"""
from .case_figure import compose_case_figure

__all__ = ["compose_case_figure"]
