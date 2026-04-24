"""Smoke test: all public package symbols import cleanly."""


def test_package_imports():
    import scccvgben
    from scccvgben.models import CCVGAE, ENCODER_REGISTRY
    from scccvgben.graphs import build
    from scccvgben.training import fit_one, LOCKED_CONFIG
    from scccvgben.stats import wilcoxon_signed_rank_with_holm, cliff_delta
    from scccvgben.data import load_dataset, load_reused_csv

    assert scccvgben.__version__
