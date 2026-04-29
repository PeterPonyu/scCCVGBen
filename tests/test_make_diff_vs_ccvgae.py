from __future__ import annotations

from scripts import make_diff_vs_ccvgae as diff


def _doc(body: str) -> str:
    return "\n".join(
        [
            r"% \DIFadd{comment-only}",
            r"\providecommand{\DIFadd}[1]{#1}",
            r"\providecommand{\DIFdel}[1]{}",
            r"\begin{document}",
            body,
            r"\end{document}",
        ]
    )


def test_marker_counts_ignore_comments_and_preamble_definitions() -> None:
    counts = diff.marker_counts(_doc("unchanged body"))

    assert counts.add == 0
    assert counts.delete == 0


def test_validate_tex_requires_bidirectional_body_evidence() -> None:
    add_only = _doc(r"Body \DIFadd{new text}.")
    del_only = _doc(r"Body \DIFdel{old text}.")

    assert any("DIFdel" in error for error in diff.validate_tex(add_only))
    assert any("DIFadd" in error for error in diff.validate_tex(del_only))


def test_validate_tex_accepts_body_level_add_and_delete_with_metadata() -> None:
    text = "\n".join(
        [
            "% Baseline: baseline.tex",
            "% Baseline-SHA256: abc",
            "% Current: current.tex",
            "% Current-SHA256: def",
            "% Method: test",
            "% Prior-artifact-rejected: markerless artifact",
            _doc(r"Body \DIFdel{old text}\DIFadd{new text}."),
        ]
    )

    assert diff.validate_tex(text) == []
