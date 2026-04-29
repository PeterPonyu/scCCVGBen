from __future__ import annotations

import re

from scripts import audit_submission_ready as audit


def test_public_text_redacts_sensitive_matches_by_default(tmp_path, monkeypatch) -> None:
    target = tmp_path / "public.tex"
    raw_token = "sk-" + ("A" * 24)
    target.write_text(f"safe line\nsecret {raw_token}\n", encoding="utf-8")

    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(audit, "TEXT_TARGETS", [tmp_path])
    monkeypatch.setattr(
        audit,
        "DENY_PATTERNS",
        [("token-like secret", re.compile(r"sk-[A-Za-z0-9]+"))],
    )

    issues = audit.audit_public_text(set())

    assert len(issues) == 1
    assert issues[0].where == "public.tex:2"
    assert "redacted match" in issues[0].message
    assert "sha256=" in issues[0].message
    assert raw_token not in issues[0].message


def test_debug_snippets_are_capped_and_redacted(tmp_path, monkeypatch) -> None:
    target = tmp_path / "public.tex"
    raw_path = "/home/alice/private/source/sn-article.tex"
    target.write_text(raw_path, encoding="utf-8")

    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(audit, "TEXT_TARGETS", [target])
    monkeypatch.setattr(
        audit,
        "DENY_PATTERNS",
        [("local absolute path", re.compile(r"/home/[^\s{}\"']+"))],
    )

    issues = audit.audit_public_text(set(), include_snippets=True)

    assert len(issues) == 1
    assert "debug_snippet=<LOCAL_PATH>" in issues[0].message
    assert "/home/alice" not in issues[0].message


def test_package_hygiene_blocks_path_bearing_sidecars_only(tmp_path, monkeypatch) -> None:
    root = tmp_path / "manuscript" / "scccvgben"
    root.mkdir(parents=True)
    forbidden_fls = root / "sn-article.fls"
    forbidden_aux = root / "path-bearing.aux"
    harmless_aux = root / "harmless.aux"
    forbidden_fls.write_text("PWD /home/alice/project\n", encoding="utf-8")
    forbidden_aux.write_text("INPUT ./sn-jnl.cls\n", encoding="utf-8")
    harmless_aux.write_text("\\relax\n", encoding="utf-8")

    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(audit, "PUBLIC_SOURCE_ROOTS", (root,))
    monkeypatch.setattr(audit, "_git_tracked_public_files", lambda: set())

    issues = audit.audit_package_hygiene()
    wheres = {issue.where for issue in issues}
    messages = "\n".join(issue.message for issue in issues)

    assert "manuscript/scccvgben/sn-article.fls" in wheres
    assert "manuscript/scccvgben/path-bearing.aux" in wheres
    assert "manuscript/scccvgben/harmless.aux" not in wheres
    assert "/home/alice" not in messages


def test_package_hygiene_blocks_hidden_runtime_state(tmp_path, monkeypatch) -> None:
    root = tmp_path / "manuscript" / "scccvgben"
    hidden = root / ".omx"
    hidden.mkdir(parents=True)
    (hidden / "session.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(audit, "PUBLIC_SOURCE_ROOTS", (root,))
    monkeypatch.setattr(audit, "_git_tracked_public_files", lambda: set())

    issues = audit.audit_package_hygiene()

    assert any(issue.where == "manuscript/scccvgben/.omx" for issue in issues)
    assert all("hidden runtime state" in issue.message for issue in issues)



def test_true_diff_marker_counts_ignore_comments_and_preamble() -> None:
    tex = r"""
% \DIFadd{commented}
\providecommand{\DIFadd}[1]{#1}
\providecommand{\DIFdel}[1]{#1}
\begin{document}
% \DIFdel{commented body}
\DIFadd{added body}
\DIFdel{deleted body}
\end{document}
"""
    assert audit._body_dif_marker_counts(tex) == (1, 1)


def test_true_diff_marker_counts_reject_markerless_body() -> None:
    tex = r"""
\providecommand{\DIFadd}[1]{#1}
\providecommand{\DIFdel}[1]{#1}
\begin{document}
No body-level change evidence.
\end{document}
"""
    assert audit._body_dif_marker_counts(tex) == (0, 0)


def test_audit_true_diff_rejects_pseudo_review_copy(tmp_path, monkeypatch) -> None:
    diff_tex = tmp_path / "diff.tex"
    diff_pdf = tmp_path / "diff.pdf"
    diff_tex.write_text(
        r"""
% Buildable review copy regenerated from manuscript/scccvgben/sn-article.tex.
\providecommand{\DIFadd}[1]{#1}
\providecommand{\DIFdel}[1]{#1}
\begin{document}
No evidence.
\end{document}
""",
        encoding="utf-8",
    )
    diff_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(audit, "DIFF_TEX", diff_tex)
    monkeypatch.setattr(audit, "DIFF_PDF", diff_pdf)

    issues = audit.audit_true_diff()

    assert any("body-level DIFadd and DIFdel" in issue.message for issue in issues)
    assert any("pseudo-diff" in issue.message for issue in issues)
