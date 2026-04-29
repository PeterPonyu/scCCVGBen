from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_ai_tone.py"
spec = importlib.util.spec_from_file_location("audit_ai_tone", SCRIPT)
audit_ai_tone = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit_ai_tone
spec.loader.exec_module(audit_ai_tone)


def test_tone_audit_omits_snippets_by_default(tmp_path: Path) -> None:
    target = tmp_path / "paper.tex"
    target.write_text("This flagship method gives a result.\n", encoding="utf-8")

    findings = audit_ai_tone.audit_paths([target])

    assert [item.rule_id for item in findings] == ["TONE001_FLAGSHIP_FRAMING"]
    assert findings[0].snippet is None
    assert findings[0].snippet_hash


def test_tone_audit_ignores_bibliography_and_reviewer_comments(tmp_path: Path) -> None:
    target = tmp_path / "paper.tex"
    target.write_text(
        "\\begin{thebibliography}{1}\n"
        "A state-of-the-art flagship title.\n"
        "\\end{thebibliography}\n"
        "\\rcomment{This should mention the state-of-the-art and a flagship system.}\n",
        encoding="utf-8",
    )

    assert audit_ai_tone.audit_paths([target]) == []


def test_editorial_guard_flags_changed_protected_tokens(tmp_path: Path) -> None:
    baseline = tmp_path / "before.tex"
    current = tmp_path / "after.tex"
    baseline.write_text("Result $+0.288$ with Fig.~\\ref{fig01} and \\cite{key2024}.\n", encoding="utf-8")
    current.write_text("Result $+0.289$ with Fig.~\\ref{fig01} and \\cite{key2024}.\n", encoding="utf-8")

    issues = audit_ai_tone.editorial_guard(baseline, current)

    assert any(issue.category == "numbers" for issue in issues)
