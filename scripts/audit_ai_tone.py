#!/usr/bin/env python3
"""Audit manuscript prose for AI-like or overpromotional scientific tone.

The audit is intentionally conservative.  Default output reports rule IDs,
paths, line numbers, severities, and categories only; it does not emit
manuscript snippets.  Use ``--debug-snippets`` only for local editorial work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO = Path(__file__).resolve().parent.parent
MANUSCRIPT = REPO / "manuscript" / "scccvgben" / "sn-article.tex"
SUBMISSION_DIR = REPO / "manuscript" / "scccvgben" / "submission"
DEFAULT_TARGETS = [MANUSCRIPT, *sorted(SUBMISSION_DIR.glob("*.tex"))]

COMMAND_ARG_PATTERNS = [
    re.compile(r"\\(?:label|ref|eqref|autoref|pageref|cite\w*|url|email)\{[^{}]*\}"),
    re.compile(r"\\(?:includegraphics|input)(?:\[[^\]]*\])?\{[^{}]*\}"),
]
MATH_RE = re.compile(r"\$[^$]*\$")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?")


@dataclass(frozen=True)
class ToneRule:
    rule_id: str
    severity: str
    category: str
    pattern: re.Pattern[str]
    guidance: str


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    category: str
    path: str
    line: int
    guidance: str
    snippet_hash: str | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class GuardIssue:
    rule_id: str
    severity: str
    category: str
    path: str
    detail: str
    baseline_count: int
    current_count: int


TONE_RULES: tuple[ToneRule, ...] = (
    ToneRule(
        "TONE001_FLAGSHIP_FRAMING",
        "error",
        "promotional-framing",
        re.compile(r"\bflagship\b", re.I),
        "Replace marketing-style flagship framing with default, reference, or evaluated configuration.",
    ),
    ToneRule(
        "TONE002_COLLOQUIAL_HONEST_PICTURE",
        "error",
        "colloquial-framing",
        re.compile(r"\bhonest\s+picture\b", re.I),
        "Use direct scientific phrasing such as explicit account or clearer decomposition.",
    ),
    ToneRule(
        "TONE003_UNQUALIFIED_STRONGEST_AGGREGATE",
        "error",
        "unsupported-superlative",
        re.compile(r"\bstrongest\s+aggregate(?:\s+[A-Za-z-]+){0,6}\s+reference\b", re.I),
        "Replace strongest aggregate claims with cohort- and metric-qualified ranking language.",
    ),
    ToneRule(
        "TONE004_NATURAL_NEXT_STEP_CLICHE",
        "warning",
        "generic-transition",
        re.compile(r"\bnatural\s+next\s+step\b", re.I),
        "Use specific future-work language rather than generic next-step phrasing.",
    ),
    ToneRule(
        "TONE005_BUILT_AROUND_IDEA",
        "warning",
        "informal-framing",
        re.compile(r"\bbuilt\s+around\s+the\s+idea\b", re.I),
        "State the tested design premise directly.",
    ),
    ToneRule(
        "TONE006_PAYS_OFF",
        "warning",
        "colloquial-framing",
        re.compile(r"\bpays\s+off\b", re.I),
        "Use measured-effect language rather than colloquial phrasing.",
    ),
    ToneRule(
        "TONE007_STATE_OF_THE_ART",
        "warning",
        "inflated-novelty-language",
        re.compile(r"\bstate[- ]of[- ]the[- ]art\b", re.I),
        "Keep state-of-the-art wording only in reviewer quotations or cited titles.",
    ),
)

TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "numbers",
        re.compile(
            r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:[,{}]\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\\%|%)?"
        ),
    ),
    ("citation_keys", re.compile(r"\\cite\w*\{([^{}]+)\}")),
    ("labels", re.compile(r"\\label\{([^{}]+)\}")),
    ("refs", re.compile(r"\\(?:ref|eqref|autoref|pageref)\{([^{}]+)\}")),
    ("dataset_ids", re.compile(r"\b(?:GS[EM]\d+|SR[ARPX]\d+|ERP\d+)\b")),
    ("supplementary_labels", re.compile(r"\b(?:Supplementary\s+)?(?:Fig(?:ure)?|Table)s?~?S\d+\b", re.I)),
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _strip_latex_comment(line: str) -> str:
    escaped = False
    for idx, char in enumerate(line):
        if char == "\\":
            escaped = not escaped
            continue
        if char == "%" and not escaped:
            return line[:idx]
        escaped = False
    return line


def _remove_command_args(line: str) -> str:
    cleaned = line
    for pattern in COMMAND_ARG_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def _prose_line(line: str) -> str:
    line = _strip_latex_comment(line)
    line = _remove_command_args(line)
    line = MATH_RE.sub(" ", line)
    line = LATEX_COMMAND_RE.sub(" ", line)
    return re.sub(r"[{}\[\]~]+", " ", line)


def _line_mask(lines: Sequence[str]) -> list[bool]:
    """Return True for lines that should be audited as author prose."""
    mask = [True] * len(lines)
    in_bibliography = False
    reviewer_depth = 0
    for idx, raw in enumerate(lines):
        line = _strip_latex_comment(raw)
        if r"\begin{thebibliography}" in line:
            in_bibliography = True
        if in_bibliography:
            mask[idx] = False
            if r"\end{thebibliography}" in line:
                in_bibliography = False
            continue

        starts_reviewer = r"\rcomment" in line
        if reviewer_depth > 0 or starts_reviewer:
            mask[idx] = False
            if starts_reviewer and reviewer_depth == 0:
                start = line.find(r"\rcomment")
                segment = line[start:]
            else:
                segment = line
            reviewer_depth += segment.count("{") - segment.count("}")
            reviewer_depth = max(reviewer_depth, 0)
            continue
    return mask


def _snippet_hash(line: str) -> str:
    return hashlib.sha256(line.strip().encode("utf-8", errors="ignore")).hexdigest()[:12]


def _redacted_snippet(line: str) -> str:
    text = re.sub(r"\s+", " ", _strip_latex_comment(line)).strip()
    text = re.sub(r"\bGS[EM]\d+\b", "<DATASET_ID>", text)
    text = re.sub(r"[-+]?\d+(?:\.\d+)?", "<NUM>", text)
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def audit_paths(paths: Iterable[Path], *, debug_snippets: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if not path.exists():
            findings.append(
                Finding(
                    "TONE000_MISSING_TARGET",
                    "error",
                    "missing-target",
                    _relative(path),
                    0,
                    "Tone target does not exist.",
                )
            )
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        mask = _line_mask(lines)
        for line_no, raw in enumerate(lines, start=1):
            if not mask[line_no - 1]:
                continue
            prose = _prose_line(raw)
            if not prose.strip():
                continue
            for rule in TONE_RULES:
                if rule.pattern.search(prose):
                    findings.append(
                        Finding(
                            rule.rule_id,
                            rule.severity,
                            rule.category,
                            _relative(path),
                            line_no,
                            rule.guidance,
                            snippet_hash=_snippet_hash(raw),
                            snippet=_redacted_snippet(raw) if debug_snippets else None,
                        )
                    )
    return findings


def _token_counter(text: str) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for category, pattern in TOKEN_PATTERNS:
        if category == "citation_keys":
            for group in pattern.findall(text):
                for key in group.split(","):
                    key = key.strip()
                    if key:
                        counter[(category, key)] += 1
            continue
        for match in pattern.findall(text):
            token = match if isinstance(match, str) else match[0]
            counter[(category, token)] += 1
    return counter


def editorial_guard(baseline: Path, current: Path) -> list[GuardIssue]:
    base_text = baseline.read_text(encoding="utf-8", errors="ignore")
    curr_text = current.read_text(encoding="utf-8", errors="ignore")
    base = _token_counter(base_text)
    curr = _token_counter(curr_text)
    issues: list[GuardIssue] = []
    for category in sorted({key[0] for key in base} | {key[0] for key in curr}):
        base_count = sum(count for (cat, _), count in base.items() if cat == category)
        curr_count = sum(count for (cat, _), count in curr.items() if cat == category)
        if base_count != curr_count:
            issues.append(
                GuardIssue(
                    "TOKENGUARD001_CHANGED_PROTECTED_TOKENS",
                    "error",
                    category,
                    _relative(current),
                    "protected token multiset changed; review numeric/citation/label/ref preservation",
                    base_count,
                    curr_count,
                )
            )
            continue
        if Counter({key: count for key, count in base.items() if key[0] == category}) != Counter(
            {key: count for key, count in curr.items() if key[0] == category}
        ):
            issues.append(
                GuardIssue(
                    "TOKENGUARD001_CHANGED_PROTECTED_TOKENS",
                    "error",
                    category,
                    _relative(current),
                    "protected token identities changed; review numeric/citation/label/ref preservation",
                    base_count,
                    curr_count,
                )
            )
    return issues


def _finding_dict(finding: Finding) -> dict[str, object]:
    data: dict[str, object] = {
        "rule_id": finding.rule_id,
        "severity": finding.severity,
        "category": finding.category,
        "path": finding.path,
        "line": finding.line,
        "guidance": finding.guidance,
    }
    if finding.snippet_hash:
        data["snippet_hash"] = finding.snippet_hash
    if finding.snippet is not None:
        data["snippet"] = finding.snippet
    return data


def _guard_dict(issue: GuardIssue) -> dict[str, object]:
    return {
        "rule_id": issue.rule_id,
        "severity": issue.severity,
        "category": issue.category,
        "path": issue.path,
        "detail": issue.detail,
        "baseline_count": issue.baseline_count,
        "current_count": issue.current_count,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        dest="paths",
        help="Target TeX file to audit; repeatable. Defaults to manuscript and submission TeX.",
    )
    parser.add_argument(
        "--include-diff",
        action="store_true",
        help="Also audit manuscript/scccvgben/diff_vs_ccvgae/sn-article-vs-ccvgae.diff.tex if present.",
    )
    parser.add_argument(
        "--debug-snippets",
        action="store_true",
        help="Local-only: include capped, redacted snippets in output.",
    )
    parser.add_argument("--baseline", type=Path, help="Baseline TeX file for protected-token guard.")
    parser.add_argument("--current", type=Path, help="Current TeX file for protected-token guard.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    targets = list(args.paths) if args.paths else list(DEFAULT_TARGETS)
    if args.include_diff:
        targets.append(REPO / "manuscript" / "scccvgben" / "diff_vs_ccvgae" / "sn-article-vs-ccvgae.diff.tex")

    findings = audit_paths(targets, debug_snippets=args.debug_snippets)
    guard_issues: list[GuardIssue] = []
    if bool(args.baseline) != bool(args.current):
        raise SystemExit("--baseline and --current must be provided together")
    if args.baseline and args.current:
        guard_issues = editorial_guard(args.baseline, args.current)

    error_count = sum(1 for item in findings if item.severity == "error") + len(guard_issues)
    warning_count = sum(1 for item in findings if item.severity == "warning")
    payload = {
        "ok": error_count == 0,
        "targets": [_relative(path) for path in targets],
        "summary": {
            "errors": error_count,
            "warnings": warning_count,
            "findings": len(findings),
            "token_guard_errors": len(guard_issues),
        },
        "findings": [_finding_dict(item) for item in findings],
        "token_guard": [_guard_dict(item) for item in guard_issues],
        "snippets_included": bool(args.debug_snippets),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"AI-tone audit: {error_count} error(s), {warning_count} warning(s)")
        for finding in findings:
            print(
                f"{finding.severity.upper()} {finding.rule_id} "
                f"{finding.path}:{finding.line} {finding.category}"
            )
        for issue in guard_issues:
            print(f"ERROR {issue.rule_id} {issue.path} {issue.category}: {issue.detail}")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
