#!/usr/bin/env python3
"""Generate and validate the scCCVGBen-vs-CCVGAE diff artifact.

The generator is intentionally fail-closed: the artifact is accepted only when
it builds and contains both body-level latexdiff add and delete evidence after
``\\begin{document}``.  Metadata uses repository-relative paths only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "CCVGAE" / "CCVGAE_snLaTeX" / "sn-article.tex"
CURRENT = REPO / "manuscript" / "scccvgben" / "sn-article.tex"
DIFF_DIR = REPO / "manuscript" / "scccvgben" / "diff_vs_ccvgae"
OUTPUT_TEX = DIFF_DIR / "sn-article-vs-ccvgae.diff.tex"
OUTPUT_PDF = DIFF_DIR / "sn-article-vs-ccvgae.diff.pdf"
SUPPORT_FILES = ("sn-jnl.cls", "sn-vancouver-num.bst", "references.bib")
FLOAT_ENVS = ("figure", "figure*", "table", "table*", "longtable")
INPUT_LINES = {r"\input{_supplementary_table_s1}", r"\input{_effects_table}"}
SIDECAR_SUFFIXES = (".aux", ".out", ".log", ".fls", ".fdb_latexmk", ".synctex.gz")


@dataclass(frozen=True)
class MarkerCounts:
    add: int
    delete: int


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    proc = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"


def _strip_latex_comments(tex: str) -> str:
    lines: list[str] = []
    for line in tex.splitlines():
        escaped = False
        for idx, char in enumerate(line):
            if char == "\\":
                escaped = not escaped
                continue
            if char == "%" and not escaped:
                line = line[:idx]
                break
            escaped = False
        lines.append(line)
    return "\n".join(lines)


def _document_body(tex: str) -> str:
    stripped = _strip_latex_comments(tex)
    if r"\begin{document}" not in stripped:
        return ""
    return stripped.split(r"\begin{document}", 1)[1]


def marker_counts(tex: str) -> MarkerCounts:
    body = _document_body(tex)
    if not body:
        return MarkerCounts(0, 0)
    body = re.sub(r"\\(?:providecommand|newcommand|renewcommand|DeclareRobustCommand)\s*\*?\s*\{?\\DIF(?:add|del)[^\n]*", "", body)
    return MarkerCounts(
        add=len(re.findall(r"\\DIFadd(?:FL)?\s*\{", body)),
        delete=len(re.findall(r"\\DIFdel(?:FL)?\s*\{", body)),
    )


def _remove_display_envs(tex: str) -> str:
    kept_lines = ["% OMITTED_INPUT_TABLE_FOR_DIFF_BUILD_STABILITY" if line.strip() in INPUT_LINES else line for line in tex.splitlines()]
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"^.*\\input\{_(?:supplementary_table_s1|effects_table)\}.*$", "% OMITTED_INPUT_TABLE_FOR_DIFF_BUILD_STABILITY", cleaned, flags=re.M)
    for env in FLOAT_ENVS:
        pattern = re.compile(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", re.S)
        cleaned = pattern.sub(f"\n% OMITTED_{env.replace('*', 'STAR')}_FOR_DIFF_BUILD_STABILITY\n", cleaned)
    return cleaned


def _normalise_graphicspath(tex: str) -> str:
    replacement = r"\graphicspath{{../}{../../../figures/}{figures/}}"
    if r"\graphicspath" in tex:
        return re.sub(r"\\graphicspath\{[^\n]*\}", lambda _m: replacement, tex, count=1)
    return tex.replace(r"\begin{document}", replacement + "\n\n" + r"\begin{document}", 1)


def _metadata_header(method: str) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return "\n".join(
        [
            "% TRUE_DIFF_ARTIFACT: buildable latexdiff comparison against documented CCVGAE baseline.",
            f"% Baseline: {_rel(BASELINE)}",
            f"% Baseline-SHA256: {_sha256(BASELINE)}",
            f"% Current: {_rel(CURRENT)}",
            f"% Current-SHA256: {_sha256(CURRENT)}",
            f"% Git-Revision: {_git_revision()}",
            f"% Generated-UTC: {generated_at}",
            f"% Method: {method}",
            "% Prior-artifact-rejected: previous buildable review copy had zero body-level DIFadd/DIFdel evidence.",
            "% Build-stability-note: display floats/longtables are omitted before latexdiff to avoid invalid nested caption/table markup; prose, equations, sections, references, and bibliography remain diffed.",
            "% Redaction-note: restricted accessions from the baseline are replaced with neutral placeholders before building. ",
            "% Confidentiality-note: metadata uses repository-relative paths only; no local absolute paths are embedded. ",
            "",
        ]
    )


def _copy_support_files() -> None:
    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    for name in SUPPORT_FILES:
        src = CURRENT.parent / name
        if src.exists():
            shutil.copy2(src, DIFF_DIR / name)


def _cleanup_sidecars(stem: str = "sn-article-vs-ccvgae.diff") -> None:
    for suffix in SIDECAR_SUFFIXES:
        for path in DIFF_DIR.glob(f"{stem}*{suffix}"):
            path.unlink(missing_ok=True)


def _run_latexdiff() -> str:
    if shutil.which("latexdiff") is None:
        raise RuntimeError("latexdiff is not available")
    tmp = DIFF_DIR / ".diff_build"
    tmp.mkdir(parents=True, exist_ok=True)
    old = tmp / "baseline.no-floats.tex"
    new = tmp / "current.no-floats.tex"
    old.write_text(_remove_display_envs(_read(BASELINE)), encoding="utf-8")
    new.write_text(_remove_display_envs(_read(CURRENT)), encoding="utf-8")
    cmd = [
        "latexdiff",
        "--exclude-textcmd=textbf",
        "--exclude-textcmd=section",
        "--exclude-textcmd=subsection",
        str(old),
        str(new),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("latexdiff failed to produce output")
    tex = _normalise_graphicspath(proc.stdout)
    tex = re.sub(r"^%DIF (?:DEL|ADD) /[^\n]*$", "", tex, flags=re.M)
    tex = re.sub(r"^%DIF < %% Please do not use \\input\{\.\.\.\} to include other tex files\.\s*%%?\s*$", "", tex, flags=re.M)
    # The original CCVGAE manuscript contains accession text now classified as restricted.
    # Keep the comparison buildable without carrying that nonpublic string into submission artifacts.
    tex = re.sub(r"GSM82486(?:68|69|70|72|73|75)", "restricted accession", tex)
    shutil.rmtree(tmp, ignore_errors=True)
    return _metadata_header("latexdiff-no-display-floats") + tex


def validate_tex(tex: str) -> list[str]:
    errors: list[str] = []
    if r"\begin{document}" not in tex:
        errors.append("diff TeX has no document body")
    if "Buildable review copy regenerated" in tex or "markerless copied review" in tex:
        errors.append("diff TeX still carries pseudo-diff/review-copy wording")
    counts = marker_counts(tex)
    if counts.add < 1 or counts.delete < 1:
        errors.append(f"diff lacks bidirectional body-level evidence (DIFadd={counts.add}, DIFdel={counts.delete})")
    for required in ("% Baseline:", "% Baseline-SHA256:", "% Current:", "% Current-SHA256:", "% Method:", "% Prior-artifact-rejected:"):
        if required not in tex:
            errors.append(f"diff metadata missing {required.rstrip(':')}")
    if re.search(r"(?:/home/|/Users/|\\Users\\|[A-Za-z]:\\Users\\)", tex):
        errors.append("diff TeX contains a local absolute path")
    return errors


def _build_pdf() -> None:
    proc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", OUTPUT_TEX.name],
        cwd=DIFF_DIR,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    _cleanup_sidecars()
    if proc.returncode != 0 or not OUTPUT_PDF.exists():
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
        raise RuntimeError(f"diff PDF build failed\n{tail}")


def _check_pdf_text() -> list[str]:
    if not OUTPUT_PDF.exists():
        return ["diff PDF is missing"]
    if shutil.which("pdftotext") is None:
        return ["pdftotext unavailable for diff PDF validation"]
    proc = subprocess.run(["pdftotext", str(OUTPUT_PDF), "-"], capture_output=True, text=True, timeout=30, check=False)
    if proc.returncode != 0 or len(proc.stdout.strip()) < 100:
        return ["diff PDF text is not extractable"]
    return []


def generate(*, build: bool = True) -> None:
    if not BASELINE.exists():
        raise RuntimeError(f"baseline not found: {_rel(BASELINE)}")
    if not CURRENT.exists():
        raise RuntimeError(f"current manuscript not found: {_rel(CURRENT)}")
    _copy_support_files()
    tex = _run_latexdiff()
    errors = validate_tex(tex)
    if errors:
        raise RuntimeError("; ".join(errors))
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    if build:
        _build_pdf()


def check() -> tuple[bool, dict[str, object]]:
    errors: list[str] = []
    if not OUTPUT_TEX.exists():
        errors.append("diff TeX is missing")
        tex = ""
    else:
        tex = _read(OUTPUT_TEX)
        errors.extend(validate_tex(tex))
    if OUTPUT_PDF.exists():
        errors.extend(_check_pdf_text())
    else:
        errors.append("diff PDF is missing")
    counts = marker_counts(tex)
    report: dict[str, object] = {
        "ok": not errors,
        "errors": errors,
        "diff_tex": _rel(OUTPUT_TEX),
        "diff_pdf": _rel(OUTPUT_PDF),
        "body_markers": {"DIFadd": counts.add, "DIFdel": counts.delete},
        "baseline": _rel(BASELINE),
        "current": _rel(CURRENT),
    }
    return not errors, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the existing diff artifact without regenerating it.")
    parser.add_argument("--no-build", action="store_true", help="Generate TeX only; skip PDF build.")
    parser.add_argument("--json", action="store_true", help="Emit JSON status.")
    args = parser.parse_args(argv)

    try:
        if not args.check:
            generate(build=not args.no_build)
        ok, report = check()
    except Exception as exc:  # noqa: BLE001
        report = {"ok": False, "errors": [str(exc)], "diff_tex": _rel(OUTPUT_TEX), "diff_pdf": _rel(OUTPUT_PDF)}
        ok = False

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if ok:
            counts = report.get("body_markers", {})
            print(f"diff ok: {report['diff_tex']} ({counts})")
        else:
            for error in report.get("errors", []):
                print(f"ERROR: {error}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
