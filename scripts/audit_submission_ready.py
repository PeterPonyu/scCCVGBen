#!/usr/bin/env python3
"""Audit submission-ready public artifacts for metadata consistency and leaks.

Checks are intentionally local and deterministic by default.  Use
``--check-external`` to resolve DOI/URL links through the network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypedDict

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "benchmark_manifest.csv"
DATASETS_JSON = REPO / "site" / "data" / "datasets.json"
SUMMARY_JSON = REPO / "site" / "data" / "summary.json"
METRICS_JSON = REPO / "site" / "data" / "metrics.json"
MANUSCRIPT = REPO / "manuscript" / "scccvgben" / "sn-article.tex"
SUPP_TABLE = REPO / "manuscript" / "scccvgben" / "_supplementary_table_s1.tex"
REFERENCES_BIB = REPO / "manuscript" / "scccvgben" / "references.bib"
SUBMISSION_DIR = REPO / "manuscript" / "scccvgben" / "submission"
FIGURES_DIR = REPO / "figures"
MANUSCRIPT_DIR = REPO / "manuscript" / "scccvgben"
DIFF_DIR = MANUSCRIPT_DIR / "diff_vs_ccvgae"

DIFF_TEX = DIFF_DIR / "sn-article-vs-ccvgae.diff.tex"
DIFF_PDF = DIFF_DIR / "sn-article-vs-ccvgae.diff.pdf"
EXPECTED_MANUSCRIPT_FILES = (
    "_supplementary_table_s1.tex",
    "_effects_table.tex",
    "fig_supp_nextjs.pdf",
    "fig_online_resource_integration.pdf",
)
WEB_OUT = REPO / "webapp" / "out"
WEB_BASE_PATH = "/scccvgben-next"
MAX_INCLUDED_WIDTH_CM = 17.0
MAX_INCLUDED_HEIGHT_CM = 21.0
PT_PER_CM = 72.0 / 2.54


class WebResourceFigureSpec(TypedDict):
    supplementary_number: int
    caption_terms: tuple[str, ...]


WEB_RESOURCE_FIGURES: dict[str, WebResourceFigureSpec] = {
    "fig_supp_nextjs": {
        "supplementary_number": 1,
        "caption_terms": (
            "Next.js interactive benchmark explorer",
            "Dataset browser",
            "Online-resource integration",
        ),
    },
    "fig_online_resource_integration": {
        "supplementary_number": 2,
        "caption_terms": (
            "Online resource integration",
            "scPortal",
            "public graph",
        ),
    },
}

# URLs that resolve only after an out-of-band deploy. Empty by default; add
# entries here only while a deploy is in flight, then remove them once the
# URL serves a 200 so the audit catches future deploy regressions.
PENDING_DEPLOY_URLS: frozenset[str] = frozenset()

TEXT_TARGETS = [
    REPO / "site" / "data",
    REPO / "site" / "static",
    REPO / "site" / "content" / "datasets",
    REPO / "site" / "content" / "methods",
    REPO / "site" / "content" / "metrics",
    REPO / "webapp" / "out",
    SUPP_TABLE,
    MANUSCRIPT,
    SUBMISSION_DIR,
    DIFF_DIR,
]

ALLOWED_SUFFIXES = {".html", ".txt", ".json", ".xml", ".md", ".tex", ".csv"}

PUBLIC_SOURCE_ROOTS = (MANUSCRIPT_DIR,)
HIDDEN_RUNTIME_NAMES = {
    ".omx",
    ".omc",
    ".codex",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
ALWAYS_PATH_BEARING_SIDECAR_SUFFIXES = {".fls", ".fdb_latexmk"}
CONDITIONAL_SIDECAR_SUFFIXES = {".aux", ".out", ".log"}
SIDECAR_SUFFIXES = ALWAYS_PATH_BEARING_SIDECAR_SUFFIXES | CONDITIONAL_SIDECAR_SUFFIXES
SIDECAR_PATH_METADATA_PATTERN = re.compile(
    r"(/home/|/Users/|\\Users\\|[A-Za-z]:\\Users\\|TEXINPUTS|PWD=|INPUT\s+\./|OUTPUT\s+\./|INPUT\s+/|OUTPUT\s+/)",
    re.I,
)

DENY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "local absolute path",
        re.compile(
            r"(?:/home/[^\s{}\"']+|/Users/[^\s{}\"']+|[A-Za-z]:\\Users\\[^\s{}\"']+|/workspace/[^\s{}\"']*|workspace/data)",
            re.I,
        ),
    ),
    (
        "hidden runtime state reference",
        re.compile(r"(?<![\w.-])(?:\.omx|\.omc|\.codex|\.pytest_cache|\.ruff_cache|\.mypy_cache)(?:[\\/]|$)", re.I),
    ),
    (
        "token-like secret",
        re.compile(
            r"\b(?:ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"
        ),
    ),
    ("LaTeX build path metadata", SIDECAR_PATH_METADATA_PATTERN),
    ("h5ad path", re.compile(r"\.h5ad\b", re.I)),
    ("raw manifest key field", re.compile(r"\bfilename_key\b", re.I)),
    ("10x internal matrix suffix", re.compile(r"filtered_(?:peak|feature)_bc_matrix|cellrangerGenome", re.I)),
    ("restricted accession leak", re.compile(r"\b(?:GSE266511|GSM82486(?:68|69|70|72|73|75))\b", re.I)),
    ("reviewer/private wording leak", re.compile(r"reviewer-private|private GEO|private accession|scheduled for public release", re.I)),
    (
        "draft placeholder leak",
        re.compile(
            r"\b(?:TODO|TBD)\b|placeholder (?:numbers|panels|cells)|\bpending\s+GPU\b|\[Institution\]|GPU authorization|will be replaced",
            re.I,
        ),
    ),
    ("internal alias", re.compile(r"\b(?:irall|wtko|hemato)\b", re.I)),
]

@dataclass
class Issue:
    severity: str
    where: str
    message: str


@dataclass
class FigureBlock:
    start: int
    body: str
    label: str | None
    includes: list[tuple[str, str]]


def _read_manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_risky_raw_key(key: str) -> bool:
    """Only exact-match raw identifiers that are filename/sample-shaped.

    Some legacy manifest keys are plain biological labels (for example
    ``lung``), which are valid public tissue words and must not be denylisted.
    """
    if not key:
        return False
    if re.search(r"filtered_(?:peak|feature)_bc_matrix|cellranger|\\.h5ad", key, re.I):
        return True
    if re.search(r"GS[EM]\\d+[_-].+", key, re.I):
        return True
    if re.search(r"[_-]GS[EM]\\d+", key, re.I):
        return True
    if re.search(r"^GS[EM]\\d+[A-Za-z].*(?:Hm|Mm|Cancer|Dev|Aged|Batch|Times|Niche)", key):
        return True
    return False


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, max(offset, 0)) + 1


def _redact_debug_snippet(text: str) -> str:
    replacements = (
        (re.compile(r"/home/[^\s{}\"']+", re.I), "<LOCAL_PATH>"),
        (re.compile(r"/Users/[^\s{}\"']+", re.I), "<LOCAL_PATH>"),
        (re.compile(r"[A-Za-z]:\\Users\\[^\s{}\"']+", re.I), "<LOCAL_PATH>"),
        (re.compile(r"/workspace/[^\s{}\"']*", re.I), "<LOCAL_PATH>"),
        (
            re.compile(
                r"\b(?:ghp_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|sk-[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16})\b"
            ),
            "<TOKEN>",
        ),
        (re.compile(r"\b(?:GSE266511|GSM82486(?:68|69|70|72|73|75))\b", re.I), "<RESTRICTED_ACCESSION>"),
        (re.compile(r"filtered_(?:peak|feature)_bc_matrix|cellrangerGenome", re.I), "<INTERNAL_SOURCE>"),
    )
    redacted = text.replace("\n", " ")
    for pattern, replacement in replacements:
        redacted = pattern.sub(replacement, redacted)
    redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted[:80] + ("…" if len(redacted) > 80 else "")


def _redacted_match_issue(
    *,
    severity: str,
    path: Path,
    label: str,
    text: str,
    match: re.Match[str],
    include_snippets: bool = False,
) -> Issue:
    matched_text = match.group(0)
    digest = hashlib.sha256(matched_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    where = f"{_rel(path)}:{_line_number(text, match.start())}"
    message = f"{label}: redacted match (sha256={digest}; chars={len(matched_text)})"
    if include_snippets:
        message = f"{message}; debug_snippet={_redact_debug_snippet(matched_text)}"
    return Issue(severity, where, message)


def _git_tracked_public_files() -> set[Path]:
    roots = [str(root.relative_to(REPO)) for root in PUBLIC_SOURCE_ROOTS if root.exists()]
    if not roots:
        return set()
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "--", *roots],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return set()
    if proc.returncode != 0:
        return set()
    return {REPO / line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _public_source_files() -> set[Path]:
    files: set[Path] = set()
    for root in PUBLIC_SOURCE_ROOTS:
        if not root.exists():
            continue
        files.update(path for path in root.rglob("*") if path.is_file())
    files.update(path for path in _git_tracked_public_files() if path.exists())
    return files


def _has_path_bearing_sidecar_metadata(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in ALWAYS_PATH_BEARING_SIDECAR_SUFFIXES:
        return True
    if suffix not in CONDITIONAL_SIDECAR_SUFFIXES or not path.exists():
        return False
    return bool(SIDECAR_PATH_METADATA_PATTERN.search(_read_text(path)))


def audit_package_hygiene() -> list[Issue]:
    issues: list[Issue] = []
    tracked = _git_tracked_public_files()

    for root in PUBLIC_SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name in HIDDEN_RUNTIME_NAMES:
                issues.append(
                    Issue(
                        "error",
                        _rel(path),
                        "hidden runtime state is forbidden under public manuscript/source scope",
                    )
                )

    for path in sorted(_public_source_files()):
        if path.suffix.lower() not in SIDECAR_SUFFIXES:
            continue
        if not _has_path_bearing_sidecar_metadata(path):
            continue
        state = "tracked" if path in tracked else "present"
        issues.append(
            Issue(
                "error",
                _rel(path),
                (
                    f"{state} path-bearing LaTeX sidecar is forbidden in public/source scope; "
                    "remove/untrack it or exclude it with a curated package manifest"
                ),
            )
        )
    return issues


def _text_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in ALLOWED_SUFFIXES:
                yield root
            continue
        for p in root.rglob("*"):
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            if not p.is_file() or p.suffix not in ALLOWED_SUFFIXES:
                continue
            # Bundle chunks are executable web code, not submission metadata;
            # page HTML/RSC text remains scanned above.
            if "/_next/static/" in p.as_posix():
                continue
            yield p


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def audit_public_text(raw_keys: set[str], *, include_snippets: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    exact_key_pattern = re.compile("|".join(re.escape(k) for k in sorted(raw_keys, key=len, reverse=True) if k)) if raw_keys else None
    for path in _text_files(TEXT_TARGETS):
        text = _read_text(path)
        for label, pattern in DENY_PATTERNS:
            m = pattern.search(text)
            if m:
                issues.append(
                    _redacted_match_issue(
                        severity="error",
                        path=path,
                        label=label,
                        text=text,
                        match=m,
                        include_snippets=include_snippets,
                    )
                )
        if exact_key_pattern:
            m = exact_key_pattern.search(text)
            if m:
                issues.append(
                    _redacted_match_issue(
                        severity="error",
                        path=path,
                        label="raw internal dataset identifier",
                        text=text,
                        match=m,
                        include_snippets=include_snippets,
                    )
                )
    return issues


def audit_dataset_json(manifest_rows: list[dict[str, str]]) -> list[Issue]:
    issues: list[Issue] = []
    datasets = json.loads(DATASETS_JSON.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    if len(datasets) != 200:
        issues.append(Issue("error", str(DATASETS_JSON.relative_to(REPO)), f"expected 200 records, found {len(datasets)}"))
    ids = [d.get("id", "") for d in datasets]
    if len(ids) != len(set(ids)):
        issues.append(Issue("error", str(DATASETS_JSON.relative_to(REPO)), "dataset IDs are not unique"))
    modality_counts = {m: sum(1 for d in datasets if d.get("modality") == m) for m in ("scRNA", "scATAC")}
    if modality_counts != {"scRNA": 100, "scATAC": 100}:
        issues.append(Issue("error", str(DATASETS_JSON.relative_to(REPO)), f"unexpected modality balance {modality_counts}"))
    restricted = [d for d in datasets if d.get("release_status") == "restricted"]
    if len(restricted) != 6:
        issues.append(Issue("error", str(DATASETS_JSON.relative_to(REPO)), f"expected 6 restricted rows, found {len(restricted)}"))
    for d in datasets:
        if "filename_key" in d:
            issues.append(Issue("error", str(DATASETS_JSON.relative_to(REPO)), "filename_key present in public JSON"))
        if d.get("release_status") == "restricted":
            if d.get("geo_url") or d.get("GSE") != "restricted" or d.get("pubmed_id"):
                issues.append(Issue("error", d.get("id", "<unknown>"), "restricted row exposes accession/URL/PubMed"))
        elif str(d.get("GSE", "")).startswith(("GSE", "GSM")) and d.get("geo_url"):
            if str(d["GSE"]) not in str(d["geo_url"]):
                issues.append(Issue("warning", d.get("id", "<unknown>"), "GEO URL does not contain accession"))
        if int(d.get("cell_count") or 0) <= 0 and d.get("cell_count_status") != "not_reported":
            issues.append(Issue("warning", d.get("id", "<unknown>"), "non-positive cell count without not_reported status"))
    displayed_metrics = sum(len(v) for v in metrics.values())
    if displayed_metrics != 20 or summary.get("metrics_total") != 20:
        issues.append(Issue("error", str(METRICS_JSON.relative_to(REPO)), f"expected 20 display metrics, found {displayed_metrics}; summary={summary.get('metrics_total')}"))
    return issues


def audit_web_export() -> list[Issue]:
    issues: list[Issue] = []
    index = WEB_OUT / "index.html"
    if not index.exists():
        return [Issue("warning", str(index.relative_to(REPO)), "Next.js export not found; skipped subpath audit")]
    html = index.read_text(encoding="utf-8", errors="ignore")
    rel = str(index.relative_to(REPO))
    if re.search(r'(?:href|src)="/_next/static/', html):
        issues.append(Issue("error", rel, "root-relative Next.js asset path; build with NEXT_PUBLIC_BASE_PATH=/scccvgben-next"))
    if re.search(r'href="/(?:datasets|methods|metrics|resource-integration|supplementary-figure)(?:/|")', html):
        issues.append(Issue("error", rel, "root-relative internal link; use next/link and build with the public base path"))
    if f'{WEB_BASE_PATH}/_next/static/' not in html:
        issues.append(Issue("error", rel, f"missing {WEB_BASE_PATH} asset prefix in exported index"))
    return issues


def _strip_comments(tex: str) -> str:
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


def _compact_tex(tex: str) -> str:
    return re.sub(r"\s+", " ", tex).strip()


def _find_figure_blocks(tex: str) -> list[FigureBlock]:
    blocks: list[FigureBlock] = []
    figure_re = re.compile(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.S)
    include_re = re.compile(
        r"\\includegraphics(?:\[(?P<options>[^\]]*)\])?\{(?P<target>[^}]+)\}"
    )
    for match in figure_re.finditer(tex):
        body = match.group(0)
        label_match = re.search(r"\\label\{([^}]+)\}", body)
        includes = [
            (include.group("options") or "", include.group("target"))
            for include in include_re.finditer(body)
        ]
        blocks.append(
            FigureBlock(
                start=match.start(),
                body=body,
                label=label_match.group(1) if label_match else None,
                includes=includes,
            )
        )
    return blocks


def _supplementary_figure_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for match in re.finditer(r"Supplementary\s+Fig(?:ure)?s?\.?~?([^.;)]{0,120})", text):
        snippet = match.group(1)
        for start, end in re.findall(r"S(\d+)\s*--\s*S?(\d+)", snippet):
            numbers.update(range(int(start), int(end) + 1))
        for number in re.findall(r"S(\d+)", snippet):
            numbers.add(int(number))
    return numbers


def _graphics_stem(target: str) -> str:
    return Path(target).name.removesuffix(".pdf")


def _graphics_pdf_path(target: str) -> Path | None:
    target_path = Path(target)
    names = [target_path]
    if target_path.suffix.lower() != ".pdf":
        names.append(target_path.with_suffix(".pdf"))
    candidates: list[Path] = []
    for name in names:
        if name.is_absolute():
            candidates.append(name)
        else:
            candidates.extend([MANUSCRIPT_DIR / name, FIGURES_DIR / name, REPO / name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_size_cm(pdf: Path) -> tuple[float, float] | None:
    proc = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        return None
    match = re.search(r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", proc.stdout)
    if not match:
        return None
    width_pt, height_pt = float(match.group(1)), float(match.group(2))
    return width_pt / PT_PER_CM, height_pt / PT_PER_CM


def _graphics_options(raw_options: str) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    for item in (part.strip() for part in raw_options.split(",")):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            options[key.strip()] = value.strip()
        else:
            options[item] = True
    return options


def _length_cm(expr: str | bool | None) -> float | None:
    if expr is None or expr is True:
        return None
    cleaned = str(expr).replace(" ", "")
    text_width_aliases = {"\\textwidth", "\\linewidth", "\\columnwidth"}
    if cleaned in text_width_aliases:
        return MAX_INCLUDED_WIDTH_CM
    if cleaned == "\\textheight":
        return MAX_INCLUDED_HEIGHT_CM
    match = re.fullmatch(r"([0-9.]+)\\(?:textwidth|linewidth|columnwidth)", cleaned)
    if match:
        return float(match.group(1)) * MAX_INCLUDED_WIDTH_CM
    match = re.fullmatch(r"([0-9.]+)\\textheight", cleaned)
    if match:
        return float(match.group(1)) * MAX_INCLUDED_HEIGHT_CM
    match = re.fullmatch(r"([0-9.]+)cm", cleaned)
    if match:
        return float(match.group(1))
    match = re.fullmatch(r"([0-9.]+)in", cleaned)
    if match:
        return float(match.group(1)) * 2.54
    match = re.fullmatch(r"([0-9.]+)pt", cleaned)
    if match:
        return float(match.group(1)) / PT_PER_CM
    return None


def _included_size_cm(
    natural_width_cm: float,
    natural_height_cm: float,
    raw_options: str,
) -> tuple[float, float]:
    options = _graphics_options(raw_options)
    width_cm = _length_cm(options.get("width"))
    height_cm = _length_cm(options.get("height"))
    keep_aspect = bool(options.get("keepaspectratio"))
    if width_cm and height_cm:
        if keep_aspect:
            scale = min(width_cm / natural_width_cm, height_cm / natural_height_cm)
            return natural_width_cm * scale, natural_height_cm * scale
        return width_cm, height_cm
    if width_cm:
        scale = width_cm / natural_width_cm
        return width_cm, natural_height_cm * scale
    if height_cm:
        scale = height_cm / natural_height_cm
        return natural_width_cm * scale, height_cm
    return natural_width_cm, natural_height_cm


def _format_size(width_cm: float, height_cm: float) -> str:
    return f"{width_cm:.2f}x{height_cm:.2f} cm"


def audit_references(check_external: bool = False, max_external: int | None = None) -> list[Issue]:
    issues: list[Issue] = []
    tex = _strip_comments(MANUSCRIPT.read_text(encoding="utf-8", errors="ignore"))
    cited: set[str] = set()
    for m in re.finditer(r"\\cite(?:[a-zA-Z*]*)?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(",") if k.strip())
    inline_bib = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    missing = cited - inline_bib
    if missing:
        issues.append(Issue("error", str(MANUSCRIPT.relative_to(REPO)), f"missing inline bibitems for cited keys: {', '.join(sorted(missing))}"))
    unused_inline = sorted(inline_bib - cited)
    if unused_inline:
        issues.append(Issue("warning", str(MANUSCRIPT.relative_to(REPO)), f"unused inline bibitems: {', '.join(unused_inline[:12])}"))

    if REFERENCES_BIB.exists():
        bib_text = REFERENCES_BIB.read_text(encoding="utf-8", errors="ignore")
        bib_entries = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
        bib_only = sorted(bib_entries - cited)
        if bib_only:
            issues.append(Issue("warning", str(REFERENCES_BIB.relative_to(REPO)), f"bib file entries not cited in manuscript: {', '.join(bib_only[:12])}"))

    doi_like = sorted({
        doi.rstrip(".,;:")
        for doi in re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", tex)
    })
    for doi in doi_like:
        if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", doi):
            issues.append(Issue("error", str(MANUSCRIPT.relative_to(REPO)), f"malformed DOI: {doi}"))

    if check_external:
        for doi in doi_like[: max_external or None]:
            quoted = urllib.parse.quote(doi, safe="/")
            handle_url = f"https://doi.org/api/handles/{quoted}"
            try:
                req = urllib.request.Request(handle_url, headers={"User-Agent": "scCCVGBen-submission-audit/1.0"})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    if payload.get("responseCode") != 1:
                        issues.append(Issue("warning", f"https://doi.org/{doi}", f"DOI handle responseCode={payload.get('responseCode')}"))
            except Exception as exc:  # noqa: BLE001
                issues.append(Issue("warning", f"https://doi.org/{doi}", f"DOI handle unresolved: {exc.__class__.__name__}"))

        urls = [
            url.rstrip(".,;:")
            for url in sorted(set(re.findall(r"https?://[^}\s]+", tex)))
            if "doi.org/" not in url
        ]
        if max_external:
            urls = urls[:max_external]
        for url in urls:
            if url in PENDING_DEPLOY_URLS or url.rstrip("/") + "/" in PENDING_DEPLOY_URLS:
                continue
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "scCCVGBen-submission-audit/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    if resp.status >= 400:
                        issues.append(Issue("warning", url, f"HTTP {resp.status}"))
            except urllib.error.HTTPError as exc:
                # Some endpoints reject HEAD but resolve with GET; try a tiny GET.
                try:
                    req_get = urllib.request.Request(url, method="GET", headers={"User-Agent": "scCCVGBen-submission-audit/1.0"})
                    with urllib.request.urlopen(req_get, timeout=12) as resp:
                        if resp.status >= 400:
                            issues.append(Issue("warning", url, f"HTTP {resp.status}"))
                except Exception as exc2:  # noqa: BLE001
                    issues.append(Issue("warning", url, f"unresolved: {exc.code}/{exc2.__class__.__name__}"))
            except Exception as exc:  # noqa: BLE001
                issues.append(Issue("warning", url, f"unresolved: {exc.__class__.__name__}"))
    return issues


ONLINE_RESOURCE_URLS = {
    "companion": "https://peterponyu.github.io/scccvgben-next/",
    "companion_resource": "https://peterponyu.github.io/scccvgben-next/resource-integration/",
    "homepage": "https://peterponyu.github.io/",
    "scportal": "https://peterponyu.github.io/scportal/",
    "public_graph_manifest": "https://peterponyu.github.io/public-graph.manifest.json",
    "sitemap": "https://peterponyu.github.io/sitemap.xml",
    "scportal_manifest_control": "https://peterponyu.github.io/scportal/public-graph.manifest.json",
}


def _url_text(url: str, timeout: int = 15) -> tuple[int | None, str, str | None]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scCCVGBen-submission-audit/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="ignore"), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), "", None
    except Exception as exc:  # noqa: BLE001
        return None, "", exc.__class__.__name__


def _has_strong_public_graph_claim() -> bool:
    targets = [MANUSCRIPT, REPO / "webapp" / "src" / "app" / "page.tsx", REPO / "webapp" / "src" / "app" / "resource-integration" / "page.tsx", REPO / "webapp" / "src" / "app" / "supplementary-figure" / "page.tsx"]
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in targets if path.exists())
    strong_patterns = (
        "registered against the broader scPortal public graph",
        "linked from PeterPonyu's homepage and scPortal public graph",
        "linked from PeterPonyu&apos;s homepage and scPortal discovery hub",
        "Homepage → scPortal → scCCVGBen",
    )
    return any(pattern in text for pattern in strong_patterns)


def audit_online_resource_graph(check_external: bool = False) -> list[Issue]:
    """Validate reciprocal public-graph evidence for the online-resource claim.

    Local audits remain deterministic.  Network checks run only under
    ``--check-external`` because they depend on deployed GitHub Pages state.
    """
    if not check_external:
        return []

    issues: list[Issue] = []
    strong_claim = _has_strong_public_graph_claim()
    severity = "error" if strong_claim else "warning"

    expected_200 = (
        "companion",
        "companion_resource",
        "homepage",
        "scportal",
        "public_graph_manifest",
        "sitemap",
    )
    fetched: dict[str, tuple[int | None, str, str | None]] = {}
    for key in (*expected_200, "scportal_manifest_control"):
        fetched[key] = _url_text(ONLINE_RESOURCE_URLS[key])

    for key in expected_200:
        status, _text, error = fetched[key]
        if status != 200:
            msg = f"expected HTTP 200 for online-resource URL, found {status or error}"
            issues.append(Issue(severity, ONLINE_RESOURCE_URLS[key], msg))

    control_status, _control_text, _control_error = fetched["scportal_manifest_control"]
    if control_status not in (404, 200):
        issues.append(
            Issue(
                "warning",
                ONLINE_RESOURCE_URLS["scportal_manifest_control"],
                f"expected 404 control (manifest normally rooted at homepage) or a mirrored 200, found {control_status}",
            )
        )

    manifest_status, manifest_text, _manifest_error = fetched["public_graph_manifest"]
    manifest_site: dict[str, object] | None = None
    if manifest_status == 200:
        try:
            manifest_payload = json.loads(manifest_text)
            sites = manifest_payload.get("sites", [])
            for site in sites:
                site_text = json.dumps(site, sort_keys=True).lower()
                if site.get("id") == "scccvgben" or "scccvgben-next" in site_text or site.get("name") == "scCCVGBen":
                    manifest_site = site
                    break
        except json.JSONDecodeError as exc:
            issues.append(Issue("error", ONLINE_RESOURCE_URLS["public_graph_manifest"], f"manifest JSON parse failed: {exc.__class__.__name__}"))

    if manifest_site is None:
        issues.append(Issue(severity, ONLINE_RESOURCE_URLS["public_graph_manifest"], "public graph manifest lacks an scCCVGBen entry"))
    else:
        canonical = str(manifest_site.get("canonical_url", ""))
        if canonical.rstrip("/") != ONLINE_RESOURCE_URLS["companion"].rstrip("/"):
            issues.append(Issue("error", ONLINE_RESOURCE_URLS["public_graph_manifest"], f"scCCVGBen canonical_url mismatch: {canonical}"))
        visibility = manifest_site.get("visibility", {})
        if isinstance(visibility, dict) and visibility.get("sitemap") is True:
            sitemap_status, sitemap_text, _sitemap_error = fetched["sitemap"]
            if sitemap_status == 200 and "scccvgben-next" not in sitemap_text:
                issues.append(Issue("error", ONLINE_RESOURCE_URLS["sitemap"], "sitemap omits /scccvgben-next/ despite manifest sitemap=true"))

    homepage_status, homepage_text, _homepage_error = fetched["homepage"]
    if homepage_status == 200 and not ("scccvgben-next" in homepage_text.lower() and "scccvgben" in homepage_text.lower()):
        issues.append(Issue(severity, ONLINE_RESOURCE_URLS["homepage"], "homepage does not expose an scCCVGBen route/link"))

    scportal_status, scportal_text, _scportal_error = fetched["scportal"]
    if scportal_status == 200 and not ("scccvgben-next" in scportal_text.lower() or "scccvgben" in scportal_text.lower()):
        issues.append(Issue(severity, ONLINE_RESOURCE_URLS["scportal"], "scPortal does not expose an scCCVGBen route/link"))

    return issues


def _audit_graphics_dimensions(stem: str, target: str, raw_options: str) -> list[Issue]:
    issues: list[Issue] = []
    pdf = _graphics_pdf_path(target)
    where = str(MANUSCRIPT.relative_to(REPO))
    if not pdf or not pdf.exists():
        issues.append(Issue("error", where, f"{stem} include target is missing: {target}"))
        return issues
    size = _pdf_page_size_cm(pdf)
    rel_pdf = str(pdf.relative_to(REPO)) if pdf.is_relative_to(REPO) else str(pdf)
    if size is None:
        issues.append(Issue("warning", rel_pdf, "could not read PDF page size with pdfinfo"))
        return issues

    raw_width_cm, raw_height_cm = size
    if raw_width_cm > MAX_INCLUDED_WIDTH_CM or raw_height_cm > MAX_INCLUDED_HEIGHT_CM:
        issues.append(
            Issue(
                "error",
                rel_pdf,
                (
                    f"raw web-resource page is {_format_size(raw_width_cm, raw_height_cm)}; "
                    f"policy is <= {_format_size(MAX_INCLUDED_WIDTH_CM, MAX_INCLUDED_HEIGHT_CM)}"
                ),
            )
        )

    included_width_cm, included_height_cm = _included_size_cm(
        raw_width_cm,
        raw_height_cm,
        raw_options,
    )
    if included_width_cm > MAX_INCLUDED_WIDTH_CM or included_height_cm > MAX_INCLUDED_HEIGHT_CM:
        options = raw_options or "natural size"
        issues.append(
            Issue(
                "error",
                where,
                (
                    f"{stem} included as {_format_size(included_width_cm, included_height_cm)} "
                    f"from {target} [{options}]; policy is <= "
                    f"{_format_size(MAX_INCLUDED_WIDTH_CM, MAX_INCLUDED_HEIGHT_CM)}"
                ),
            )
        )
    return issues


def audit_web_resource_figures() -> list[Issue]:
    issues: list[Issue] = []
    tex = _strip_comments(MANUSCRIPT.read_text(encoding="utf-8", errors="ignore"))
    blocks = _find_figure_blocks(tex)
    appendix_start = tex.find(r"\section{Supplementary Material}")
    if appendix_start < 0:
        issues.append(Issue("error", str(MANUSCRIPT.relative_to(REPO)), "missing Supplementary Material section"))

    declared_numbers: dict[int, list[str]] = {}
    for block in blocks:
        block_numbers = _supplementary_figure_numbers(block.body)
        if block.label:
            label_match = re.match(r"figS(\d+)", block.label)
            if label_match:
                block_numbers.add(int(label_match.group(1)))
        for number in block_numbers:
            declared_numbers.setdefault(number, []).append(block.label or "<unlabelled>")
    for number, labels in sorted(declared_numbers.items()):
        if len(labels) > 1:
            issues.append(
                Issue(
                    "error",
                    str(MANUSCRIPT.relative_to(REPO)),
                    f"Supplementary Figure S{number} is declared by multiple blocks: {', '.join(labels)}",
                )
            )

    mentioned_numbers = _supplementary_figure_numbers(tex)
    missing_numbers = sorted(mentioned_numbers - set(declared_numbers))
    if missing_numbers:
        joined = ", ".join(f"S{number}" for number in missing_numbers)
        issues.append(
            Issue(
                "error",
                str(MANUSCRIPT.relative_to(REPO)),
                f"Supplementary Figure mention(s) lack matching figure blocks: {joined}",
            )
        )

    for stem, spec in WEB_RESOURCE_FIGURES.items():
        expected_number = int(spec["supplementary_number"])
        expected_terms = tuple(str(term).lower() for term in spec["caption_terms"])
        figure_pdf = FIGURES_DIR / f"{stem}.pdf"
        manuscript_pdf = MANUSCRIPT_DIR / f"{stem}.pdf"
        if not manuscript_pdf.exists():
            issues.append(
                Issue(
                    "error",
                    str(manuscript_pdf.relative_to(REPO)),
                    f"required submission PDF is missing for {stem}",
                )
            )
        if figure_pdf.exists() and manuscript_pdf.exists() and _hash_file(figure_pdf) != _hash_file(manuscript_pdf):
            issues.append(
                Issue(
                    "error",
                    str(manuscript_pdf.relative_to(REPO)),
                    f"manuscript copy differs from figures/{stem}.pdf; rerun make_figure_supp_nextjs.py",
                )
            )

        matching_blocks: list[tuple[FigureBlock, list[tuple[str, str]]]] = []
        for block in blocks:
            matching_includes = [
                (options, target)
                for options, target in block.includes
                if _graphics_stem(target) == stem
            ]
            if matching_includes:
                matching_blocks.append((block, matching_includes))

        if not matching_blocks:
            issues.append(
                Issue(
                    "error",
                    str(MANUSCRIPT.relative_to(REPO)),
                    f"{stem}.pdf is not included by a manuscript figure block",
                )
            )
            continue

        for block, includes in matching_blocks:
            label = block.label or "<unlabelled>"
            if appendix_start >= 0 and block.start < appendix_start:
                issues.append(
                    Issue(
                        "error",
                        str(MANUSCRIPT.relative_to(REPO)),
                        f"{stem}.pdf is included before the supplementary appendix in {label}",
                    )
                )
            block_numbers = _supplementary_figure_numbers(block.body)
            if block.label:
                label_match = re.match(r"figS(\d+)", block.label)
                if label_match:
                    block_numbers.add(int(label_match.group(1)))
            if expected_number not in block_numbers:
                issues.append(
                    Issue(
                        "error",
                        str(MANUSCRIPT.relative_to(REPO)),
                        f"{stem}.pdf must be declared as Supplementary Figure S{expected_number}",
                    )
                )
            if not re.fullmatch(fr"figS{expected_number}(?:[_A-Za-z0-9-]+)?", label):
                issues.append(
                    Issue(
                        "error",
                        str(MANUSCRIPT.relative_to(REPO)),
                        f"{stem}.pdf label should encode S{expected_number}, found {label}",
                    )
                )
            compact_block = _compact_tex(block.body).lower()
            for term in expected_terms:
                if term not in compact_block:
                    issues.append(
                        Issue(
                            "error",
                            str(MANUSCRIPT.relative_to(REPO)),
                            f"{stem}.pdf caption is missing route/content term: {term}",
                        )
                    )
            for raw_options, target in includes:
                issues.extend(_audit_graphics_dimensions(stem, target, raw_options))
    return issues


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


def _body_dif_marker_counts(tex: str) -> tuple[int, int]:
    body = _document_body(tex)
    if not body:
        return 0, 0
    body = re.sub(
        r"\\(?:providecommand|newcommand|renewcommand|DeclareRobustCommand)\s*\*?\s*\{?\\DIF(?:add|del)[^\n]*",
        "",
        body,
    )
    return (
        len(re.findall(r"\\DIFadd(?:FL)?\s*\{", body)),
        len(re.findall(r"\\DIFdel(?:FL)?\s*\{", body)),
    )


def audit_true_diff() -> list[Issue]:
    issues: list[Issue] = []
    if not DIFF_TEX.exists():
        return [Issue("error", _rel(DIFF_TEX), "true-diff TeX artifact is missing")]
    tex = DIFF_TEX.read_text(encoding="utf-8", errors="ignore")
    add_count, del_count = _body_dif_marker_counts(tex)
    if add_count < 1 or del_count < 1:
        issues.append(
            Issue(
                "error",
                _rel(DIFF_TEX),
                f"true-diff validation requires body-level DIFadd and DIFdel markers after begin{{document}}; found add={add_count}, delete={del_count}",
            )
        )
    if "Buildable review copy regenerated" in tex:
        issues.append(Issue("error", _rel(DIFF_TEX), "pseudo-diff review-copy header is forbidden"))
    for token in ("% Baseline:", "% Baseline-SHA256:", "% Current:", "% Current-SHA256:", "% Method:", "% Prior-artifact-rejected:"):
        if token not in tex:
            issues.append(Issue("error", _rel(DIFF_TEX), f"true-diff metadata missing {token.rstrip(':')}"))
    if re.search(r"(?:/home/|/Users/|\\Users\\|[A-Za-z]:\\Users\\)", tex):
        issues.append(Issue("error", _rel(DIFF_TEX), "true-diff TeX metadata/body contains a local absolute path"))
    if not DIFF_PDF.exists():
        issues.append(Issue("error", _rel(DIFF_PDF), "true-diff PDF is missing"))
    else:
        proc = subprocess.run(["pdftotext", str(DIFF_PDF), "-"], capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode != 0 or len(proc.stdout.strip()) < 100:
            issues.append(Issue("error", _rel(DIFF_PDF), "true-diff PDF text is not extractable"))
    return issues


def _extract_labels(tex: str) -> list[str]:
    return re.findall(r"\\label\{([^{}]+)\}", _strip_comments(tex))


def _extract_refs(tex: str) -> list[str]:
    return re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^{}]+)\}", _strip_comments(tex))


def audit_manuscript_consistency() -> list[Issue]:
    issues: list[Issue] = []
    if not MANUSCRIPT.exists():
        return [Issue("error", _rel(MANUSCRIPT), "main manuscript TeX is missing")]
    tex = MANUSCRIPT.read_text(encoding="utf-8", errors="ignore")
    stripped = _strip_comments(tex)
    labels = _extract_labels(tex)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        issues.append(Issue("error", _rel(MANUSCRIPT), f"duplicate labels: {', '.join(duplicate_labels[:20])}"))
    missing_refs = sorted(set(_extract_refs(tex)) - set(labels))
    if missing_refs:
        issues.append(Issue("error", _rel(MANUSCRIPT), f"references without labels: {', '.join(missing_refs[:20])}"))
    for name in EXPECTED_MANUSCRIPT_FILES:
        if not (MANUSCRIPT_DIR / name).exists():
            issues.append(Issue("error", _rel(MANUSCRIPT_DIR / name), "required manuscript companion artifact is missing"))
    for expected in (r"\input{_supplementary_table_s1}", r"\input{_effects_table}"):
        if expected not in stripped:
            issues.append(Issue("error", _rel(MANUSCRIPT), f"missing required table include {expected}"))
    for label in ("figS1_nextjs", "figS2_online_resource"):
        if label not in set(labels):
            issues.append(Issue("error", _rel(MANUSCRIPT), f"missing supplementary figure label {label}"))
    if "??" in stripped:
        issues.append(Issue("error", _rel(MANUSCRIPT), "unresolved TeX reference marker '??' present in source"))
    effect_text = (MANUSCRIPT_DIR / "_effects_table.tex").read_text(encoding="utf-8", errors="ignore") if (MANUSCRIPT_DIR / "_effects_table.tex").exists() else ""
    effect_captions = re.findall(r"\\caption\{Effect sizes behind Fig\.", effect_text)
    if effect_text and len(effect_captions) != 9:
        issues.append(Issue("warning", _rel(MANUSCRIPT_DIR / "_effects_table.tex"), f"expected 9 effect-size longtable captions for S2--S10 sequence, found {len(effect_captions)}"))
    pdf = MANUSCRIPT_DIR / "sn-article.pdf"
    if pdf.exists() and subprocess.run(["bash", "-lc", "command -v pdftotext"], capture_output=True, text=True).returncode == 0:
        proc = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode != 0 or "scCCVGBen" not in proc.stdout or "Supplementary" not in proc.stdout:
            issues.append(Issue("warning", _rel(pdf), "PDF text extraction lacks expected safe manuscript anchors"))
    return issues


def audit_pdf_text(*, include_snippets: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    pdftotext = subprocess.run(["bash", "-lc", "command -v pdftotext"], capture_output=True, text=True)
    if pdftotext.returncode != 0:
        return [Issue("warning", "pdf", "pdftotext not available; skipped PDF text audit")]
    pdfs = [
        REPO / "figures" / "fig_supp_nextjs.pdf",
        REPO / "figures" / "fig_online_resource_integration.pdf",
        REPO / "manuscript" / "scccvgben" / "fig_supp_nextjs.pdf",
        REPO / "manuscript" / "scccvgben" / "fig_online_resource_integration.pdf",
        REPO / "manuscript" / "scccvgben" / "sn-article.pdf",
    ]
    manuscript_dir = REPO / "manuscript" / "scccvgben"
    if manuscript_dir.exists():
        pdfs.extend(sorted(manuscript_dir.rglob("*.pdf")))
    if SUBMISSION_DIR.exists():
        pdfs.extend(sorted(SUBMISSION_DIR.glob("*.pdf")))
    for pdf in sorted(set(pdfs)):
        if not pdf.exists():
            continue
        proc = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            issues.append(Issue("warning", str(pdf.relative_to(REPO)), "could not extract PDF text"))
            continue
        for label, pattern in DENY_PATTERNS:
            m = pattern.search(proc.stdout)
            if m:
                issues.append(
                    _redacted_match_issue(
                        severity="error",
                        path=pdf,
                        label=label,
                        text=proc.stdout,
                        match=m,
                        include_snippets=include_snippets,
                    )
                )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-external", action="store_true", help="Resolve DOI/URL links over the network.")
    parser.add_argument("--max-external", type=int, default=None, help="Limit external checks for smoke testing.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--debug-snippets",
        action="store_true",
        help="Local-only: include capped, redacted snippets for leak debugging.",
    )
    args = parser.parse_args(argv)

    manifest_rows = _read_manifest_rows()
    raw_keys = {
        r.get("filename_key", "")
        for r in manifest_rows
        if _is_risky_raw_key(r.get("filename_key", ""))
    }
    issues: list[Issue] = []
    issues.extend(audit_package_hygiene())
    issues.extend(audit_dataset_json(manifest_rows))
    issues.extend(audit_web_export())
    issues.extend(audit_public_text(raw_keys, include_snippets=args.debug_snippets))
    issues.extend(audit_references(check_external=args.check_external, max_external=args.max_external))
    issues.extend(audit_online_resource_graph(check_external=args.check_external))
    issues.extend(audit_web_resource_figures())
    issues.extend(audit_true_diff())
    issues.extend(audit_manuscript_consistency())
    issues.extend(audit_pdf_text(include_snippets=args.debug_snippets))

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    if args.json:
        print(json.dumps({"errors": [i.__dict__ for i in errors], "warnings": [i.__dict__ for i in warnings]}, indent=2))
    else:
        for issue in issues:
            print(f"{issue.severity.upper()}: {issue.where}: {issue.message}")
        print(f"audit complete: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
