#!/usr/bin/env python3
r"""Build manuscript/scccvgben/_supplementary_table_s1.tex.

The table uses the same public-safe dataset records as the website so the
submission-ready manuscript never exposes internal filename stems, local paths,
or source-code file names. Rows are identified by GEO/GSM accessions with
numeric suffixes only when an accession maps to multiple benchmark entries.
"""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "manuscript" / "scccvgben" / "_supplementary_table_s1.tex"

# Allow direct execution from any working directory.
sys.path.insert(0, str(REPO_ROOT))
from scripts.build_site_data import build_datasets  # noqa: E402


def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters in plain text fields."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for char, repl in replacements:
        s = s.replace(char, repl)
    return s


def _cells(value: int, status: str) -> str:
    if value > 0:
        return r"\num{" + str(value) + r"}"
    return r"--" if status == "not_reported" else r"\num{0}"


def build_table(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append(r"% Auto-generated public metadata table; internal identifiers are intentionally omitted.")
    lines.append(r"\begin{longtable}{lllrll>{\raggedright\arraybackslash}p{2.8cm}}")
    lines.append(
        r"\caption*{\textbf{Supplementary Table S1:} Benchmark cohort metadata. "
        r"All 200 public dataset records are listed with their accession-safe dataset ID, modality, "
        r"GEO/GSM accession, cell count, species, tissue annotation and GEO URL.} \\"
    )
    lines.append(r"\label{tab:supp_s1} \\")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset ID} & \textbf{Modality} & \textbf{Accession} & "
        r"\textbf{Cells} & \textbf{Species} & \textbf{Tissue} & \textbf{Source} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{7}{c}{\tablename\ \thetable{} (continued)} \\")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset ID} & \textbf{Modality} & \textbf{Accession} & "
        r"\textbf{Cells} & \textbf{Species} & \textbf{Tissue} & \textbf{Source} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for row in rows:
        dataset_id = _tex_escape(str(row.get("id", "")))
        modality = _tex_escape(str(row.get("modality", "")))
        accession = _tex_escape(str(row.get("GSE", "")))
        n_cells = int(row.get("cell_count") or 0)
        cells_str = _cells(n_cells, str(row.get("cell_count_status", "")))
        species = _tex_escape(str(row.get("species", "")))
        tissue = _tex_escape(str(row.get("tissue", "")))
        url = str(row.get("geo_url", "")).strip()
        url_cell = r"\href{" + url + r"}{GEO link}" if url else ""
        lines.append(" & ".join([dataset_id, modality, accession, cells_str, species, tissue, url_cell]) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_datasets()
    if len(rows) != 200:
        print(f"ERROR: expected 200 public dataset rows, found {len(rows)}", file=sys.stderr)
        sys.exit(1)
    OUT.write_text(build_table(rows), encoding="utf-8")
    print(f"Written {OUT} ({len(rows)} public dataset rows)")


if __name__ == "__main__":
    main()
