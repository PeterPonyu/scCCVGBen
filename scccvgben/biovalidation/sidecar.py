"""Small tabular sidecars for regenerable bio-validation figures.

The single-case figures are still the canonical cached visual artifacts, but
the paired manuscript figures need tabular provenance so their compact panels
can be redrawn without scraping text from PDFs.  This module intentionally
contains only lightweight JSON-serialisable summaries derived from a
``run_case`` payload.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SIDECAR_VERSION = 5
LABEL_MAP_FILENAME = "celltype_label_map.json"


def case_sidecar_path(case_id: str, out_dir: Path) -> Path:
    """Return the stable sidecar path for one bio-validation case."""
    return out_dir / f"fig_biovalidation_case_{case_id}.sidecar.json"


def _safe_label_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the durable subset of one marker-derived label-map record."""
    return {
        "case_id": str(record.get("case_id", "")),
        "cluster_id": str(record.get("cluster_id", "")),
        "short_label": str(record.get("short_label", "")),
        "full_label": str(record.get("full_label", "")),
        "label_kind": str(record.get("label_kind", "")),
        "confidence": str(record.get("confidence", "")),
        "marker_genes": [str(g) for g in (record.get("marker_genes") or [])],
        "evidence_note": str(record.get("evidence_note", "")),
        "n_cells_in_marker_subsample": record.get("n_cells_in_marker_subsample"),
    }


def _sort_label_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple[int, str]:
        cluster_id = str(record.get("cluster_id", ""))
        try:
            return (int(cluster_id), cluster_id)
        except ValueError:
            return (10**9, cluster_id)

    return sorted(records, key=key)


def load_case_label_evidence(case_id: str, out_dir: Path) -> dict[str, Any]:
    """Load marker-derived cluster label evidence for ``case_id`` if present.

    The label-map file is intentionally small JSON metadata emitted by the
    marker-audit lane.  Reading it here does not touch h5ad files and keeps
    future sidecar rebuilds aligned with the committed cluster-label contract.
    """
    label_map_path = out_dir / LABEL_MAP_FILENAME
    if not label_map_path.exists():
        return {}
    try:
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    cases = label_map.get("cases") or {}
    case_payload = cases.get(case_id) or {}
    raw_records = case_payload.get("records")
    if raw_records is None:
        raw_records = [
            record for record in label_map.get("records", [])
            if str(record.get("case_id", "")) == case_id
        ]
    records = _sort_label_records([
        _safe_label_record(record)
        for record in raw_records
    ])
    if not records and case_id not in (label_map.get("curated_cases") or {}):
        return {}

    source = (
        case_payload.get("cell_type_label_source")
        or (label_map.get("curated_cases") or {}).get(case_id, {}).get("cell_type_label_source")
        or ("marker_gene_inferred" if records else "curated_source_obs")
    )
    return {
        "cell_type_label_source": source,
        "cell_type_label_map": f"figures/biovalidation/{LABEL_MAP_FILENAME}",
        "cell_type_label_map_schema": label_map.get("schema_version"),
        "cell_type_label_cluster_key": case_payload.get("cluster_key") or label_map.get("cluster_key"),
        "cluster_label_evidence": records,
    }


def _short_condition_label(label: str) -> str:
    """Compact biological/sample labels for dense paired figure panels."""
    s = str(label)
    if s.endswith(".csv.gz"):
        s = s.removesuffix(".csv.gz")
    if len(s) <= 12:
        return s
    if s.startswith("GSM") and "_sample" in s:
        tail = s.split("_sample", 1)[-1]
        if tail:
            return f"s{tail[:10]}"
    s = s.replace("sample", "s")
    return s[:11] + "…"


def json_records(df: Any) -> list[dict[str, Any]]:
    """Convert a pandas-like DataFrame to JSON-safe records."""
    if df is None or getattr(df, "empty", True):
        return []
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                value = None
            clean[str(key)] = value
        records.append(clean)
    return records


def json_matrix(matrix: Any) -> list[list[float]]:
    """Convert a numeric matrix to JSON-safe finite floats."""
    if matrix is None:
        return []
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return []
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return [[float(value) for value in row] for row in arr.tolist()]


def top_gene_rows(top_k_df: Any) -> list[dict[str, Any]]:
    """Return the top-ranked gene record per latent coordinate."""
    if top_k_df is None or getattr(top_k_df, "empty", True):
        return []
    rows: list[dict[str, Any]] = []
    for dim, group in top_k_df.sort_values(["dim", "rank"]).groupby("dim", sort=True):
        rec = group.iloc[0]
        rows.append({
            "dim": int(dim),
            "rank": int(rec.get("rank", 0)),
            "gene": str(rec.get("gene", "")),
            "rho": float(rec.get("rho", 0.0)),
        })
    return rows


def summarize_condition_latent(
    latent: np.ndarray,
    condition: Any,
    *,
    n_dims_show: int = 5,
    max_conditions: int = 8,
) -> dict[str, Any]:
    """Return compact condition × latent summaries for direct paired redraws."""
    z = np.asarray(latent, dtype=float)
    if z.ndim != 2 or z.size == 0:
        return {"dims": [], "conditions": [], "records": []}

    cond = np.asarray([str(x) for x in list(condition)])
    if cond.shape[0] != z.shape[0]:
        return {"dims": [], "conditions": [], "records": []}

    dims = list(range(min(n_dims_show, z.shape[1])))
    raw_labels, counts = np.unique(cond, return_counts=True)
    order = np.argsort(-counts)
    labels = [raw_labels[i] for i in order]
    if len(labels) > max_conditions:
        keep = labels[: max_conditions - 1]
        labels = keep + ["other"]
    else:
        keep = labels

    categories: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for label in labels:
        if label == "other":
            mask = ~np.isin(cond, keep)
            raw = "other"
        else:
            mask = cond == label
            raw = label
        if not mask.any():
            continue
        categories.append({
            "label": _short_condition_label(raw),
            "raw_label": raw,
            "n": int(mask.sum()),
        })

    for dim in dims:
        vals = z[:, dim]
        center = float(np.nanmedian(vals))
        iqr = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
        scale = iqr / 1.349 if iqr > 1e-8 else float(np.nanstd(vals))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        zvals = np.clip((vals - center) / scale, -2.5, 2.5)
        for cat in categories:
            raw = cat["raw_label"]
            mask = ~np.isin(cond, keep) if raw == "other" else cond == raw
            sub = zvals[mask]
            raw_sub = vals[mask]
            if sub.size == 0:
                continue
            records.append({
                "dim": int(dim),
                "condition": cat["label"],
                "raw_condition": raw,
                "n": int(sub.size),
                "median_z": float(np.nanmedian(sub)),
                "q25_z": float(np.nanpercentile(sub, 25)),
                "q75_z": float(np.nanpercentile(sub, 75)),
                "median_raw": float(np.nanmedian(raw_sub)),
            })

    return {"dims": dims, "conditions": categories, "records": records}


def build_case_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON-serialisable sidecar payload for one computed case."""
    case = payload["case"]
    top_k = payload.get("top_k_genes_df")
    return {
        "version": SIDECAR_VERSION,
        "case_id": case.case_id,
        "title": case.title,
        "theme": case.theme,
        "accession": case.accession,
        "encoder": case.encoder,
        "condition_obs": case.condition_obs,
        "cell_type_obs": case.cell_type_obs,
        "condition_is_inferred": case.condition_obs is None,
        "cell_type_is_inferred": case.cell_type_obs is None,
        "n_obs": payload.get("n_obs"),
        "n_obs_subsampled": payload.get("n_obs_subsampled"),
        "n_vars_post_hvg": payload.get("n_vars_post_hvg"),
        "condition_latent": summarize_condition_latent(
            np.asarray(payload.get("latent")),
            payload.get("condition"),
            n_dims_show=5,
            max_conditions=8,
        ),
        "latent_corr": json_matrix(payload.get("latent_corr")),
        "top_gene_rows": top_gene_rows(top_k),
        "top_k_genes_df": json_records(top_k),
        "enrichment": json_records(payload.get("enrichment_df")),
    }


def save_case_sidecar(payload: dict[str, Any], out_dir: Path) -> Path:
    """Persist the sidecar next to the single-case figure artifacts."""
    case = payload["case"]
    path = case_sidecar_path(case.case_id, out_dir)
    sidecar = build_case_sidecar(payload)
    sidecar.update(load_case_label_evidence(case.case_id, out_dir))
    path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
