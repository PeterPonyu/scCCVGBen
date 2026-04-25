#!/usr/bin/env python3
"""fetch_geo_scrna.py — Proxy-API GEO downloader for the 45 new scRNA datasets.

Ported from scCCVGBen's data curation patterns; proxy-API download only.

Downloads filtered_feature_bc_matrix.h5 or .h5ad files from GEO via
GEOparse (primary) or NCBI E-utilities HTTPS API (fallback). No FTP paths.

Inclusion rubric (spec lines 73-78):
  1. Diversity-first: tissues/conditions/organisms under-represented in existing 55.
  2. Pre-filtered preferred (QC'd feature matrices).
  3. Small-and-clean preferred: 3k-30k cells ideal.
  4. Reproducibility floor: public GEO/ENA accession with raw counts.
  5. Tiebreak: peer-reviewed publication preferred over preprint-only.

Usage:
    python scripts/fetch_geo_scrna.py --target 45 --candidate-csv data/scrna_candidate_pool.csv \\
        --out workspace/data/scrna/ [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    import GEOparse  # PyPI: GEOparse (camel case)
    HAS_GEOPARSE = True
except ImportError:
    HAS_GEOPARSE = False

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

# ---------------------------------------------------------------------------
# Logging — console + file
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "workspace" / "logs"

log = logging.getLogger(__name__)

_MAX_FILE_BYTES = 2 * 1024 ** 3  # 2 GB atlas-rejection threshold

# NCBI E-utilities base
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Exponential back-off delays for 429 rate-limiting (seconds)
_BACKOFF_DELAYS = [2, 4, 8, 16]


def _setup_log_file() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / "fetch_geo.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # avoid duplicate handlers if called twice in tests
    if not any(isinstance(h, logging.FileHandler) and "fetch_geo" in str(getattr(h, "baseFilename", ""))
               for h in root.handlers):
        root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        root.addHandler(sh)


# ---------------------------------------------------------------------------
# Candidate CSV helpers
# ---------------------------------------------------------------------------

def _read_candidate_pool(csv_path: Path) -> list[dict]:
    """Read candidate accessions from the manually-curated CSV.

    Expected columns: GSE, description, tissue, organism, priority (optional).
    """
    if not csv_path.exists():
        log.error(
            "Candidate pool CSV not found: %s\n"
            "Please pre-populate it with 60 GSE accessions per the inclusion rubric.",
            csv_path,
        )
        sys.exit(1)

    candidates: list[dict] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            candidates.append(dict(row))
    log.info("Loaded %d candidate accessions from %s", len(candidates), csv_path)
    return candidates


# ---------------------------------------------------------------------------
# Network helpers with back-off
# ---------------------------------------------------------------------------

def _http_get(url: str, retries: int = 4) -> bytes:
    """HTTPS GET with exponential back-off on 429 / transient errors."""
    for attempt, delay in enumerate([0] + _BACKOFF_DELAYS[:retries - 1], start=1):
        if delay:
            log.info("Back-off %ds before retry %d: %s", delay, attempt, url)
            time.sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                log.warning("Rate-limited (429) on attempt %d: %s", attempt, url)
                if attempt >= retries:
                    raise
            else:
                raise
        except urllib.error.URLError:
            if attempt >= retries:
                raise
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


def _get_file_size_bytes(url: str) -> Optional[int]:
    """HEAD request to get Content-Length without downloading."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _download_file(url: str, dest: Path, retries: int = 4) -> None:
    """Download url to dest with byte-range resume support.

    Resume rules:
    - If dest exists and size == server Content-Length, skip (complete).
    - If dest exists and size < Content-Length, send Range: bytes=N- and
      append (so partial downloads are never re-done from byte 0).
    - If server returns 416 (Range Not Satisfiable), delete partial + retry.
    Progress logged at 10-second intervals so long downloads show life.
    """
    total = _get_file_size_bytes(url)
    for attempt, delay in enumerate([0] + _BACKOFF_DELAYS[: retries - 1], start=1):
        if delay:
            log.info("Back-off %ds before retry %d", delay, attempt)
            time.sleep(delay)
        try:
            current = dest.stat().st_size if dest.exists() else 0
            if total is not None and current == total:
                log.info("Already complete (idempotent): %s (%d B)", dest.name, total)
                return
            headers = {}
            mode = "wb"
            if current > 0 and total is not None and current < total:
                log.info("Resuming at byte %d/%d: %s", current, total, dest.name)
                headers["Range"] = f"bytes={current}-"
                mode = "ab"
            else:
                current = 0
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, mode) as fh:
                chunk_size = 1024 * 1024  # 1 MB
                downloaded = current
                last_log = time.time()
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_log > 10:
                        if total:
                            log.info(
                                "  ... %d/%d MB (%.1f%%) -> %s",
                                downloaded >> 20, total >> 20,
                                100.0 * downloaded / total, dest.name,
                            )
                        else:
                            log.info("  ... %d MB -> %s", downloaded >> 20, dest.name)
                        last_log = time.time()
            final = dest.stat().st_size
            log.info("Download complete: %s (%d B)", dest.name, final)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                log.warning("Range 416 on partial %s — deleting and restarting", dest.name)
                dest.unlink(missing_ok=True)
                continue
            if exc.code == 429:
                log.warning("Rate-limited (429) on attempt %d", attempt)
                if attempt >= retries:
                    raise
            else:
                raise
        except Exception:
            if attempt >= retries:
                raise
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


# ---------------------------------------------------------------------------
# GEO metadata helpers
# ---------------------------------------------------------------------------

def _geoparse_supplementary_files(gse: str, cache_dir: Path) -> list[dict]:
    """Return list of {url, filename} dicts for all supplementary files of gse.

    Uses GEOparse library (primary path).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    gse_obj = GEOparse.get_GEO(geo=gse, destdir=str(cache_dir), silent=True)
    files: list[dict] = []

    def _append(url: str) -> None:
        url = url.strip()
        if not url or url.lower() == "none":
            return
        # Rewrite ftp:// -> https:// to stay proxy-API only
        if url.startswith("ftp://ftp.ncbi.nlm.nih.gov"):
            url = "https://" + url[len("ftp://"):]
        files.append({"url": url, "filename": url.split("/")[-1]})

    # 1) GSE-level supplementary (rarely contains cellranger outputs but cheap)
    for key, vals in gse_obj.metadata.items():
        if key.startswith("supplementary_file"):
            for v in vals:
                _append(v)

    # 2) Per-GSM supplementary — this is where cellranger mtx trios live
    for gsm_id, gsm in gse_obj.gsms.items():
        for key, vals in gsm.metadata.items():
            if key.startswith("supplementary_file"):
                for v in vals:
                    _append(v)
    return files


def _eutils_supplementary_files(gse: str) -> list[dict]:
    """Fallback: fetch supplementary file list via NCBI E-utilities HTTPS."""
    # Step 1: esearch to get internal UID
    search_url = (
        f"{_EUTILS_BASE}/esearch.fcgi?db=gds&term={gse}[Accession]&retmode=json"
    )
    log.info("E-utilities esearch: %s", search_url)
    data = _http_get(search_url)
    import json
    result = json.loads(data)
    ids = result.get("esearchresult", {}).get("idlist", [])
    if not ids:
        log.info("E-utilities: no GDS record found for %s", gse)
        return []

    uid = ids[0]
    # Step 2: efetch to get full record text
    fetch_url = f"{_EUTILS_BASE}/efetch.fcgi?db=gds&id={uid}&retmode=text"
    log.info("E-utilities efetch uid=%s for %s", uid, gse)
    text = _http_get(fetch_url).decode("utf-8", errors="replace")

    # Parse supplementary file URLs from text
    files: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ftp://") or line.startswith("https://"):
            # Convert ftp:// → https:// to stay proxy-API only
            url = line.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
            files.append({"url": url, "filename": url.split("/")[-1]})
    return files


def _get_supplementary_files(gse: str, cache_dir: Path) -> list[dict]:
    """Get supplementary files preferring GEOparse, falling back to E-utilities."""
    if HAS_GEOPARSE:
        try:
            log.info("GEOparse: fetching metadata for %s", gse)
            return _geoparse_supplementary_files(gse, cache_dir)
        except Exception as exc:
            log.warning("GEOparse failed for %s (%s); falling back to E-utilities", gse, exc)
    log.info("E-utilities fallback for %s", gse)
    return _eutils_supplementary_files(gse)


# ---------------------------------------------------------------------------
# Core per-GSE download function
# ---------------------------------------------------------------------------

def _extract_metadata_from_gse(gse: str, row: dict, cache_dir: Path) -> dict:
    """Return {tissue, organism, cell_type} from row CSV or GEOparse metadata."""
    tissue = row.get("tissue", "")
    organism = row.get("organism", "")
    cell_type = row.get("cell_type", "")

    if HAS_GEOPARSE and (not tissue or not organism):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            gse_obj = GEOparse.get_GEO(geo=gse, destdir=str(cache_dir), silent=True)
            meta = gse_obj.metadata
            if not organism:
                organism = "; ".join(meta.get("organism_ch1", meta.get("organism", [""])))
            if not tissue:
                tissue = "; ".join(meta.get("source_name_ch1", meta.get("tissue", [""])))
        except Exception:
            pass
    return {"tissue": tissue, "organism": organism, "cell_type": cell_type}


def _read_10x_mtx_robust(trio_dir: Path):
    """Read a cellranger mtx trio without scanpy's version-detection bugs.

    scanpy.read_10x_mtx() hits KeyError when a v3-named `features.tsv.gz` has
    only 2 columns (v2 content), because its v3 reader unconditionally renames
    column index 2 to 'feature_types'. Many GSEs under my control publish v2
    content under either genes.tsv.gz (standard v2) or features.tsv.gz (authors
    who renamed without updating column count).

    This manual reader auto-detects:
      - features.tsv.gz (v3) or genes.tsv.gz (v2) by glob
      - 2-column (v2) vs 3-column (v3) content by actual column count

    Returns cells-as-rows AnnData.
    """
    import pandas as pd
    import scipy.io as sio
    import scipy.sparse as sp
    import anndata as ad

    barcodes_path = next(trio_dir.glob("barcodes.tsv*"), None)
    feat_path = (next(trio_dir.glob("features.tsv*"), None)
                 or next(trio_dir.glob("genes.tsv*"), None))
    mtx_path = next(trio_dir.glob("matrix.mtx*"), None)
    if not (barcodes_path and feat_path and mtx_path):
        raise FileNotFoundError(
            f"mtx trio incomplete in {trio_dir} — have: "
            f"barcodes={barcodes_path}, features/genes={feat_path}, matrix={mtx_path}"
        )

    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].astype(str).values
    features = pd.read_csv(feat_path, header=None, sep="\t", dtype=str)
    mtx = sio.mmread(str(mtx_path))
    # 10x mtx stores genes-as-rows, cells-as-cols; AnnData expects cells-as-rows
    X = sp.csr_matrix(mtx.T)

    if features.shape[1] >= 2:
        var = pd.DataFrame(index=features[1].astype(str).values)
        var["gene_id"] = features[0].astype(str).values
    else:
        var = pd.DataFrame(index=features[0].astype(str).values)
    if features.shape[1] >= 3:
        var["feature_type"] = features[2].astype(str).values

    obs = pd.DataFrame(index=barcodes)
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Sanity check: reject pathological shapes that would poison downstream.
    # Real scRNA maxes out ~1-2M cells; >5M signals a multi-sample aggregate
    # where barcodes.tsv.gz concatenates every well from every sample without
    # de-duplication, or the mtx was mis-transposed at the source.
    if adata.n_obs > 5_000_000:
        raise ValueError(
            f"rejected n_obs={adata.n_obs} (> 5M cells is unphysical for a "
            f"single cellranger sample). Source trio likely aggregates across "
            f"samples or has transposed dimensions."
        )
    # Also reject impossible cells/gene ratios that imply mis-transpose
    if adata.n_obs > 0 and adata.n_vars > 0:
        ratio = adata.n_obs / adata.n_vars
        if ratio > 1000:
            raise ValueError(
                f"rejected n_obs/n_vars ratio {ratio:.0f} (too cell-heavy; "
                f"suggests barcode space wasn't emptyDrops-filtered)"
            )
    return adata


def _pick_target_file(files: list[dict]) -> Optional[dict]:
    """Select ONLY cellranger-standard filtered_feature_bc_matrix.h5.

    (Legacy API: returns a single file or None. Retained so existing tests
    pass; new code should call _pick_cellranger_outputs which also finds
    mtx trios.)
    """
    for f in files:
        name = f["filename"].lower()
        if "filtered_feature_bc_matrix" in name and name.endswith(".h5"):
            return f
    for f in files:
        name = f["filename"].lower()
        if "filtered_feature_bc_matrix" in name and name.endswith(".h5.gz"):
            return f
    return None


_TRIO_SUFFIXES = (
    ("barcodes.tsv.gz", "barcodes"),
    ("barcodes.tsv",    "barcodes"),
    ("features.tsv.gz", "features"),
    ("features.tsv",    "features"),
    ("genes.tsv.gz",    "features"),
    ("genes.tsv",       "features"),
    ("matrix.mtx.gz",   "matrix"),
    ("matrix.mtx",      "matrix"),
)


def _sample_key(name: str) -> Optional[tuple[str, str]]:
    """Strip a cellranger-trio suffix; return (prefix_key, slot) or None.

    Separator before the suffix may be `_`, `-`, or `.`; all trailing
    separators are stripped so sibling files align to the same key.

    Examples that group correctly with this helper:
      GSM5008737_RNA_3P-barcodes.tsv.gz       -> ("gsm5008737_rna_3p", "barcodes")
      GSE167118_Bac17B_CD3pos_barcodes.tsv.gz -> ("gse167118_bac17b_cd3pos", "barcodes")
      GSM5617891_snRNA_FCtr_matrix.mtx.gz     -> ("gsm5617891_snrna_fctr",   "matrix")
    """
    low = name.lower()
    for suf, slot in _TRIO_SUFFIXES:
        if low.endswith(suf):
            prefix = low[: -len(suf)].rstrip("_-.")
            if prefix:
                return prefix, slot
    return None


def _pick_cellranger_outputs(files: list[dict]) -> list[dict]:
    """Select cellranger-standard output: .h5 OR prefix-aligned mtx trio.

    Priority:
      1) Single filtered_feature_bc_matrix.h5 / .h5.gz  -> returns [f]
      2) Any complete mtx trio sharing a prefix — GSM-level or GSE-level —
         {barcodes.tsv.gz, features.tsv.gz (or genes.tsv.gz), matrix.mtx.gz}
         -> returns [barcodes, features, matrix]

    Both are cellranger `count` standard outputs — .h5 is the HDF5 single-file
    form; the trio is the 10x Genomics 'filtered_feature_bc_matrix/' dir form.
    """
    # Priority 1: single .h5 file
    single = _pick_target_file(files)
    if single is not None:
        return [single]

    # Priority 2: group by stripped prefix
    groups: dict[str, dict[str, dict]] = {}
    for f in files:
        key_slot = _sample_key(f["filename"])
        if key_slot is None:
            continue
        key, slot = key_slot
        groups.setdefault(key, {})[slot] = f

    for key, slots in groups.items():
        if {"barcodes", "features", "matrix"}.issubset(slots.keys()):
            return [slots["barcodes"], slots["features"], slots["matrix"]]

    return []


def _download_gse(gse: str, row: dict, out_dir: Path, cache_dir: Path, dry_run: bool) -> Optional[Path]:
    """Attempt to download a GSE dataset. Returns path to .h5ad on success.

    Steps:
      1. Check idempotency — skip if output already exists and non-empty.
      2. Fetch supplementary file list.
      3. Filter to filtered_feature_bc_matrix.h5 / .h5ad.
      4. Reject files > 2 GB (atlas-scale).
      5. Download to {out_dir}/{gse}_raw.h5 or {gse}_raw.h5ad.
      6. Convert 10x h5 → h5ad via scanpy.read_10x_h5().
      7. Attach metadata to adata.uns.
    """
    h5ad_out = out_dir / f"{gse}_new.h5ad"

    # Idempotency guard
    if h5ad_out.exists() and h5ad_out.stat().st_size > 0:
        log.info("Already exists (idempotent skip): %s", h5ad_out)
        return h5ad_out

    if dry_run:
        log.info("[dry-run] Would download %s -> %s", gse, h5ad_out)
        return None

    # --- fetch file list ---
    try:
        files = _get_supplementary_files(gse, cache_dir)
    except Exception as exc:
        log.warning("Could not fetch file list for %s: %s", gse, exc)
        return None

    targets = _pick_cellranger_outputs(files)
    if not targets:
        log.info("no cellranger output for %s (checked %d files)", gse, len(files))
        return None

    # --- atlas-size rejection (applies only to single-.h5 path; trio is small) ---
    if len(targets) == 1:
        size = _get_file_size_bytes(targets[0]["url"])
        if size is not None and size > _MAX_FILE_BYTES:
            log.warning(
                "SKIP %s: file %s is %.1f GB > 2 GB atlas limit",
                gse, targets[0]["filename"], size / 1024 ** 3,
            )
            return None

    if not HAS_SCANPY or not HAS_ANNDATA:
        log.error("scanpy/anndata not available; cannot convert downloads for %s", gse)
        return None

    adata = None
    raw_paths: list[Path] = []

    try:
        if len(targets) == 1 and targets[0]["filename"].lower().endswith((".h5", ".h5.gz")):
            # .h5 path: single file
            f = targets[0]
            raw_dest = out_dir / f"{gse}_raw_{f['filename']}"
            log.info("Downloading %s -> %s", f["url"], raw_dest)
            _download_file(f["url"], raw_dest)
            raw_paths.append(raw_dest)
            if raw_dest.suffix == ".gz":
                import gzip, shutil
                h5_path = raw_dest.with_suffix("")
                with gzip.open(raw_dest, "rb") as fin, open(h5_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                raw_paths.append(h5_path)
                adata = sc.read_10x_h5(str(h5_path))
            else:
                adata = sc.read_10x_h5(str(raw_dest))
            adata.var_names_make_unique()

        elif len(targets) == 3:
            # mtx trio path: download 3 files into per-GSM subdir with cellranger
            # canonical names barcodes.tsv.gz / features.tsv.gz / matrix.mtx.gz
            import re
            m = re.match(r"(GSM\d+)", targets[0]["filename"], re.IGNORECASE)
            gsm = m.group(1).upper() if m else f"{gse}_GSM"
            trio_dir = out_dir / f"{gse}_raw" / gsm
            trio_dir.mkdir(parents=True, exist_ok=True)

            # Always save the features/genes file as features.tsv.gz (v3 name).
            # scanpy.read_10x_mtx v3 reader handles both 2-col v2 content and
            # 3-col v3 content; the missing feature_types column is silently
            # dropped. Saving as genes.tsv.gz triggers scanpy's legacy reader
            # which in some versions still insists on features.tsv.gz — that
            # inconsistency caused the cryptic "Errno 2" failures.
            slot_to_canonical = {
                "barcodes": "barcodes.tsv.gz",
                "features": "features.tsv.gz",
                "matrix":   "matrix.mtx.gz",
            }
            for f in targets:
                name = f["filename"].lower()
                if name.endswith(("barcodes.tsv.gz", "barcodes.tsv")):
                    slot = "barcodes"
                elif name.endswith(("features.tsv.gz", "features.tsv",
                                    "genes.tsv.gz", "genes.tsv")):
                    slot = "features"
                else:
                    slot = "matrix"
                dest = trio_dir / slot_to_canonical[slot]
                log.info("Downloading %s -> %s", f["url"], dest)
                _download_file(f["url"], dest)
                raw_paths.append(dest)

            log.info("Reading 10x_mtx dir: %s", trio_dir)
            adata = _read_10x_mtx_robust(trio_dir)
            adata.var_names_make_unique()

        else:
            log.warning("Unrecognised target set for %s: %d files", gse, len(targets))
            return None

    except Exception as exc:
        log.warning("Download/convert failed for %s: %s", gse, exc)
        for p in raw_paths:
            try: p.unlink()
            except Exception: pass
        return None

    # --- attach metadata ---
    meta = _extract_metadata_from_gse(gse, row, cache_dir)
    adata.uns["source_gse"] = gse
    adata.uns["tissue"] = meta["tissue"]
    adata.uns["organism"] = meta["organism"]
    adata.uns["cell_type"] = meta["cell_type"]

    # --- write h5ad ---
    try:
        adata.write_h5ad(str(h5ad_out))
        log.info(
            "Saved %s: %d cells x %d genes -> %s",
            gse, adata.n_obs, adata.n_vars, h5ad_out,
        )
    except Exception as exc:
        log.warning("Failed to write h5ad for %s: %s", gse, exc)
        return None

    # Clean up raw downloads to save disk
    for p in raw_paths:
        try: p.unlink()
        except Exception: pass
    # Remove per-GSM subdir if empty
    for p in raw_paths:
        parent = p.parent
        if parent.name.startswith(gse) or parent.parent.name == f"{gse}_raw":
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                gse_raw = out_dir / f"{gse}_raw"
                if gse_raw.exists() and not any(gse_raw.iterdir()):
                    gse_raw.rmdir()
            except Exception:
                pass

    return h5ad_out


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------

def _write_failure_log(failures: list[tuple[str, str, str]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fail_csv = LOG_DIR / "fetch_failures.csv"
    with open(fail_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gse", "error_type", "message"])
        for row in failures:
            writer.writerow(row)
    log.info("Failure log written: %s (%d entries)", fail_csv, len(failures))


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=int, default=45,
        help="Number of datasets to successfully download (default: 45)."
    )
    parser.add_argument(
        "--candidate-pool", type=int, default=60,
        help="Max candidates to consider from the pool (default: 60)."
    )
    parser.add_argument(
        "--candidate-csv",
        default=str(REPO_ROOT / "data" / "scrna_candidate_pool.csv"),
        help="Path to manually-curated candidate accessions CSV."
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "workspace" / "data" / "scrna_geo"),
        help="Output directory for downloaded .h5ad files (separate from "
             "workspace/data/scrna/ which holds symlinks to on-host raw data)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print candidate list and exit without downloading."
    )
    args = parser.parse_args()

    _setup_log_file()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO_ROOT / "workspace" / "geo_cache"

    candidates = _read_candidate_pool(Path(args.candidate_csv))
    candidates = candidates[: args.candidate_pool]

    log.info(
        "Target: %d datasets. Candidate pool: %d. Dry-run: %s. GEOparse: %s",
        args.target, len(candidates), args.dry_run, HAS_GEOPARSE,
    )

    if args.dry_run:
        print(f"\n[dry-run] Would attempt download of up to {args.target} GSEs:")
        for i, row in enumerate(candidates[: args.target], 1):
            gse = row.get("GSE", row.get("gse", "UNKNOWN"))
            desc = row.get("description", "")
            tissue = row.get("tissue", "")
            organism = row.get("organism", "")
            print(f"  {i:3d}. {gse}  tissue={tissue!r}  organism={organism!r}  {desc}")
        print(f"\nCandidate CSV: {args.candidate_csv}")
        print(f"GEOparse available: {HAS_GEOPARSE}")
        print("Re-run without --dry-run to execute downloads.")
        return

    succeeded = 0
    failures: list[tuple[str, str, str]] = []

    for row in candidates:
        if succeeded >= args.target:
            break
        gse = row.get("GSE", row.get("gse", "UNKNOWN"))
        try:
            result = _download_gse(gse, row, out_dir, cache_dir, dry_run=False)
        except Exception as exc:
            log.error("Unexpected error for %s: %s", gse, exc)
            failures.append((gse, type(exc).__name__, str(exc)))
            continue

        if result is not None and result.exists() and result.stat().st_size > 0:
            succeeded += 1
            log.info("SUCCESS %d/%d: %s", succeeded, args.target, gse)
        else:
            failures.append((gse, "SkipOrEmpty", "no valid output produced"))

    _write_failure_log(failures)

    log.info(
        "Fetch complete: %d succeeded, %d failed/skipped.",
        succeeded, len(failures),
    )

    if succeeded < args.target:
        log.warning(
            "Only %d/%d datasets downloaded. Add more candidates to %s and re-run.",
            succeeded, args.target, args.candidate_csv,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
