from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / "workspace"
RESULTS = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "scccvgben" / "data"
DATASETS_CSV = DATA_DIR / "datasets.csv"
FIGURES = REPO_ROOT / "figures"

DATA_SCRNA = WORKSPACE / "data" / "scrna"
DATA_SCATAC = WORKSPACE / "data" / "scatac"
REUSED_SCRNA_BASELINES = WORKSPACE / "reused_results" / "scrna_baselines"
REUSED_SCATAC_BASELINES = WORKSPACE / "reused_results" / "scatac_baselines"
REUSED_AXISA_GAT_SCRNA = WORKSPACE / "reused_results" / "axisA_GAT_scrna"

# upstream sources (read-only)
CCVGAE_ROOT = Path("/home/zeyufu/LAB/CCVGAE")
CG_DL_MERGED = CCVGAE_ROOT / "CG_results" / "CG_dl_merged"
CG_ATACS_TABLES = CCVGAE_ROOT / "CG_results" / "CG_atacs" / "tables"

# Forbidden paths (must NOT read from these per spec Hard Boundary Rule 1)
FORBIDDEN_READS = (CCVGAE_ROOT / "CCVGAE_supplement",)
