#!/usr/bin/env bash
# Direct-env-var D2 OAT resume for the 9 missing settings.
# Bypasses the broken run_d2_hyperparam.py shard/worker plumbing.
# Each setting writes to results/hyperparam/<axis>__<value>__GSE183904_GastricHmCancer.csv

set -e

REPO="${SCCCVGBEN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

DS="${SCCCVGBEN_D2_DATASET:-workspace/data/scrna/GSE183904_GastricHmCancer.h5ad}"
EPOCHS=100
OUT_BASE=$REPO/results/hyperparam
LOG_BASE=$REPO/workspace/logs/d2-direct-$(date +%Y%m%d-%H%M%S)

mkdir -p "$OUT_BASE"

run_setting() {
  local axis=$1 val=$2 envvar=$3
  local label="${axis}__${val}__GSE183904_GastricHmCancer"
  local out_dir="$OUT_BASE/.tmp_${axis}_${val}"
  mkdir -p "$out_dir"
  echo "[d2-direct] $label START $(date +%H:%M:%S)"
  $envvar python "$REPO/scripts/run_encoder_sweep.py" \
    --datasets-glob "$DS" \
    --epochs $EPOCHS \
    --out "$out_dir" \
    > "$LOG_BASE-${label}.log" 2>&1
  if [ -f "$out_dir/GSE183904_GastricHmCancer.csv" ]; then
    cp "$out_dir/GSE183904_GastricHmCancer.csv" "$OUT_BASE/${label}.csv"
    echo "[d2-direct] $label DONE $(date +%H:%M:%S) -> $OUT_BASE/${label}.csv"
  else
    echo "[d2-direct] $label FAILED $(date +%H:%M:%S) — see $LOG_BASE-${label}.log"
  fi
  rm -rf "$out_dir"
}

# 9 missing OAT settings (defaults already in 4 existing CSVs)
run_setting alpha 0.1 "SCCCVGBEN_ALPHA=0.1"
run_setting alpha 1.0 "SCCCVGBEN_ALPHA=1.0"
run_setting w_adj 0.0 "SCCCVGBEN_W_ADJ=0.0"
run_setting w_adj 0.5 "SCCCVGBEN_W_ADJ=0.5"
run_setting w_adj 2.0 "SCCCVGBEN_W_ADJ=2.0"
run_setting dropout 0.0 "SCCCVGBEN_DROPOUT=0.0"
run_setting dropout 0.2 "SCCCVGBEN_DROPOUT=0.2"
run_setting hidden_dim 64 "SCCCVGBEN_HIDDEN_DIM=64"
run_setting hidden_dim 256 "SCCCVGBEN_HIDDEN_DIM=256"

echo "[d2-direct] ALL DONE $(date +%H:%M:%S)"
