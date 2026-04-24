#!/usr/bin/env bash
# Health monitor — run every ~60s in a loop
WS=/home/zeyufu/LAB/scCCVGBen
echo "=========================="
date -u +"%F %T UTC"
echo "=========================="
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
echo ""
echo "--- workers ---"
pgrep -af "run_encoder_sweep|run_graph_sweep|run_baseline_backfill" | head
echo ""
echo "--- progress snapshot ---"
printf "  Axis A CSVs: "
ls $WS/results/encoder_sweep/*.csv 2>/dev/null | wc -l
printf "  Axis B CSVs: "
ls $WS/results/graph_sweep/*.csv 2>/dev/null | wc -l
printf "  Axis C CSVs: "
ls $WS/results/baselines/*.csv 2>/dev/null | wc -l
echo ""
echo "--- total rows across axes (each = 1 completed (dataset, method) pair) ---"
for axis in encoder_sweep graph_sweep baselines; do
    cnt=$(find $WS/results/$axis -name "*.csv" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1 - NR_FILES}')
    nf=$(ls $WS/results/$axis/*.csv 2>/dev/null | wc -l)
    printf "  $axis: $(find $WS/results/$axis -name "*.csv" -exec cat {} + 2>/dev/null | wc -l) lines in $nf files\n"
done
echo ""
echo "--- log tails (last error OR last progress in each log) ---"
for f in $WS/workspace/logs/axis_A_shard{0,1,2,3}.log $WS/workspace/logs/axis_B_shard{0,1}.log $WS/workspace/logs/axis_C_sweep.log; do
    [ -f "$f" ] || continue
    echo "$(basename $f):"
    grep -E "ERROR|Traceback|failed|✓ " "$f" 2>/dev/null | tail -1
done
