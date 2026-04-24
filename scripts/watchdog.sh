#!/usr/bin/env bash
WS=/home/zeyufu/LAB/scCCVGBen
# Loop every 90s, emit one status line + alerts
while true; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    n_workers=$(pgrep -f "run_encoder_sweep|run_graph_sweep|run_baseline_backfill" | wc -l)
    rows_a=$(find $WS/results/encoder_sweep -name "*.csv" -exec cat {} + 2>/dev/null | wc -l)
    rows_b=$(find $WS/results/graph_sweep -name "*.csv" -exec cat {} + 2>/dev/null | wc -l)
    rows_c=$(find $WS/results/baselines -name "*.csv" -exec cat {} + 2>/dev/null | wc -l)
    # Alert conditions (prefix ALERT so Monitor picks them up)
    if [ "$n_workers" -lt 7 ]; then
        echo "ALERT worker-count dropped to $n_workers/7 at $(date +%T)"
    fi
    if [ "$temp" -gt 85 ]; then
        echo "ALERT GPU temp $temp°C at $(date +%T)"
    fi
    # Progress heartbeat every 15 min
    if [ $(($(date +%M) % 15)) -eq 0 ] && [ $(date +%S) -lt 10 ]; then
        echo "PROGRESS gpu=${util}% mem=${mem}MiB workers=$n_workers A=$rows_a B=$rows_b C=$rows_c t=$(date +%T)"
    fi
    sleep 90
done
