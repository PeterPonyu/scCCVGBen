# STAGE 2 Training State Snapshot

Recorded: 2026-04-24 ~16:10 UTC (4.5 h into STAGE 2)

## Running workers (6 active)

| PID | Axis | Shard | Command |
|-----|------|-------|---------|
| 143652 | A | 0/4 | `python scripts/run_encoder_sweep.py --epochs 100 --shard 0/4` |
| 143653 | A | 1/4 | (same, shard 1) |
| 143654 | A | 2/4 | (same, shard 2) |
| 143655 | A | 3/4 | (same, shard 3) |
| 611713 | B | 0/2 | `python scripts/run_graph_sweep.py --epochs 100 --shard 0/2` |
| 611714 | B | 1/2 | (same, shard 1) |

- **Axis C** (baseline backfill) killed at ~16:07 UTC after 4.5h stall on ICA
  (previous PID 143658 consumed 590% CPU without progress). To resume:
  `python scripts/run_baseline_backfill.py > workspace/logs/axis_C_sweep.log 2>&1 &`
  Recommended: wait for Axis A + B to finish before relaunching to avoid
  GPU/CPU resource contention that caused the stall.

## Progress at snapshot

| Axis | Rows completed | Files | Target | % |
|------|---------------|-------|--------|---|
| A (12 encoders × 100 ds) | 158 | 13 | ~1145 | 14% |
| B (4 new graphs × 100 ds) | 87 | 19 | ~400 | 22% |
| C (13 baselines × 100 ds) | 4 (frozen) | 2 | ~585 | <1% |

GPU RTX 4080 Laptop: 64% util, 10.4/12.3 GB, 58 °C.

## Resume safety (live)

- Per-(dataset, method) row-level append in `results/{encoder,graph}_sweep/*.csv`;
  re-run of any worker reads existing rows and skips already-done methods.
- Watchdog (PID 144380) emits ALERTs on worker-count drop / GPU overheat,
  PROGRESS heartbeat every 15 min → `workspace/logs/watchdog.log`.

## Fixes applied mid-run (commits)

- `842765c` — `build_gaussian_threshold` accepts `k` kwarg (ignored).
- `a6961b5` — gaussian_threshold default threshold 0.5 → 0.9 (avoid dense-graph
  OOM on 3000-cell subsample; 0.3% density instead of 40%).

## ETA

- Axis A: ~25 more hours
- Axis B: ~15 more hours (after gaussian_threshold fix)
- Axis C: standalone ~3-5 h after A/B finish (GPU unshared)

## Next phase after training completes

1. `python scripts/reconcile_result_schema.py` — consolidate results with
   reused CG_dl_merged into CCVGAE-format per-dataset CSVs.
2. Build paper figures + GitHub Pages (current user request).
