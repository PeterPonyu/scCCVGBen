# scCCVGBen

A 200-dataset single-cell representation-learning benchmark and a Python
package for the centroid-coupled variational graph attention autoencoder
(100 scRNA-seq + 100 scATAC-seq, balanced).

The repository carries:

- the dataset catalogue with GEO-verified metadata,
- the method and metric registries,
- the Python package used to regenerate the benchmark,
- the Hugo source for the public site,
- the Next.js companion explorer (`webapp/`).

## Install

```bash
pip install -e ".[dev]"
```

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md). Sweep scripts are idempotent and resume
per `(dataset, method)` row.

## Layout

- `scccvgben/` — Python package
- `scripts/` — data curation and sweep runners
- `data/` — dataset metadata CSVs
- `figures/` — generated figures (raw screenshots in `figures/site_shots/` stay local)
- `site/` — Hugo source for the public site
- `webapp/` — Next.js benchmark explorer
- `tests/` — smoke tests

## Figures

- **Figure 1 — site and benchmark overview.** Render with `scripts/make_figure1_site.py`.
- **Figure 2 — model architecture and benchmark axes.** Render with `scripts/make_figure2_model_architecture.py`.
- Generated figure binaries are not committed by the figure-pipeline PR.

## Sites

- Hugo public site: <https://peterponyu.github.io/scCCVGBen/>
- Next.js companion explorer: <https://peterponyu.github.io/scccvgben-next/>

## License

MIT — see [`LICENSE`](LICENSE).
