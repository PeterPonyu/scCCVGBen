# scCCVGBen

A benchmark companion site and Python package for graph variational
autoencoders on 200 single-cell omics datasets (100 scRNA-seq + 100 scATAC-seq).

The repository hosts:

- the dataset catalog with GEO-verified metadata,
- the method and metric registries,
- the Python package used to regenerate results,
- the Hugo source for the public site.

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
- `figures/` — generated figures
- `site/` — Hugo source
- `tests/` — smoke tests

## Site

Published at <https://peterponyu.github.io/scCCVGBen/>.

## License

MIT — see [`LICENSE`](LICENSE).
