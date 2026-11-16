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
pip install scccvgben
```

The reference graph model also needs `torch-sparse` (`pip install "scccvgben[sparse]"`, with a wheel that matches your PyTorch build). To regenerate the benchmark, clone the repository and run `pip install -e ".[dev,full]"`.

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

## Citation

If you use scCCVGBen, please cite the article:

> Fu Z, Fu J, Chen C, Zhang K, Wang J, Ran T, Wang S. scCCVGBen for benchmarking of single-cell representation learning anchored on a centroid-coupled variational graph attention autoencoder across scRNA-seq and scATAC-seq. *Frontiers in Genetics*. 2026;17:1822168. https://doi.org/10.3389/fgene.2026.1822168

```bibtex
@article{fu2026scccvgben,
  author  = {Fu, Zeyu and Fu, Jiawei and Chen, Chunlin and Zhang, Keyang and Wang, Junping and Ran, Tianfei and Wang, Song},
  title   = {scCCVGBen for benchmarking of single-cell representation learning anchored on a centroid-coupled variational graph attention autoencoder across scRNA-seq and scATAC-seq},
  journal = {Frontiers in Genetics},
  volume  = {17},
  pages   = {1822168},
  year    = {2026},
  doi     = {10.3389/fgene.2026.1822168}
}
```

GitHub's "Cite this repository" button reads the same record from `CITATION.cff`.

## License

MIT — see [`LICENSE`](LICENSE).
