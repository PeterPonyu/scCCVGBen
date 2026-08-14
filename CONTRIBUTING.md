# Contributing to scCCVGBen

This repository is a **downstream public mirror** of an active scientific
software project. It contains source code (the `scccvgben` Python package,
generation scripts, regression tests, the Hugo project page, and the Next.js
companion explorer) so that readers and collaborators can clone, build,
and reproduce the published results.

**It is not the development repository.** Internal working documents and
revision history do not live here.

## Pull requests and issues are not accepted on this repository

All active development happens in a **private companion repository**.
Pull requests opened against this public mirror will be closed with a
redirect comment; issues will be triaged and either closed or duplicated
into the private repo at the maintainer's discretion.

If you are a collaborator and need access, contact the maintainer through
known channels, **not via this repository**.

## What lives where

| Concern | This repo (`scCCVGBen`) | Private companion repo |
|---|---|---|
| Python package source (`scccvgben/`) | yes | yes (authoritative) |
| Pipeline / figure generation scripts (`scripts/`) | yes (sync'd) | yes (authoritative) |
| Regression tests (`tests/`) | yes | yes |
| Examples (`examples/`) | yes | yes |
| Hugo project page source (`site/`) | yes | yes (authoritative) |
| Next.js companion (`webapp/`) | yes | yes (authoritative) |
| Internal working documents, revision history, process notes | **no** | yes |
| Large raw data, model checkpoints | **no** (out-of-band) | **no** (out-of-band) |

## Reproducing results

See [`REPRODUCE.md`](REPRODUCE.md). The Python package can be installed with
`pip install -e ".[dev]"` and the pipeline scripts in `scripts/` are
idempotent and resume per `(dataset, method)` row.

## Citing

If you use this code, please cite the associated publication (link will be
added when available). For the time being, this repository can be cited via
its current commit hash.
