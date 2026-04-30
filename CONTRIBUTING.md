# Contributing to scCCVGBen

This repository is a **downstream public mirror** of an active scientific
software project. It contains source code (the `scccvgben` Python package,
generation scripts, regression tests, the Hugo project page, and the Next.js
companion explorer) so that journal reviewers and readers can clone, build,
and reproduce the published results.

**It is not the development repository.** Manuscript text, paper figures,
review-response materials, journal submission packages, and revision history
do not live here.

## Pull requests and issues are not accepted on this repository

All active development happens in a **private companion repository**.
Pull requests opened against this public mirror will be closed with a
redirect comment; issues will be triaged and either closed or duplicated
into the private repo at the maintainer's discretion.

If you are a co-author or invited collaborator and need access during the
review window, contact the corresponding author through the journal's
editorial system, **not via direct email or this repository**.

## What lives where

| Concern | This repo (`scCCVGBen`) | Private repo (`scCCVGBen-assets`) |
|---|---|---|
| Python package source (`scccvgben/`) | yes | yes (authoritative) |
| Pipeline / figure generation scripts (`scripts/`) | yes (sync'd) | yes (authoritative) |
| Regression tests (`tests/`) | yes | yes |
| Examples (`examples/`) | yes | yes |
| Hugo project page source (`site/`) | yes | yes (authoritative) |
| Next.js companion (`webapp/`) | yes | yes (authoritative) |
| Manuscript TeX, PDF, references | **no** | yes |
| Submission package, cover letter, rebuttals | **no** | yes |
| Historical revisions / round-diffs (`archive/`) | **no** | yes |
| Session handoff notes | **no** | yes (`handoff/`) |
| Asset manifest, sync policy, governance docs | **no** | yes |
| Large raw data, model checkpoints | **no** (out-of-band) | **no** (out-of-band) |

## Reproducing results

See [`REPRODUCE.md`](REPRODUCE.md). The Python package can be installed with
`pip install -e ".[dev]"` and the pipeline scripts in `scripts/` are
idempotent and resume per `(dataset, method)` row.

## Citing

If you use this code, please cite the associated manuscript (link will be
added on publication). For the time being, this repository can be cited via
its current commit hash.
