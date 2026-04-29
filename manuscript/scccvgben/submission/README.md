# Submission package — scCCVGBen revision

## Compiling the letters

Run from inside this `submission/` directory:

```bash
make all        # build all three PDFs
make cover      # cover_letter.pdf only
make r1         # rebuttal_reviewer_1.pdf only
make r2         # rebuttal_reviewer_2.pdf only
make clean      # remove auxiliary and PDF files
```

Requires a standard TeX Live installation with `mdframed`, `microtype`,
`parskip`, `enumitem`, `booktabs`, `hyperref`.

## Next.js deploy — manual prerequisites

Before the GitHub Actions workflow
(`.github/workflows/webapp-deploy.yml`) can publish to
`https://peterponyu.github.io/scccvgben-next/`, two one-time manual
steps are required:

1. **Create the sibling repo**: go to github.com/new and create
   `peterponyu/scccvgben-next` as an empty public repository.
   Leave the default branch as `main`.

2. **Add the deploy secret**: in the *source* repo
   (`peterponyu/scCCVGBen`), go to
   Settings → Secrets and variables → Actions → New repository secret.
   Name: `WEBAPP_DEPLOY_TOKEN`.
   Value: a fine-grained personal access token with
   **Contents: Read and write** permission scoped to the
   `peterponyu/scccvgben-next` repository.

Once those two steps are done, any push to `main` that touches the webapp,
public site data, benchmark manifest, or site-data build scripts will trigger
the build-and-deploy workflow. The workflow builds with the `/scccvgben-next`
base path used by GitHub Pages project sites.

## Cover letter — load-bearing sentence

The following sentence in `cover_letter.tex` is the pivot of the
title-evolution argument and must not be paraphrased away:

> "The algorithmic contribution is unchanged, but the manuscript now reads
> as a benchmark companion rather than a method-introduction paper."

## Submission-readiness checks

The rebuttal PDFs should be regenerated with `make all` after every text edit.
Before packaging, run the repository-level submission audit from the project
root; it scans the public site payloads, manuscript text, generated figures,
and this submission directory for unresolved draft markers, local paths,
restricted accession leaks, and stale source-token wording.
