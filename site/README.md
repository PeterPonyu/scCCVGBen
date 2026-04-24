# scCCVGBen — Hugo Site

This is the Hugo project for scCCVGBen.

**Theme**: `hugo-book` (https://github.com/alex-shpak/hugo-book)

## First-time setup

After cloning the repo, add the theme as a git submodule:

```bash
cd site
git submodule add https://github.com/alex-shpak/hugo-book themes/hugo-book
hugo --minify
```

## Local preview

```bash
cd site
hugo server --minify
# open http://localhost:1313/scCCVGBen/
```

## Data

The dataset table is populated from `site/data/datasets.json`.  That file is
generated from `scccvgben/data/datasets.csv` by:

```bash
python scripts/build_datasets_json.py
```

Run this command after updating `datasets.csv` and before deploying.

## Deployment

The site is auto-deployed via `.github/workflows/pages.yml` on every push to `main`.
The workflow runs `hugo --minify` and publishes `site/public/` to GitHub Pages.
