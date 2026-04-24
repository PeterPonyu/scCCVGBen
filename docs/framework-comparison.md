# Framework choice for the site: Hugo vs Astro / Next.js / Nuxt

A brief retrospective on the frontend stack picked for this repository's
companion site.

## Workload

- ~260 static pages (200 datasets, 32 methods, 26 metrics, a few hand-written
  chapters).
- Inputs are CSV/JSON files on disk; no database or per-request computation.
- Interactivity: one filterable table of 200 rows, a few Vega-Lite charts fed
  from one summary JSON, hover cards.
- Hosting: GitHub Pages — only static files.
- Operator: one researcher editing Markdown.

## Scoring (1 poor fit, 5 great fit)

| Dimension | Hugo | Astro | Next.js | Nuxt |
|---|---|---|---|---|
| GitHub Pages static export out of the box | 5 | 5 | 4 | 4 |
| Zero Node toolchain to preview | 5 | 1 | 1 | 1 |
| Auto-generate 200 pages from one CSV | 5 | 4 | 3 | 3 |
| Sidebar/search/dark-mode theme available | 5 | 3 | 2 | 2 |
| Client-side `<script>` interactivity | 5 | 5 | 5 | 5 |
| SSG cost for 260 pages | 5 | 4 | 3 | 3 |
| Learning curve for a researcher | 5 | 3 | 2 | 2 |
| Surface area a reviewer can audit | 5 | 3 | 2 | 2 |
| Hot-reload iteration speed | 5 | 4 | 4 | 4 |
| **Total** | **45** | **32** | **26** | **26** |

## Where each alternative falls short

### Next.js / Nuxt
Static export works, but `app/` router + React/Vue islands are heavier than
this site needs. Node toolchain must be installed before you can preview a
paragraph change. Reviewers end up auditing a bundler output.

### Astro
A fair fight with Hugo on this workload. Loses only on:
- Needs `pnpm`/`npm`; Hugo is one binary.
- Closest book-style theme (Starlight) requires more wiring than `hugo-book`.
- Data-driven loops still want TypeScript.

Astro is the realistic migration target if Hugo stops being a good fit.

## What Hugo costs

1. Hugo + `hugo-book` version compatibility requires a theme pin
   (`hugo-book@v10`, Hugo 0.124).
2. Go template syntax is unfamiliar to Python/JS authors.
3. `hugo-book` is a doc theme; paper-companion styling is layered on with
   `assets/custom.css`.
4. No component system — each chart lives in a `<script>` block.
5. Data comes from `site/data/*.json` for server-side templates and
   `site/static/` for client-side `fetch`. Two dump paths are maintained by
   `scripts/build_site_data.py`.

## Why Hugo was actually picked

A Hugo scaffold was already in the repository when the site work started,
and the decision was to go with it rather than re-evaluate. That is a valid
decision, but it should have been called out at the time rather than framed
as a fresh choice.
