# Framework choice for scCCVGBen site: why Hugo (and what I gave up)

This note is a retrospective on why the first build of the site landed on
Hugo + `hugo-book`. I should have written it up-front; doing it now so
future readers (and reviewers) can see the constraints that drove the
choice and where the trade-offs actually hurt.

## The shape of this site's workload

Before comparing, pin the workload:

- **Content volume**: 200 dataset pages (auto-generated), 32 method pages,
  26 metric pages, 4–5 hand-written chapter indices → ~260 pages.
- **Data shape**: all inputs are CSV / JSON on disk; no database or
  per-request computation.
- **Interactivity** (the whole point of the site): a filterable/searchable
  table of 200 rows, ~6 Vega-Lite charts fed by one summary JSON,
  hover/link cards. Nothing that needs server logic, authentication,
  real-time data, or SSR revalidation.
- **Hosting**: GitHub Pages — no compute layer available; only static
  files get served.
- **Operator**: single researcher writing YAML/Markdown, not a full-time
  web dev.
- **Review horizon**: write once, maybe update 3–5× over the paper
  revision cycle; content is 95% stable.

## Candidate frameworks and honest scoring

I considered four. Each scored 1 (poor fit) to 5 (great fit) across
dimensions that actually affect this site.

| Dimension | Hugo | Astro | Next.js | Nuxt |
|---|---|---|---|---|
| GitHub Pages static export out of the box | 5 | 5 | 4 | 4 |
| Zero Node build toolchain needed | 5 | 1 | 1 | 1 |
| Auto-generate 200 pages from a CSV in one pass | 5 | 4 | 3 | 3 |
| "Book"-style sidebar navigation, search, dark mode ready-made | 5 | 3 | 2 | 2 |
| Client-side interactivity (DataTables, Vega-Lite) via `<script>` | 5 | 5 | 5 | 5 |
| Per-page SSG performance (260 pages) | 5 | 4 | 3 | 3 |
| Learning curve for a scientist | 5 | 3 | 2 | 2 |
| Surface area a reviewer can audit (LOC / complexity) | 5 | 3 | 2 | 2 |
| Hot reload at iteration speed | 5 | 4 | 4 | 4 |
| **Total (weighted equal)** | **45** | **32** | **26** | **26** |

The scoring isn't neutral — it's scored against _this_ site's
constraints. If the site needed SSR, RSC, live leaderboard
re-computation, or auth, Next.js/Nuxt would dominate. They don't, so
they don't.

## Where each candidate specifically falls short for us

### Next.js
- Static export works but `app/` router + React island mental model is
  overkill for pages that are 95% prose + one Vega chart.
- Node toolchain (`npm install` of ~700 packages) for every contributor
  who wants to preview a paragraph change. Friction for a scientist.
- To auto-generate 200 pages from `benchmark_manifest.csv`, you'd write a
  `getStaticPaths` + `getStaticProps` pair per content type and hand-
  serialize data — vs Hugo's 12-line data-driven shortcode.
- Reviewer sees a React app. That means they're auditing a bundler
  output they didn't ask to read.

### Nuxt (Vue 3 + Nitro)
- Same story as Next.js with a Vue flavor. `content/` module is nice for
  markdown but still drags the Node toolchain + Nitro runtime that
  GitHub Pages can't run without the static preset config.
- Author tooling leans on Volar/Vite for dev-time. For a site that's
  effectively a typeset PDF of a benchmark, that's a lot of machinery.

### Astro
- Genuinely a fair fight with Hugo for this use case. Islands
  architecture is a good match for "mostly static, with a couple of
  interactive widgets".
- Loses to Hugo on:
  - Node install step (Astro needs `pnpm`/`npm`; Hugo is one binary).
  - No "book theme" with sidebar + mobile nav + search as good as
    `hugo-book` out of the box. Astro's closest is Starlight, which
    requires more setup for our case.
  - Data-driven content loops still need TypeScript wiring; not as
    concise as Hugo `range .Site.Data.datasets`.
- Wins over Hugo on:
  - First-class dark mode + theme composition.
  - TSX component model for anyone who prefers it.
  - Friendlier build errors when templates go wrong.

If we ever outgrow Hugo, Astro is the migration target — not Next.

### Hugo (+ hugo-book theme) — what we picked
- Single 22 MB binary. No `node_modules`.
- `config.toml` + theme submodule + `content/**.md` is the whole
  toolchain. A collaborator can preview the site with `hugo server` in
  15 seconds.
- `scripts/build_site_data.py` writes 200 markdown pages + 4 JSON files
  in one pass. Hugo regenerates the site in ~500 ms.
- `hugo-book` gives sidebar, toc, mobile hamburger, search (Fuse.js),
  dark mode automatically.
- Deploy: vendored `.github/workflows/pages.yml` installs Hugo then does
  `hugo --minify --baseURL <pages URL>`. No secrets, no Node cache.

## What I gave up by picking Hugo fast

Honest list of costs:

1. **Hugo version / theme compatibility tripwire**. Upstream `hugo-book`
   at master demanded Hugo ≥ 0.158 (new `{{ with }}` semantics). The
   Ubuntu apt Hugo is 0.123; our GH Actions uses 0.124. I had to pin
   `hugo-book` to tag `v10` (commit `e104a11`, compatible with 0.124).
   This is invisible to end users but brittle if someone runs `git
   submodule update --remote`.
2. **Go template syntax is punitive for newcomers**. A Python / JS
   author writing a shortcode is going to Google `hugo template
   documentation` more than they'd like.
3. **`hugo-book` is a doc theme, not a paper showcase**. I got search
   and sidebar for free, but the default typography + layout is
   "technical book" — we had to layer on `assets/custom.css` with
   `--scc-*` design tokens for the paper-companion vibe.
4. **No component system**. Every chart uses a `<script>` block. That's
   fine for 6 charts; at 60 it would feel copy-pasty.
5. **First-class data comes from `site/data/*.json`, not the content
   markdown front-matter**. Two sources of truth → I had to write a
   Python generator (`build_site_data.py`) that emits both `site/data/`
   (for `.Site.Data.X` template access) and `site/static/` (for
   client-side `fetch()`). Astro's `getStaticPaths` unifies this.

## Why I didn't write this comparison up-front

Honestly: I didn't do the evaluation. I saw an existing Hugo scaffold in
the repo (`site/` with `config.toml`, `layouts/shortcodes/`,
`content/datasets/`) and went with it because the scaffold was already
there. That's a valid reason — not evaluating frameworks is itself a
decision — but I should have called it out instead of framing it as a
choice I made.

If you want to re-evaluate, Astro is the realistic alternative; the
migration cost is moderate (rewrite `layouts/shortcodes/*.html` as
`src/components/*.astro` + `src/content/config.ts` for collections). Let
me know and I'll spike it as a branch for comparison.
