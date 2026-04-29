# scCCVGBen Webapp

Static Next.js 15 site mirroring the Hugo benchmark site.

## Requirements

- Node.js 20+
- npm 10+

## Development

```bash
npm install
npm run dev
```

## Build (static export)

```bash
npm run build
# output is in out/
```

## Lint

```bash
npm run lint
```

## Data

Pages read JSON data from `../site/data/` relative to this directory. Ensure the
Hugo site data files are present before building.
