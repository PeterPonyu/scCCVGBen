import { existsSync, readFileSync } from 'node:fs';

const html = readFileSync(new URL('../webapp/out/index.html', import.meta.url), 'utf8');
const required = [
  'https://peterponyu.github.io/scccvgben-next/',
  'name="robots" content="index, follow"',
  'property="og:url"',
  'PeterPonyu/scCCVGBen',
  'PeterPonyu/scccvgben-next',
];

const missing = required.filter((token) => !html.includes(token));

if (missing.length > 0) {
  console.error(`Missing required metadata tokens:\n${missing.map((token) => `- ${token}`).join('\n')}`);
  process.exitCode = 1;
}

const sitemapUrl = 'https://peterponyu.github.io/scccvgben-next/';
const sitemapPath = new URL('../webapp/out/sitemap.xml', import.meta.url);

if (!existsSync(sitemapPath)) {
  console.error('Missing exported sitemap: webapp/out/sitemap.xml');
  process.exitCode = 1;
} else if (!readFileSync(sitemapPath, 'utf8').includes(sitemapUrl)) {
  console.error(`Missing canonical site URL in sitemap: ${sitemapUrl}`);
  process.exitCode = 1;
}
