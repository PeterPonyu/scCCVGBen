import { readFileSync } from 'node:fs';

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
