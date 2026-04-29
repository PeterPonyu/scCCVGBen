import { getDatasets, getMethods, getMetrics, getSummary } from '@/lib/data';

const graphCards = [
  {
    title: 'Homepage',
    url: 'https://peterponyu.github.io/',
    body: 'Identity root, public graph manifest, sitemap entry and route-strip card for scCCVGBen.',
    cta: 'Open homepage context',
  },
  {
    title: 'scPortal',
    url: 'https://peterponyu.github.io/scportal/',
    body: 'Discovery hub that now surfaces scCCVGBen alongside dataset, benchmark and model routes.',
    cta: 'Open scPortal hub',
  },
  {
    title: 'scCCVGBen',
    url: 'https://peterponyu.github.io/scccvgben-next/',
    body: 'Publication companion with per-dataset cards, figure browser and sanitized benchmark metadata.',
    cta: 'Open companion site',
  },
];

const scPortalRoutes = [
  ['Datasets', 'https://peterponyu.github.io/scportal/datasets/'],
  ['Benchmarks', 'https://peterponyu.github.io/scportal/benchmarks/'],
];

export default async function ResourceIntegrationPage() {
  const [datasets, methods, metrics, summary] = await Promise.all([
    getDatasets(),
    getMethods(),
    getMetrics(),
    getSummary(),
  ]);
  const totalMethods = methods.scCCVGBen_encoders.length + methods.graph_constructions.length + methods.baselines.length;
  const totalMetrics = metrics.clustering.length + metrics.dre.length + metrics.lse.length;

  return (
    <main className="mx-auto max-w-6xl overflow-x-hidden px-4 py-8 sm:px-6 sm:py-12">
      <div className="rounded-[2rem] border border-slate-200 bg-white p-5 shadow-sm sm:p-8">
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-teal-700 sm:text-lg lg:text-2xl lg:tracking-[0.28em]">Online resource integration</p>
        <h1 className="mt-3 text-3xl font-black tracking-tight text-slate-950 sm:text-4xl">scPortal public graph integration</h1>
        <p className="mt-4 max-w-3xl text-base leading-7 text-slate-600 sm:text-lg lg:text-2xl">
          The polished scCCVGBen Next.js export is registered in the PeterPonyu public graph, discoverable from the homepage and surfaced through the scPortal discovery hub. This page is the figure/screenshot source for the online-resource integration panel.
        </p>
      </div>

      <section className="mt-8 grid gap-5 lg:grid-cols-3">
        {graphCards.map((card, index) => (
          <article key={card.title} className="relative rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="absolute right-4 top-4 text-6xl font-black text-slate-100">{index + 1}</div>
            <h2 className="relative text-xl font-black text-slate-950">{card.title}</h2>
            <p className="relative mt-3 text-base leading-6 text-slate-600 sm:text-lg lg:text-xl">{card.body}</p>
            <a className="relative mt-5 block break-words text-sm font-semibold leading-6 text-teal-700 hover:text-teal-800 sm:text-base lg:text-lg" href={card.url}>
              {card.url.replace('https://', '')}
            </a>
            <span className="relative mt-2 inline-flex text-sm font-black uppercase tracking-wide text-slate-400">{card.cta}</span>
          </article>
        ))}
      </section>

      <section className="mt-5 rounded-3xl border border-teal-100 bg-teal-50/70 p-5 shadow-sm">
        <p className="text-lg font-black uppercase tracking-[0.2em] text-teal-700">scPortal route checks</p>
        <div className="mt-3 flex flex-wrap gap-3">
          {scPortalRoutes.map(([label, url]) => (
            <a key={url} href={url} className="rounded-2xl border border-teal-200 bg-white px-4 py-2 text-sm font-bold text-teal-800 shadow-sm hover:border-teal-400 sm:text-base lg:text-lg">
              {label}: {url.replace('https://peterponyu.github.io/', '')}
            </a>
          ))}
        </div>
      </section>

      <section className="mt-8 rounded-[2rem] border border-slate-200 bg-slate-950 p-8 text-white shadow-xl shadow-slate-900/10">
        <div className="grid gap-6 lg:grid-cols-[1fr_2fr]">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-teal-200 sm:text-lg lg:text-2xl">Manifest facts</p>
            <h2 className="mt-2 text-2xl font-black">Public resource contract</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {[
              ['Role', 'benchmark_resource'],
              ['Boundary', 'public'],
              ['Indexing', 'index_follow'],
              ['Restricted rows', `${summary.restricted_datasets ?? 0} redacted`],
              ['Records', `${datasets.length}`],
              ['Methods', `${totalMethods}`],
              ['Metrics', `${totalMetrics}`],
              ['Graph ID', 'scccvgben'],
            ].map(([k, v]) => (
              <div key={k} className="rounded-2xl bg-white/10 p-4 ring-1 ring-white/10">
                <div className="text-sm font-semibold uppercase tracking-wide text-slate-300 sm:text-base lg:text-lg">{k}</div>
                <div className="mt-1 font-mono text-lg font-bold text-teal-100">{v}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
