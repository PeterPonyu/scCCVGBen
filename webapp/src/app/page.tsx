import Link from 'next/link';
import SummaryCharts from '@/components/SummaryCharts';
import { getDatasets, getMethods, getMetrics, getSummary } from '@/lib/data';

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

export default async function HomePage() {
  const [datasets, methods, metrics, summary] = await Promise.all([
    getDatasets(),
    getMethods(),
    getMetrics(),
    getSummary(),
  ]);

  const totalMethods =
    methods.scCCVGBen_encoders.length +
    methods.graph_constructions.length +
    methods.baselines.length;
  const displayedMetrics = metrics.clustering.length + metrics.dre.length + metrics.lse.length;
  const scRNA = datasets.filter((d) => d.modality === 'scRNA').length;
  const scATAC = datasets.filter((d) => d.modality === 'scATAC').length;
  const restricted = datasets.filter((d) => d.release_status === 'restricted').length;

  const heroStats = [
    {
      value: datasets.length,
      label: 'Benchmark records',
      sub: `${scRNA} scRNA · ${scATAC} scATAC`,
      href: '/datasets',
      accent: 'from-teal-500 to-teal-600',
      bg: 'bg-teal-50',
      border: 'border-teal-200',
      textAccent: 'text-teal-700',
    },
    {
      value: totalMethods,
      label: 'Methods',
      sub: `${methods.scCCVGBen_encoders.length} encoders · ${methods.graph_constructions.length} graph builders · ${methods.baselines.length} baselines`,
      href: '/methods',
      accent: 'from-violet-500 to-violet-600',
      bg: 'bg-violet-50',
      border: 'border-violet-200',
      textAccent: 'text-violet-700',
    },
    {
      value: displayedMetrics,
      label: 'Display metrics',
      sub: `${metrics.clustering.length} BEN · ${metrics.dre.length} DRE · ${metrics.lse.length} LSE`,
      href: '/metrics',
      accent: 'from-amber-500 to-amber-600',
      bg: 'bg-amber-50',
      border: 'border-amber-200',
      textAccent: 'text-amber-700',
    },
    {
      value: restricted,
      label: 'Redacted rows',
      sub: 'Public-safe restricted-access records',
      href: '/datasets',
      accent: 'from-slate-500 to-slate-600',
      bg: 'bg-slate-50',
      border: 'border-slate-200',
      textAccent: 'text-slate-700',
    },
  ];

  return (
    <div>
      <section className="gradient-hero border-b border-slate-200 px-4 py-16 sm:px-6">
        <div className="mx-auto grid max-w-7xl items-center gap-10 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="max-w-3xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-teal-200 bg-white/80 px-3 py-1 text-xs font-semibold text-teal-700 shadow-sm">
              Single-cell graph representation benchmark · static public resource
            </div>
            <h1 className="text-4xl font-bold tracking-tight text-slate-950 sm:text-6xl">
              scCCVGBen
            </h1>
            <p className="mt-5 text-lg leading-8 text-slate-600">
              A polished Next.js companion site for the scCCVGBen manuscript, generated from sanitized
              benchmark metadata and designed to double as a publication-grade online resource figure.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link
                href="/datasets"
                className="inline-flex items-center gap-2 rounded-xl bg-teal-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-700"
              >
                Browse datasets
                <span aria-hidden="true">→</span>
              </Link>
              <Link
                href="/resource-integration"
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-teal-400 hover:text-teal-700"
              >
                View resource graph
              </Link>
              <a
                href="https://doi.org/10.1101/2025.02.01.636000"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white/70 px-5 py-3 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-teal-400 hover:text-teal-700"
              >
                Paper DOI
              </a>
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/70 bg-white/80 p-5 shadow-2xl shadow-teal-900/10 backdrop-blur">
            <div className="rounded-3xl border border-slate-200 bg-slate-950 p-5 text-white shadow-inner">
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-teal-200">Public graph</p>
                  <h2 className="mt-1 text-xl font-bold">Homepage → scPortal → scCCVGBen</h2>
                </div>
                <div className="rounded-full bg-teal-400/15 px-3 py-1 text-xs font-semibold text-teal-100">
                  indexable
                </div>
              </div>
              <div className="mt-5 grid gap-3">
                {[
                  ['PeterPonyu homepage', 'Identity root and route strip'],
                  ['scPortal', 'Discovery hub for public resources'],
                  ['scCCVGBen', `${formatNumber(datasets.length)} records · ${totalMethods} methods · ${summary.metrics_total} metrics`],
                ].map(([title, sub], index) => (
                  <div key={title} className="flex items-center gap-3 rounded-2xl bg-white/8 p-3 ring-1 ring-white/10">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-teal-400 text-sm font-black text-slate-950">
                      {index + 1}
                    </div>
                    <div>
                      <div className="text-sm font-semibold">{title}</div>
                      <div className="text-xs text-slate-300">{sub}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-7xl space-y-12 px-4 py-10 sm:px-6">
        <section>
          <h2 className="mb-4 text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
            Benchmark at a glance
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {heroStats.map((s) => (
              <Link
                key={s.label}
                href={s.href}
                className={`card-hover block rounded-3xl border ${s.border} ${s.bg} p-5 shadow-sm`}
              >
                <div className={`mb-1 text-4xl font-bold ${s.textAccent}`}>{s.value}</div>
                <div className="text-sm font-semibold text-slate-700">{s.label}</div>
                <div className="mt-1 text-xs leading-relaxed text-slate-500">{s.sub}</div>
                <div className={`mt-4 h-1 rounded-full bg-gradient-to-r ${s.accent} opacity-70`} />
              </Link>
            ))}
          </div>
        </section>

        <SummaryCharts summary={summary} />

        <section>
          <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Explore the resource</h2>
              <p className="mt-2 text-sm text-slate-600">Publication-ready pages use public-safe identifiers and server-rendered metadata.</p>
            </div>
            <Link href="/supplementary-figure" className="text-sm font-semibold text-teal-700 hover:text-teal-800">
              Open figure layout →
            </Link>
          </div>
          <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
            {[
              ['Datasets', `/datasets`, `${datasets.length} public-safe records with searchable accession, modality, species and tissue metadata.`, 'teal'],
              ['Methods', '/methods', `${methods.scCCVGBen_encoders.length} encoders, ${methods.graph_constructions.length} graph builders and ${methods.baselines.length} baselines.`, 'violet'],
              ['Metrics', '/metrics', `${summary.metrics_total} publication-display metrics across BEN, DRE and LSE suites.`, 'amber'],
            ].map(([title, href, body, tone]) => (
              <Link key={title} href={href} className="card-hover block rounded-3xl border border-slate-200 bg-white p-6 shadow-sm hover:border-teal-200">
                <div className={`mb-4 h-2 w-16 rounded-full ${tone === 'teal' ? 'bg-teal-500' : tone === 'violet' ? 'bg-violet-500' : 'bg-amber-500'}`} />
                <h3 className="font-bold text-slate-900">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-500">{body}</p>
              </Link>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
