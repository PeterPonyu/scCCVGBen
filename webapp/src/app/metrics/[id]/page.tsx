import { getMetrics } from '@/lib/data';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import type { Metric } from '@/lib/types';

export async function generateStaticParams() {
  const metrics = await getMetrics();
  const all: Metric[] = [
    ...metrics.clustering,
    ...metrics.dre,
    ...metrics.lse,
  ];
  return all.map((m) => ({ id: encodeURIComponent(m.name) }));
}

interface Props {
  params: Promise<{ id: string }>;
}

const SUITE_META: Record<string, { label: string; band: string; pill: string }> = {
  'Clustering (BEN)':                  { label: 'BEN — Clustering',                        band: 'from-orange-400 to-amber-400',   pill: 'bg-orange-100 text-orange-700 border border-orange-200' },
  'Dimensionality Reduction (DRE)':    { label: 'DRE — Dimensionality Reduction',           band: 'from-teal-400 to-teal-600',      pill: 'bg-teal-100 text-teal-700 border border-teal-200' },
  'Latent Space Evaluation (LSE)':     { label: 'LSE — Latent Space Evaluation',            band: 'from-violet-400 to-violet-600',  pill: 'bg-violet-100 text-violet-700 border border-violet-200' },
};

export default async function MetricDetailPage({ params }: Props) {
  const { id } = await params;
  const decodedId = decodeURIComponent(id);
  const metrics = await getMetrics();

  const suiteMap: Record<string, string> = {};
  metrics.clustering.forEach((m) => (suiteMap[m.name] = 'Clustering (BEN)'));
  metrics.dre.forEach((m) => (suiteMap[m.name] = 'Dimensionality Reduction (DRE)'));
  metrics.lse.forEach((m) => (suiteMap[m.name] = 'Latent Space Evaluation (LSE)'));

  const all: Metric[] = [
    ...metrics.clustering,
    ...metrics.dre,
    ...metrics.lse,
  ];
  const metric = all.find((m) => m.name === decodedId);
  if (!metric) notFound();

  const suiteName = suiteMap[metric.name] ?? 'Clustering (BEN)';
  const suiteMeta = SUITE_META[suiteName] ?? SUITE_META['Clustering (BEN)'];

  // Directionality
  const isLower = /lower.better|lower is better/i.test(metric.note ?? '');
  const direction = isLower
    ? { icon: '↓', label: 'lower is better', color: 'text-amber-600', bg: 'bg-amber-50 border-amber-200' }
    : { icon: '↑', label: 'higher is better', color: 'text-teal-600', bg: 'bg-teal-50 border-teal-200' };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10">
      {/* Back link */}
      <Link
        href="/metrics"
        className="inline-flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-teal-700 transition-colors mb-6"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Metrics
      </Link>

      {/* Hero header with color band */}
      <div className="rounded-2xl overflow-hidden border border-slate-200 shadow-sm mb-8">
        <div className={`h-2 bg-gradient-to-r ${suiteMeta.band}`} />
        <div className="bg-white px-6 py-6">
          <div className="flex flex-wrap items-start gap-3">
            <h1 className="text-3xl font-bold tracking-tight text-slate-900 break-all" style={{ fontFamily: 'var(--font-mono)' }}>
              {metric.name}
            </h1>
            <span className={`mt-1 inline-flex items-center px-3 py-0.5 rounded-full text-xs font-semibold ${suiteMeta.pill}`}>
              {suiteMeta.label}
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-2">{metric.description}</p>
        </div>
      </div>

      <div className="max-w-2xl space-y-5">
        {/* Directionality card */}
        <div className={`rounded-2xl border ${direction.bg} p-5 flex items-center gap-4`}>
          <div className={`text-4xl font-bold ${direction.color}`}>{direction.icon}</div>
          <div>
            <div className="text-sm font-semibold text-slate-700 capitalize">{direction.label}</div>
            <div className="text-xs text-slate-500 mt-0.5">
              Optimization direction for this metric
            </div>
          </div>
        </div>

        {/* Detail card */}
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6 space-y-5">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">
              Description
            </span>
            <p className="text-sm text-slate-700 leading-relaxed">{metric.description}</p>
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">
              Implementation Reference
            </span>
            <p className="text-sm text-slate-700 leading-relaxed">{metric.source}</p>
          </div>
          {metric.note && (
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">
                Note
              </span>
              <p className="text-sm text-slate-500 italic">{metric.note}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
