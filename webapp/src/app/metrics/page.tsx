import Link from 'next/link';
import { getMetrics } from '@/lib/data';
import type { Metric } from '@/lib/types';

// Higher = better by convention for these metrics
const HIGHER_BETTER = new Set([
  'ARI', 'NMI', 'AMI', 'FMI', 'Silhouette', 'silhouette',
  'Trustworthiness', 'KNN Accuracy', 'BatchKL',
  'cLISI', 'iLISI', 'KBET',
]);

function directionality(metric: Metric): { icon: string; label: string; color: string } {
  const name = metric.name;
  // Check note field for direction hints
  const noteHigh = /higher.better|higher is better/i.test(metric.note ?? '');
  const noteLow  = /lower.better|lower is better/i.test(metric.note ?? '');
  if (noteLow)  return { icon: '↓', label: 'lower is better',  color: 'text-amber-600' };
  if (noteHigh) return { icon: '↑', label: 'higher is better', color: 'text-teal-600' };
  // Heuristic by name
  if (HIGHER_BETTER.has(name)) return { icon: '↑', label: 'higher is better', color: 'text-teal-600' };
  return { icon: '↑', label: 'higher is better', color: 'text-teal-600' };
}

function MetricCard({ metric, suite }: { metric: Metric; suite: string }) {
  const dir = directionality(metric);
  return (
    <Link
      href={`/metrics/${encodeURIComponent(metric.name)}`}
      className="card-hover block bg-white border border-slate-200 rounded-2xl p-4 shadow-sm hover:border-teal-200 transition-colors"
    >
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span className="font-mono text-sm font-semibold text-slate-900 break-all leading-snug">
          {metric.name}
        </span>
        <span
          className={`shrink-0 text-sm font-bold ${dir.color}`}
          title={dir.label}
        >
          {dir.icon}
        </span>
      </div>
      <p className="text-xs text-slate-500 leading-relaxed">{metric.description}</p>
      {metric.note && (
        <p className="text-xs text-slate-400 mt-1.5 italic">{metric.note}</p>
      )}
      <p className="text-xs text-slate-400 mt-1.5">{metric.source}</p>
    </Link>
  );
}

export default async function MetricsPage() {
  const metrics = await getMetrics();

  const suites = [
    {
      key: 'clustering',
      title: 'Clustering — BEN',
      subtitle: `${metrics.clustering.length} metrics`,
      items: metrics.clustering,
      headerStyle: 'border-orange-200 bg-orange-50',
      titleColor: 'text-orange-800',
      countColor: 'text-orange-600',
      band: 'from-orange-400 to-amber-400',
    },
    {
      key: 'dre',
      title: 'Dimensionality Reduction — DRE',
      subtitle: `${metrics.dre.length} metrics`,
      items: metrics.dre,
      headerStyle: 'border-teal-200 bg-teal-50',
      titleColor: 'text-teal-800',
      countColor: 'text-teal-600',
      band: 'from-teal-400 to-teal-600',
    },
    {
      key: 'lse',
      title: 'Latent Space Evaluation — LSE',
      subtitle: `${metrics.lse.length} metrics`,
      items: metrics.lse,
      headerStyle: 'border-violet-200 bg-violet-50',
      titleColor: 'text-violet-800',
      countColor: 'text-violet-600',
      band: 'from-violet-400 to-violet-600',
    },
  ];

  const total = metrics.clustering.length + metrics.dre.length + metrics.lse.length;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 space-y-10">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">Metrics</h1>
        <p className="text-slate-500 mt-1 text-sm">{total} metrics across 3 evaluation suites.</p>
      </div>

      {suites.map((s) => (
        <section key={s.key}>
          <div className={`rounded-xl border ${s.headerStyle} px-4 py-3 mb-4`}>
            <div className="flex items-center gap-3">
              <div className={`h-3 w-1 rounded-full bg-gradient-to-b ${s.band}`} />
              <div>
                <h2 className={`text-base font-bold ${s.titleColor}`}>{s.title}</h2>
                <p className={`text-xs ${s.countColor} mt-0.5`}>{s.subtitle}</p>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {s.items.map((m) => (
              <MetricCard key={m.name} metric={m} suite={s.key} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
