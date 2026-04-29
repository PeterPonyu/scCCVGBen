import Link from 'next/link';
import { getMethods } from '@/lib/data';
import type { Method } from '@/lib/types';

const FAMILY_COLORS: Record<string, { pill: string; border: string; bg: string }> = {
  attention:      { pill: 'bg-violet-100 text-violet-700 border border-violet-200', border: 'border-violet-200', bg: 'hover:bg-violet-50/40' },
  'message-pass': { pill: 'bg-sky-100 text-sky-700 border border-sky-200',         border: 'border-sky-200',    bg: 'hover:bg-sky-50/40' },
  graph:          { pill: 'bg-teal-100 text-teal-700 border border-teal-200',       border: 'border-teal-200',   bg: 'hover:bg-teal-50/40' },
  baseline:       { pill: 'bg-slate-100 text-slate-600 border border-slate-200',    border: 'border-slate-200',  bg: 'hover:bg-slate-50' },
};

const FAMILY_LABEL: Record<string, string> = {
  attention:      'Attention',
  'message-pass': 'Msg-Pass',
  graph:          'Graph',
  baseline:       'Baseline',
};

function MethodCard({ method }: { method: Method }) {
  const style = FAMILY_COLORS[method.family] ?? FAMILY_COLORS.baseline;
  return (
    <Link
      href={`/methods/${encodeURIComponent(method.name)}`}
      className={`card-hover block bg-white border ${style.border} rounded-2xl p-4 shadow-sm transition-colors ${style.bg}`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="font-mono font-semibold text-slate-900 text-sm leading-snug break-all">
          {method.name}
        </span>
        <span className={`shrink-0 text-xs px-2 py-0.5 rounded-full font-semibold ${style.pill}`}>
          {FAMILY_LABEL[method.family] ?? method.family}
        </span>
      </div>
      <p className="text-xs text-slate-500 leading-relaxed">{method.description}</p>
    </Link>
  );
}

export default async function MethodsPage() {
  const methods = await getMethods();

  const groups = [
    {
      key: 'encoders',
      title: 'Graph Encoders',
      subtitle: `${methods.scCCVGBen_encoders.length} encoder variants`,
      items: methods.scCCVGBen_encoders,
      headerStyle: 'border-violet-300 bg-violet-50',
      titleColor: 'text-violet-800',
      countColor: 'text-violet-600',
    },
    {
      key: 'graph',
      title: 'Graph Constructions',
      subtitle: `${methods.graph_constructions.length} graph strategies`,
      items: methods.graph_constructions,
      headerStyle: 'border-teal-300 bg-teal-50',
      titleColor: 'text-teal-800',
      countColor: 'text-teal-600',
    },
    {
      key: 'baselines',
      title: 'Comparators / Baselines',
      subtitle: `${methods.baselines.length} baseline methods`,
      items: methods.baselines,
      headerStyle: 'border-slate-300 bg-slate-50',
      titleColor: 'text-slate-800',
      countColor: 'text-slate-600',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 space-y-10">
      {/* Page header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">Methods</h1>
        <p className="text-slate-500 mt-1 text-sm">
          {methods.scCCVGBen_encoders.length} encoders &middot;{' '}
          {methods.graph_constructions.length} graph constructions &middot;{' '}
          {methods.baselines.length} baselines
        </p>
      </div>

      {/* Grouped sections */}
      {groups.map((g) => (
        <section key={g.key}>
          {/* Section header band */}
          <div className={`rounded-xl border ${g.headerStyle} px-4 py-3 mb-4 flex items-center justify-between`}>
            <div>
              <h2 className={`text-base font-bold ${g.titleColor}`}>{g.title}</h2>
              <p className={`text-xs ${g.countColor} mt-0.5`}>{g.subtitle}</p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {g.items.map((m) => (
              <MethodCard key={m.name} method={m} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
