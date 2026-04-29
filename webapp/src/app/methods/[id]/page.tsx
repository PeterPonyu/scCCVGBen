import { getMethods } from '@/lib/data';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import type { Method } from '@/lib/types';

export async function generateStaticParams() {
  const methods = await getMethods();
  const all: Method[] = [
    ...methods.scCCVGBen_encoders,
    ...methods.graph_constructions,
    ...methods.baselines,
  ];
  return all.map((m) => ({ id: encodeURIComponent(m.name) }));
}

interface Props {
  params: Promise<{ id: string }>;
}

const FAMILY_META: Record<string, { label: string; pill: string; band: string }> = {
  attention:      { label: 'Attention-based encoder',     pill: 'bg-violet-100 text-violet-700 border border-violet-200', band: 'from-violet-500 to-violet-600' },
  'message-pass': { label: 'Message-passing encoder',     pill: 'bg-sky-100 text-sky-700 border border-sky-200',         band: 'from-sky-500 to-sky-600' },
  graph:          { label: 'Graph construction strategy', pill: 'bg-teal-100 text-teal-700 border border-teal-200',       band: 'from-teal-500 to-teal-600' },
  baseline:       { label: 'Baseline / comparator',       pill: 'bg-slate-100 text-slate-600 border border-slate-200',    band: 'from-slate-400 to-slate-500' },
};

export default async function MethodDetailPage({ params }: Props) {
  const { id } = await params;
  const decodedId = decodeURIComponent(id);
  const methods = await getMethods();

  const all: Method[] = [
    ...methods.scCCVGBen_encoders,
    ...methods.graph_constructions,
    ...methods.baselines,
  ];
  const method = all.find((m) => m.name === decodedId);
  if (!method) notFound();

  const meta = FAMILY_META[method.family] ?? FAMILY_META.baseline;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10">
      {/* Back link */}
      <Link
        href="/methods"
        className="inline-flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-teal-700 transition-colors mb-6"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Methods
      </Link>

      {/* Hero header */}
      <div className="rounded-2xl overflow-hidden border border-slate-200 shadow-sm mb-8">
        {/* Color band at top */}
        <div className={`h-2 bg-gradient-to-r ${meta.band}`} />
        <div className="bg-white px-6 py-6">
          <div className="flex flex-wrap items-start gap-3">
            <h1 className="text-3xl font-bold tracking-tight text-slate-900" style={{ fontFamily: 'var(--font-mono)' }}>
              {method.name}
            </h1>
            <span className={`mt-1 inline-flex items-center px-3 py-0.5 rounded-full text-xs font-semibold ${meta.pill}`}>
              {method.family}
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-2">{meta.label}</p>
        </div>
      </div>

      {/* Detail card */}
      <div className="max-w-2xl rounded-2xl border border-slate-200 bg-white shadow-sm p-6 space-y-5">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">
            Family
          </span>
          <span className={`inline-flex items-center px-3 py-0.5 rounded-full text-xs font-semibold ${meta.pill}`}>
            {method.family}
          </span>
        </div>
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">
            Description
          </span>
          <p className="text-sm text-slate-700 leading-relaxed">{method.description}</p>
        </div>
      </div>
    </div>
  );
}
