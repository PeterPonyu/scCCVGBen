import type { Summary } from '@/lib/types';

type ChartDatum = { label: string; value: number; tone?: string };

const toneClass: Record<string, string> = {
  teal: 'bg-teal-500',
  sky: 'bg-sky-500',
  amber: 'bg-amber-500',
  violet: 'bg-violet-500',
  slate: 'bg-slate-500',
};

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <article className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900">{title}</h3>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">{subtitle}</p>
        </div>
      </div>
      {children}
    </article>
  );
}

function Bars({ data }: { data: ChartDatum[] }) {
  const max = Math.max(...data.map((d) => d.value), 1);
  return (
    <div className="space-y-2.5">
      {data.map((d) => (
        <div key={d.label} className="grid grid-cols-[88px_1fr_44px] items-center gap-3 text-xs">
          <span className="truncate font-medium text-slate-600" title={d.label}>{d.label}</span>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full rounded-full ${toneClass[d.tone ?? 'teal']}`}
              style={{ width: `${Math.max(4, (d.value / max) * 100)}%` }}
            />
          </div>
          <span className="text-right font-mono text-slate-500">{formatNumber(d.value)}</span>
        </div>
      ))}
    </div>
  );
}

function Timeline({ data }: { data: ChartDatum[] }) {
  const max = Math.max(...data.map((d) => d.value), 1);
  return (
    <div className="flex h-40 items-end gap-1.5 rounded-2xl bg-slate-50 px-3 pb-4 pt-6">
      {data.map((d) => (
        <div key={d.label} className="flex min-w-0 flex-1 flex-col items-center gap-1">
          <div
            className="w-full rounded-t-lg bg-gradient-to-t from-sky-500 to-teal-400"
            style={{ height: `${Math.max(10, (d.value / max) * 112)}px` }}
            title={`${d.label}: ${d.value}`}
          />
          <span className="w-10 -rotate-45 text-[10px] text-slate-400">{d.label}</span>
        </div>
      ))}
    </div>
  );
}

export default function SummaryCharts({ summary }: { summary: Summary }) {
  const cellBins = summary.cell_count_hist.map((d) => ({ label: d.bin, value: d.count, tone: 'teal' }));
  const years = summary.submission_year_timeline.slice(-12).map((d) => ({ label: String(d.year), value: d.count }));
  const tissues = Object.entries(summary.tissue_top10)
    .slice(0, 8)
    .map(([label, value], index) => ({ label: label.replace(/_/g, ' '), value, tone: index % 2 ? 'sky' : 'violet' }));
  const modality = Object.entries(summary.modality).map(([label, value]) => ({
    label,
    value,
    tone: label === 'scRNA' ? 'teal' : 'amber',
  }));

  return (
    <section className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Cohort analytics</h2>
          <p className="mt-2 max-w-2xl text-sm text-slate-600">
            Data-driven summaries rendered directly in the Next.js export; these panels replace the earlier placeholder chart cards.
          </p>
        </div>
        <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-500">
          {summary.restricted_datasets ?? 0} restricted rows redacted publicly
        </div>
      </div>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <ChartCard title="Cell-count distribution" subtitle={`Positive-count range ${formatNumber(summary.cell_count_range.min)}–${formatNumber(summary.cell_count_range.max)} cells; ${summary.cell_count_missing ?? 0} entries report counts as unavailable.`}>
          <Bars data={cellBins} />
        </ChartCard>
        <ChartCard title="Submission year timeline" subtitle="Last 12 submission years represented in the public manifest.">
          <Timeline data={years} />
        </ChartCard>
        <ChartCard title="Top tissues" subtitle="Canonical tissue labels inferred from manifest and GEO text.">
          <Bars data={tissues} />
        </ChartCard>
        <ChartCard title="Modality balance" subtitle="Benchmark design enforces equal scRNA/scATAC coverage.">
          <Bars data={modality} />
        </ChartCard>
      </div>
    </section>
  );
}
