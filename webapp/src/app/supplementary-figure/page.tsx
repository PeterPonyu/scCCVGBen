import SummaryCharts from '@/components/SummaryCharts';
import { getDatasets, getMethods, getMetrics, getSummary } from '@/lib/data';

function fmt(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

function fmtCompact(value: number): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 1, notation: 'compact' }).format(value);
}

function Panel({ letter, title, children }: { letter: string; title: string; children: React.ReactNode }) {
  return (
    <section className="figure-panel rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center gap-3">
        <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-teal-600 text-xl font-black text-white">{letter}</span>
        <h2 className="text-xl font-bold uppercase tracking-[0.18em] text-slate-700">{title}</h2>
      </div>
      {children}
    </section>
  );
}

export default async function SupplementaryFigurePage() {
  const [datasets, methods, metrics, summary] = await Promise.all([
    getDatasets(),
    getMethods(),
    getMetrics(),
    getSummary(),
  ]);
  const publicExample = datasets.find((d) => d.id === 'GSE115571') ?? datasets.find((d) => d.release_status !== 'restricted') ?? datasets[0];
  const restricted = datasets.filter((d) => d.release_status === 'restricted').length;
  const methodTotal = methods.scCCVGBen_encoders.length + methods.graph_constructions.length + methods.baselines.length;
  const metricTotal = metrics.clustering.length + metrics.dre.length + metrics.lse.length;

  return (
    <main className="mx-auto max-w-[1500px] bg-slate-50 px-5 py-5 print:bg-white">
      <div className="mb-4 rounded-[2rem] border border-teal-100 bg-gradient-to-r from-teal-50 via-white to-amber-50 p-5">
        <p className="text-xl font-semibold uppercase tracking-[0.28em] text-teal-700">Supplementary Figure S1 source route</p>
        <h1 className="mt-1 text-3xl font-black tracking-tight text-slate-950">Next.js online benchmark resource</h1>
        <p className="mt-1 max-w-4xl text-xl leading-5 text-slate-600">
          This route is a browser-rendered companion layout for publication screenshots. It reuses the same sanitized JSON payload as the exported site and presents a compact six-panel online-resource summary.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel letter="A" title="Landing summary">
          <div className="grid grid-cols-2 gap-3">
            {[
              ['Records', datasets.length, 'balanced scRNA/scATAC'],
              ['Methods', methodTotal, 'encoders + graph + baselines'],
              ['Metrics', metricTotal, 'publication-display scores'],
              ['Redacted', restricted, 'restricted public rows'],
            ].map(([label, value, sub]) => (
              <div key={String(label)} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-2xl font-black text-teal-700">{value}</div>
                <div className="text-xl font-bold text-slate-800">{label}</div>
                <div className="text-xl leading-5 text-slate-500">{sub}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 rounded-2xl border border-teal-100 bg-teal-50 p-3">
            <div className="text-xl font-bold uppercase tracking-[0.18em] text-teal-700">Publication-safe contract</div>
            <p className="mt-1 text-xl leading-5 text-slate-700">
              The figure, static export and supplementary table are generated from the same sanitized manifest: public accessions remain linkable, while restricted rows keep cohort-level reproducibility without disclosing nonpublic accession or file tokens.
            </p>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3 text-xl">
            {[
              ['JSON export', 'sanitized site/data payload'],
              ['Static pages', '261 generated routes'],
              ['Submission audit', '0 leak findings'],
              ['Public graph', 'homepage + scPortal linked'],
            ].map(([label, value]) => (
              <div key={label} className="rounded-2xl border border-slate-200 bg-white p-2.5">
                <div className="font-semibold uppercase tracking-wide text-slate-400">{label}</div>
                <div className="mt-1 font-bold text-slate-800">{value}</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel letter="B" title="Cohort analytics">
          <div className="figure-chart-slim">
            <SummaryCharts summary={summary} />
          </div>
        </Panel>

        <Panel letter="C" title="Dataset browser">
          <div className="overflow-hidden rounded-2xl border border-slate-200">
            <table className="w-full table-fixed text-left text-lg">
              <thead className="bg-slate-100 text-slate-500">
                <tr>
                  <th className="w-[27%] px-2 py-1.5">ID</th><th className="w-[19%] px-2 py-1.5">Modality</th><th className="w-[17%] px-2 py-1.5">Species</th><th className="w-[18%] px-2 py-1.5">Tissue</th><th className="w-[19%] px-2 py-1.5 text-right">Cells</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {datasets.filter((d) => d.release_status !== 'restricted').slice(0, 5).map((d) => (
                  <tr key={d.id}>
                    <td className="px-2 py-1.5 font-mono font-semibold text-teal-700">{d.id}</td>
                    <td className="px-2 py-1.5">{d.modality}</td>
                    <td className="px-2 py-1.5 capitalize">{d.species}</td>
                    <td className="truncate px-2 py-1.5" title={d.tissue}>{d.tissue}</td>
                    <td className="px-2 py-1.5 text-right font-mono">{fmtCompact(d.cell_count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel letter="D" title="Per-record provenance">
          <div className="space-y-2 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xl">
            <h3 className="font-mono text-lg font-black text-slate-950">{publicExample.id}</h3>
            <p className="leading-5 text-slate-600">{publicExample.geo_title}</p>
            <dl className="grid grid-cols-2 gap-2 text-xl">
              {[
                ['Accession', publicExample.GSE],
                ['Modality', publicExample.modality],
                ['Species', publicExample.species],
                ['Tissue', publicExample.tissue],
                ['Cells', fmt(publicExample.cell_count)],
                ['Release', publicExample.release_status ?? 'public'],
              ].map(([k, v]) => (
                <div key={k} className="rounded-xl bg-white p-2.5">
                  <dt className="font-semibold uppercase tracking-wide text-slate-400">{k}</dt>
                  <dd className="mt-1 font-medium text-slate-800">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </Panel>

        <Panel letter="E" title="Methods and metrics">
          <div className="grid grid-cols-2 gap-3 text-xl">
            {[
              ['Graph encoders', methods.scCCVGBen_encoders.length, 'attention and message passing'],
              ['Graph builders', methods.graph_constructions.length, 'neighbourhood definitions'],
              ['Baselines', methods.baselines.length, 'external comparators'],
              ['Metric suites', 3, `${metricTotal} display metrics`],
            ].map(([label, value, sub]) => (
              <div key={String(label)} className="rounded-2xl border border-slate-200 bg-white p-3">
                <div className="text-xl font-black text-violet-700">{value}</div>
                <div className="font-bold text-slate-800">{label}</div>
                <div className="leading-5 text-slate-500">{sub}</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel letter="F" title="Online resource integration">
          <div className="grid gap-2.5">
            {[
              ['PeterPonyu homepage', 'Identity root lists scCCVGBen in the public graph route strip.'],
              ['scPortal', 'Discovery hub connects datasets, benchmarks, models and public tools.'],
              ['scCCVGBen resource', 'Static Next.js export serves searchable benchmark metadata and paper figures.'],
            ].map(([title, text], index) => (
              <div key={title} className="flex items-start gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-teal-600 text-xl font-black text-white">{index + 1}</span>
                <div>
                  <div className="font-bold text-slate-900">{title}</div>
                  <p className="mt-1 text-xl leading-5 text-slate-600">{text}</p>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </main>
  );
}
