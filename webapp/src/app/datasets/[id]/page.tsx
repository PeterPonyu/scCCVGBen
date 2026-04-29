import { getDatasets } from '@/lib/data';
import Link from 'next/link';
import { notFound } from 'next/navigation';

export async function generateStaticParams() {
  const datasets = await getDatasets();
  return datasets.map((d) => ({ id: d.id }));
}

interface Props {
  params: Promise<{ id: string }>;
}

const modalityStyle = (m: string) =>
  m === 'scRNA'
    ? 'bg-teal-100 text-teal-700 border border-teal-200'
    : 'bg-amber-100 text-amber-700 border border-amber-200';

function firstPubmedId(value: string): string {
  return value.match(/\d{6,}/)?.[0] ?? '';
}

export default async function DatasetDetailPage({ params }: Props) {
  const { id } = await params;
  const datasets = await getDatasets();
  const dataset = datasets.find((d) => d.id === id);

  if (!dataset) notFound();

  const isRestricted = dataset.release_status === 'restricted';
  const pubmedId = firstPubmedId(dataset.pubmed_id ?? '');

  const metaFields: Array<{ label: string; value: string | number }> = [
    { label: 'GEO Accession', value: dataset.GSE },
    { label: 'Modality', value: dataset.modality },
    { label: 'Species', value: dataset.species },
    { label: 'Tissue', value: dataset.tissue },
    { label: 'Category', value: dataset.category },
    { label: 'Cell Count', value: dataset.cell_count > 0 ? dataset.cell_count.toLocaleString() : 'Not reported' },
    { label: 'Submission Date', value: dataset.submission_date },
    { label: 'PubMed ID', value: dataset.pubmed_id },
    { label: 'Source Name', value: dataset.source_name },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10">
      {/* Back link */}
      <Link
        href="/datasets"
        className="inline-flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-teal-700 transition-colors mb-6"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Datasets
      </Link>

      {/* Hero row */}
      <div className="flex flex-wrap items-start gap-3 mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900" style={{ fontFamily: 'var(--font-mono)' }}>
          {dataset.id}
        </h1>
        <span className={`mt-1 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${modalityStyle(dataset.modality)}`}>
          {dataset.modality}
        </span>
        {isRestricted && (
          <span className="mt-1 inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-red-100 text-red-700 text-xs font-semibold border border-red-200">
            Restricted
          </span>
        )}
      </div>

      {/* Two-column layout */}
      <div className="flex flex-col md:flex-row gap-8">
        {/* LEFT — sticky metadata sidebar (1/3) */}
        <aside className="md:w-80 shrink-0">
          <div className="sticky top-20 rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100" style={{ background: 'var(--color-slate-light)' }}>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Key Statistics
              </span>
            </div>
            {/* Cell count highlight */}
            <div className="px-4 py-4 border-b border-slate-100">
              <div className="text-2xl font-bold" style={{ color: 'var(--color-primary)', fontFamily: 'var(--font-mono)' }}>
                {dataset.cell_count > 0 ? dataset.cell_count.toLocaleString() : '—'}
              </div>
              <div className="text-xs text-slate-500 mt-0.5">cells profiled</div>
            </div>
            <dl className="divide-y divide-slate-100">
              {metaFields.filter((f) => f.label !== 'Cell Count').map(({ label, value }) => (
                <div key={label} className="px-4 py-2.5 flex flex-col gap-0.5">
                  <dt className="text-xs font-medium text-slate-400 uppercase tracking-wide">{label}</dt>
                  <dd className="text-sm text-slate-800 break-words">{value}</dd>
                </div>
              ))}
            </dl>
          </div>
        </aside>

        {/* RIGHT — description + accession links (2/3) */}
        <div className="flex-1 space-y-6">
          {/* GEO title */}
          <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">GEO Title</h2>
            <p className="text-slate-800 leading-relaxed">{dataset.geo_title}</p>
          </div>

          {/* Description */}
          <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Description</h2>
            <p className="text-slate-700 text-sm leading-relaxed">{dataset.description}</p>
          </div>

          {/* Accession links */}
          <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Accession Links</h2>
            <div className="flex flex-wrap gap-3">
              {dataset.geo_url && dataset.GSE !== 'restricted' ? (
                <a
                  href={dataset.geo_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold border border-teal-200 text-teal-700 hover:bg-teal-50 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                  GEO: {dataset.GSE}
                </a>
              ) : (
                <span className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold border border-slate-200 text-slate-500 bg-slate-50">
                  Accession redacted for public display
                </span>
              )}
              {pubmedId && (
                <a
                  href={`https://pubmed.ncbi.nlm.nih.gov/${pubmedId}/`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                  PubMed: {dataset.pubmed_id}
                </a>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
