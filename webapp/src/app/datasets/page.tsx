import { getDatasets } from '@/lib/data';
import DatasetSearch from '@/components/DatasetSearch';
import type { DatasetListItem } from '@/components/DatasetSearch';

export default async function DatasetsPage() {
  const datasets = await getDatasets();
  const scRNA = datasets.filter((d) => d.modality === 'scRNA').length;
  const scATAC = datasets.filter((d) => d.modality === 'scATAC').length;
  const restricted = datasets.filter((d) => d.release_status === 'restricted').length;
  const listItems: DatasetListItem[] = datasets.map((d) => ({
    id: d.id,
    modality: d.modality,
    species: d.species,
    tissue: d.tissue,
    cell_count: d.cell_count,
    release_status: d.release_status,
  }));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Datasets</h1>
          <p className="text-slate-500 mt-1 text-sm">
            {datasets.length} benchmark records spanning scRNA and scATAC modalities
            {restricted ? ` (${restricted} restricted-access rows redacted for public display)` : ''}.
          </p>
        </div>
        {/* Modality summary pills */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-teal-100 text-teal-700 border border-teal-200">
            scRNA
            <span className="font-bold">{scRNA}</span>
          </span>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-100 text-amber-700 border border-amber-200">
            scATAC
            <span className="font-bold">{scATAC}</span>
          </span>
        </div>
      </div>
      <DatasetSearch datasets={listItems} />
    </div>
  );
}
