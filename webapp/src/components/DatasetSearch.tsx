'use client';

import Link from 'next/link';
import { useState } from 'react';
import type { Dataset } from '@/lib/types';

export type DatasetListItem = Pick<Dataset, 'id' | 'modality' | 'species' | 'tissue' | 'cell_count' | 'release_status'>;
type SortKey = 'id' | 'modality' | 'species' | 'tissue' | 'cell_count';
type SortDir = 'asc' | 'desc';

export default function DatasetSearch({ datasets }: { datasets: DatasetListItem[] }) {
  const [query, setQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const filtered = query.trim()
    ? datasets.filter(
        (d) =>
          d.id.toLowerCase().includes(query.toLowerCase()) ||
          d.tissue.toLowerCase().includes(query.toLowerCase()) ||
          d.species.toLowerCase().includes(query.toLowerCase()) ||
          d.modality.toLowerCase().includes(query.toLowerCase())
      )
    : datasets;

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === 'number' && typeof bv === 'number') {
      return sortDir === 'asc' ? av - bv : bv - av;
    }
    const as = String(av).toLowerCase();
    const bs = String(bv).toLowerCase();
    return sortDir === 'asc' ? as.localeCompare(bs) : bs.localeCompare(as);
  });

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) {
      return <span className="text-slate-300 ml-1">↕</span>;
    }
    return (
      <span className="ml-1" style={{ color: 'var(--color-primary)' }}>
        {sortDir === 'asc' ? '↑' : '↓'}
      </span>
    );
  }

  const modalityStyle = (m: string) =>
    m === 'scRNA'
      ? 'bg-teal-100 text-teal-700 border border-teal-200'
      : 'bg-amber-100 text-amber-700 border border-amber-200';

  return (
    <div className="space-y-4">
      {/* Search bar */}
      <div className="relative max-w-sm">
        <svg
          className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
        </svg>
        <input
          type="search"
          placeholder="Search ID, tissue, species, modality…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-xl bg-white shadow-sm focus:outline-none focus:ring-2 focus:border-transparent"
          style={{ '--tw-ring-color': 'var(--color-primary)' } as React.CSSProperties}
        />
      </div>

      <p className="text-xs text-slate-500">
        Showing <span className="font-semibold text-slate-700">{sorted.length}</span> of {datasets.length} datasets
      </p>

      {/* Sortable table */}
      <div className="overflow-x-auto rounded-2xl border border-slate-200 shadow-sm bg-white">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-slate-200" style={{ background: 'var(--color-slate-light)' }}>
              {(
                [
                  { key: 'id', label: 'Dataset ID' },
                  { key: 'modality', label: 'Modality' },
                  { key: 'species', label: 'Species' },
                  { key: 'tissue', label: 'Tissue' },
                  { key: 'cell_count', label: 'Cells' },
                ] as { key: SortKey; label: string }[]
              ).map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => toggleSort(key)}
                  className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500 cursor-pointer select-none hover:text-teal-700 transition-colors ${key === 'cell_count' ? 'text-right' : 'text-left'}`}
                >
                  {label}
                  <SortIcon col={key} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {sorted.map((d) => (
              <tr
                key={d.id}
                className="hover:bg-teal-50/40 transition-colors"
              >
                <td className="px-4 py-2.5">
                  <Link
                    href={`/datasets/${d.id}`}
                    className="font-mono text-xs font-semibold hover:underline"
                    style={{ color: 'var(--color-primary)' }}
                  >
                    {d.id}
                  </Link>
                  {d.release_status === 'restricted' && (
                    <span className="ml-2 rounded-full border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                      restricted
                    </span>
                  )}
                </td>
                <td className="px-4 py-2.5">
                  <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${modalityStyle(d.modality)}`}>
                    {d.modality}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-slate-600 capitalize">{d.species}</td>
                <td className="px-4 py-2.5 text-slate-600">{d.tissue}</td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-slate-700">
                  {d.cell_count > 0 ? d.cell_count.toLocaleString() : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
