import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: 'scCCVGBen — Single-Cell Graph VAE Benchmark',
  description:
    'A comprehensive benchmark of graph-encoder variational autoencoders for single-cell omics.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col" style={{ background: 'var(--color-slate-light)', fontFamily: 'var(--font-sans)' }}>
        {/* Sticky top navbar — NOT a sidebar */}
        <header className="sticky top-0 z-20 bg-white/90 backdrop-blur border-b border-slate-200 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
            {/* Brand */}
            <Link
              href="/"
              className="flex items-center gap-2 text-base font-bold tracking-tight shrink-0"
              style={{ color: 'var(--color-primary)' }}
            >
              <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg text-white text-xs font-bold"
                    style={{ background: 'var(--color-primary)' }}>
                sc
              </span>
              <span className="text-slate-800">CCVGBen</span>
            </Link>

            {/* Primary nav tabs */}
            <nav className="flex items-center gap-1 text-sm font-medium">
              <Link
                href="/datasets"
                className="px-3 py-1.5 rounded-lg transition-colors text-slate-600 hover:bg-teal-50 hover:text-teal-700"
              >
                Datasets
              </Link>
              <Link
                href="/methods"
                className="px-3 py-1.5 rounded-lg transition-colors text-slate-600 hover:bg-teal-50 hover:text-teal-700"
              >
                Methods
              </Link>
              <Link
                href="/metrics"
                className="px-3 py-1.5 rounded-lg transition-colors text-slate-600 hover:bg-teal-50 hover:text-teal-700"
              >
                Metrics
              </Link>
              <Link
                href="/resource-integration"
                className="hidden sm:inline-flex px-3 py-1.5 rounded-lg transition-colors text-slate-600 hover:bg-teal-50 hover:text-teal-700"
              >
                Resource
              </Link>
            </nav>

            {/* Search shortcut badge */}
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-slate-200 text-xs text-slate-400 cursor-pointer hover:border-teal-300 transition-colors select-none">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
              </svg>
              <span>Search datasets…</span>
            </div>
          </div>
        </header>

        <main className="flex-1 w-full">{children}</main>

        {/* Footer */}
        <footer className="border-t border-slate-200 bg-white mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
            <span className="font-medium text-slate-700">scCCVGBen</span>
            <div className="flex items-center gap-4">
              <a
                href="https://doi.org/10.1101/2025.02.01.636000"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-teal-600 transition-colors"
              >
                Paper (bioRxiv)
              </a>
              <a
                href="https://github.com/PeterPonyu/scCCVGBen"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-teal-600 transition-colors"
              >
                GitHub
              </a>
              <span>Data: 2026-04-28</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
