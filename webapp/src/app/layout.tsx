import type { Metadata } from 'next';
import SiteHeader from '@/components/SiteHeader';
import { CITE } from '@/lib/cite';
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
        <SiteHeader />

        <main className="flex-1 w-full">{children}</main>

        <footer className="mt-12 border-t border-slate-200 bg-white">
          <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-4 py-5 text-[13px] text-slate-500 sm:flex-row sm:px-6">
            <span className="font-medium text-slate-700">scCCVGBen</span>
            <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
              <a
                href={CITE.doi}
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors hover:text-teal-600"
              >
                DOI
              </a>
              <a
                href={CITE.code}
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors hover:text-teal-600"
              >
                Code
              </a>
              <a href={CITE.site} className="transition-colors hover:text-teal-600">
                Site
              </a>
              <a href={CITE.homepage} className="transition-colors hover:text-teal-600">
                Homepage
              </a>
              <a href={CITE.scportal} className="transition-colors hover:text-teal-600">
                SCPortal
              </a>
              <span>Data: {CITE.dataSnapshot}</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
