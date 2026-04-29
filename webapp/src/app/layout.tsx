import type { Metadata } from 'next';
import SiteHeader from '@/components/SiteHeader';
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
