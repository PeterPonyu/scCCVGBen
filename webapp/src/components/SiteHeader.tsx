'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

const navItems = [
  { href: '/datasets', label: 'Datasets' },
  { href: '/methods', label: 'Methods' },
  { href: '/metrics', label: 'Metrics' },
  { href: '/resource-integration', label: 'Resource' },
];

function isActive(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

function navClass(active: boolean): string {
  return [
    'rounded-lg px-3 py-2 text-sm font-semibold transition-colors',
    active
      ? 'bg-teal-50 text-teal-700 ring-1 ring-teal-100'
      : 'text-slate-600 hover:bg-teal-50 hover:text-teal-700',
  ].join(' ');
}

export default function SiteHeader() {
  const pathname = usePathname() || '/';
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 shadow-sm backdrop-blur">
      <div className="mx-auto flex min-h-14 max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-2 sm:px-6 lg:flex-nowrap">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2 text-base font-bold tracking-tight"
          style={{ color: 'var(--color-primary)' }}
          onClick={() => setMenuOpen(false)}
        >
          <span
            className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-xs font-bold text-white"
            style={{ background: 'var(--color-primary)' }}
          >
            sc
          </span>
          <span className="text-slate-800">CCVGBen</span>
        </Link>

        <nav className="hidden min-w-0 flex-1 items-center justify-center gap-1 lg:flex" aria-label="Primary navigation">
          {navItems.map((item) => (
            <Link key={item.href} href={item.href} className={navClass(isActive(pathname, item.href))}>
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="hidden shrink-0 items-center gap-2 lg:flex">
          <a
            href="https://peterponyu.github.io/scCCVGBen/"
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-500 transition-colors hover:border-teal-300 hover:text-teal-700"
          >
            Hugo atlas
          </a>
          <a
            href="https://peterponyu.github.io/scportal/"
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-500 transition-colors hover:border-teal-300 hover:text-teal-700"
          >
            scPortal
          </a>
        </div>

        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 transition-colors hover:border-teal-300 hover:text-teal-700 lg:hidden"
          aria-expanded={menuOpen}
          aria-controls="scccvgben-mobile-nav"
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span>{menuOpen ? 'Close' : 'Menu'}</span>
          <span aria-hidden="true">{menuOpen ? '×' : '☰'}</span>
        </button>
      </div>

      <div
        id="scccvgben-mobile-nav"
        className={`${menuOpen ? 'block' : 'hidden'} border-t border-slate-200 bg-white px-4 py-3 lg:hidden`}
      >
        <nav className="mx-auto grid max-w-7xl gap-2" aria-label="Mobile navigation">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={navClass(isActive(pathname, item.href))}
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </Link>
          ))}
          <div className="mt-2 grid gap-2 border-t border-slate-100 pt-3 text-sm font-semibold">
            <a href="https://peterponyu.github.io/scCCVGBen/" className="text-slate-600 hover:text-teal-700">
              Hugo benchmark atlas
            </a>
            <a href="https://peterponyu.github.io/scportal/" className="text-slate-600 hover:text-teal-700">
              scPortal discovery hub
            </a>
          </div>
        </nav>
      </div>
    </header>
  );
}
