#!/usr/bin/env python3
"""Render publication figures from the polished Next.js static export.

Outputs:
  figures/fig_supp_nextjs.{png,pdf}
  figures/fig_online_resource_integration.{png,pdf}
  manuscript/scccvgben/fig_supp_nextjs.pdf
  manuscript/scccvgben/fig_online_resource_integration.pdf

The script captures browser-rendered routes instead of stitching LaTeX panels.
When the export is built for a GitHub Pages project path, pass the same
``--base-path`` used by Next.js (for example ``/scccvgben-next``).
"""
from __future__ import annotations

import argparse
import http.server
import shutil
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEB_OUT = ROOT / "webapp" / "out"
FIGURES = ROOT / "figures"
MANUSCRIPT = ROOT / "manuscript" / "scccvgben"
DPI = 300
MAX_INCLUDED_WIDTH_CM = 17.0
MAX_INCLUDED_HEIGHT_CM = 21.0

ROUTES = {
    # Capture windows are sized for print readability.  The submission audit
    # enforces the final manuscript inclusion envelope (<=17 cm x <=21 cm)
    # because LaTeX include options, not raw screenshot dimensions alone,
    # determine the final printed size.
    "fig_supp_nextjs": {
        "route": "/supplementary-figure/",
        "window": "1100,2050",
    },
    "fig_online_resource_integration": {
        "route": "/resource-integration/",
        "window": "1100,1100",
    },
}
MANUSCRIPT_COPY_STEMS = tuple(ROUTES)


def _chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("No Chrome/Chromium executable found for headless screenshots")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
        return


def _normalize_base_path(base_path: str | None) -> str:
    if not base_path:
        return ""
    cleaned = "/" + base_path.strip("/")
    return "" if cleaned == "/" else cleaned


def _serve(directory: Path, base_path: str = "") -> tuple[socketserver.TCPServer, int]:
    class Handler(QuietHandler):
        def translate_path(self, path: str) -> str:
            if base_path and (path == base_path or path.startswith(f"{base_path}/")):
                stripped = path[len(base_path):] or "/"
                return super().translate_path(stripped)
            return super().translate_path(path)

        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, directory=str(directory), **kwargs)

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = int(httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _wait_for(url: str) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as resp:  # noqa: S310 - local server only
                if resp.status < 500:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {url}")


def _capture(url: str, out_png: Path, window: str) -> None:
    chrome = _chrome()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        f"--window-size={window}",
        f"--screenshot={out_png}",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        raise RuntimeError(f"Chrome screenshot failed for {url}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    if not out_png.exists() or out_png.stat().st_size == 0:
        raise RuntimeError(f"Chrome did not create screenshot {out_png}")


def _png_to_pdf(png: Path, pdf: Path) -> None:
    image = Image.open(png).convert("RGB")
    image.save(pdf, "PDF", resolution=DPI)


def _png_natural_size_cm(png: Path) -> tuple[float, float]:
    image = Image.open(png)
    width_px, height_px = image.size
    return width_px / DPI * 2.54, height_px / DPI * 2.54


def _dimension_policy_line(stem: str, png: Path) -> str:
    width_cm, height_cm = _png_natural_size_cm(png)
    scaled_height_at_max_width = height_cm * (MAX_INCLUDED_WIDTH_CM / width_cm)
    natural_status = "PASS" if width_cm <= MAX_INCLUDED_WIDTH_CM and height_cm <= MAX_INCLUDED_HEIGHT_CM else "FAIL"
    width_scaled_status = "PASS" if scaled_height_at_max_width <= MAX_INCLUDED_HEIGHT_CM else "FAIL"
    return (
        f"dimension {stem}: natural={width_cm:.2f}x{height_cm:.2f} cm "
        f"({natural_status} <= {MAX_INCLUDED_WIDTH_CM:g}x{MAX_INCLUDED_HEIGHT_CM:g} cm); "
        f"at {MAX_INCLUDED_WIDTH_CM:g} cm width height={scaled_height_at_max_width:.2f} cm "
        f"({width_scaled_status})"
    )


def render_figures(web_out: Path, out_dir: Path, base_path: str = "") -> list[Path]:
    if not (web_out / "index.html").exists():
        raise FileNotFoundError(f"Next.js export not found at {web_out}; run `npm run build` in webapp first")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = _normalize_base_path(base_path)
    httpd, port = _serve(web_out, base_path)
    written: list[Path] = []
    try:
        root_url = f"http://127.0.0.1:{port}/"
        _wait_for(root_url)
        for stem, spec in ROUTES.items():
            route_url = f"http://127.0.0.1:{port}{base_path}{spec['route']}"
            _wait_for(route_url)
            png = out_dir / f"{stem}.png"
            pdf = out_dir / f"{stem}.pdf"
            _capture(route_url, png, str(spec["window"]))
            _png_to_pdf(png, pdf)
            written.extend([png, pdf])
    finally:
        httpd.shutdown()
        httpd.server_close()
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-out", type=Path, default=WEB_OUT, help="Static Next.js export directory")
    parser.add_argument("--out-dir", type=Path, default=FIGURES, help="Figure output directory")
    parser.add_argument(
        "--base-path",
        default="",
        help="Optional deployment base path to strip while serving the export, e.g. /scccvgben-next",
    )
    parser.add_argument("--no-manuscript-copy", action="store_true", help="Do not copy web-resource PDFs into manuscript directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    written = render_figures(args.web_out, args.out_dir, args.base_path)
    if not args.no_manuscript_copy:
        MANUSCRIPT.mkdir(parents=True, exist_ok=True)
        for stem in MANUSCRIPT_COPY_STEMS:
            src = args.out_dir / f"{stem}.pdf"
            dst = MANUSCRIPT / f"{stem}.pdf"
            shutil.copy2(src, dst)
            written.append(dst)
    for path in written:
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    for stem in ROUTES:
        png = args.out_dir / f"{stem}.png"
        if png.exists():
            print(_dimension_policy_line(stem, png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
