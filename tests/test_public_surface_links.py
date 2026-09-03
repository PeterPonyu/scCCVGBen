from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HOMEPAGE = "https://peterponyu.github.io/"
SCPORTAL = "https://peterponyu.github.io/scportal/"
AUTOSELECT = "https://peterponyu.github.io/scportal/autoselect/"
ATLAS = "https://peterponyu.github.io/scCCVGBen/"
COMPANION = "https://peterponyu.github.io/scccvgben-next/"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_source_surfaces_close_the_public_series_graph():
    texts = [
        read("webapp/src/lib/cite.ts"),
        read("webapp/src/components/SiteHeader.tsx"),
        read("webapp/src/app/layout.tsx"),
        read("site/config.toml"),
        read("site/layouts/partials/docs/inject/content-before.html"),
        read("site/layouts/partials/docs/inject/footer.html"),
    ]
    all_text = "\n".join(texts)
    for url in (HOMEPAGE, SCPORTAL, AUTOSELECT, ATLAS, COMPANION):
        assert url in all_text, f"missing public link {url}"
    for text in texts:
        assert not re.search(r"(?:localhost|127\.0\.0\.1|file:|/home/)", text, re.IGNORECASE)
        assert not re.search(
            r"github\.com/[^/]+/[^\s\"']+/(?:\.git/)?(?:secrets|private)",
            text,
            re.IGNORECASE,
        )

    assert "autoselect: '" + AUTOSELECT + "'" in texts[0]
    assert "{ href: CITE.autoselect, label: 'AutoSelect' }" in texts[1]
    assert "CITE.autoselect" in texts[2]
    assert 'SeriesAutoSelect = "' + AUTOSELECT + '"' in texts[3]
    assert texts[4].count("{{ $autoselect }}") == 1
    assert texts[5].count("{{ $autoselect }}") == 1


def test_public_surface_urls_keep_one_trailing_slash():
    for url in (HOMEPAGE, SCPORTAL, AUTOSELECT, ATLAS, COMPANION):
        assert url.endswith("/")
        assert not url.endswith("//")
