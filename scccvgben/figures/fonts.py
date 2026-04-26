"""Font helpers shared by manuscript figure renderers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def font_dir_candidates() -> tuple[Path, ...]:
    """Return likely locations for the lab Arial font bundle."""
    env = os.environ.get("SCCCVGBEN_FONT_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    root = _repo_root()
    candidates.extend([
        root / "fonts",
        root / "site" / "static" / "fonts",
        root.parent / "FONTS",
        root.parent / "fonts",
        Path("/home/zeyufu/LAB/FONTS"),
    ])
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return tuple(unique)


@lru_cache(maxsize=1)
def arial_font_paths() -> dict[str, Path]:
    """Find Arial regular/bold/italic faces in the configured font folders."""
    names = {
        "regular": ("Arial.ttf", "arial.ttf"),
        "bold": ("Arial Bold.ttf", "Arial-Bold.ttf", "arialbd.ttf"),
        "italic": ("Arial Italic.ttf", "Arial-Italic.ttf", "ariali.ttf"),
        "bold_italic": (
            "Arial Bold Italic.ttf",
            "Arial-BoldItalic.ttf",
            "Arial Bold Italic.otf",
            "arialbi.ttf",
        ),
    }
    found: dict[str, Path] = {}
    for directory in font_dir_candidates():
        if not directory.is_dir():
            continue
        for face, candidates in names.items():
            if face in found:
                continue
            for name in candidates:
                path = directory / name
                if path.exists():
                    found[face] = path
                    break
    return found


def arial_font_path(*, bold: bool = False, italic: bool = False) -> Path | None:
    """Return a concrete Arial face path for PIL or ``None`` if unavailable."""
    paths = arial_font_paths()
    if bold and italic:
        return paths.get("bold_italic") or paths.get("bold") or paths.get("italic") or paths.get("regular")
    if bold:
        return paths.get("bold") or paths.get("regular")
    if italic:
        return paths.get("italic") or paths.get("regular")
    return paths.get("regular")


def register_arial_with_matplotlib() -> bool:
    """Register local Arial TTFs with Matplotlib and report availability."""
    try:
        from matplotlib import font_manager
    except Exception:
        return False
    registered = False
    for path in arial_font_paths().values():
        try:
            font_manager.fontManager.addfont(str(path))
            registered = True
        except Exception:
            continue
    return registered
