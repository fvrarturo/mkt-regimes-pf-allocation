"""Helper utilities for locating key directories inside `main_project`.

All Section 1 scripts rely on these helpers to avoid hard-coding fragile
relative paths (e.g., references to the old `main_project2` folder). Each
function caches the discovered location so repeated calls stay cheap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]


def _candidate_roots(start_path: Path) -> Iterable[Path]:
    """Yield candidate directories starting from *start_path* up to `/`."""
    if start_path.is_dir():
        yield start_path
    yield from start_path.parents


@lru_cache(maxsize=None)
def get_project_root(start_path: Optional[PathLike] = None) -> Path:
    """
    Locate the canonical `main_project` root (contains `data/` + `s1_macro_vars/`).
    """
    path = Path(start_path) if start_path else Path(__file__)
    path = path.resolve()

    for candidate in _candidate_roots(path):
        if (candidate / "data").exists() and (candidate / "s1_macro_vars").exists():
            return candidate

    raise RuntimeError(
        f"Could not locate project root starting from {path}. "
        "Ensure you are executing scripts from within the main_project tree."
    )


@lru_cache(maxsize=None)
def get_data_dir(start_path: Optional[PathLike] = None) -> Path:
    """Return `<project_root>/data`."""
    return get_project_root(start_path) / "data"


@lru_cache(maxsize=None)
def get_section_root(start_path: Optional[PathLike] = None) -> Path:
    """Return the `s1_macro_vars` directory."""
    return get_project_root(start_path) / "s1_macro_vars"


__all__ = ["get_project_root", "get_data_dir", "get_section_root"]
