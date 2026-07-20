"""Project-wide configuration: seeds and paths.

Single source of truth for reproducibility. Every stochastic step in the
project — the synthetic panel generator, any bootstrap inference, any
resampling — draws its randomness from the seed defined here, and the
notebooks resolve their file locations through the paths defined here. This
is what lets the synthetic validation be re-run bit-for-bit from a clean
checkout: the generating seed is committed, not incidental.

Prefer `rng()` (a seeded ``numpy`` Generator) for new code. `set_global_seeds()`
is provided for libraries that still read global state (legacy ``numpy``,
the stdlib ``random`` module, and ``PYTHONHASHSEED``).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
# The one global seed. Committed on purpose: the synthetic panel's "true"
# planted effects are only reproducible because this value is fixed.
SEED: int = 20260620  # 2026-06-20 — the day the 2026 World Cup group stage begins.


def rng(seed: int | None = None) -> np.random.Generator:
    """Return a seeded NumPy ``Generator``.

    This is the preferred randomness source for the project. Passing an
    explicit ``seed`` lets a caller spin off an independent, still-reproducible
    stream (e.g. per-bootstrap-replicate) without disturbing the global one.
    """
    return np.random.default_rng(SEED if seed is None else seed)


def set_global_seeds(seed: int | None = None) -> int:
    """Seed all global RNG state for reproducibility.

    Seeds the stdlib ``random`` module, legacy ``numpy`` global RNG, and
    ``PYTHONHASHSEED``. Prefer the explicit `rng()` Generator in new code;
    use this to pin libraries that still read global state. Returns the seed
    used.
    """
    s = SEED if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    return s


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolved relative to this file so notebooks and scripts agree regardless of
# the working directory they are launched from.
SRC_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SRC_DIR.parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"


__all__ = [
    "SEED",
    "rng",
    "set_global_seeds",
    "SRC_DIR",
    "ROOT_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "NOTEBOOKS_DIR",
]
