"""Reusable difference-in-differences, event-study, and CS-estimator functions.

STATUS: stub — implemented in Stage 5 (see PROJECT_BRIEF.md §9.5).

The single source of truth for the estimation machinery shared across the
notebooks:

  * the canonical 2x2 DiD and its parallel-trends test;
  * the event-study (dynamic DiD) specification — leads and lags;
  * the staggered-treatment layer: a naive TWFE baseline for contrast against
    the Callaway & Sant'Anna estimator (via the `differences` package);
  * robustness helpers — placebo tests, HonestDiD-style sensitivity bounds,
    and clustered / wild-cluster-bootstrap inference for the few-treated-
    clusters problem (PROJECT_BRIEF.md §5).

Factored out of the notebooks so every stage calls the same, tested code.
"""

from __future__ import annotations

raise NotImplementedError("did.py is a Stage 5 deliverable; not implemented yet.")
