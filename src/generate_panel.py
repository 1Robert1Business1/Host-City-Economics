"""Seeded synthetic panel generator with known, planted causal effects.

STATUS: stub — implemented in Stage 2 (see PROJECT_BRIEF.md §9.2).

This module builds the synthetic panel that carries the project's
proof-of-correctness (PROJECT_BRIEF.md §6). It plants a *known* treatment
effect by construction so the estimators can be shown to recover it:

  * a clean average treatment effect for the canonical 2x2 recovery check;
  * a staggered-adoption panel with heterogeneous effects, where naive TWFE
    is biased and Callaway & Sant'Anna recovers the truth;
  * a pure-displacement ("Orlando") scenario whose apparent gains net to zero.

All randomness is drawn from the committed seed in `config.py`, so the panel
is reproducible bit-for-bit from a clean checkout.
"""

from __future__ import annotations

raise NotImplementedError(
    "generate_panel.py is a Stage 2 deliverable; not implemented yet."
)
