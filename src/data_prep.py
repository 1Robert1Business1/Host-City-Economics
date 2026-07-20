"""QCEW download, county->metro aggregation, and treated/control construction.

STATUS: stub — implemented in Stage 3 (see PROJECT_BRIEF.md §9.3).

Builds the real analysis panel from public BLS Quarterly Census of Employment
and Wages (QCEW) data:

  * pull leisure & hospitality employment via the QCEW Open Data Access API;
  * aggregate county-level industry data up to metros by hand, because as of
    Q2 2025 BLS publishes MSA data as totals only (PROJECT_BRIEF.md §4);
  * assemble host (treated) vs. comparable non-host (control) metros, matched
    on pre-tournament size, industry mix, and baseline tourism.

Scope is US-only by design (§4): the cleanest free city-level economic data is
US; Mexican/Canadian city data is noted as a limitation, not faked.
"""

from __future__ import annotations

raise NotImplementedError(
    "data_prep.py is a Stage 3 deliverable; not implemented yet."
)
