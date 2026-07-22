"""Seeded synthetic panel generator with known, planted causal effects.

This module builds the synthetic panels that carry the project's
proof-of-correctness (PROJECT_BRIEF.md §6). The true effect is planted by
construction, so the estimators can be *checked* rather than merely run: if
the 2x2 DiD does not recover the planted ATT, the estimator is wrong.

Three panels are provided, each isolating one thing the project must prove:

  * `generate_simple_panel` (Stage 2A) — one treatment date, one homogeneous
    effect. The recovery check: does the 2x2 DiD find a planted ATT?
  * `generate_staggered_panel` (Stage 2B) — multiple cohorts adopting at
    different dates with effects that differ *across cohorts* and *grow with
    exposure*. The bias check: naive TWFE fails here, Callaway & Sant'Anna
    does not.
  * `generate_displacement_panel` (Stage 2B) — a two-sector panel where a
    visible gain in the event-facing sector is exactly offset by a loss
    elsewhere, netting to zero. The honesty check: a naive reading reports a
    windfall that is not there.

The data-generating process for the simple case is the textbook DiD model:

    y_it = base_level + unit_fe_i + time_fe_t + true_att * D_it + noise_it
    D_it = 1 if city i is treated AND period t >= treat_period, else 0

Two properties matter and both are by construction:

  * **Parallel trends holds.** Treated and control cities share the *same*
    time effects ``time_fe_t`` and there is no group-specific trend, so in the
    absence of treatment their expected paths move in parallel. This is what
    makes the pre-period a clean test bed: a pre-trends test *should* pass
    here, and if it fails the test itself is suspect.
  * **The ATT is exactly ``true_att``.** The effect is homogeneous across
    treated cities and post periods, so the average treatment effect on the
    treated equals the planted constant with no aggregation subtleties.

All randomness derives from the committed seed in `config.py`, so the panel is
reproducible bit-for-bit from a clean checkout. Every generating parameter
lives in `PanelParams` and is stamped onto the returned frame's ``.attrs``, so
a notebook never has to hardcode the truth it is trying to recover.

Outcome units are illustrative: think of ``y`` as metro leisure & hospitality
employment in thousands of jobs, the real outcome used from Stage 3 onward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import SEED


@dataclass(frozen=True)
class PanelParams:
    """Generating parameters for the simple synthetic panel.

    Defaults are the committed configuration: changing them changes the truth
    the notebook checks against, because the notebook reads `true_att` back
    from here rather than restating it.
    """

    n_cities: int = 50
    """Number of cities (panel units)."""

    n_periods: int = 12
    """Number of equally-spaced time periods, numbered 1..n_periods."""

    treat_period: int = 7
    """First treated period. Periods < this are pre-treatment for everyone."""

    share_treated: float = 0.5
    """Fraction of cities assigned to the treated group."""

    true_att: float = 3.0
    """THE PLANTED EFFECT. Thousands of jobs added to treated cities post-treatment."""

    base_level: float = 100.0
    """Baseline outcome level (thousands of jobs) before city/time variation."""

    sd_unit: float = 5.0
    """SD of time-invariant city fixed effects (cities differ in size)."""

    time_trend: float = 0.5
    """Slope of the common linear time trend, shared by treated and control."""

    sd_time: float = 1.0
    """SD of common time shocks (business cycle / seasonality), shared by all."""

    sd_noise: float = 2.0
    """SD of idiosyncratic city-period noise."""

    seed: int = SEED
    """Committed random seed; defaults to the project-wide seed in config.py."""

    def __post_init__(self) -> None:
        if not 1 < self.treat_period <= self.n_periods:
            raise ValueError(
                f"treat_period must leave at least one pre-period and fall within "
                f"the panel: got {self.treat_period} with n_periods={self.n_periods}"
            )
        if not 0.0 < self.share_treated < 1.0:
            raise ValueError(
                f"share_treated must leave both a treated and a control group: "
                f"got {self.share_treated}"
            )

    @property
    def n_treated(self) -> int:
        """Number of treated cities."""
        return int(round(self.n_cities * self.share_treated))

    @property
    def n_pre_periods(self) -> int:
        """Number of pre-treatment periods."""
        return self.treat_period - 1

    @property
    def n_post_periods(self) -> int:
        """Number of post-treatment periods."""
        return self.n_periods - self.treat_period + 1


def generate_simple_panel(params: PanelParams | None = None) -> pd.DataFrame:
    """Generate a balanced synthetic city panel with a known planted ATT.

    Parameters
    ----------
    params
        Generating parameters. Defaults to the committed `PanelParams()`.

    Returns
    -------
    pandas.DataFrame
        Long-format panel, one row per city-period, sorted by city then period,
        with columns:

        ``city``
            City identifier (``city_00`` ...).
        ``period``
            Integer time period, 1..n_periods.
        ``treated``
            1 if the city is ever treated, else 0 (time-invariant group flag).
        ``post``
            1 if ``period >= treat_period``, else 0 (time flag, both groups).
        ``d``
            Treatment indicator ``treated * post`` — the actual regressor.
        ``cohort``
            Treatment period for treated cities, ``NaN`` for never-treated.
            Carried for forward-compatibility with staggered estimators.
        ``true_effect``
            The planted effect applied to this observation (``true_att * d``).
            Committed in the data so the truth is auditable, not just asserted.
        ``y``
            The outcome (thousands of jobs).

        The generating parameters are attached to ``df.attrs["params"]`` and
        the planted effect to ``df.attrs["true_att"]``.
    """
    p = params if params is not None else PanelParams()
    rng = np.random.default_rng(p.seed)

    cities = [f"city_{i:02d}" for i in range(p.n_cities)]
    periods = np.arange(1, p.n_periods + 1)

    # Treatment assignment is random but seeded: which cities host is not
    # correlated with their fixed effects, so there is no selection bias to
    # confound the recovery check.
    treated_idx = rng.choice(p.n_cities, size=p.n_treated, replace=False)
    is_treated = np.zeros(p.n_cities, dtype=int)
    is_treated[treated_idx] = 1

    # City fixed effects: cities differ in level (size), constant over time.
    unit_fe = rng.normal(0.0, p.sd_unit, size=p.n_cities)

    # Common time effects: a shared trend plus shared shocks. Because these are
    # identical for treated and control, parallel trends holds by construction.
    time_fe = p.time_trend * (periods - 1) + rng.normal(0.0, p.sd_time, size=p.n_periods)

    # Build the long panel via a cartesian product of cities and periods.
    city_grid, period_grid = np.meshgrid(np.arange(p.n_cities), periods, indexing="ij")
    city_grid = city_grid.ravel()
    period_grid = period_grid.ravel()

    treated_col = is_treated[city_grid]
    post_col = (period_grid >= p.treat_period).astype(int)
    d_col = treated_col * post_col
    true_effect = p.true_att * d_col

    noise = rng.normal(0.0, p.sd_noise, size=city_grid.size)

    y = (
        p.base_level
        + unit_fe[city_grid]
        + time_fe[period_grid - 1]
        + true_effect
        + noise
    )

    df = pd.DataFrame(
        {
            "city": [cities[i] for i in city_grid],
            "period": period_grid,
            "treated": treated_col,
            "post": post_col,
            "d": d_col,
            "cohort": np.where(treated_col == 1, float(p.treat_period), np.nan),
            "true_effect": true_effect,
            "y": y,
        }
    ).sort_values(["city", "period"], ignore_index=True)

    # Stamp the truth onto the frame so downstream code reads it rather than
    # restating it. (pandas does not guarantee attrs survive every operation,
    # so treat this as provenance, not as a data dependency.)
    df.attrs["params"] = asdict(p)
    df.attrs["true_att"] = p.true_att

    return df


# --------------------------------------------------------------------------- #
# Stage 2B: staggered adoption with heterogeneous, exposure-growing effects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StaggeredParams:
    """Generating parameters for the staggered-adoption panel.

    The effect structure here is the whole point and is deliberately *not*
    homogeneous. Two-way fixed effects is biased under staggered timing only
    when treatment effects are heterogeneous — with a single constant effect
    TWFE is fine and the demonstration would collapse. So effects are planted
    that vary **across cohorts** and **grow with exposure**:

        effect(g, t) = base_effect_g + growth_g * (t - g)   for t >= g

    Growth with exposure is what makes the "forbidden comparisons" bite: when
    TWFE uses an already-treated city as a control for a later-treated one, the
    control's own effect is still rising, and that rise is subtracted from the
    later city's change. The predicted consequence is a **downward**-biased
    TWFE estimate.
    """

    cohort_specs: tuple[tuple[int, float, float], ...] = (
        (5, 1.0, 0.6),
        (9, 1.5, 0.4),
        (13, 2.0, 0.2),
    )
    """``(adoption_period, base_effect, growth_per_period)`` per treated cohort.

    Read: the cohort adopting at period 5 jumps 1.0 on impact then gains 0.6
    per period of exposure; the period-9 cohort starts higher (1.5) but grows
    more slowly (0.4); the period-13 cohort starts highest (2.0) and grows
    slowest (0.2). Both the level and the slope differ by cohort — heterogeneity
    on two dimensions at once.
    """

    n_per_cohort: int = 15
    """Cities in each treated cohort."""

    n_never_treated: int = 15
    """Cities never treated. These are the clean comparison group for CS."""

    n_periods: int = 16
    """Number of periods, numbered 1..n_periods."""

    base_level: float = 100.0
    sd_unit: float = 5.0
    time_trend: float = 0.5
    sd_time: float = 1.0
    sd_noise: float = 2.0
    seed: int = SEED

    def __post_init__(self) -> None:
        if not self.cohort_specs:
            raise ValueError("need at least one treated cohort")
        adopt = [g for g, _, _ in self.cohort_specs]
        if len(set(adopt)) != len(adopt):
            raise ValueError(f"cohort adoption periods must be unique: {adopt}")
        if min(adopt) < 2:
            raise ValueError(
                f"every cohort needs at least one pre-period: earliest adoption "
                f"is {min(adopt)}"
            )
        if max(adopt) > self.n_periods:
            raise ValueError(
                f"cohort adopts at {max(adopt)} but panel ends at {self.n_periods}"
            )
        if self.n_never_treated < 1:
            raise ValueError(
                "a never-treated group is required as the Callaway & Sant'Anna "
                "comparison group"
            )

    @property
    def n_cities(self) -> int:
        return len(self.cohort_specs) * self.n_per_cohort + self.n_never_treated

    def effect(self, cohort: float, period: float) -> float:
        """Planted effect for a cohort at a period. Zero before adoption."""
        for g, base, growth in self.cohort_specs:
            if g == cohort:
                return 0.0 if period < g else base + growth * (period - g)
        raise KeyError(f"unknown cohort {cohort}")


def generate_staggered_panel(params: StaggeredParams | None = None) -> pd.DataFrame:
    """Generate a staggered-adoption panel with heterogeneous, growing effects.

    Returns
    -------
    pandas.DataFrame
        Long-format panel with columns ``city``, ``period``, ``cohort``
        (``NaN`` for never-treated), ``treated`` (ever-treated flag), ``d``
        (treated-and-post indicator), ``exposure`` (``period - cohort``, ``NaN``
        for never-treated), ``true_effect`` (the planted effect for that
        observation), and ``y``.

        ``df.attrs`` carries the parameters, the true overall ATT (the mean
        planted effect across treated observations), and the true event-study
        profile by exposure — so the notebook reads the truth back instead of
        restating it.
    """
    p = params if params is not None else StaggeredParams()
    rng = np.random.default_rng(p.seed)

    # Assign cities to cohorts: treated cohorts first, then never-treated.
    cohort_of_city: list[float] = []
    for g, _, _ in p.cohort_specs:
        cohort_of_city.extend([float(g)] * p.n_per_cohort)
    cohort_of_city.extend([np.nan] * p.n_never_treated)
    cohort_arr = np.array(cohort_of_city, dtype=float)

    cities = [f"city_{i:02d}" for i in range(p.n_cities)]
    periods = np.arange(1, p.n_periods + 1)

    unit_fe = rng.normal(0.0, p.sd_unit, size=p.n_cities)
    # Common to every city, so parallel trends holds absent treatment.
    time_fe = p.time_trend * (periods - 1) + rng.normal(0.0, p.sd_time, size=p.n_periods)

    city_grid, period_grid = np.meshgrid(np.arange(p.n_cities), periods, indexing="ij")
    city_grid = city_grid.ravel()
    period_grid = period_grid.ravel()

    cohort_col = cohort_arr[city_grid]
    treated_col = (~np.isnan(cohort_col)).astype(int)
    with np.errstate(invalid="ignore"):
        d_col = np.where(
            np.isnan(cohort_col), 0, (period_grid >= cohort_col).astype(int)
        ).astype(int)
        exposure = np.where(np.isnan(cohort_col), np.nan, period_grid - cohort_col)

    # Planted effect, cohort-varying and exposure-growing.
    true_effect = np.zeros(city_grid.size, dtype=float)
    for g, base, growth in p.cohort_specs:
        sel = (cohort_col == g) & (period_grid >= g)
        true_effect[sel] = base + growth * (period_grid[sel] - g)

    noise = rng.normal(0.0, p.sd_noise, size=city_grid.size)

    y = (
        p.base_level
        + unit_fe[city_grid]
        + time_fe[period_grid - 1]
        + true_effect
        + noise
    )

    df = pd.DataFrame(
        {
            "city": [cities[i] for i in city_grid],
            "period": period_grid,
            "cohort": cohort_col,
            "treated": treated_col,
            "d": d_col,
            "exposure": exposure,
            "true_effect": true_effect,
            "y": y,
        }
    ).sort_values(["city", "period"], ignore_index=True)

    treated_obs = df.loc[df["d"] == 1]
    df.attrs["params"] = asdict(p)
    # The estimand: average planted effect across treated unit-periods.
    df.attrs["true_att"] = float(treated_obs["true_effect"].mean())
    df.attrs["true_event_profile"] = (
        treated_obs.groupby("exposure")["true_effect"].mean().to_dict()
    )

    return df


# --------------------------------------------------------------------------- #
# Stage 2B: displacement / pure substitution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DisplacementParams:
    """Generating parameters for the displacement ("Orlando") panel.

    Simulates the substitution and crowding-out critique that thirty years of
    mega-event economics keeps finding (PROJECT_BRIEF.md §3). The host city's
    event-facing sector genuinely gains ``gross_boost``. But that activity is
    pulled from the city's other tourism sector — football fans displacing the
    theme-park visitors who would otherwise have come.

    With ``displacement_share = 1.0`` the offset is exact: **the true net effect
    on total local employment is zero.** An analyst who measures only the
    event-facing sector reports a windfall that does not exist.
    """

    n_cities: int = 50
    n_periods: int = 12
    treat_period: int = 7
    share_treated: float = 0.5

    gross_boost: float = 4.0
    """Visible gain in the event-facing sector for treated cities post-treatment."""

    displacement_share: float = 1.0
    """Fraction of the gross boost pulled from the other sector.

    ``1.0`` is pure substitution (zero net effect) — the committed default and
    the case the notebook demonstrates. ``0.0`` would be pure new activity.
    """

    base_event: float = 40.0
    """Baseline event-facing employment (e.g. accommodation & food)."""

    base_other: float = 60.0
    """Baseline other-tourism employment (e.g. amusement & recreation)."""

    sd_unit: float = 4.0
    time_trend: float = 0.3
    sd_time: float = 0.8
    sd_noise: float = 1.5
    seed: int = SEED

    def __post_init__(self) -> None:
        if not 0.0 <= self.displacement_share <= 1.0:
            raise ValueError(
                f"displacement_share must be in [0, 1]: got {self.displacement_share}"
            )

    @property
    def displaced(self) -> float:
        """Amount pulled out of the other sector."""
        return self.gross_boost * self.displacement_share

    @property
    def true_net_effect(self) -> float:
        """The honest estimand: net effect on TOTAL employment."""
        return self.gross_boost - self.displaced


def generate_displacement_panel(
    params: DisplacementParams | None = None,
) -> pd.DataFrame:
    """Generate a two-sector panel where apparent gains are pure substitution.

    Returns
    -------
    pandas.DataFrame
        Long-format panel with ``city``, ``period``, ``treated``, ``post``,
        ``d``, ``cohort``, the two sector outcomes ``y_event`` and ``y_other``,
        their sum ``y_total``, and the planted truths ``true_gross_effect``,
        ``true_displaced_effect``, ``true_net_effect``.
    """
    p = params if params is not None else DisplacementParams()
    rng = np.random.default_rng(p.seed)

    n_treated = int(round(p.n_cities * p.share_treated))
    cities = [f"city_{i:02d}" for i in range(p.n_cities)]
    periods = np.arange(1, p.n_periods + 1)

    treated_idx = rng.choice(p.n_cities, size=n_treated, replace=False)
    is_treated = np.zeros(p.n_cities, dtype=int)
    is_treated[treated_idx] = 1

    # Independent fixed effects and shocks per sector.
    unit_fe_e = rng.normal(0.0, p.sd_unit, size=p.n_cities)
    unit_fe_o = rng.normal(0.0, p.sd_unit, size=p.n_cities)
    time_fe_e = p.time_trend * (periods - 1) + rng.normal(0.0, p.sd_time, size=p.n_periods)
    time_fe_o = p.time_trend * (periods - 1) + rng.normal(0.0, p.sd_time, size=p.n_periods)

    city_grid, period_grid = np.meshgrid(np.arange(p.n_cities), periods, indexing="ij")
    city_grid = city_grid.ravel()
    period_grid = period_grid.ravel()

    treated_col = is_treated[city_grid]
    post_col = (period_grid >= p.treat_period).astype(int)
    d_col = treated_col * post_col

    gross = p.gross_boost * d_col          # what the event-facing sector gains
    displaced = -p.displaced * d_col       # what the other sector loses
    net = gross + displaced                # zero when displacement_share == 1

    y_event = (
        p.base_event
        + unit_fe_e[city_grid]
        + time_fe_e[period_grid - 1]
        + gross
        + rng.normal(0.0, p.sd_noise, size=city_grid.size)
    )
    y_other = (
        p.base_other
        + unit_fe_o[city_grid]
        + time_fe_o[period_grid - 1]
        + displaced
        + rng.normal(0.0, p.sd_noise, size=city_grid.size)
    )

    df = pd.DataFrame(
        {
            "city": [cities[i] for i in city_grid],
            "period": period_grid,
            "treated": treated_col,
            "post": post_col,
            "d": d_col,
            "cohort": np.where(treated_col == 1, float(p.treat_period), np.nan),
            "y_event": y_event,
            "y_other": y_other,
            "y_total": y_event + y_other,
            "true_gross_effect": gross,
            "true_displaced_effect": displaced,
            "true_net_effect": net,
        }
    ).sort_values(["city", "period"], ignore_index=True)

    df.attrs["params"] = asdict(p)
    df.attrs["true_gross_effect"] = float(p.gross_boost)
    df.attrs["true_displaced_effect"] = float(-p.displaced)
    df.attrs["true_net_effect"] = float(p.true_net_effect)

    return df


__all__ = [
    "PanelParams",
    "generate_simple_panel",
    "StaggeredParams",
    "generate_staggered_panel",
    "DisplacementParams",
    "generate_displacement_panel",
]
