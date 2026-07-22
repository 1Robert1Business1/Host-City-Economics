"""Seeded synthetic panel generator with known, planted causal effects.

This module builds the synthetic panels that carry the project's
proof-of-correctness (PROJECT_BRIEF.md §6). The true effect is planted by
construction, so the estimators can be *checked* rather than merely run: if
the 2x2 DiD does not recover the planted ATT, the estimator is wrong.

Stage 2A implements the simple case: a balanced panel of cities observed over
equally-spaced periods, a single treatment date, treated vs. never-treated
groups, and one homogeneous treatment effect.

The data-generating process is deliberately the textbook DiD model:

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


__all__ = ["PanelParams", "generate_simple_panel"]
