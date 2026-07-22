"""Reusable difference-in-differences estimators and diagnostics.

The single source of truth for the estimation machinery. Everything here is
written against *column names*, not against a particular dataset, so the same
functions run on the synthetic panel (Stage 2) and on the real QCEW metro
panel (Stage 4) with no re-implementation.

Stage 2A provides the two pieces the synthetic recovery gate needs:

  * `did_2x2` — the canonical 2x2 difference-in-differences estimator.
  * `pretrends_test` — the pre-treatment parallel-trends check.

Parallel trends is the assumption the whole design rests on (PROJECT_BRIEF.md
§5.1), so the diagnostic ships alongside the estimator rather than as an
afterthought.

Inference note: standard errors are clustered on the panel unit by default.
DiD residuals are serially correlated within a unit, and unclustered standard
errors are badly anti-conservative in that setting. The few-treated-clusters
problem (a handful of host cities) and the wild cluster bootstrap that
addresses it are taken up in Stage 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


# --------------------------------------------------------------------------- #
# Canonical 2x2 DiD
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiD2x2Result:
    """Result of a canonical 2x2 difference-in-differences estimation."""

    att: float
    """The DiD point estimate: the treated-vs-control change in the outcome."""

    se: float
    """Standard error of the ATT (clustered on unit unless disabled)."""

    tstat: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    alpha: float
    """Significance level used for the confidence interval (0.05 -> 95% CI)."""

    n_obs: int
    n_units: int
    n_treated_units: int
    cluster: bool

    def contains(self, value: float) -> bool:
        """Whether the confidence interval covers ``value``.

        This is the recovery check: for a synthetic panel with a planted
        effect, ``result.contains(true_att)`` must be True.
        """
        return bool(self.ci_lower <= value <= self.ci_upper)

    def summary(self, planted: float | None = None) -> str:
        """Human-readable one-block summary, optionally against a planted truth."""
        pct = int(round((1 - self.alpha) * 100))
        lines = [
            f"Canonical 2x2 DiD",
            f"  ATT estimate     : {self.att:+.4f}",
            f"  Std. error       : {self.se:.4f}"
            f" ({'clustered by unit' if self.cluster else 'unclustered'})",
            f"  t / p            : {self.tstat:+.3f} / {self.pvalue:.4g}",
            f"  {pct}% CI           : [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
            f"  N                : {self.n_obs} obs, {self.n_units} units"
            f" ({self.n_treated_units} treated)",
        ]
        if planted is not None:
            covered = self.contains(planted)
            lines += [
                f"  Planted ATT      : {planted:+.4f}",
                f"  Recovery error   : {self.att - planted:+.4f}",
                f"  CI covers truth  : {'YES' if covered else 'NO'}",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def did_2x2(
    data: pd.DataFrame,
    *,
    outcome: str = "y",
    unit: str = "city",
    time: str = "period",
    treated: str = "treated",
    treat_period: float,
    cluster: bool = True,
    alpha: float = 0.05,
) -> DiD2x2Result:
    """Estimate the canonical 2x2 difference-in-differences.

    Fits the textbook interaction specification

        y_it = b0 + b1*treated_i + b2*post_t + b3*(treated_i x post_t) + e_it

    where ``post_t = 1{t >= treat_period}``. The coefficient ``b3`` is the DiD
    estimate. With a single treatment date and a never-treated control group
    this is numerically identical to the group-means calculation

        (Ybar_treat,post - Ybar_treat,pre) - (Ybar_ctrl,post - Ybar_ctrl,pre)

    but the regression form also delivers standard errors and a confidence
    interval.

    Parameters
    ----------
    data
        Long-format panel, one row per unit-period.
    outcome, unit, time, treated
        Column names. ``treated`` must be a time-invariant 0/1 group flag
        (1 for ever-treated units), *not* the treated-and-post indicator.
    treat_period
        First treated period. Periods >= this value are "post".
    cluster
        Cluster standard errors on ``unit`` (default, and recommended).
    alpha
        1 - confidence level. 0.05 gives a 95% CI.

    Returns
    -------
    DiD2x2Result
    """
    required = {outcome, unit, time, treated}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"data is missing required column(s): {sorted(missing)}")

    d = data[[outcome, unit, time, treated]].dropna().copy()

    treat_flag = d[treated].astype(float).to_numpy()
    unique_flags = set(np.unique(treat_flag))
    if not unique_flags <= {0.0, 1.0}:
        raise ValueError(
            f"column '{treated}' must be a 0/1 group flag; found values "
            f"{sorted(unique_flags)[:5]}"
        )
    if unique_flags != {0.0, 1.0}:
        raise ValueError(
            f"column '{treated}' must contain both treated and control units; "
            f"found only {sorted(unique_flags)}"
        )

    post = (d[time].to_numpy() >= treat_period).astype(float)
    if post.all() or not post.any():
        raise ValueError(
            f"treat_period={treat_period} leaves no pre- or no post-period; "
            f"observed {time} range is [{d[time].min()}, {d[time].max()}]"
        )

    X = pd.DataFrame(
        {
            "const": 1.0,
            "treated": treat_flag,
            "post": post,
            "treated_post": treat_flag * post,
        },
        index=d.index,
    )
    y = d[outcome].astype(float).to_numpy()

    model = sm.OLS(y, X)
    if cluster:
        res = model.fit(
            cov_type="cluster", cov_kwds={"groups": d[unit].to_numpy()}
        )
    else:
        res = model.fit()

    ci = res.conf_int(alpha=alpha)
    ci_lower, ci_upper = float(ci.loc["treated_post", 0]), float(ci.loc["treated_post", 1])

    units = d[[unit, treated]].drop_duplicates(subset=unit)

    return DiD2x2Result(
        att=float(res.params["treated_post"]),
        se=float(res.bse["treated_post"]),
        tstat=float(res.tvalues["treated_post"]),
        pvalue=float(res.pvalues["treated_post"]),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        n_obs=int(d.shape[0]),
        n_units=int(units.shape[0]),
        n_treated_units=int((units[treated].astype(float) == 1).sum()),
        cluster=cluster,
    )


# --------------------------------------------------------------------------- #
# Pre-trends / parallel-trends diagnostic
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PretrendsResult:
    """Result of a pre-treatment parallel-trends check."""

    leads: pd.DataFrame
    """Per-period treated-vs-control gaps in the pre-period.

    Columns: ``period``, ``relative_period`` (period - treat_period),
    ``estimate``, ``se``, ``ci_lower``, ``ci_upper``, ``is_base``. The base
    period is normalised to exactly 0 by construction and carried in the table
    so it can be plotted.
    """

    joint_stat: float
    """F statistic for the joint null that every pre-treatment lead is zero."""

    joint_pvalue: float
    joint_df: tuple[float, float] | None
    base_period: float
    alpha: float
    n_leads: int

    @property
    def passes(self) -> bool:
        """True when we FAIL to reject the joint null of no differential pre-trends.

        Note the logic: this is a test we hope to *fail to reject*. Passing is
        an absence of evidence against parallel trends, not proof of it — the
        honest framing the brief insists on (PROJECT_BRIEF.md §5.1).
        """
        return bool(self.joint_pvalue > self.alpha)

    def summary(self) -> str:
        """Human-readable summary of the joint test."""
        verdict = (
            "PASS - no evidence of differential pre-trends"
            if self.passes
            else "FAIL - pre-trends differ; DiD design is contaminated"
        )
        return "\n".join(
            [
                "Pre-trends (parallel-trends) test",
                f"  Base period      : {self.base_period:g}",
                f"  Leads tested     : {self.n_leads}",
                f"  Joint F          : {self.joint_stat:.4f}",
                f"  Joint p-value    : {self.joint_pvalue:.4f}",
                f"  Verdict (a={self.alpha:g}) : {verdict}",
            ]
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def pretrends_test(
    data: pd.DataFrame,
    *,
    outcome: str = "y",
    unit: str = "city",
    time: str = "period",
    treated: str = "treated",
    treat_period: float,
    base_period: float | None = None,
    cluster: bool = True,
    alpha: float = 0.05,
) -> PretrendsResult:
    """Test for differential pre-treatment trends between treated and control.

    Restricted to pre-treatment observations only, so the test cannot be
    contaminated by the treatment effect itself. Estimates a treated-vs-control
    gap for each pre-period relative to ``base_period``

        y_it = a + b*treated_i + sum_p [ g_p * 1{t=p} + lead_p * treated_i*1{t=p} ]

    (base period omitted), then jointly tests H0: all ``lead_p`` = 0.

    Under parallel trends the leads are zero: the two groups' pre-treatment
    paths are the same up to a level shift. A significant joint test means the
    groups were already diverging before treatment, which invalidates the DiD.

    Parameters
    ----------
    base_period
        Period the gaps are measured against. Defaults to the last pre-period
        (``treat_period - 1``), the standard normalisation.

    Returns
    -------
    PretrendsResult
    """
    required = {outcome, unit, time, treated}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"data is missing required column(s): {sorted(missing)}")

    pre = data.loc[data[time] < treat_period, [outcome, unit, time, treated]].dropna().copy()
    if pre.empty:
        raise ValueError(f"no pre-treatment observations before treat_period={treat_period}")

    periods = np.sort(pre[time].unique())
    if base_period is None:
        base_period = float(periods.max())
    if base_period not in set(periods.tolist()):
        raise ValueError(
            f"base_period={base_period} is not a pre-treatment period; "
            f"available: {periods.tolist()}"
        )

    lead_periods = [p for p in periods if p != base_period]
    if not lead_periods:
        raise ValueError(
            "need at least two pre-treatment periods to test for differential "
            f"trends; found only {periods.tolist()}"
        )

    treat_flag = pre[treated].astype(float).to_numpy()
    time_vals = pre[time].to_numpy()

    cols: dict[str, np.ndarray | float] = {"const": 1.0, "treated": treat_flag}
    lead_names: list[str] = []
    for p in lead_periods:
        period_dummy = (time_vals == p).astype(float)
        cols[f"t_{p:g}"] = period_dummy               # common time effect
        name = f"lead_{p:g}"
        cols[name] = period_dummy * treat_flag        # differential gap at p
        lead_names.append(name)

    X = pd.DataFrame(cols, index=pre.index)
    y = pre[outcome].astype(float).to_numpy()

    model = sm.OLS(y, X)
    if cluster:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": pre[unit].to_numpy()})
    else:
        res = model.fit()

    # Joint Wald test: every pre-treatment lead is zero.
    restriction = ", ".join(f"{n} = 0" for n in lead_names)
    wald = res.wald_test(restriction, use_f=True, scalar=True)
    joint_stat = float(np.squeeze(wald.statistic))
    joint_pvalue = float(np.squeeze(wald.pvalue))
    joint_df = (
        (float(wald.df_num), float(wald.df_denom))
        if hasattr(wald, "df_num") and hasattr(wald, "df_denom")
        else None
    )

    ci = res.conf_int(alpha=alpha)
    rows = []
    for p in periods:
        if p == base_period:
            rows.append(
                {
                    "period": float(p),
                    "relative_period": float(p) - float(treat_period),
                    "estimate": 0.0,
                    "se": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "is_base": True,
                }
            )
        else:
            name = f"lead_{p:g}"
            rows.append(
                {
                    "period": float(p),
                    "relative_period": float(p) - float(treat_period),
                    "estimate": float(res.params[name]),
                    "se": float(res.bse[name]),
                    "ci_lower": float(ci.loc[name, 0]),
                    "ci_upper": float(ci.loc[name, 1]),
                    "is_base": False,
                }
            )

    leads = pd.DataFrame(rows).sort_values("period", ignore_index=True)

    return PretrendsResult(
        leads=leads,
        joint_stat=joint_stat,
        joint_pvalue=joint_pvalue,
        joint_df=joint_df,
        base_period=float(base_period),
        alpha=alpha,
        n_leads=len(lead_names),
    )


__all__ = ["DiD2x2Result", "did_2x2", "PretrendsResult", "pretrends_test"]
