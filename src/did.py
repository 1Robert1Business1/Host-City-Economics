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

import re
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


# --------------------------------------------------------------------------- #
# Naive two-way fixed effects (the biased baseline)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TWFEResult:
    """Result of a static two-way fixed-effects DiD regression."""

    att: float
    se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_obs: int
    cluster: bool

    def contains(self, value: float) -> bool:
        """Whether the confidence interval covers ``value``."""
        return bool(self.ci_lower <= value <= self.ci_upper)

    def summary(self, truth: float | None = None) -> str:
        pct = int(round((1 - self.alpha) * 100))
        lines = [
            "Naive two-way fixed effects (static)",
            f"  Estimate         : {self.att:+.4f}",
            f"  Std. error       : {self.se:.4f}"
            f" ({'clustered by unit' if self.cluster else 'unclustered'})",
            f"  {pct}% CI           : [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
            f"  N                : {self.n_obs} obs",
        ]
        if truth is not None:
            lines += [
                f"  True ATT         : {truth:+.4f}",
                f"  Bias             : {self.att - truth:+.4f}"
                f"  ({(self.att - truth) / truth * 100:+.1f}%)",
                f"  CI covers truth  : {'YES' if self.contains(truth) else 'NO'}",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


def twfe_did(
    data: pd.DataFrame,
    *,
    outcome: str = "y",
    unit: str = "city",
    time: str = "period",
    treatment: str = "d",
    cluster: bool = True,
    alpha: float = 0.05,
) -> TWFEResult:
    """Estimate the static TWFE DiD: ``y ~ D | unit + time`` (via pyfixest).

    This is the estimator the modern staggered-DiD literature warns about
    (PROJECT_BRIEF.md §5.3). Under staggered timing with heterogeneous effects
    it implicitly uses already-treated units as controls for later-treated ones
    — "forbidden comparisons" — and is biased. It is included precisely so the
    bias can be *shown* rather than asserted.

    ``treatment`` must be the treated-and-post indicator (1 while a unit is
    under treatment), not the ever-treated group flag.
    """
    import pyfixest as pf

    required = {outcome, unit, time, treatment}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"data is missing required column(s): {sorted(missing)}")

    fml = f"{outcome} ~ {treatment} | {unit} + {time}"
    vcov = {"CRV1": unit} if cluster else "iid"
    fit = pf.feols(fml, data=data, vcov=vcov)

    tidy = fit.tidy()
    row = tidy.loc[treatment]
    lo_col, hi_col = f"{alpha / 2 * 100:g}%", f"{(1 - alpha / 2) * 100:g}%"
    if lo_col in tidy.columns and hi_col in tidy.columns:
        ci_lower, ci_upper = float(row[lo_col]), float(row[hi_col])
    else:  # non-default alpha: fall back to the fitted CI at that level
        ci = fit.confint(alpha=alpha)
        ci_lower, ci_upper = float(ci.loc[treatment].iloc[0]), float(ci.loc[treatment].iloc[1])

    return TWFEResult(
        att=float(row["Estimate"]),
        se=float(row["Std. Error"]),
        pvalue=float(row["Pr(>|t|)"]),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        n_obs=int(data.shape[0]),
        cluster=cluster,
    )


# --------------------------------------------------------------------------- #
# Callaway & Sant'Anna
# --------------------------------------------------------------------------- #
def _flatten_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the `differences` MultiIndex columns to their leaf names."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[-1] if isinstance(c, tuple) else c for c in out.columns]
    return out


@dataclass(frozen=True)
class CSResult:
    """Result of a Callaway & Sant'Anna (2021) estimation."""

    att: float
    """Overall ATT (the 'simple' aggregation)."""

    se: float
    ci_lower: float
    ci_upper: float
    alpha: float
    group_time: pd.DataFrame
    """Group-time ATT(g,t) table."""

    event_study: pd.DataFrame
    """Event-study aggregation: columns ``relative_period``, ``ATT``,
    ``std_error``, ``lower``, ``upper``."""

    def contains(self, value: float) -> bool:
        """Whether the overall-ATT confidence interval covers ``value``."""
        return bool(self.ci_lower <= value <= self.ci_upper)

    def summary(self, truth: float | None = None) -> str:
        pct = int(round((1 - self.alpha) * 100))
        lines = [
            "Callaway & Sant'Anna (staggered-robust)",
            f"  Overall ATT      : {self.att:+.4f}",
            f"  Std. error       : {self.se:.4f}",
            f"  {pct}% CI           : [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
        ]
        if truth is not None:
            lines += [
                f"  True ATT         : {truth:+.4f}",
                f"  Error            : {self.att - truth:+.4f}"
                f"  ({(self.att - truth) / truth * 100:+.1f}%)",
                f"  CI covers truth  : {'YES' if self.contains(truth) else 'NO'}",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


def callaway_santanna(
    data: pd.DataFrame,
    *,
    outcome: str = "y",
    unit: str = "city",
    time: str = "period",
    cohort: str = "cohort",
    control_group: str = "never_treated",
    est_method: str = "dr",
    base_period: str = "varying",
    alpha: float = 0.05,
    boot_iterations: int = 0,
    random_state: int | None = None,
) -> CSResult:
    """Estimate group-time ATTs with Callaway & Sant'Anna via `differences`.

    The estimator that fixes what TWFE breaks (PROJECT_BRIEF.md §5.3). It
    estimates a separate ATT(g,t) for every cohort-period cell using only
    *clean* comparisons — never-treated (or not-yet-treated) units — and then
    aggregates. No already-treated unit is ever used as a control, so the
    forbidden comparisons that bias TWFE cannot arise.

    Parameters
    ----------
    cohort
        Column holding each unit's treatment period, ``NaN`` for never-treated.
    control_group
        ``'never_treated'`` or ``'not_yet_treated'``.
    est_method
        ``differences`` estimation method; ``'dr'`` is doubly-robust.
    boot_iterations
        If > 0, use the multiplier bootstrap for uniform confidence bands.
    """
    from differences import ATTgt

    required = {outcome, unit, time, cohort}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"data is missing required column(s): {sorted(missing)}")

    panel = data.set_index([unit, time])

    att = ATTgt(data=panel, cohort_column=cohort, base_period=base_period)
    att.fit(
        f"{outcome} ~ 1",
        est_method=est_method,
        control_group=control_group,
        alpha=alpha,
        boot_iterations=boot_iterations,
        random_state=random_state,
        progress_bar=False,
    )

    simple = _flatten_cs(att.aggregate("simple", alpha=alpha))
    event = _flatten_cs(att.aggregate("event", alpha=alpha)).reset_index()
    group_time = _flatten_cs(att.results())

    row = simple.iloc[0]
    return CSResult(
        att=float(row["ATT"]),
        se=float(row["std_error"]),
        ci_lower=float(row["lower"]),
        ci_upper=float(row["upper"]),
        alpha=alpha,
        group_time=group_time,
        event_study=event,
    )


# --------------------------------------------------------------------------- #
# Goodman-Bacon decomposition — the mechanism behind the TWFE bias
# --------------------------------------------------------------------------- #
def _means_2x2(
    frame: pd.DataFrame, outcome: str, treated_mask: np.ndarray, post_mask: np.ndarray
) -> float:
    """Group-means 2x2 DiD on a subsample."""
    y = frame[outcome].to_numpy()
    tp = y[treated_mask & post_mask].mean()
    tq = y[treated_mask & ~post_mask].mean()
    cp = y[~treated_mask & post_mask].mean()
    cq = y[~treated_mask & ~post_mask].mean()
    return float((tp - tq) - (cp - cq))


def goodman_bacon(
    data: pd.DataFrame,
    *,
    outcome: str = "y",
    unit: str = "city",
    time: str = "period",
    cohort: str = "cohort",
) -> pd.DataFrame:
    """Decompose the static TWFE estimate into its underlying 2x2 comparisons.

    Goodman-Bacon (2021) shows the TWFE coefficient is a weighted average of
    every possible 2x2 DiD in the data. Three kinds arise under staggered
    timing:

      * **treated vs never-treated** — a clean comparison;
      * **earlier vs later** (the later cohort not yet treated) — also clean;
      * **later vs earlier** (the earlier cohort *already treated*, serving as
        the control) — the **forbidden comparison**. When the already-treated
        control's own effect is still growing, that growth is subtracted from
        the later cohort's change and the 2x2 is biased downward, dragging the
        TWFE average with it.

    Requires a balanced panel and no covariates, in which case the weighted sum
    of the comparisons reproduces the TWFE coefficient exactly — which is used
    as a built-in correctness check in the notebook.

    Returns
    -------
    pandas.DataFrame
        One row per comparison with columns ``kind``, ``treated_group``,
        ``control_group``, ``estimate``, ``weight``, ``forbidden``.
    """
    d = data[[outcome, unit, time, cohort]].copy()
    periods = np.sort(d[time].unique())
    n_periods = len(periods)

    units = d[[unit, cohort]].drop_duplicates(subset=unit)
    n_total = len(units)

    treated_cohorts = np.sort(units[cohort].dropna().unique())
    has_never = units[cohort].isna().any()

    def share(g: float | None) -> float:
        if g is None:
            return float(units[cohort].isna().sum()) / n_total
        return float((units[cohort] == g).sum()) / n_total

    def dbar(g: float | None) -> float:
        """Fraction of periods the group spends treated."""
        if g is None:
            return 0.0
        return float((periods >= g).sum()) / n_periods

    rows: list[dict] = []

    # (1) each treated cohort vs the never-treated group
    if has_never:
        n_u = share(None)
        never_mask_units = set(units.loc[units[cohort].isna(), unit])
        for g in treated_cohorts:
            n_k = share(g)
            sub = d[(d[cohort] == g) | (d[cohort].isna())]
            treated_mask = (sub[cohort] == g).to_numpy()
            post_mask = (sub[time] >= g).to_numpy()
            est = _means_2x2(sub, outcome, treated_mask, post_mask)
            nhat = n_k / (n_k + n_u)
            dk = dbar(g)
            num = (n_k + n_u) ** 2 * nhat * (1 - nhat) * dk * (1 - dk)
            rows.append(
                {
                    "kind": "treated vs never-treated",
                    "treated_group": g,
                    "control_group": np.nan,
                    "estimate": est,
                    "_num": num,
                    "forbidden": False,
                }
            )
        del never_mask_units

    # (2) timing pairs
    for i, g_k in enumerate(treated_cohorts):
        for g_l in treated_cohorts[i + 1 :]:
            n_k, n_l = share(g_k), share(g_l)
            dk, dl = dbar(g_k), dbar(g_l)
            nhat = n_k / (n_k + n_l)
            pair = d[(d[cohort] == g_k) | (d[cohort] == g_l)]

            # (2a) earlier vs later, restricted to before the later adopts
            win = pair[pair[time] < g_l]
            est_early = _means_2x2(
                win, outcome,
                (win[cohort] == g_k).to_numpy(),
                (win[time] >= g_k).to_numpy(),
            )
            num_early = (
                ((n_k + n_l) * (1 - dl)) ** 2
                * nhat * (1 - nhat)
                * ((dk - dl) / (1 - dl))
                * ((1 - dk) / (1 - dl))
            )
            rows.append(
                {
                    "kind": "earlier vs later (not yet treated)",
                    "treated_group": g_k,
                    "control_group": g_l,
                    "estimate": est_early,
                    "_num": num_early,
                    "forbidden": False,
                }
            )

            # (2b) later vs earlier — the FORBIDDEN comparison
            win = pair[pair[time] >= g_k]
            est_late = _means_2x2(
                win, outcome,
                (win[cohort] == g_l).to_numpy(),
                (win[time] >= g_l).to_numpy(),
            )
            num_late = (
                ((n_k + n_l) * dk) ** 2
                * nhat * (1 - nhat)
                * (dl / dk)
                * ((dk - dl) / dk)
            )
            rows.append(
                {
                    "kind": "later vs earlier (ALREADY treated)",
                    "treated_group": g_l,
                    "control_group": g_k,
                    "estimate": est_late,
                    "_num": num_late,
                    "forbidden": True,
                }
            )

    out = pd.DataFrame(rows)
    out["weight"] = out["_num"] / out["_num"].sum()
    out = out.drop(columns="_num")
    return out[
        ["kind", "treated_group", "control_group", "estimate", "weight", "forbidden"]
    ]


# --------------------------------------------------------------------------- #
# Stacked event study (matched design) with a joint pre-trends Wald test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StackedEventStudy:
    """Result of a stacked (Cengiz et al. 2019) event-study estimation."""

    coefs: pd.DataFrame
    """Event-time coefficients: ``event_time``, ``estimate``, ``se``,
    ``ci_lower``, ``ci_upper`` (the reference period is included at 0)."""

    pretrends_stat: float
    """Joint Wald statistic that all pre-treatment leads are zero (chi-square)."""

    pretrends_pvalue: float
    pretrends_df: int
    pre_rms: float
    """Root-mean-square of the pre-treatment lead coefficients (effect-size view)."""

    n_stacks: int
    n_obs: int
    k_pre: int
    k_post: int
    ref: int
    alpha: float

    @property
    def passes(self) -> bool:
        """Fail to reject the joint null of no differential pre-trends."""
        return bool(self.pretrends_pvalue > self.alpha)

    def summary(self) -> str:
        verdict = (
            "PASS - no evidence of differential pre-trends"
            if self.passes
            else "FAIL - pre-trends differ; design contaminated"
        )
        return "\n".join(
            [
                "Stacked event study (matched controls)",
                f"  sub-experiments  : {self.n_stacks}",
                f"  window           : [{-self.k_pre}, +{self.k_post}] (ref {self.ref})",
                f"  pre-trend RMS    : {self.pre_rms:.4f} log points",
                f"  joint Wald chi2  : {self.pretrends_stat:.3f} (df {self.pretrends_df})",
                f"  joint p-value    : {self.pretrends_pvalue:.4f}",
                f"  verdict (a={self.alpha:g}) : {verdict}",
            ]
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


def stacked_event_study(
    panel: pd.DataFrame,
    matches: dict[str, list[str]],
    *,
    outcome: str = "log_lh",
    unit: str = "metro",
    time: str = "period",
    cohort: str = "cohort",
    k_pre: int = 8,
    k_post: int = 6,
    ref: int = -1,
    cluster: bool = True,
    alpha: float = 0.05,
) -> StackedEventStudy:
    """Stacked event study for a matched treated-vs-control design.

    For each treated unit (a key of ``matches``) a "clean" 2x2-style
    sub-experiment is built from that unit and *its own* matched controls over
    the event window ``[cohort - k_pre, cohort + k_post]``; the sub-experiments
    are stacked and estimated jointly with sub-experiment-specific unit and
    time fixed effects (Cengiz, Dube, Lindner & Zipperer 2019). This realises
    the matched-control strategy exactly — each host is compared only with its
    matches — and avoids the "forbidden comparisons" TWFE makes under staggered
    timing.

    The event-time coefficients are relative to ``ref`` (default the quarter
    before treatment). The pre-treatment leads (event time < ``ref``) are jointly
    tested against zero: this is the parallel-trends check.

    Standard errors are clustered on ``unit`` by default. With few treated
    clusters the cluster-robust variance is anti-conservative (the few-clusters
    problem, PROJECT_BRIEF.md §5.4) -- flagged here and addressed with a wild
    cluster bootstrap in Stage 4.
    """
    import pyfixest as pf
    from scipy import stats

    cohort_of = panel.dropna(subset=[cohort]).groupby(unit)[cohort].first().to_dict()

    parts = []
    for host, controls in matches.items():
        g = cohort_of.get(host)
        if g is None:
            raise ValueError(f"{host} has no cohort in the panel")
        members = [host, *controls]
        sub = panel[
            panel[unit].isin(members)
            & panel[time].between(g - k_pre, g + k_post)
        ].copy()
        sub["event_time"] = (sub[time] - g).astype(int)
        sub["treated_num"] = (sub[unit] == host).astype(int)
        sub["stack_id"] = host
        parts.append(sub)

    stack = pd.concat(parts, ignore_index=True)
    stack["unit_stack"] = stack[unit] + "__" + stack["stack_id"]
    stack["time_stack"] = stack[time].astype(str) + "__" + stack["stack_id"]

    vcov = {"CRV1": unit} if cluster else "iid"
    fit = pf.feols(
        f"{outcome} ~ i(event_time, treated_num, ref={ref}) "
        f"| unit_stack + time_stack",
        data=stack,
        vcov=vcov,
    )

    tidy = fit.tidy()
    names = list(tidy.index)
    et = {}
    for nm in names:
        m = re.search(r"event_time::(-?\d+)", str(nm))
        if m:
            et[nm] = int(m.group(1))

    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for nm, e in et.items():
        r = tidy.loc[nm]
        est, se = float(r["Estimate"]), float(r["Std. Error"])
        rows.append(
            {
                "event_time": e,
                "estimate": est,
                "se": se,
                "ci_lower": est - z * se,
                "ci_upper": est + z * se,
            }
        )
    rows.append({"event_time": ref, "estimate": 0.0, "se": 0.0,
                 "ci_lower": 0.0, "ci_upper": 0.0})
    coefs = pd.DataFrame(rows).sort_values("event_time", ignore_index=True)

    # Joint pre-trends Wald test: b' V^{-1} b over the lead coefficients.
    lead_names = [nm for nm, e in et.items() if e < ref]
    b = fit.coef().loc[lead_names].to_numpy()
    V = fit._vcov
    idx = [fit._coefnames.index(nm) for nm in lead_names]
    Vsub = V[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(Vsub, b))
    df = len(lead_names)
    pval = float(stats.chi2.sf(stat, df))
    pre_rms = float(np.sqrt(np.mean(b**2)))

    return StackedEventStudy(
        coefs=coefs,
        pretrends_stat=stat,
        pretrends_pvalue=pval,
        pretrends_df=df,
        pre_rms=pre_rms,
        n_stacks=len(matches),
        n_obs=int(stack.shape[0]),
        k_pre=k_pre,
        k_post=k_post,
        ref=ref,
        alpha=alpha,
    )


__all__ = [
    "DiD2x2Result",
    "did_2x2",
    "PretrendsResult",
    "pretrends_test",
    "TWFEResult",
    "twfe_did",
    "CSResult",
    "callaway_santanna",
    "goodman_bacon",
    "StackedEventStudy",
    "stacked_event_study",
]
