"""QCEW download, county->metro aggregation, and treated/control construction.

Builds the real analysis panel for the Super Bowl anchor (PROJECT_BRIEF.md §4;
anchor + control decision made in Stage 3): quarterly metro-level leisure &
hospitality employment from the BLS Quarterly Census of Employment and Wages,
for Super Bowl host metros (2015-2025) and a comparable non-host control pool.

Data source
-----------
QCEW Open Data Access "slice" API:

    https://data.bls.gov/cew/data/api/{year}/{qtr}/industry/{industry}.csv

Empirically (checked Stage 3) the slice API serves 2014-2025, so the whole
Super Bowl window is covered without the historical bulk files that a 1994
anchor would have required.

A note on the brief's MSA wrinkle
---------------------------------
The brief expected the MSA industry breakdown to have been discontinued in
Q2 2025, forcing a manual county->metro aggregation. Checked against the live
API, that is **not** currently the case: the MSA-level L&H series (aggregation
level 43) is still published through at least 2025 Q3. So this module uses
QCEW's official MSA aggregates as the metro series -- and `aggregate_counties_
to_metro` + `validate_metro_aggregation` demonstrate and *verify* that a manual
county->metro sum reproduces those aggregates exactly (ratio 1.000), so the
series is interchangeable and the county method is on hand if the MSA breakdown
is ever dropped.

Outcome
-------
Metro leisure & hospitality employment: QCEW industry supersector ``1026``,
private ownership (``own_code`` 5), MSA aggregation level (``agglvl_code`` 43),
quarterly employment = mean of the three monthly employment levels.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DATA_DIR, SEED

# --------------------------------------------------------------------------- #
# API / classification constants
# --------------------------------------------------------------------------- #
SLICE_API = "https://data.bls.gov/cew/data/api/{year}/{qtr}/industry/{industry}.csv"
_USER_AGENT = "host-city-economics portfolio research; github.com/1Robert1Business1/host-city-economics"

INDUSTRY_LH = "1026"       # Leisure & hospitality supersector
OWN_PRIVATE = "5"          # Private ownership
AGGLVL_LH_MSA = "43"       # MSA, by supersector
AGGLVL_LH_COUNTY = "73"    # County, by supersector

INDUSTRY_TOTAL = "10"      # Total, all industries
OWN_TOTAL = "0"            # Total covered
AGGLVL_TOTAL_MSA = "40"    # MSA, total all industries

PANEL_START_YEAR = 2014
PANEL_END_YEAR = 2025      # 2025 Q4 is the latest available (2026 Q1 not yet released)

RAW_DIR = DATA_DIR / "raw"  # git-ignored cache of downloaded slices


# --------------------------------------------------------------------------- #
# The study metros
# --------------------------------------------------------------------------- #
# Super Bowl host metros, 2015-2025. Value: (CBSA C-code, [host years]).
# The Super Bowl is played in early February, so treatment lands in Q1 of the
# host year. Phoenix hosts twice in-window (2015, 2023); under the absorbing-
# treatment convention its cohort is the first, 2015, and the 2023 game is a
# second dose while already treated (noted as a caveat downstream).
HOST_METROS: dict[str, tuple[str, list[int]]] = {
    "Phoenix": ("C3806", [2015, 2023]),
    "San Jose": ("C4194", [2016]),      # Levi's Stadium, Santa Clara
    "Houston": ("C2642", [2017]),
    "Minneapolis": ("C3346", [2018]),
    "Atlanta": ("C1206", [2019]),
    "Miami": ("C3310", [2020]),
    "Tampa": ("C4530", [2021]),
    "Los Angeles": ("C3108", [2022]),   # SoFi Stadium, Inglewood
    "Las Vegas": ("C2982", [2024]),
    "New Orleans": ("C3538", [2025]),
}

# Comparable non-host metros: large, host-caliber (major-league / top population)
# metros that did NOT host a Super Bowl in 2014-2025. Dallas ('11), Detroit
# ('06), Indianapolis-adjacent etc. hosted only before the window, so they are
# clean never-treated here. New York (hosted 2014, at the panel boundary) is
# deliberately excluded from both groups.
CONTROL_METROS: dict[str, str] = {
    "Chicago": "C1698",
    "Dallas": "C1910",
    "Washington": "C4790",
    "Philadelphia": "C3798",
    "Boston": "C1446",
    "Seattle": "C4266",
    "Denver": "C1974",
    "San Diego": "C4174",
    "Detroit": "C1982",
    "Orlando": "C3674",
    "Nashville": "C3498",
    "Charlotte": "C1674",
    "Baltimore": "C1258",
    "St. Louis": "C4118",
    "Portland": "C3890",
    "Pittsburgh": "C3830",
    "San Antonio": "C4170",
    "Austin": "C1242",
}

# Excluded controls: their private L&H MSA aggregate is disclosure-suppressed
# (disclosure_code 'N') in ~30% of quarters, too often to interpolate honestly.
# Documented here rather than silently dropped. Cincinnati (C1714) and
# Kansas City (C2814) were in the candidate pool and removed on this ground.
SUPPRESSED_METROS: dict[str, str] = {
    "Cincinnati": "C1714",
    "Kansas City": "C2814",
}

# Complete constituent counties for host metros with tractable county sets,
# used to VALIDATE that a manual county->metro sum reproduces the QCEW MSA
# aggregate. Multi-county giants (Atlanta ~29, Minneapolis ~16, Houston ~9)
# are taken from the official MSA aggregate directly.
HOST_METRO_COUNTIES: dict[str, list[str]] = {
    "Phoenix": ["04013", "04021"],                       # Maricopa, Pinal
    "Las Vegas": ["32003"],                              # Clark
    "Miami": ["12086", "12011", "12099"],               # Miami-Dade, Broward, Palm Beach
    "San Jose": ["06085", "06069"],                     # Santa Clara, San Benito
    "Los Angeles": ["06037", "06059"],                  # Los Angeles, Orange
    "Tampa": ["12057", "12103", "12101", "12053"],      # Hillsborough, Pinellas, Pasco, Hernando
    "New Orleans": ["22071", "22051", "22087", "22103",  # Orleans, Jefferson, St. Bernard, St. Tammany
                    "22089", "22075", "22095", "22093"],  # St. Charles, Plaquemines, St. John, St. James
}


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def yq_to_period(year: int, qtr: int) -> int:
    """Integer period index; 2014 Q1 = 0, each quarter +1."""
    return (year - PANEL_START_YEAR) * 4 + (qtr - 1)


def period_to_yq(period: int) -> tuple[int, int]:
    """Inverse of `yq_to_period`."""
    return PANEL_START_YEAR + period // 4, period % 4 + 1


def host_cohort_period(metro: str) -> int:
    """First in-window Super Bowl of a host metro, as a period index (Q1)."""
    _ccode, years = HOST_METROS[metro]
    return yq_to_period(min(years), 1)


# --------------------------------------------------------------------------- #
# Download (with on-disk caching of raw slices)
# --------------------------------------------------------------------------- #
def _fetch_slice(
    year: int, qtr: int, industry: str, *, cache_dir=RAW_DIR, pause: float = 0.3
) -> pd.DataFrame:
    """Fetch one industry-slice CSV (all areas), caching the raw file on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"industry_{industry}_{year}_q{qtr}.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)

    url = SLICE_API.format(year=year, qtr=qtr, industry=industry)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", "replace")
            path.write_text(raw, encoding="utf-8")
            time.sleep(pause)  # be polite to the BLS server
            return pd.read_csv(io.StringIO(raw), dtype=str)
        except urllib.error.HTTPError as e:  # noqa: PERF203
            if e.code == 404:
                raise FileNotFoundError(f"QCEW slice not published: {url}") from e
            last_err = e
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err!r}")


def _quarterly_emp(df: pd.DataFrame) -> pd.Series:
    """Quarterly employment = mean of the three monthly employment levels.

    Disclosure-suppressed cells (``disclosure_code`` 'N') carry zeros in the
    employment fields; these are returned as ``NaN`` (genuinely missing), never
    as a real zero.
    """
    months = ["month1_emplvl", "month2_emplvl", "month3_emplvl"]
    vals = df[months].apply(pd.to_numeric, errors="coerce")
    emp = vals.mean(axis=1)
    suppressed = df["disclosure_code"].astype(str).str.upper().eq("N")
    emp = emp.mask(suppressed | (emp <= 0))
    return emp


def download_metro_panel(
    *,
    start_year: int = PANEL_START_YEAR,
    end_year: int = PANEL_END_YEAR,
    cache_dir=RAW_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download and assemble the metro-quarter panel for all study metros.

    Pulls the L&H (industry 1026) and total-employment (industry 10) MSA
    aggregates for every quarter, filters to the study metros, and returns a
    tidy frame with one row per metro-quarter.

    Returns
    -------
    pandas.DataFrame
        Columns: ``metro``, ``ccode``, ``year``, ``qtr``, ``period``,
        ``lh_emp`` (L&H employment), ``total_emp`` (all-industry employment),
        ``lh_share`` (L&H / total), ``is_host``.
    """
    metros = {**{m: c for m, (c, _y) in HOST_METROS.items()}, **CONTROL_METROS}
    ccode_to_metro = {c: m for m, c in metros.items()}
    wanted = set(metros.values())

    rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        for qtr in range(1, 5):
            lh = _fetch_slice(year, qtr, INDUSTRY_LH, cache_dir=cache_dir)
            tot = _fetch_slice(year, qtr, INDUSTRY_TOTAL, cache_dir=cache_dir)

            lh_m = lh[
                (lh.own_code == OWN_PRIVATE)
                & (lh.agglvl_code == AGGLVL_LH_MSA)
                & (lh.area_fips.isin(wanted))
            ].copy()
            lh_m["lh_emp"] = _quarterly_emp(lh_m)

            tot_m = tot[
                (tot.own_code == OWN_TOTAL)
                & (tot.agglvl_code == AGGLVL_TOTAL_MSA)
                & (tot.area_fips.isin(wanted))
            ].copy()
            tot_m["total_emp"] = _quarterly_emp(tot_m)

            lh_map = lh_m.set_index("area_fips")["lh_emp"].to_dict()
            tot_map = tot_m.set_index("area_fips")["total_emp"].to_dict()

            for ccode in wanted:
                rows.append(
                    {
                        "metro": ccode_to_metro[ccode],
                        "ccode": ccode,
                        "year": year,
                        "qtr": qtr,
                        "period": yq_to_period(year, qtr),
                        "lh_emp": lh_map.get(ccode, np.nan),
                        "total_emp": tot_map.get(ccode, np.nan),
                    }
                )
        if verbose:
            print(f"  downloaded {year} (Q1-Q4)")

    panel = pd.DataFrame(rows)
    panel["lh_share"] = panel["lh_emp"] / panel["total_emp"]
    panel["is_host"] = panel["metro"].isin(HOST_METROS).astype(int)
    return panel.sort_values(["metro", "period"], ignore_index=True)


def clean_panel(
    panel: pd.DataFrame, *, min_coverage: float = 0.9, verbose: bool = True
) -> pd.DataFrame:
    """Trim unpublished trailing quarters and interpolate scattered gaps.

    Two distinct kinds of missingness are handled differently and honestly:

      * **Trailing unpublished quarters.** The most recent one or two quarters
        may not yet carry MSA aggregates for any metro (the ~6-month QCEW lag).
        Quarters where fewer than ``min_coverage`` of metros have data are
        dropped from the panel end.
      * **Scattered internal suppression.** A metro-quarter whose L&H aggregate
        is disclosure-suppressed is filled by log-linear interpolation from its
        own neighbouring quarters. Reported, not silent.

    Returns a copy with a filled ``lh_emp`` and a boolean ``lh_imputed`` column
    flagging interpolated cells.
    """
    df = panel.copy()
    n_metros = df["metro"].nunique()

    # 1) drop trailing globally-unpublished quarters
    cov = df.groupby("period")["lh_emp"].apply(lambda s: s.notna().mean())
    good_periods = cov[cov >= min_coverage].index
    last_good = int(good_periods.max())
    dropped = sorted(p for p in df["period"].unique() if p > last_good)
    if dropped and verbose:
        yqs = ", ".join(f"{y}Q{q}" for y, q in map(period_to_yq, dropped))
        print(f"  trimmed {len(dropped)} unpublished trailing quarter(s): {yqs}")
    df = df[df["period"] <= last_good].copy()

    # 2) interpolate scattered internal suppression, per metro, in log space
    df = df.sort_values(["metro", "period"]).reset_index(drop=True)
    df["lh_imputed"] = False
    filled = []
    for metro, g in df.groupby("metro"):
        s = g.set_index("period")["lh_emp"]
        missing = s[s.isna()].index.tolist()
        if not missing:
            continue
        log_s = np.log(s)
        log_filled = log_s.interpolate(method="index", limit_area="inside")
        s_filled = np.exp(log_filled)
        for p in missing:
            if pd.notna(s_filled.get(p)):
                df.loc[(df.metro == metro) & (df.period == p), "lh_emp"] = s_filled[p]
                df.loc[(df.metro == metro) & (df.period == p), "lh_imputed"] = True
                filled.append((metro, *period_to_yq(p)))
    if filled and verbose:
        cells = ", ".join(f"{m} {y}Q{q}" for m, y, q in filled)
        print(f"  interpolated {len(filled)} suppressed cell(s): {cells}")

    # recompute derived columns and flag any still-missing metros
    df["lh_share"] = df["lh_emp"] / df["total_emp"]
    still = df.loc[df["lh_emp"].isna(), "metro"].unique().tolist()
    if still and verbose:
        print(f"  WARNING: metros with unfillable gaps (edge-of-panel): {still}")
    return df.sort_values(["metro", "period"], ignore_index=True)


def suppression_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-metro count of disclosure-suppressed (missing) L&H cells, pre-clean."""
    rep = (
        panel.assign(missing=panel["lh_emp"].isna())
        .groupby("metro")
        .agg(n_quarters=("period", "size"), n_suppressed=("missing", "sum"))
    )
    rep["pct_suppressed"] = (rep["n_suppressed"] / rep["n_quarters"] * 100).round(1)
    return rep.sort_values("n_suppressed", ascending=False)


# --------------------------------------------------------------------------- #
# County -> metro aggregation (the documented step, validated)
# --------------------------------------------------------------------------- #
def aggregate_counties_to_metro(
    counties: list[str], year: int, qtr: int, *, cache_dir=RAW_DIR
) -> float:
    """Sum county-level private L&H employment to a metro total for one quarter.

    The manual county->metro step the brief calls for. Used by
    `validate_metro_aggregation` to confirm it reproduces the QCEW MSA
    aggregate.
    """
    lh = _fetch_slice(year, qtr, INDUSTRY_LH, cache_dir=cache_dir)
    cty = lh[
        (lh.own_code == OWN_PRIVATE)
        & (lh.agglvl_code == AGGLVL_LH_COUNTY)
        & (lh.area_fips.isin(counties))
    ].copy()
    return float(_quarterly_emp(cty).sum())


def validate_metro_aggregation(
    year: int = 2019, qtr: int = 1, *, cache_dir=RAW_DIR, tol: float = 1e-6
) -> pd.DataFrame:
    """Check county-sum == QCEW MSA aggregate for host metros with county lists.

    Returns a frame with the county-sum, the official MSA aggregate, and their
    ratio for each validated metro. A correct crosswalk gives ratio 1.000.
    """
    lh = _fetch_slice(year, qtr, INDUSTRY_LH, cache_dir=cache_dir)
    msa = lh[(lh.own_code == OWN_PRIVATE) & (lh.agglvl_code == AGGLVL_LH_MSA)].copy()
    msa["lh_emp"] = _quarterly_emp(msa)
    msa_map = msa.set_index("area_fips")["lh_emp"].to_dict()

    rows = []
    for metro, counties in HOST_METRO_COUNTIES.items():
        ccode = HOST_METROS[metro][0]
        csum = aggregate_counties_to_metro(counties, year, qtr, cache_dir=cache_dir)
        mval = msa_map.get(ccode, np.nan)
        rows.append(
            {
                "metro": metro,
                "n_counties": len(counties),
                "county_sum": csum,
                "qcew_msa": mval,
                "ratio": csum / mval if mval else np.nan,
                "ok": bool(abs(csum - mval) <= tol * max(mval, 1.0)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Treated / control construction + matching
# --------------------------------------------------------------------------- #
def add_treatment_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``cohort`` (host's first Super Bowl period; NaN for controls),
    ``treated`` (ever-host flag) and ``d`` (treated-and-post indicator)."""
    out = panel.copy()
    cohort = {m: float(host_cohort_period(m)) for m in HOST_METROS}
    out["cohort"] = out["metro"].map(cohort)  # NaN for controls
    out["treated"] = out["metro"].isin(HOST_METROS).astype(int)
    out["d"] = ((out["cohort"].notna()) & (out["period"] >= out["cohort"])).astype(int)
    return out


@dataclass(frozen=True)
class MatchResult:
    """Matched controls for each host, plus the covariates matched on."""

    matches: dict[str, list[str]]
    covariates: pd.DataFrame
    match_vars: tuple[str, ...]
    k: int


def metro_covariates(
    panel: pd.DataFrame, *, pre_end_period: int, window: int = 8
) -> pd.DataFrame:
    """Pre-treatment structural covariates per metro, for matching.

    Deliberately matches on **structural** comparability -- size (log L&H and
    log total employment) and tourism intensity (L&H share) -- averaged over the
    ``window`` quarters ending at ``pre_end_period``. It does NOT match on the
    pre-trend slope: matching on the very thing the parallel-trends test
    examines would manufacture a pass. Pre-trends are tested separately.
    """
    lo = pre_end_period - window + 1
    pre = panel[(panel.period >= lo) & (panel.period <= pre_end_period)]
    g = pre.groupby("metro")
    cov = pd.DataFrame(
        {
            "log_lh": np.log(g["lh_emp"].mean()),
            "log_total": np.log(g["total_emp"].mean()),
            "lh_share": g["lh_share"].mean(),
        }
    )
    cov["is_host"] = cov.index.isin(HOST_METROS).astype(int)
    return cov


def match_controls(
    panel: pd.DataFrame,
    *,
    pre_end_period: int,
    k: int = 4,
    window: int = 8,
    match_vars: tuple[str, ...] = ("log_lh", "log_total", "lh_share"),
    seed: int = SEED,
) -> MatchResult:
    """k-nearest-neighbour match of each host to control metros.

    Distance is Euclidean in standardised covariate space (each covariate
    z-scored across metros). The committed `SEED` only breaks exact distance
    ties, so the matching is deterministic.
    """
    cov = metro_covariates(panel, pre_end_period=pre_end_period, window=window)
    z = cov[list(match_vars)].copy()
    z = (z - z.mean()) / z.std(ddof=0)

    controls = cov.index[cov.is_host == 0].tolist()
    rng = np.random.default_rng(seed)

    matches: dict[str, list[str]] = {}
    for host in [m for m in cov.index if cov.loc[m, "is_host"] == 1]:
        d = np.sqrt(((z.loc[controls] - z.loc[host]) ** 2).sum(axis=1))
        jitter = pd.Series(rng.normal(0, 1e-9, len(controls)), index=controls)
        order = (d + jitter).sort_values().index.tolist()
        matches[host] = order[:k]
    return MatchResult(matches=matches, covariates=cov, match_vars=match_vars, k=k)


__all__ = [
    "HOST_METROS",
    "CONTROL_METROS",
    "SUPPRESSED_METROS",
    "HOST_METRO_COUNTIES",
    "yq_to_period",
    "period_to_yq",
    "host_cohort_period",
    "download_metro_panel",
    "clean_panel",
    "suppression_report",
    "aggregate_counties_to_metro",
    "validate_metro_aggregation",
    "add_treatment_columns",
    "metro_covariates",
    "match_controls",
    "MatchResult",
]
