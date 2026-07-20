# Host-City Economics — Project Brief

A difference-in-differences study of whether hosting football's biggest tournament actually moves a city's economy — and an honest reckoning with how much of the apparent effect is real.

This document is the specification for the project: the question, the design, the data, the methods, and what "done" looks like. It sits alongside the code so that a recruiter skimming for thirty seconds and an econometrician reading every line both understand the work and the reasoning behind it.

## 1. The question

When a city hosts World Cup matches, boosters promise a windfall — fans, hotels full, restaurants packed, jobs created. The 2026 tournament, hosted across the United States, Mexico, and Canada and running June–July 2026, is being sold on exactly that promise right now. This project asks the causal question underneath the promise: does hosting matches actually raise local economic activity relative to what would have happened anyway — and if there is an effect, how much of it is genuine new activity versus money that simply moved around?

That second clause is the whole point. It is easy to show that spending rises in a host city during a tournament. It is hard, and far more honest, to show whether that rise is real economic gain or an accounting illusion created by locals reallocating spending and regular visitors staying away.

## 2. Why this is a causal-inference problem, not a chart

A before-and-after comparison — "spending was higher during the tournament" — is not evidence of an effect. Everything changes in summer; the tournament is tangled up with seasonality, the economic cycle, and a dozen other things. To isolate the causal effect you need a counterfactual: what would the host city have done without the tournament? Difference-in-differences (DiD) builds that counterfactual from comparable non-host cities, on the logic that if host and non-host cities moved in parallel before the tournament, the non-hosts show you where the hosts would have gone without it. The gap that opens up after is the causal estimate.

This is the standard tool for exactly this kind of natural experiment, and using it correctly — with its assumptions stated, tested, and stress-tested — is the skill the project demonstrates.

## 3. The honest-rigour spine — this is what makes the project credible

There is a large, settled economics literature on mega-event impact, and its central finding is uncomfortable for boosters: ex-post academic studies routinely find near-zero or fractional effects compared to the rosy ex-ante projections (Baade & Matheson, *Going for the Gold: The Economics of the Olympics*, Journal of Economic Perspectives 2016). Baade and Matheson's ex-post analysis found the 1994 US World Cup produced losses for host cities, not gains. The reasons are specific and testable:

* **Substitution.** Locals have a roughly fixed entertainment budget. Money spent on the tournament is money not spent elsewhere in the same city — a reallocation, not new activity. Economists argue local spending should be largely excluded from any genuine impact estimate.
* **Crowding out / displacement.** Host cities are often already busy in summer. Football fans displace the tourists who would otherwise have come. The clearest example: in 1994, Orlando — home to Disneyworld and already full in summer — was the worst-performing host city, because soccer fans likely displaced higher-spending theme-park visitors.
* **Leakage.** Revenue flows to non-local owners (hotel chains, FIFA itself) and leaves the local economy.

A portfolio project that estimated a positive effect and stopped there would be making exactly the mistake this literature spent thirty years documenting. So the spine of this project is: estimate the local effect cleanly, then interrogate honestly how much of it survives the substitution-and-displacement critique. The mature conclusion is not "the World Cup made cities £X richer." It is "here is the local divergence I can measure, here is why the naive reading overstates the true economic gain, and here is what the data can and cannot tell us." That skepticism — siding with the empirical literature over the boosters — is the senior signal, the equivalent of refusing to oversell a model that looks too good.

## 4. The data

### The constraint that shapes everything

The 2026 tournament's own economic outcome data does not exist yet and will not for months. US metro employment data (the cleanest free city-level economic series) is published by the BLS on a roughly six-month lag — Q2 2026 figures arrive in late November 2026, Q3 in early 2027. You cannot run a headline estimate on a tournament still in progress. This is not a limitation to apologise for; it is a fact of the data calendar, and the design accounts for it directly.

### The design that follows

The live 2026 tournament is the framing and motivation. The headline causal estimate runs on a past, fully-settled mega-event where the outcomes are completely observed — the 1994 US World Cup is the natural anchor (same country as the 2026 hosts, the event the foundational critique studied), with a more recent settled event as an alternative if its data is cleaner. A synthetic-data validation carries the core methodological demonstration, which is where most of the technical credibility is won (see §6).

### The outcome variable and source

Primary: US metro-level leisure & hospitality employment, from the BLS Quarterly Census of Employment and Wages (QCEW). Free, downloadable as CSV via the QCEW Open Data Access API, covering ~95% of US jobs by county and industry. One real wrinkle, caught in research and built into the plan: as of Q2 2025, BLS publishes metro-area (MSA) data as totals only — the industry breakdown is no longer at MSA level. So the leisure-&-hospitality slice is taken at county level (still published by industry) and aggregated up to metros manually. This is a known, documented step, not a surprise.

Employment is the right outcome because it is the variable the academic literature actually uses for sectoral mega-event analysis (Baumann, Engelhardt & Matheson on World Cup labour-market effects), it is free and granular, and it sidesteps the paywalls around the more direct measures. Honest about proxies: free card-spending and hotel-occupancy data at city level effectively does not exist (Mastercard, STR, and AirDNA are all paid), so employment is the best free outcome rather than the theoretically ideal one — and the brief says so.

### Treated, control, and the asymmetry

Treated units are host metros; control units are comparable non-host metros matched on pre-tournament characteristics (size, industry mix, baseline tourism). One honest scoping decision: the cleanest free city-level economic data is US-only — Mexican (INEGI) and Canadian (Statistics Canada CMA) city data is thinner and harder to harmonise — so the rigorous estimate is built on US host-vs-non-host cities, with the cross-country picture noted as a limitation rather than faked.

## 5. The method

1. **The canonical 2×2 DiD and the parallel-trends assumption** — stated plainly, tested on pre-treatment data, and visualised. Parallel trends is the assumption the whole design rests on; if host and control cities were already diverging before the tournament, the estimate is contaminated, and showing the pre-trends honestly is non-negotiable.
2. **The event-study (dynamic DiD) specification** — effects estimated by time relative to treatment (leads and lags). The event-study plot is the hero figure of any credible DiD: flat pre-treatment coefficients (parallel trends holding) followed by a post-treatment jump (the effect) is the single most convincing visual in the field.
3. **The modern staggered-treatment layer** — this is the methodological centrepiece. World Cup matches arrive in different cities on different dates, so treatment timing is staggered. The recent econometrics literature (Goodman-Bacon 2021; Callaway & Sant'Anna 2021; Sun & Abraham 2021; de Chaisemartin & D'Haultfœuille; Borusyak et al.) has shown that the naive two-way fixed-effects (TWFE) estimator is biased under staggered timing with heterogeneous effects — it implicitly uses already-treated cities as controls for later-treated ones ("forbidden comparisons"), producing negative weights and a distorted estimate. The project demonstrates this problem and applies a modern estimator (Callaway & Sant'Anna) that fixes it. Being aware of this — rather than running a naive TWFE and calling it done — is exactly what a senior econometrician checks for, and it is the difference between a 2018-era and a current causal project.
4. **Robustness and falsification, treated as core not appendix:** placebo tests (fake treatment dates, fake treated cities — an effect that shows up under a placebo means the design is broken), HonestDiD-style sensitivity bounds on the parallel-trends assumption, and clustered standard errors with attention to the few-treated-clusters problem (a handful of host cities means standard inference understates uncertainty; a wild cluster bootstrap is the honest fix).

### Tooling

The entire analysis runs in Python — no R dependency. The `differences` package implements the Callaway & Sant'Anna estimator natively (staggered timing, group-time ATTs, event-study aggregation, multiplier bootstrap); `pyfixest` / `linearmodels` cover the fixed-effects and TWFE baselines; `statsmodels` and `scipy` handle the 2×2 and inference. Versions are pinned at build time.

## 6. The synthetic validation — where the method is proven

Because the real data is observational and its true causal effect is unknown, the project includes a synthetic panel where the true effect is planted by construction, and shows the methods recover it. This does triple duty:

* **Proves the estimator works** — plant a known average treatment effect, show the DiD recovers it within confidence bounds.
* **Demonstrates the staggered-DiD problem in a controlled setting** — build a staggered-adoption panel with heterogeneous effects, show naive TWFE returns the wrong (biased) number while Callaway & Sant'Anna recovers the true one. This is the clearest possible illustration that the author understands the modern critique, not just the textbook 2×2.
* **Makes the substitution critique concrete** — plant a pure-displacement scenario (a city where apparent "gains" are entirely spending pulled from elsewhere, netting to zero real effect) and show that a naive impact reading would falsely report a windfall. This is the Orlando story, simulated: the analysis that looks like it found an effect, didn't.

The synthetic half is labelled clearly as synthetic and its generating parameters and seed are committed, so it is fully reproducible — the same transparency discipline the rest of the work demands.

## 7. What "done" looks like — the standard

For the recruiter / hiring manager (first thirty seconds): the repository is clean and professional; the README opens with the question in plain language, the event-study hero plot, and the headline finding stated honestly (likely: "a small or negligible local effect, consistent with thirty years of economics research, not the windfall boosters claim"); and the synthetic-vs-real structure is clear.

For the senior reviewer (the deep read): parallel trends is tested and shown, not assumed; the staggered-treatment problem is recognised and handled with a modern estimator; placebo and sensitivity tests are present; inference accounts for the few-clusters problem; the substitution/crowding-out critique is engaged with the actual literature; the synthetic validation proves the methods recover known effects; and the limitations — US-only, employment-as-proxy, observational data, the discontinued MSA series — are named by the author rather than left for the reader to find.

Both must be true. The first gets the project looked at; the second gets the author trusted. And the project's headline honesty — that the credible finding is probably a small or null effect, siding with the literature against the hype — is itself the differentiator. A null result, rigorously defended, is a more sophisticated portfolio piece than a fabricated win.

## 8. Repository structure

Following the portfolio conventions — lowercase-hyphen naming, one project per repo, numbered notebooks, no catch-all folders:

```
host-city-economics/
├── README.md                     # written last, from real results
├── PROJECT_BRIEF.md              # this document
├── requirements.txt              # pinned dependencies
├── src/
│   ├── generate_panel.py         # seeded synthetic panel generator (known planted effects)
│   ├── data_prep.py              # QCEW download, county→metro aggregation, treated/control build
│   └── did.py                    # reusable DiD / event-study / CS-estimator functions
├── notebooks/
│   ├── 01_synthetic-validation.ipynb   # plant effects; 2x2, TWFE bias, Callaway-Sant'Anna recovery
│   ├── 02_data-and-design.ipynb        # real data build, treated/control matching, parallel-trends checks
│   └── 03_estimation-and-robustness.ipynb  # event study, CS estimator, placebo, sensitivity, inference
├── data/
│   └── ...                        # QCEW extracts + synthetic panel (committed where small)
└── results/
    └── *.png                      # event-study and diagnostic figures for the README
```

## 9. The build sequence

Built in dependency order — the README is written last, from real results, never from an assumed conclusion.

1. **Skeleton + environment** — structure, pinned `requirements.txt`, seeds, git initialised so history is honest from the first commit.
2. **Synthetic panel generator + validation notebook (`01_`)** — built first, deliberately. Planting known effects and showing the estimators (including the TWFE-bias-vs-Callaway-Sant'Anna demonstration) recover them proves the machinery is correct before it touches messy real data. This is the methodological proof-of-correctness, the equivalent of the correctness gates in the forecasting project.
3. **Real data build + design notebook (`02_`)** — QCEW extract, county→metro aggregation, treated/control construction and matching, and the all-important pre-treatment parallel-trends check.
4. **Estimation + robustness notebook (`03_`)** — event study, Callaway & Sant'Anna estimate, placebo tests, sensitivity bounds, clustered/bootstrap inference.
5. **Reusable `src/` modules** — the DiD/event-study functions factored out as a clean, documented module, single source of truth across the notebooks.
6. **README** — written last, from the actual results, in the established portfolio shape, with the honest finding (likely small/null) framed as the credible result it is, and the substitution critique foregrounded.

Commits stage by stage, so the history reflects a real build.

## 10. The honest finding, anticipated

It is worth stating the likely conclusion up front, because the project is designed to reach it honestly rather than avoid it: the credible local effect of hosting is probably small, possibly null, and far below booster projections — and the data cannot cleanly separate even that small effect from substitution and displacement. If the analysis instead finds a robust positive effect that survives the placebo and sensitivity tests, that stands too — the rule, as always, is that the conclusion follows the evidence. But a rigorously defended "the windfall isn't really there" is the result most consistent with the literature, and the more impressive thing to be able to demonstrate and defend.

The synthetic panel in this project is generated for demonstration and is clearly labelled. The real-data analysis uses public BLS QCEW employment data and is observational; its causal claims are bounded by the assumptions stated in the analysis.
