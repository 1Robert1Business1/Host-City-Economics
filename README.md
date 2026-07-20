# Host-City Economics

A difference-in-differences study of whether hosting football's biggest
tournament actually moves a city's economy — and an honest reckoning with how
much of the apparent effect is real.

> **Status: under construction.** This README is written *last*, from real
> results, never from an assumed conclusion (see `PROJECT_BRIEF.md` §9). The
> full specification — question, design, data, methods, and what "done" means —
> lives in [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

## Build sequence

Built in dependency order; commits stage by stage so the history reflects a
real build.

1. **Skeleton + environment** — structure, pinned `requirements.txt`, seeds,
   git. ✅ *current stage*
2. Synthetic panel generator + validation notebook (`01_`) — the
   proof-of-correctness, built first.
3. Real data build + design notebook (`02_`).
4. Estimation + robustness notebook (`03_`).
5. Reusable `src/` modules factored out.
6. README — written from the actual results.

## Repository layout

```
host-city-economics/
├── README.md              # this file — written last, from real results
├── PROJECT_BRIEF.md       # the specification and source of truth
├── requirements.txt       # pinned dependencies
├── src/
│   ├── config.py          # seeds + paths — single source of reproducibility
│   ├── generate_panel.py  # seeded synthetic panel (known planted effects)
│   ├── data_prep.py       # QCEW download, county→metro aggregation
│   └── did.py             # reusable DiD / event-study / CS-estimator functions
├── notebooks/
│   ├── 01_synthetic-validation.ipynb
│   ├── 02_data-and-design.ipynb
│   └── 03_estimation-and-robustness.ipynb
├── data/                  # QCEW extracts + synthetic panel (committed where small)
└── results/               # event-study and diagnostic figures
```

## Environment (Windows / PowerShell)

```powershell
# From the repo root, create and activate a virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the pinned causal stack
python -m pip install --upgrade pip
pip install -r requirements.txt

# Register the kernel for the notebooks
python -m ipykernel install --user --name host-city-economics
```

On macOS/Linux the equivalent is `python3 -m venv .venv && source
.venv/bin/activate && pip install -r requirements.txt`.
