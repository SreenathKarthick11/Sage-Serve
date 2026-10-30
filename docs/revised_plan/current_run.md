# Running SageServe — Complete Guide

> **Current state:** The pipeline has versioned data support (ingest → ETL → forecast) with
> content-addressed, immutable artifact directories and an append-only lineage ledger.
> This doc covers every step to run from a clean checkout all the way to a SageServe
> simulation with ARIMA forecasting.

---

## Repository layout

```
Sage-Serve/                         ← repo root
├── mlops/                          ← MLOps venv + all pipeline code
│   ├── .venv/                      ← Python virtual environment
│   ├── requirements.txt
│   ├── pipeline/                   ← versioning, lineage, orchestrator
│   │   ├── run_pipeline.py         ← main entry-point  (python -m pipeline.run_pipeline)
│   │   ├── versioning.py           ← content-addressed IDs + manifests
│   │   ├── lineage.py              ← lineage ledger reader
│   │   └── paths.py                ← DATA_ROOT, SOURCES, STAGES
│   ├── trace_sim/                  ← trace generators
│   │   ├── generate_random_trace.py   (static source)
│   │   └── __main__.py                (streaming source)
│   ├── etl_module/                 ← ETL + validation
│   │   └── etl.py
│   └── forecasting/
│       └── arima_forecaster.py
│
├── data/                           ← versioned artifacts (next to mlops/)
│   ├── static/
│   │   ├── raw/<version>/          ← trace.csv + manifest.json
│   │   ├── clean/<version>/        ← validated_trace.csv + manifest.json
│   │   └── forecasts/<version>/    ← forecast_<model>_region<r>.csv + manifest.json
│   ├── streaming/                  ← same layout for streaming source
│   └── lineage/lineage.jsonl       ← append-only ledger
│
└── SageServe-original/             ← simulator
    ├── .venv/                      ← separate Python venv
    ├── requirements.txt
    └── run.py                      ← main simulator entry-point (Hydra)
```

---

## Versioning model

Every artifact is an **immutable directory** inside `data/<source>/<stage>/<version>/`.

| Stage      | Version ID derivation                                 |
|------------|-------------------------------------------------------|
| `raw`      | `SHA-256(file content)[:12]`                          |
| `clean`    | `SHA-256({raw artifact id, CLEAN_PARAMS})[:12]`       |
| `forecasts`| `SHA-256({clean artifact id, FORECAST_PARAMS})[:12]`  |

- **Same inputs + same params → same ID → stage is skipped** (idempotent).
- Each directory contains a `manifest.json` with id, parents, params, file hashes, timestamp, git commit.
- `data/<source>/<stage>/latest.json` is an atomic pointer to the newest version.
- `data/lineage/lineage.jsonl` is an append-only ledger; one JSON line per artifact.

---

## 0. First-time setup

> Run these once. The `.venv` inside `mlops/` is already created in this repo.

```bash
cd ~/repos/Sage-Serve/mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To override where data is written (default = `../data` relative to `mlops/`):

```bash
export MLOPS_DATA_ROOT=/some/other/path
```

---

## 1. Run the full pipeline (ingest → ETL → forecast)

All `python -m` commands **must be run from inside `mlops/`** with the venv active.

```bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate

# Static source: generates 5 000 random requests (seed 42)
python -m pipeline.run_pipeline --source static

# Streaming source: simulate 10 minutes of real-time traffic
python -m pipeline.run_pipeline --source streaming --duration 600
```

What happens under the hood:

```
[1/3] ingest  → data/static/raw/<version>/trace.csv  (+ manifest.json)
[2/3] etl     → data/static/clean/<version>/validated_trace.csv  (+ manifest.json)
[3/3] forecast → data/static/forecasts/<version>/forecast_<model>_region<r>.csv  (× 12 files + manifest.json)
```

After the run, a lineage tree is printed to stdout, e.g.:

```
=== lineage ===
static/forecasts/ce827f5cc85c  [2026-10-27T03:34:08+00:00, git 8dea009] n_forecast_files=12
    params: {"arima_order": [1, 1, 1], ...}
    <- static/clean/2c95d94fd840  [2026-10-27T03:34:05+00:00, git 8dea009] rows_in=68881 rows_out=68881
        params: {"cleaning_version": "1", ...}
        <- static/raw/d51facaf7e65  [2026-10-27T03:34:02+00:00, git 8dea009] status=complete rows=68881
            params: {"num_requests": 5000, "seed": 42, ...}
```

---

## 2. Run individual stages

You can run any subset of stages and skip the rest.

```bash
# Only ingest (produce raw version)
python -m pipeline.run_pipeline --source static --stages ingest

# Only ETL on the latest raw version
python -m pipeline.run_pipeline --source static --stages etl

# Only forecast on the latest clean version
python -m pipeline.run_pipeline --source static --stages forecast

# ETL + forecast, skipping ingest (reuse latest raw)
python -m pipeline.run_pipeline --source static --stages etl,forecast

# Forecast on a specific clean version (not latest)
python -m pipeline.run_pipeline --source static --stages forecast --input-version 2c95d94fd840
```

Each stage module can also be invoked directly:

```bash
# Static trace generator  (creates data/static/raw/<version>/)
python -m trace_sim.generate_random_trace --num-requests 5000 --seed 42

# Streaming trace server  (Ctrl+C to stop; creates data/streaming/raw/<version>/)
python -m trace_sim --duration 600 --speedup 60 --rate-scale 0.002 --seed 42

# ETL on latest raw
python -m etl_module.run_etl --source static

# ARIMA forecaster on latest clean
python -m forecasting.arima_forecaster --source static
python -m forecasting.arima_forecaster --source static --input-version 2c95d94fd840
```

---

## 3. Inspect the lineage ledger

```bash
# List all artifacts (all sources)
python -m pipeline.lineage list

# List only static artifacts
python -m pipeline.lineage list --source static

# Show the full lineage tree for the latest forecast version
python -m pipeline.lineage show static/forecasts/latest

# Show a specific forecast version
python -m pipeline.lineage show static/forecasts/ce827f5cc85c
```

Current artifacts (as of last run):

| Artifact                            | Version        | Created (UTC)        |
|-------------------------------------|----------------|----------------------|
| `static/raw`                        | `d51facaf7e65` | 2026-10-27 03:34:02  |
| `static/clean`                      | `2c95d94fd840` | 2026-10-27 03:34:05  |
| `static/forecasts`                  | `ce827f5cc85c` | 2026-10-27 03:34:08  |

---

## 4. Find the forecast path for SageServe

SageServe needs the directory that contains the `forecast_<model>_region<r>.csv` files.

```bash
# Print the path of the latest static forecast version
python -c "
from pipeline.versioning import resolve_version
print(resolve_version('static', 'forecasts', 'latest'))
"
```

This prints something like:

```
/home/sreenath-karthick/repos/Sage-Serve/data/static/forecasts/ce827f5cc85c
```

Or just read `data/static/forecasts/latest.json`:

```bash
cat ../data/static/forecasts/latest.json
# → {"version": "ce827f5cc85c"}
```

So the full forecast directory is:

```
../data/static/forecasts/ce827f5cc85c/
```

---

## 5. Run SageServe (simulator)

Switch to the SageServe environment (separate venv, separate Python):

```bash
deactivate                         # leave the mlops venv
cd ~/repos/Sage-Serve/SageServe-original
source .venv/bin/activate
pip install -r requirements.txt    # first time only
```

### 5a. Without forecasting (short-term scaling only)

```bash
python run.py \
  trace.path="$PWD/../data/traces/random_trace.csv" \
  short_term_scaling=True \
  long_term_scaling=False
```

### 5b. With versioned ARIMA forecasting (long-term scaling enabled)

Use the versioned forecast directory produced by Stage 3:

```bash
FORECAST_DIR="$PWD/../data/static/forecasts/ce827f5cc85c"

python run.py \
  trace.path="$PWD/../data/traces/random_trace.csv" \
  global_arbiter.arima_traces="$FORECAST_DIR" \
  short_term_scaling=True \
  long_term_scaling=True
```

Or resolve it dynamically from the latest pointer:

```bash
FORECAST_DIR=$(python -c "
import sys; sys.path.insert(0, '../mlops')
from pipeline.versioning import resolve_version
print(resolve_version('static', 'forecasts', 'latest'))
")

python run.py \
  trace.path="$PWD/../data/traces/random_trace.csv" \
  global_arbiter.arima_traces="$FORECAST_DIR" \
  short_term_scaling=True \
  long_term_scaling=True
```

---

## 6. Output locations

| Output                          | Location                                             |
|---------------------------------|------------------------------------------------------|
| Raw trace (versioned)           | `data/static/raw/<version>/trace.csv`                |
| Validated trace (versioned)     | `data/static/clean/<version>/validated_trace.csv`    |
| Forecast CSVs (versioned)       | `data/static/forecasts/<version>/forecast_*.csv`     |
| Manifest + params               | `data/static/<stage>/<version>/manifest.json`        |
| Latest version pointer          | `data/static/<stage>/latest.json`                    |
| Lineage ledger                  | `data/lineage/lineage.jsonl`                         |
| Simulation results              | `experiments/results/`                               |
| ILP outputs                     | `experiments/ilp_outputs/`                           |

---

## 7. Varying parameters (creating new versions)

Running the pipeline with different parameters automatically creates a new version:

```bash
# Different seed → different raw version → different clean & forecast versions
python -m pipeline.run_pipeline --source static --num-requests 10000 --seed 99

# Different speedup for streaming
python -m pipeline.run_pipeline --source streaming --duration 300 --speedup 120
```

Because version IDs are derived from content/params, **re-running with the same parameters always hits the cache and skips the stage**.

To force a new clean version from the same raw data, bump `CLEANING_VERSION` in
[`etl_module/etl.py`](../../mlops/etl_module/etl.py):

```python
CLEANING_VERSION = "2"   # was "1"
```

---

## 8. Quick verification (smoke test)

Run this from `mlops/` to confirm all three stages pass end-to-end:

```bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate

python -m pipeline.run_pipeline --source static
python -m pipeline.lineage list --source static
python -m pipeline.lineage show static/forecasts/latest
```

Expected final output of `lineage show`:

```
static/forecasts/ce827f5cc85c  [2026-10-27T03:34:08+00:00, git 8dea009] n_forecast_files=12
    params: {"arima_order": [1, 1, 1], "bucket_seconds": 60, "train_window": 60, ...}
    <- static/clean/2c95d94fd840  ...  rows_in=68881 rows_out=68881
        <- static/raw/d51facaf7e65  ...  status=complete rows=68881
```

---

## 9. Environment variable reference

| Variable          | Default                              | Purpose                              |
|-------------------|--------------------------------------|--------------------------------------|
| `MLOPS_DATA_ROOT` | `<parent of mlops>/data`             | Override data directory location     |

---

## 10. Common errors

| Error | Fix |
|-------|-----|
| `FileNotFoundError: No versions yet for static/raw` | Run `--stages ingest` first |
| `ModuleNotFoundError: pipeline` | Run `python -m` from inside `mlops/`, not from repo root |
| `FileNotFoundError: Trace not found` | Make sure `data/traces/random_trace.csv` exists for legacy runs |
| `Version not found: static/clean/...` | Run `--stages etl` before `--stages forecast` |
| Forecast skipped (already exists) | Expected — same inputs → same version. Change params to create a new version. |
