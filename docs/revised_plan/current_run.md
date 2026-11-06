# Running SageServe  Complete Guide

> **Current state:** SageServe supports a versioned MLOps pipeline
> (ingest → ETL → forecast), with content-addressed immutable artifacts
> and an append-only lineage ledger. The forecast stage uses whichever
> model is **active in the model registry** (ARIMA, HMM, ...), so a model
> must be registered and promoted before the first forecast. The workflow
> below covers setup, the model registry, static and streaming data
> generation, validation, forecasting, simulator execution, and the
> inference entry point.

------------------------------------------------------------------------

## Repository layout

``` text
Sage-Serve/
├── mlops/
│   ├── .venv/
│   ├── requirements.txt
│   ├── pipeline/
│   │   ├── run_pipeline.py
│   │   ├── versioning.py
│   │   ├── lineage.py
│   │   └── paths.py
│   ├── trace_sim/
│   │   ├── generate_random_trace.py
│   │   └── __main__.py
│   ├── etl_module/
│   │   ├── etl.py
│   │   └── run_etl.py
│   ├── forecasting/
│   │   ├── base.py                 # BaseForecaster contract
│   │   ├── factory.py              # loads the active model, calls run_forecast()
│   │   ├── arima_forecaster.py
│   │   ├── hmm_forecaster.py
│   │   └── trace_io.py             # shared clean-trace reader
│   ├── model_registry/             # registry CODE only
│   │   ├── __init__.py
│   │   ├── location.py             # where registry data lives
│   │   ├── scripts.py              # register / card / promote logic
│   │   └── run.py                  # registry CLI
│   └── inference/
│       └── run_inference.py
│
├── data/
│   ├── static/
│   │   ├── raw/<version>/
│   │   ├── clean/<version>/
│   │   └── forecasts/<version>/
│   ├── streaming/
│   │   ├── raw/<version>/
│   │   ├── clean/<version>/
│   │   └── forecasts/<version>/
│   ├── model_registry/             # registry DATA (changes on every operation)
│   │   ├── active_model.json
│   │   └── models/<name>/<version>/
│   │       ├── config.yaml
│   │       ├── manifest.json
│   │       ├── model_card.json
│   │       └── transition_matrix.pkl   # HMM only
│   └── lineage/lineage.jsonl
│
├── SageServe-original/
│   ├── .venv/
│   ├── requirements.txt
│   └── run.py
│
└── experiments/
    ├── results/
    └── ilp_outputs/
```

`mlops/.venv` and `SageServe-original/.venv` are separate Python
environments. Activate the environment appropriate to the component
being run.

The registry **code** (`mlops/model_registry/`) and registry **data**
(`data/model_registry/`) are deliberately separate: every `register`,
`card` and `promote` changes the data, and none of it should live next to
the code.

------------------------------------------------------------------------

## Versioning model

Every pipeline artifact is stored in an immutable directory:

``` text
data/<source>/<stage>/<version>/
```

  -----------------------------------------------------------------------
  Stage                               Version ID derivation
  ----------------------------------- -----------------------------------
  `raw`                               SHA-256 of the raw trace content,
                                      truncated to 12 characters

  `clean`                             SHA-256 derived from the raw
                                      artifact ID and cleaning parameters

  `forecasts`                         SHA-256 derived from the clean
                                      artifact ID and forecasting
                                      parameters, **including the active
                                      model's metadata**
  -----------------------------------------------------------------------

-   Same inputs and parameters produce the same version ID. Existing
    artifacts are reused/skipped.
-   Because the active model's metadata is part of the forecast version,
    switching the active model (or registering a model with different
    parameters) produces a **new** forecast version from the same clean
    artifact.
-   Each artifact directory contains a `manifest.json` with metadata
    such as its ID, parent artifact(s), parameters, file hashes,
    timestamp, and Git commit.
-   `data/<source>/<stage>/latest.json` points to the latest version for
    that source and stage.
-   `data/lineage/lineage.jsonl` is an append-only ledger containing one
    record per artifact.

The versioning model applies independently to `static` and `streaming`
sources.

Model registry versions are separate from pipeline artifact versions.
They are simple sequential ids (`v1`, `v2`, ...) per model name, assigned
automatically by `register`.

------------------------------------------------------------------------

## 0. First-time setup

The MLOps virtual environment is expected to be located inside `mlops/`.

``` bash
cd ~/repos/Sage-Serve/mlops

# Create the environment only if it does not already exist
python -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt
```

To override the default data directory (`../data`, relative to
`mlops/`):

``` bash
export MLOPS_DATA_ROOT=/some/other/path
```

To override where the model registry data lives (default
`<repo>/data/model_registry`):

``` bash
export MODEL_REGISTRY_DIR=/some/other/registry
```

The simulator uses its own environment:

``` bash
cd ~/repos/Sage-Serve/SageServe-original
source .venv/bin/activate
pip install -r requirements.txt
```

Install dependencies only when setting up the corresponding environment
or when its requirements change.

**Before the first forecast:** make sure the registry has an active
model. Check with:

``` bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate
python -m model_registry.run active
```

If it prints `No active_model.json yet.`, register and promote a model
first (section 2).

------------------------------------------------------------------------

## 1. Run the full MLOps pipeline

Run all `python -m` pipeline commands from inside `mlops/`, with
`mlops/.venv` active.

``` bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate
```

The forecast stage (3/3) loads the **active model** from the registry. It
fails with `Model registry pointer not found` if no model has been
promoted to Production yet.

### 1a. Static source

Generate a static trace and run ingest → ETL → forecast:

``` bash
python -m pipeline.run_pipeline \
  --source static \
  --num-requests 5000 \
  --seed 42
```

### 1b. Streaming source

Simulate 600 seconds of traffic, with a simulation speedup of 300:

``` bash
python -m pipeline.run_pipeline \
  --source streaming \
  --duration 600 \
  --speed 300
```

The streaming run uses the pipeline's configured rate scale and seed
defaults unless explicitly overridden.

### What the pipeline does

For either source, the orchestrator runs:

``` text
[1/3] ingest
      └── data/<source>/raw/<version>/trace.csv
          data/<source>/raw/<version>/manifest.json

[2/3] ETL
      └── data/<source>/clean/<version>/validated_trace.csv
          data/<source>/clean/<version>/manifest.json

[3/3] forecast   (uses the active model from the registry)
      └── data/<source>/forecasts/<version>/
          ├── forecast_A_region0.csv
          ├── ...
          ├── forecast_D_region2.csv
          └── manifest.json
```

With four models (`A`, `B`, `C`, `D`) and three regions (`0`, `1`, `2`),
a successful forecast stage produces 12 forecast CSV files. This is the
same for every registered model type: the pipeline calls
`ForecasterFactory.run_forecast(...)`, and the active model writes its
own CSVs. The clean trace CSV is located by its columns
(`arrival_timestamp`, `model_type`, `regions`, `prompt_size`), not by
filename.

The orchestrator prints the artifact lineage after execution. If an
artifact with the same derived version already exists, the corresponding
stage is skipped. This is expected behavior, not a failure.

------------------------------------------------------------------------

## 2. Manage the model registry

The registry stores every forecasting model version with its config,
weights (if any), manifest and model card. The pipeline never names a
model: `ForecasterFactory` reads `data/model_registry/active_model.json`,
builds that model from its `config.yaml`, and calls its `run_forecast()`.

Run all registry commands from inside `mlops/` with `mlops/.venv` active:

``` bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate
```

### Lifecycle

``` text
register  ->  card  ->  promote (Production)  ->  pipeline uses it
 (stage None)          (rewrites active_model.json)
```

-   `register` validates the config against the model class, assigns the
    next version id (`v1`, `v2`, ...), writes `config.yaml` (with
    `model_info.version` set to the assigned id), the weights file if the
    model needs one, and `manifest.json`. The stage starts as `None`.
-   `card` writes `model_card.json` (required before Production).
-   `promote` moves a version to `Staging`, `Production` or `Archived`.
    Promoting to `Production` rewrites `active_model.json`.

### 2a. Register an ARIMA model (config only)

Write a config file anywhere (it is copied into the registry):

``` yaml
# my_arima.yaml
model_info:
  type: arima
  description: "ARIMA forecaster"

parameters:
  order: {p: 1, d: 1, q: 1}
  train_window_mins: 60
  future_horizon: 60
  bucket_seconds: 60
```

``` bash
python -m model_registry.run register --name arima --config my_arima.yaml \
  --metrics mae=9.2

python -m model_registry.run card --name arima --version v1 \
  --description "ARIMA(1,1,1) workload forecaster" \
  --intended-use "Per-minute prompt-size forecasting" \
  --limitations "Poor on bursty traffic"

python -m model_registry.run promote --name arima --version v1 --stage Production
```

Use the version id printed by `register` (`v1` for the first ARIMA
model, `v2` for the next, and so on).

### 2b. Register an HMM model (config + transition matrix)

HMM needs a weights file (`transition_matrix.pkl`). Provide it in one of
two ways.

``` yaml
# my_hmm.yaml
model_info:
  type: hmm
  description: "HMM workload forecaster"

parameters:
  n_states: 3
  train_window_mins: 60
  future_horizon: 60
  bucket_seconds: 60
```

**Option 1: train the transition matrix from a clean trace.**
`--train-from` takes a clean **version directory** (not the CSV file).
The path is relative to your current directory; from `mlops/`, the data
folder is one level up:

``` bash
CLEAN_DIR=$(python -c "
from pipeline.versioning import resolve_version
print(resolve_version('static', 'clean', 'latest'))
")

python -m model_registry.run register --name hmm --config my_hmm.yaml \
  --train-from "$CLEAN_DIR" --metrics mae=11.0
```

**Option 2: supply a pre-trained weights file:**

``` bash
python -m model_registry.run register --name hmm --config my_hmm.yaml \
  --weights ./transition_matrix.pkl --metrics mae=11.0
```

The weights file must be non-empty. It is expected to be a pickled dict
with `transition_matrix` and `means` (or an object exposing `predict`,
`transmat_` and `means_`).

Then create the card and promote:

``` bash
python -m model_registry.run card --name hmm --version v1 \
  --description "HMM workload forecaster" \
  --intended-use "Per-minute prompt-size forecasting" \
  --limitations "Argmax state transitions give repetitive forecasts"

python -m model_registry.run promote --name hmm --version v1 --stage Production
```

Training pools traffic across all models and regions into a single
transition matrix.

### 2c. Inspect the registry

``` bash
# All models, versions, stages, and whether a card exists
python -m model_registry.run list

# The pointer the pipeline uses
python -m model_registry.run active

# The Production manifest for one model name
python -m model_registry.run production --name arima
```

Example `list` output:

``` text
arima/v1    stage=Archived     card
hmm/v1      stage=Production   card
```

### 2d. Switch or roll back the active model

Promote the version you want. Only one version can be in Production at a
time across **all** models, so promoting one archives the previous
Production version automatically:

``` bash
python -m model_registry.run promote --name arima --version v1 --stage Production
python -m model_registry.run active
python -m pipeline.run_pipeline --source static --stages forecast
```

Archived versions can be promoted again if they still pass the
governance gate. Re-running the forecast stage after a switch creates a
new forecast version (the active model's metadata is part of the
version id).

### 2e. Registry rules

1.  **Governance gate for Production:** the version must have a
    `model_card.json`, and its manifest metrics must satisfy
    `mae <= 15.0` **or** `f1 >= 0.80`. Missing metrics fail the gate.
    Pass real, measured values with `--metrics`; do not invent numbers
    to get past the gate.
2.  **Versions are assigned automatically.** Never create version
    folders by hand; an empty or hand-made folder (empty `manifest.json`,
    empty weights file) breaks `list`, `promote` and the factory.
3.  **One Production model at a time**, across all model names.
4.  **Validation happens before anything is written.** A bad config, a
    missing or empty weights file, or a missing `--weights` /
    `--train-from` for a model that needs weights fails without creating
    a version folder.
5.  **`model_info.type` in the config must match `--name`.**
6.  The CLI accepts a global `--registry <dir>` flag **before** the
    subcommand to operate on a different registry location:
    `python -m model_registry.run --registry /path list`.

### 2f. Add a new model type

1.  Write a class subclassing `BaseForecaster` in `mlops/forecasting/`
    that implements `__init__`, `predict`, `get_metadata` and
    `run_forecast` (which reads the clean trace, forecasts, and writes
    the forecast CSVs).
2.  Declare what it needs: `REQUIRED_PARAMS` (config keys that must
    exist), `WEIGHTS_FILE` (filename, or `None`), and, if the weights can
    be learned from a trace, a `train(config, clean_dir)` classmethod.
3.  Add one line to `FORECASTER_CLASSES` in `forecasting/factory.py`.

`register` and the factory pick the new type up from that map; no other
code changes.

### Registry CLI reference

  ---------------------------------------------------------------------------------------------------
  Command                  Purpose
  ------------------------ --------------------------------------------------------------------------
  `register`               `--name`, `--config`, optional `--weights` or `--train-from`,
                           `--metrics key=value ...`; creates the next version

  `card`                   `--name`, `--version`, `--description`, `--intended-use`, `--limitations`

  `promote`                `--name`, `--version`, `--stage {Staging,Production,Archived}`

  `list`                   Show all models, versions, stages, card status

  `active`                 Show `active_model.json`

  `production`             `--name`; show that model's Production manifest
  ---------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 3. Run individual pipeline stages

Run these commands from `mlops/` with the MLOps environment active.

``` bash
# Ingest only: generate a raw artifact
python -m pipeline.run_pipeline --source static --stages ingest

# ETL only: process the latest raw artifact
python -m pipeline.run_pipeline --source static --stages etl

# Forecast only: process the latest clean artifact with the active model
python -m pipeline.run_pipeline --source static --stages forecast

# ETL + forecast, reusing the latest raw artifact
python -m pipeline.run_pipeline --source static --stages etl,forecast

# Forecast from a particular clean artifact
python -m pipeline.run_pipeline \
  --source static \
  --stages forecast \
  --input-version <clean-version>
```

Replace `static` with `streaming` to run the corresponding stage for
streaming artifacts.

### Invoke individual modules directly

``` bash
# Static trace generator
python -m trace_sim.generate_random_trace \
  --num-requests 5000 \
  --seed 42

# Streaming trace generator
python -m trace_sim \
  --duration 600 \
  --speedup 300 \
  --rate-scale 0.002 \
  --seed 42

# ETL using the latest raw artifact
python -m etl_module.run_etl --source streaming
```

The forecast stage no longer has a standalone module entry point. Run it
through the orchestrator (`--stages forecast`), which loads the active
model from the registry:

``` bash
python -m pipeline.run_pipeline --source streaming --stages forecast
python -m pipeline.run_pipeline \
  --source streaming \
  --stages forecast \
  --input-version <clean-version>
```

The orchestrator is the recommended way to run the complete pipeline
because it coordinates stages, version resolution, and lineage
reporting.

------------------------------------------------------------------------

## 4. Inspect the lineage ledger

Run from `mlops/` with the MLOps environment active.

``` bash
# List artifacts from all sources
python -m pipeline.lineage list

# List only static artifacts
python -m pipeline.lineage list --source static

# List only streaming artifacts
python -m pipeline.lineage list --source streaming

# Show the latest streaming forecast and its ancestors
python -m pipeline.lineage show streaming/forecasts/latest

# Show the latest static forecast and its ancestors
python -m pipeline.lineage show static/forecasts/latest

# Show a particular artifact
python -m pipeline.lineage show streaming/forecasts/<forecast-version>
```

A lineage tree should show the dependency chain:

``` text
<source>/forecasts/<forecast-version>
    <- <source>/clean/<clean-version>
        <- <source>/raw/<raw-version>
```

It also reports relevant parameters, timestamps, Git commit information,
and stage metadata. The forecast artifact's parameters include the
metadata of the model that produced it (type, version, window, horizon),
so you can tell which registry model generated a given forecast.

------------------------------------------------------------------------

## 5. Resolve the latest artifact paths

Run from `mlops/` with the MLOps environment active.

### Latest streaming raw trace

``` bash
STREAMING_RAW_DIR=$(python -c "
from pipeline.versioning import resolve_version
print(resolve_version('streaming', 'raw', 'latest'))
")

STREAMING_TRACE="$STREAMING_RAW_DIR/trace.csv"

echo "Streaming trace: $STREAMING_TRACE"
test -f "$STREAMING_TRACE" && echo "Streaming trace exists"
```

### Latest streaming forecast directory

``` bash
STREAMING_FORECAST_DIR=$(python -c "
from pipeline.versioning import resolve_version
print(resolve_version('streaming', 'forecasts', 'latest'))
")

echo "Streaming forecasts: $STREAMING_FORECAST_DIR"
find "$STREAMING_FORECAST_DIR" -maxdepth 1 -type f -name 'forecast_*.csv' | sort
```

### Latest static artifacts

``` bash
STATIC_RAW_DIR=$(python -c "
from pipeline.versioning import resolve_version
print(resolve_version('static', 'raw', 'latest'))
")

STATIC_FORECAST_DIR=$(python -c "
from pipeline.versioning import resolve_version
print(resolve_version('static', 'forecasts', 'latest'))
")

echo "Static raw: $STATIC_RAW_DIR"
echo "Static forecasts: $STATIC_FORECAST_DIR"
```

These commands resolve paths from the version pointers, so there is no
need to hardcode artifact version IDs.

------------------------------------------------------------------------

## 6. Run the SageServe simulator

The simulator has a separate virtual environment. The simulator must
receive a trace and, when long-term scaling is enabled, the forecast
directory generated from the same source. The forecasts it reads were
produced by whichever model was active when the forecast stage ran.

### 6a. Run with static data

``` bash
# Leave the MLOps environment
deactivate

cd ~/repos/Sage-Serve/SageServe-original
source .venv/bin/activate

# Resolve the latest static forecast directory
FORECAST_DIR=$(python -c "
import sys
sys.path.insert(0, '../mlops')
from pipeline.versioning import resolve_version
print(resolve_version('static', 'forecasts', 'latest'))
")

# Run the simulator
python run.py \
  trace.path="$PWD/../data/traces/random_trace.csv" \
  global_arbiter.arima_traces="$FORECAST_DIR" \
  short_term_scaling=True \
  long_term_scaling=True
```

The static example above uses the legacy/static simulator trace path
shown in the simulator run configuration. The config key is still named
`arima_traces`, but it simply points at the forecast directory; it works
with forecasts from any registered model type, since they all write the
same `Time` / `Predicted` CSV format.

### 6b. Run with streaming data

This is the end-to-end streaming workflow. It uses:

-   The raw streaming trace created by the streaming ingest stage.
-   The streaming forecast directory created from that streaming trace
    by the active registry model.
-   Both short-term and long-term scaling.

First, make sure you are in the simulator directory with its environment
active:

``` bash
deactivate 2>/dev/null || true
cd ~/repos/Sage-Serve/SageServe-original
source .venv/bin/activate
```

Resolve the latest streaming raw trace and forecast directory
dynamically:

``` bash
STREAMING_RAW_DIR=$(python -c "
import sys
sys.path.insert(0, '../mlops')
from pipeline.versioning import resolve_version
print(resolve_version('streaming', 'raw', 'latest'))
")

STREAMING_FORECAST_DIR=$(python -c "
import sys
sys.path.insert(0, '../mlops')
from pipeline.versioning import resolve_version
print(resolve_version('streaming', 'forecasts', 'latest'))
")

STREAMING_TRACE="$STREAMING_RAW_DIR/trace.csv"

echo "Input trace: $STREAMING_TRACE"
echo "Forecast directory: $STREAMING_FORECAST_DIR"

test -f "$STREAMING_TRACE"
test -d "$STREAMING_FORECAST_DIR"
```

Run SageServe:

``` bash
python run.py \
  trace.path="$STREAMING_TRACE" \
  global_arbiter.arima_traces="$STREAMING_FORECAST_DIR" \
  short_term_scaling=True \
  long_term_scaling=True
```

**Important:** Do not use `data/traces/random_trace.csv` for this
streaming run. That is the legacy/static trace path. The streaming run
must use `data/streaming/raw/<version>/trace.csv` and
`data/streaming/forecasts/<version>/`.

---

## 7. Run the inference module

The inference entry point is:

```text
mlops/inference/run_inference.py

```
It analyzes saved simulator measurements (detailed request CSVs, arbiter capacity CSVs, export inventory) without running LLMs or sending live inference requests.

### 7a. Basic execution

Run `run_inference.py` from the **repository root** using the MLOps virtual environment (`mlops/.venv`):

```bash
# Leave the SageServe simulator environment
deactivate 2>/dev/null || true

# Return to the repository root
cd ~/repos/Sage-Serve

# Activate the MLOps environment
source mlops/.venv/bin/activate

# Run inference analysis
python mlops/inference/run_inference.py \
  --results experiments/results \
  --forecast-model reactive-baseline \
  --out mlops/inference/reports/static_demo.csv \
  --plots-dir mlops/inference/reports/static_demo_plots

```

> **Note on `--forecast-model`:** This flag supplies an unverified descriptive label for report metadata; it does not select, execute, or evaluate a forecasting model.

### 7b. Generated outputs

Running the inference script produces the following outputs under the specified report paths:

* **Main CSV report (`<out>.csv`)**: Summary metrics with one row per experiment and model.
* **Per-window CSV (`<out_prefix>_windows.csv`)**: Metrics broken down with one row per experiment, model, and exported window.
* **Metadata JSON (`<out_prefix>.metadata.json`)**: Complete configuration, input file hashes, diagnostics, thresholds, scoring references, weights, and known limitations.
* **Diagnostic plots (`<plots_dir>/`)**: Up to 8 diagnostic charts (`fss_bar.png`, `sub_scores.png`, `latency_pct.png`, `slo_violations.png`, `cv_scatter.png`, `resource_usage.png`, `data_coverage.png`, `completions_by_window.png`). Use `--no-plots` to skip chart generation.

### 7c. Exit codes

| Exit Code | Meaning |
| --- | --- |
| **0** | Requested report outputs were successfully written. |
| **1** | Input/settings validation or report writing failed. |
| **2** | A usable report was written, but inputs/request exports were incomplete or plot generation failed. |

### 7d. Common flags & options

* `--allow-partial`: Explicitly skip invalid files or retain usable cells while recording any input/schema issues.
* `--window-boundaries <path/to/boundaries.csv>`: Pass explicit window boundary timestamps (`experiment_id,window,start_seconds,end_seconds`) for precise resource duration accounting.
* `--ttft-threshold <sec>` / `--tbt-threshold <sec>`: Custom TTFT (default >10.0s) and average TBT (default >0.2s) SLA violation benchmarks.
* `--no-plots`: Disable Matplotlib diagnostic plot generation.

---

## 8. Output locations

  ------------------------------------------------------------------------------------------
  Output                              Location
  ----------------------------------- ------------------------------------------------------
  Static raw trace                    `data/static/raw/<version>/trace.csv`

  Static validated trace              `data/static/clean/<version>/validated_trace.csv`

  Static forecast CSVs                `data/static/forecasts/<version>/forecast_*.csv`

  Streaming raw trace                 `data/streaming/raw/<version>/trace.csv`

  Streaming validated trace           `data/streaming/clean/<version>/validated_trace.csv`

  Streaming forecast CSVs             `data/streaming/forecasts/<version>/forecast_*.csv`

  Artifact manifest                   `data/<source>/<stage>/<version>/manifest.json`

  Latest version pointer              `data/<source>/<stage>/latest.json`

  Lineage ledger                      `data/lineage/lineage.jsonl`

  Active model pointer                `data/model_registry/active_model.json`

  Registered model config             `data/model_registry/models/<name>/<version>/config.yaml`

  Registered model weights (HMM)      `data/model_registry/models/<name>/<version>/transition_matrix.pkl`

  Model manifest (stage, metrics)     `data/model_registry/models/<name>/<version>/manifest.json`

  Model card                          `data/model_registry/models/<name>/<version>/model_card.json`

  Simulator results                   `experiments/results/`

  ILP outputs                         `experiments/ilp_outputs/`

  Inference outputs                   Defined by `mlops/inference/run_inference.py`
  ------------------------------------------------------------------------------------------

-------

## 9. Vary parameters to create new versions

Changing the input content or version-defining parameters creates a new
artifact version and, consequently, new downstream versions.

### Static example

``` bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate

python -m pipeline.run_pipeline \
  --source static \
  --num-requests 10000 \
  --seed 99
```

### Streaming example

``` bash
python -m pipeline.run_pipeline \
  --source streaming \
  --duration 300 \
  --speed 120
```

### Model example

Registering a model with different parameters (for example a different
ARIMA order) and promoting it changes the forecast version for the same
clean artifact:

``` bash
python -m model_registry.run register --name arima --config my_arima_v2.yaml --metrics mae=8.7
python -m model_registry.run card --name arima --version v2 \
  --description "ARIMA(1,2,1)" --intended-use "Workload forecasting" --limitations "Bursty traffic"
python -m model_registry.run promote --name arima --version v2 --stage Production
python -m pipeline.run_pipeline --source static --stages forecast
```

Re-running with identical inputs, parameters and active model may reuse
existing artifacts. This is the intended idempotent behavior.

To force a new clean version from the same raw artifact, update
`CLEANING_VERSION` in `etl_module/etl.py` (for example, from `"1"` to
`"2"`). Do this only when the cleaning logic or its intended version has
changed.

------------------------------------------------------------------------

## 10. End-to-end streaming verification

The following sequence checks the active model, runs the streaming
ingest → ETL → forecast pipeline, the SageServe simulator with the
matching streaming trace and forecasts, and the inference entry point.

Run this from a shell session. It assumes both virtual environments
already exist, their dependencies are installed, and a model has already
been registered and promoted (section 2).

``` bash
#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/repos/Sage-Serve"

echo "========================================"
echo "1. STREAMING MLOPS PIPELINE"
echo "========================================"

cd "$REPO/mlops"
source .venv/bin/activate

# Fails (non-zero exit) if no model has been promoted yet
python -m model_registry.run active
python -m model_registry.run list

python -m pipeline.run_pipeline \
  --source streaming \
  --duration 600 \
  --speed 300

python -m pipeline.lineage list --source streaming
python -m pipeline.lineage show streaming/forecasts/latest

STREAMING_RAW_DIR=$(python -c "
from pipeline.versioning import resolve_version
print(resolve_version('streaming', 'raw', 'latest'))
")

STREAMING_FORECAST_DIR=$(python -c "
from pipeline.versioning import resolve_version
print(resolve_version('streaming', 'forecasts', 'latest'))
")

STREAMING_TRACE="$STREAMING_RAW_DIR/trace.csv"

test -f "$STREAMING_TRACE"
test -d "$STREAMING_FORECAST_DIR"
test -n "$(find "$STREAMING_FORECAST_DIR" -maxdepth 1 -name 'forecast_*.csv' -print -quit)"

echo "Streaming trace: $STREAMING_TRACE"
echo "Streaming forecasts: $STREAMING_FORECAST_DIR"

deactivate

echo "========================================"
echo "2. SAGESERVE SIMULATOR — STREAMING"
echo "========================================"

cd "$REPO/SageServe-original"
source .venv/bin/activate

python run.py \
  trace.path="$STREAMING_TRACE" \
  global_arbiter.arima_traces="$STREAMING_FORECAST_DIR" \
  short_term_scaling=True \
  long_term_scaling=True

deactivate

echo "========================================"
echo "3. INFERENCE"
echo "========================================"

cd "$REPO"
source mlops/.venv/bin/activate

python3 mlops/inference/run_inference.py

echo "========================================"
echo "4. ARTIFACT CHECK"
echo "========================================"

find data/streaming -maxdepth 3 -type f | sort
find data/model_registry -maxdepth 4 -type f | sort
find experiments/results -maxdepth 2 -type f | sort
find experiments/ilp_outputs -maxdepth 2 -type f | sort

echo "End-to-end command sequence completed successfully."
```

Save the script as `verify_streaming.sh` and run:

``` bash
chmod +x verify_streaming.sh
./verify_streaming.sh
```

Because the script uses `set -euo pipefail`, it stops if a command
fails. The final success message is printed only if every command in the
sequence exits successfully.

------------------------------------------------------------------------

## 11. Quick verification commands

### Check the model registry

``` bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate

python -m model_registry.run list
python -m model_registry.run active
```

### Check streaming artifacts

``` bash
cd ~/repos/Sage-Serve

find data/streaming -maxdepth 3 -type f | sort
```

### Check lineage

``` bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate

python -m pipeline.lineage list --source streaming
python -m pipeline.lineage show streaming/forecasts/latest
```

### Check simulator outputs

``` bash
cd ~/repos/Sage-Serve

find experiments/results -maxdepth 2 -type f | sort
find experiments/ilp_outputs -maxdepth 2 -type f | sort
```

### Check the inference entry point

``` bash
cd ~/repos/Sage-Serve
source mlops/.venv/bin/activate

python3 mlops/inference/run_inference.py
```

A successful command exit confirms that the invoked process completed
without reporting an uncaught error. Validate the generated files and
application-level results as appropriate; process exit alone does not
establish model quality or correctness.

------------------------------------------------------------------------

## 12. Environment variable reference

  --------------------------------------------------------------------------
  Variable                Default                    Purpose
  ----------------------- -------------------------- -----------------------
  `MLOPS_DATA_ROOT`       `<parent of mlops>/data`   Overrides the location
                                                     where versioned MLOps
                                                     artifacts are written

  `MODEL_REGISTRY_DIR`    `<repo>/data/model_registry`  Overrides the location
                                                     of registry data
                                                     (`models/` and
                                                     `active_model.json`)
                                                     used by the registry CLI,
                                                     the factory and the
                                                     pipeline
  --------------------------------------------------------------------------

Set them in the shell before running the MLOps pipeline or registry CLI
if a non-default location is required. Ensure the simulator and
inference workflows are configured to read from the same data location
if you override `MLOPS_DATA_ROOT`.

------------------------------------------------------------------------

## 13. Common errors

  -----------------------------------------------------------------------------------------------------
  Error / symptom                                                 What to check
  --------------------------------------------------------------- -------------------------------------
  `FileNotFoundError: No versions yet for streaming/raw`          Run the streaming ingest stage first.

  `ModuleNotFoundError: pipeline`                                 Run `python -m` commands from inside
                                                                  `mlops/` with `mlops/.venv` active.

  `ModuleNotFoundError: model_registry` or `forecasting`          Run from inside `mlops/`, or set
                                                                  `PYTHONPATH=mlops` when running from
                                                                  the repository root.

  `source: no such file or directory: mlops/.venv/bin/activate`   Check the current directory. From the
                                                                  repository root use
                                                                  `source mlops/.venv/bin/activate`;
                                                                  from inside `mlops/` use
                                                                  `source .venv/bin/activate`.

  `cd: no such file or directory: mlops`                          You are probably already inside
                                                                  `mlops/`; do not append `mlops`
                                                                  again.

  `FileNotFoundError: Trace not found` in a legacy simulator run  Check whether
                                                                  `data/traces/random_trace.csv`
                                                                  exists. For the streaming workflow,
                                                                  use the versioned streaming trace
                                                                  path instead.

  Streaming simulator cannot find its trace                       Resolve `streaming/raw/latest` and
                                                                  pass its `trace.csv` to `trace.path`.

  Forecast directory is missing                                   Run the ETL and forecast stages for
                                                                  the same source, then resolve that
                                                                  source's latest forecast version.

  Simulator cannot find forecast CSVs                             Confirm `global_arbiter.arima_traces`
                                                                  points to the matching source's
                                                                  forecast directory.

  Forecast stage is skipped                                       Expected when the same parent
                                                                  artifact, parameters and active model
                                                                  already produced that version.

  `Model registry pointer not found at: .../active_model.json`    No model has been promoted to
                                                                  Production. Register, add a card, and
                                                                  promote one (section 2), then confirm
                                                                  with `model_registry.run active`.

  `Model directory does not exist`                                `active_model.json` points at a
                                                                  name/version that is not in
                                                                  `data/model_registry/models/`. Run
                                                                  `list`, then promote an existing
                                                                  version.

  `Governance Check Failed: ... model_card.json does not exist`   Run `card` for that version before
                                                                  promoting to Production.

  `Governance Check Failed: MAE/F1 ...`                           The manifest metrics do not meet the
                                                                  Production threshold (`mae <= 15.0` or
                                                                  `f1 >= 0.80`), or no metrics were
                                                                  recorded. Re-register with real
                                                                  `--metrics`.

  `Invalid or empty JSON file: <path>`                            A registry JSON file is empty or
                                                                  corrupt (usually hand-made). Delete
                                                                  that version folder and re-register it.

  `'<name>' needs --weights or --train-from`                      HMM-style models require a weights
                                                                  file or a clean trace to train from.

  `weights file missing or empty`                                 The `--weights` file does not exist or
                                                                  is zero bytes.

  `config model_info.type must be '<name>'`                       The config's `model_info.type` does
                                                                  not match `--name`.

  `config parameters missing: [...]`                              Add the listed keys to the config's
                                                                  `parameters` section.

  `No clean trace CSV in <dir> ...` when using `--train-from`     Pass the clean **version directory**,
                                                                  not the CSV file, and remember the
                                                                  path is relative to the current
                                                                  directory (`../data/...` from
                                                                  `mlops/`). The error lists the CSVs
                                                                  found and their columns.

  Inference module cannot find an input                           Check the paths and configuration
                                                                  read by
                                                                  `mlops/inference/run_inference.py`,
                                                                  and run it from the repository root
                                                                  as shown above.

  Different components read different data roots                  Check `MLOPS_DATA_ROOT` and
                                                                  `MODEL_REGISTRY_DIR` and ensure all
                                                                  stages use the intended shared
                                                                  locations.
  -----------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 14. Important workflow rules

1.  Run MLOps pipeline module commands and model registry commands from
    `mlops/` with `mlops/.venv` active.
2.  Run SageServe from `SageServe-original/` with
    `SageServe-original/.venv` active.
3.  Run `mlops/inference/run_inference.py` from the repository root with
    `mlops/.venv` active.
4.  Register, add a card to, and promote a model before the first
    forecast. The forecast stage uses whatever model is active; it never
    names a model itself.
5.  Never hand-create registry version folders. Use `register`, which
    validates inputs and assigns the version.
6.  Promote with real metrics. Only one model can be in Production at a
    time.
7.  For streaming simulation, pair the streaming raw trace with the
    streaming forecast directory. Do not substitute the legacy static
    trace.
8.  Resolve `latest` dynamically rather than hardcoding version IDs.
9.  A stage being skipped because its artifact already exists is normal
    version-cache behavior.
10. Verify simulator and inference outputs after execution; a successful
    process exit does not by itself validate output quality.