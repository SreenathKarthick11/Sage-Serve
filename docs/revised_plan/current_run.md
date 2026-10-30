# Running SageServe --- Complete Guide

> **Current state:** SageServe supports a versioned MLOps pipeline
> (ingest → ETL → forecast), with content-addressed immutable artifacts
> and an append-only lineage ledger. The workflow below covers setup,
> static and streaming data generation, validation, ARIMA forecasting,
> simulator execution, and the inference entry point.

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
│   │   └── arima_forecaster.py
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

------------------------------------------------------------------------

## Versioning model

Every artifact is stored in an immutable directory:

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
                                      parameters
  -----------------------------------------------------------------------

-   Same inputs and parameters produce the same version ID. Existing
    artifacts are reused/skipped.
-   Each artifact directory contains a `manifest.json` with metadata
    such as its ID, parent artifact(s), parameters, file hashes,
    timestamp, and Git commit.
-   `data/<source>/<stage>/latest.json` points to the latest version for
    that source and stage.
-   `data/lineage/lineage.jsonl` is an append-only ledger containing one
    record per artifact.

The versioning model applies independently to `static` and `streaming`
sources.

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

The simulator uses its own environment:

``` bash
cd ~/repos/Sage-Serve/SageServe-original
source .venv/bin/activate
pip install -r requirements.txt
```

Install dependencies only when setting up the corresponding environment
or when its requirements change.

------------------------------------------------------------------------

## 1. Run the full MLOps pipeline

Run all `python -m` pipeline commands from inside `mlops/`, with
`mlops/.venv` active.

``` bash
cd ~/repos/Sage-Serve/mlops
source .venv/bin/activate
```

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

[3/3] forecast
      └── data/<source>/forecasts/<version>/
          ├── forecast_A_region0.csv
          ├── ...
          ├── forecast_D_region2.csv
          └── manifest.json
```

With four models (`A`, `B`, `C`, `D`) and three regions (`0`, `1`, `2`),
a successful forecast stage produces 12 forecast CSV files.

The orchestrator prints the artifact lineage after execution. If an
artifact with the same derived version already exists, the corresponding
stage is skipped. This is expected behavior, not a failure.

------------------------------------------------------------------------

## 2. Run individual pipeline stages

Run these commands from `mlops/` with the MLOps environment active.

``` bash
# Ingest only: generate a raw artifact
python -m pipeline.run_pipeline --source static --stages ingest

# ETL only: process the latest raw artifact
python -m pipeline.run_pipeline --source static --stages etl

# Forecast only: process the latest clean artifact
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

# Forecast using the latest clean artifact
python -m forecasting.arima_forecaster --source streaming

# Forecast using a specific clean artifact
python -m forecasting.arima_forecaster \
  --source streaming \
  --input-version <clean-version>
```

The orchestrator is the recommended way to run the complete pipeline
because it coordinates stages, version resolution, and lineage
reporting.

------------------------------------------------------------------------

## 3. Inspect the lineage ledger

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
and stage metadata.

------------------------------------------------------------------------

## 4. Resolve the latest artifact paths

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

## 5. Run the SageServe simulator

The simulator has a separate virtual environment. The simulator must
receive a trace and, when long-term scaling is enabled, the forecast
directory generated from the same source.

### 5a. Run with static data

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
shown in the simulator run configuration.

### 5b. Run with streaming data

This is the end-to-end streaming workflow. It uses:

-   The raw streaming trace created by the streaming ingest stage.
-   The streaming ARIMA forecast directory created from that streaming
    trace.
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

------------------------------------------------------------------------

## 6. Run the inference module

The inference entry point is:

``` text
mlops/inference/run_inference.py
```

Run it from the **repository root** using the MLOps virtual environment.
Run it after the simulator workflow when performing the complete
end-to-end verification.

``` bash
# Leave the SageServe simulator environment
deactivate

# Return to the repository root
cd ~/repos/Sage-Serve

# Activate the MLOps environment
source mlops/.venv/bin/activate

# Run inference
python3 mlops/inference/run_inference.py
```

The command above is the inference invocation used in this workflow.
Keep the working directory at the repository root when invoking it, so
any repository-relative paths used by the module resolve as expected.

The exact inference inputs, model/configuration selection, and output
locations are defined by the implementation of
`mlops/inference/run_inference.py`; check that module if you need to
change those settings. Do not assume that the inference module
automatically selects a particular raw, clean, forecast, or simulator
artifact unless its implementation explicitly does so.

------------------------------------------------------------------------

## 7. Output locations

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

  Simulator results                   `experiments/results/`

  ILP outputs                         `experiments/ilp_outputs/`

  Inference outputs                   Defined by `mlops/inference/run_inference.py`
  ------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 8. Vary parameters to create new versions

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

Re-running with identical inputs and parameters may reuse existing
artifacts. This is the intended idempotent behavior.

To force a new clean version from the same raw artifact, update
`CLEANING_VERSION` in `etl_module/etl.py` (for example, from `"1"` to
`"2"`). Do this only when the cleaning logic or its intended version has
changed.

------------------------------------------------------------------------

## 9. End-to-end streaming verification

The following sequence runs the streaming ingest → ETL → forecast
pipeline, the SageServe simulator with the matching streaming trace and
forecasts, and the inference entry point.

Run this from a shell session. It assumes both virtual environments
already exist and their dependencies are installed.

``` bash
#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/repos/Sage-Serve"

echo "========================================"
echo "1. STREAMING MLOPS PIPELINE"
echo "========================================"

cd "$REPO/mlops"
source .venv/bin/activate

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

## 10. Quick verification commands

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

## 11. Environment variable reference

  --------------------------------------------------------------------------
  Variable                Default                    Purpose
  ----------------------- -------------------------- -----------------------
  `MLOPS_DATA_ROOT`       `<parent of mlops>/data`   Overrides the location
                                                     where versioned MLOps
                                                     artifacts are written

  --------------------------------------------------------------------------

Set it in the shell before running the MLOps pipeline if a non-default
data root is required. Ensure the simulator and inference workflows are
configured to read from the same data location if you override it.

------------------------------------------------------------------------

## 12. Common errors

  -----------------------------------------------------------------------------------------------------
  Error / symptom                                                 What to check
  --------------------------------------------------------------- -------------------------------------
  `FileNotFoundError: No versions yet for streaming/raw`          Run the streaming ingest stage first.

  `ModuleNotFoundError: pipeline`                                 Run `python -m` commands from inside
                                                                  `mlops/` with `mlops/.venv` active.

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
                                                                  artifact and parameters already
                                                                  produced that version.

  Inference module cannot find an input                           Check the paths and configuration
                                                                  read by
                                                                  `mlops/inference/run_inference.py`,
                                                                  and run it from the repository root
                                                                  as shown above.

  Different components read different data roots                  Check `MLOPS_DATA_ROOT` and ensure
                                                                  all stages use the intended shared
                                                                  artifact location.
  -----------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 13. Important workflow rules

1.  Run MLOps pipeline module commands from `mlops/` with `mlops/.venv`
    active.
2.  Run SageServe from `SageServe-original/` with
    `SageServe-original/.venv` active.
3.  Run `mlops/inference/run_inference.py` from the repository root with
    `mlops/.venv` active.
4.  For streaming simulation, pair the streaming raw trace with the
    streaming forecast directory. Do not substitute the legacy static
    trace.
5.  Resolve `latest` dynamically rather than hardcoding version IDs.
6.  A stage being skipped because its artifact already exists is normal
    version-cache behavior.
7.  Verify simulator and inference outputs after execution; a successful
    process exit does not by itself validate output quality.
