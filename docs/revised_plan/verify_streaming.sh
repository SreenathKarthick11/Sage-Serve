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