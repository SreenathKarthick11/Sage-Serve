# Running SageServe

This guide covers the complete workflow for generating a workload trace, generating ARIMA forecasts, and running the SageServe simulator with and without forecasting.

## 1. Generate Random Trace

From the repository root:

```bash
cd ~/repos/Sage-Serve
cd mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_random_trace.py
```

> IF file not found error : Create dir `data/traces` in root.

This generates:

```text
data/traces/random_trace.csv
```

---

## 2. Generate ARIMA Forecasts

Keep the MLOps environment active:

```bash
python forecasting/arima_forecaster.py
```

Forecasts are generated under:

```text
data/traces/forecasts/
```

---


Installed `pyenv`, then: (check chatgpt)

```bash
pyenv install 3.11.13
```

Inside the SageServe repository:

```bash
pyenv local 3.11.13
```

## 3. Run SageServe Without Forecasting

Switch to the SageServe environment:

```bash
deactivate
cd SageServe-original
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install hydra
```

Run the simulator:

```bash
python run.py \
  trace.path="$PWD/../data/traces/random_trace.csv" \
  short_term_scaling=True \
  long_term_scaling=False
```

---

## 4. Run SageServe With Forecasting

### Using the configured forecast path

```bash
python run.py \
  trace.path="$PWD/../data/traces/random_trace.csv" \
  short_term_scaling=True \
  long_term_scaling=True
```

### Explicitly specifying the forecast path

```bash
python run.py \
  trace.path="$PWD/../data/traces/random_trace.csv" \
  global_arbiter.arima_traces="$PWD/../data/traces/forecasts" \
  short_term_scaling=True \
  long_term_scaling=True
```

---

## 5. Output Locations

Simulation results are stored under:

```text
experiments/results/
```

ILP outputs are stored under:

```text
experiments/ilp_outputs/
```

Forecast artifacts are stored under:

```text
data/traces/forecasts/
```

---




