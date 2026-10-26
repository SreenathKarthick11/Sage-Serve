# SageServe Inference Report — Column Reference

`inference_report.csv` is the output of `mlops/inference/run_inference.py`.
Each **row** describes one **(model × simulation-config)** combination evaluated
against the raw per-request data stored in `experiments/results`.

The file is designed to be **comparable across runs**: tag each run with
`--forecast-model <name>` and the `forecast_model` column lets you line up
the same model/config pair from two different simulations side-by-side.

---

## Identity columns

| Column | Type | Description |
|---|---|---|
| `forecast_model` | string | Tag you supply via `--forecast-model` (e.g. `arima`, `prophet`). Use this to compare two runs that used different forecasting models. |
| `model` | string | The LLM/application model inside the simulation (`A`, `B`, `C`, `D`). |
| `run` | int | Experiment run index (matches the top-level numeric directory under `experiments/results/`). |
| `trace` | string | Workload trace name (e.g. `week-dp`). |
| `lts` | bool | Whether the long-term scaler was active in this config. |
| `sts` | bool | Whether the short-term scaler was active in this config. |
| `arbiter` | string | Arbiter variant used (e.g. `basic_arbiter`). |

---

## Volume metrics

| Column | Unit | Description |
|---|---|---|
| `n_requests` | count | Total requests observed across all simulation windows for this (model, config). |
| `n_windows` | count | Number of simulation time-windows (1 window ≈ 1 hour of simulated time). |
| `requests_per_window` | req / window | Mean request rate per window. Higher = richer training signal for a forecasting model. |

---

## Latency percentiles

All values are in **seconds**.  Derived from per-request measurements in
`experiments/results/.../detailed/<app>/<window>.csv`.

| Column | Description |
|---|---|
| `e2e_p50` | Median end-to-end response time. |
| `e2e_p95` | 95th-percentile end-to-end response time. |
| `e2e_p99` | 99th-percentile end-to-end response time.  The primary tail-latency indicator. |
| `ttft_p95` | 95th-percentile Time-to-First-Token. |
| `ttft_p99` | 99th-percentile TTFT.  Critical for streaming UX. |
| `tbt_p99` | 99th-percentile Time-Between-Tokens (inter-token gap). |

---

## SLO violation rates

Fraction of requests (0.0 – 1.0) that breached the configured threshold.

| Column | Threshold | Description |
|---|---|---|
| `viol_ttft` | TTFT > 10 s | Fraction of requests where the first token arrived after 10 seconds. |
| `viol_tbt`  | TBT > 0.2 s | Fraction of requests where any inter-token gap exceeded 200 ms. |

> **Interpretation**: `viol_ttft = 0.03` means 3 % of requests violated the
> TTFT SLO.  Zero is ideal.  Values above ~5 % indicate the configuration is
> oversubscribed under this forecasting model's scaling decisions.

---

## Variability metrics

| Column | Description |
|---|---|
| `cv_response` | Coefficient of variation (std / mean) of end-to-end response times. **Lower = more stable, easier to forecast.** A CV > 1 indicates highly bursty or bimodal latency. |
| `demand_cv` | CV of prompt-token sizes per request. **Lower = more regular demand patterns**, which benefit simple models like ARIMA. |
| `demand_mean` | Mean prompt-token count per request. |

---

## Cost metrics

Derived from VM scaling decisions in `experiments/results/.../arbiters/<region>/<window>.csv`.

| Column | Unit | Description |
|---|---|---|
| `prod_vm_h` | VM-hours | Total production (on-demand) VM-hours consumed across all windows. |
| `spot_vm_h` | VM-hours | Total spot/preemptible VM-hours consumed. |
| `mem_util_mean` | [0, 1] | Mean memory utilisation across all scaling decisions. |
| `vm_h_per_window` | VM-hours / window | Total VM-hours divided by number of windows. The primary **cost-efficiency indicator**. Lower = cheaper per unit time under this forecasting model. |

---

## FSS Sub-scores  *(each in [0, 1])*

These are the five raw components of the FSS before weighting.
They are useful for diagnosing **why** a config scored high or low.

| Column | Weight | What it captures |
|---|---|---|
| `score_slo` | 35 % | `1 − mean(viol_ttft, viol_tbt)`.  Perfect SLO compliance → 1.0. |
| `score_stability` | 25 % | `1 / (1 + cv_response)`.  Stable latency → close to 1.0.  Highly variable → close to 0. |
| `score_demand` | 20 % | `1 / (1 + demand_cv)`.  Regular demand → close to 1.0.  Bursty demand → close to 0. |
| `score_cost` | 10 % | Inverted min-max rank of `vm_h_per_window` within this report.  The cheapest config gets 1.0. |
| `score_throughput` | 10 % | Min-max rank of `requests_per_window`.  More requests = richer historical signal = 1.0. |

> **Note**: `score_cost` and `score_throughput` are **relative** to the
> configs in the same report.  Absolute cost comparisons across runs require
> manual normalisation.

---

## Forecast Suitability Score (FSS)

```
fss = (0.35 × norm(score_slo)
     + 0.25 × norm(score_stability)
     + 0.20 × norm(score_demand)
     + 0.10 × score_cost
     + 0.10 × score_throughput) × 100
```

| Column | Range | Description |
|---|---|---|
| `fss` | 0 – 100 | **The single discrete indicator.** Higher means the traffic pattern produced by the simulation is more forecastable, SLO-compliant, and cost-efficient under the current forecasting model's decisions. |

### Interpreting FSS

| FSS | What it signals |
|---|---|
| ≥ 80 | Traffic is stable, SLOs are met, demand is regular.  The forecasting model is doing well for this (model, config). |
| 50 – 79 | Moderate variability or partial SLO pressure.  The forecasting model works but may benefit from seasonal/trend awareness. |
| 30 – 49 | Bursty demand or notable SLO violations.  The forecasting model's scaling decisions are not fully covering load. |
| < 30 | Highly unstable or extremely sparse traffic.  The forecasting model is struggling with this configuration. |

> These are **guidelines**, not hard gates.  The real value of FSS is in
> **relative comparison**: if ARIMA yields FSS = 65 and Prophet yields
> FSS = 78 for the same (model D, lts=False) row, Prophet is the better
> choice for that traffic pattern.

---

## Plots generated

All plots are saved to `mlops/inference/plots/` by default.

| File | What to look for |
|---|---|
| `fss_bar.png` | Primary at-a-glance chart.  Long green bars = well-served by the current forecasting model. |
| `sub_scores.png` | Stacked bar breaking FSS into its five components.  Useful to see *why* a config scores high or low. |
| `latency_pct.png` | Grouped bar of e2e p50 / p95 / p99.  Watch for a large p95→p99 gap (tail inflation). |
| `slo_violations.png` | Dot-plot of TTFT and TBT violation rates.  Any dot far from zero warrants investigation. |
| `cv_scatter.png` | Forecastability space.  Points near the origin (low CV on both axes) are easiest to forecast, coloured by FSS. |

---

## Comparing two forecasting models

Run the inference script twice with different `--forecast-model` tags and
different `--out` files, then concatenate:

```python
import pandas as pd

arima   = pd.read_csv("mlops/inference/inference_arima.csv")
prophet = pd.read_csv("mlops/inference/inference_prophet.csv")

combined = pd.concat([arima, prophet], ignore_index=True)

# Compare FSS for the same (model, trace, lts, sts, arbiter):
pivot = combined.pivot_table(
    index=["model", "trace", "lts", "sts", "arbiter"],
    columns="forecast_model",
    values="fss",
)
print(pivot)
```

The row with the **higher FSS** is the forecasting model better suited to
that traffic pattern.  When a model registry is available, this pivot can be
computed automatically over all registered models.
