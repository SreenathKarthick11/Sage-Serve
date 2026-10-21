"""
metrics.py  –  Computes per-model inference metrics from SageServe results.

``compute_metrics(results_root)`` returns one row per (model, config) with:

  Latency percentiles   e2e_p50/p95/p99, ttft_p95/p99, tbt_p99
  SLO compliance        viol_ttft, viol_tbt  (fraction of requests violating)
  Variability           cv_response, demand_cv  (coefficient of variation)
  Cost                  prod_vm_h, spot_vm_h, mem_util_mean, vm_h_per_window
  Volume                n_requests, requests_per_window, n_windows

  --- Forecast Suitability Score (FSS) sub-scores, each in [0, 1] ---
  score_slo             SLO compliance component
  score_stability       Latency stability component
  score_demand          Demand regularity component
  score_cost            Cost efficiency component (relative to other configs)
  score_throughput      Training-signal richness component

  fss                   Weighted composite in [0, 100].
                        Higher = more forecastable, more stable, lower cost.
                        Use this as the single discrete value to compare
                        across different forecasting models / simulation runs.

FSS weights (defaults, overridable via compute_metrics):
  slo: 0.35, stability: 0.25, demand: 0.20, cost: 0.10, throughput: 0.10
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import load_detailed, load_cost

# ── SLO thresholds (seconds) ────────────────────────────────────────────────
SLO_TTFT = 10.0
SLO_TBT  = 0.20

# ── FSS default weights ─────────────────────────────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "slo":        0.35,
    "stability":  0.25,
    "demand":     0.20,
    "cost":       0.10,
    "throughput": 0.10,
}

_CFG_KEYS = ["model", "run", "trace", "lts", "sts", "arbiter"]


# ---------------------------------------------------------------------------
# VM-hour aggregation
# ---------------------------------------------------------------------------

def _vm_hours(cost_df: pd.DataFrame) -> pd.DataFrame:
    if cost_df.empty:
        return pd.DataFrame()

    group_keys = ["model", "run", "trace", "lts", "sts", "arbiter", "region", "window"]
    rows = []
    for keys, g in cost_df.groupby(group_keys):
        g = g.sort_values("timestamp")
        dt = g["timestamp"].diff().shift(-1).fillna(g["timestamp"].diff().median())
        row = dict(zip(group_keys, keys))
        row["prod_vm_h"] = float((g["prod"] * dt).sum() / 3600)
        row["spot_vm_h"] = float((g["spot"] * dt).sum() / 3600)
        row["mem_util"]  = float(g["memory_util"].mean())
        rows.append(row)

    cost_agg = pd.DataFrame(rows)
    return (
        cost_agg
        .groupby(_CFG_KEYS)
        .agg(
            prod_vm_h=("prod_vm_h", "sum"),
            spot_vm_h=("spot_vm_h", "sum"),
            mem_util_mean=("mem_util", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Latency / demand metrics
# ---------------------------------------------------------------------------

def _latency_metrics(det: pd.DataFrame) -> pd.DataFrame:
    pct = lambda p: (lambda s: float(s.quantile(p)))  # noqa: E731

    agg = (
        det.groupby(_CFG_KEYS)
        .agg(
            n_requests=("response_times", "size"),
            e2e_p50=("response_times", pct(0.50)),
            e2e_p95=("response_times", pct(0.95)),
            e2e_p99=("response_times", pct(0.99)),
            ttft_p95=("ttft_times",    pct(0.95)),
            ttft_p99=("ttft_times",    pct(0.99)),
            tbt_p99=("tbt_times",      pct(0.99)),
            response_mean=("response_times", "mean"),
            response_std=("response_times",  "std"),
            demand_mean=("prompt_sizes",     "mean"),
            demand_std=("prompt_sizes",      "std"),
        )
        .reset_index()
    )

    agg["cv_response"] = (
        agg["response_std"] / agg["response_mean"].replace(0, np.nan)
    ).fillna(0.0)

    agg["demand_cv"] = (
        agg["demand_std"] / agg["demand_mean"].replace(0, np.nan)
    ).fillna(0.0)

    # SLO violation fractions
    for col, slo, short in [
        ("ttft_times", SLO_TTFT, "ttft"),
        ("tbt_times",  SLO_TBT,  "tbt"),
    ]:
        viol = (
            det.groupby(_CFG_KEYS)[col]
            .apply(lambda s, lim=slo: float((s > lim).mean()))
            .reset_index(name=f"viol_{short}")
        )
        agg = agg.merge(viol, on=_CFG_KEYS, how="left")

    n_windows = (
        det.groupby(_CFG_KEYS)["window"]
        .nunique()
        .reset_index(name="n_windows")
    )
    agg = agg.merge(n_windows, on=_CFG_KEYS, how="left")
    agg["requests_per_window"] = agg["n_requests"] / agg["n_windows"].clip(lower=1)

    return agg.drop(columns=["response_mean", "response_std", "demand_std"])


# ---------------------------------------------------------------------------
# FSS computation – returns sub-scores + composite
# ---------------------------------------------------------------------------

def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _compute_fss(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """
    Compute per-row FSS sub-scores (each in [0,1]) and the composite FSS [0,100].
    Returns the same DataFrame with extra columns appended.
    """
    out = df.copy()

    # score_slo: higher SLO compliance = higher score
    viol_mean = (
        out.get("viol_ttft", pd.Series(0.0, index=out.index)) +
        out.get("viol_tbt",  pd.Series(0.0, index=out.index))
    ) / 2
    out["score_slo"] = (1.0 - viol_mean.clip(0.0, 1.0))

    # score_stability: lower response-time CV = more stable = more forecastable
    out["score_stability"] = 1.0 / (1.0 + out["cv_response"].clip(lower=0))

    # score_demand: lower demand CV = more regular traffic patterns
    out["score_demand"] = 1.0 / (1.0 + out["demand_cv"].clip(lower=0))

    # score_cost: lower vm_h_per_window = better (inverted min-max rank)
    if "vm_h_per_window" in out.columns:
        out["score_cost"] = 1.0 - _minmax(out["vm_h_per_window"])
    else:
        out["score_cost"] = 0.5

    # score_throughput: more requests per window = richer training signal
    out["score_throughput"] = _minmax(out["requests_per_window"])

    # FSS: weighted sum of min-max normalised sub-scores * 100
    out["fss"] = (
        weights["slo"]        * _minmax(out["score_slo"])       +
        weights["stability"]  * _minmax(out["score_stability"])  +
        weights["demand"]     * _minmax(out["score_demand"])     +
        weights["cost"]       * out["score_cost"]                +
        weights["throughput"] * out["score_throughput"]
    ) * 100.0

    out["fss"] = out["fss"].clip(0.0, 100.0)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    results_root: str,
    fss_weights: dict[str, float] | None = None,
    forecast_model_tag: str = "baseline",
) -> pd.DataFrame:
    """
    Compute inference metrics + FSS for every (model, config) in results_root.

    Parameters
    ----------
    results_root : str | Path
        Path to ``experiments/results`` directory.
    fss_weights : dict, optional
        Override FSS sub-score weights.
        Keys: {'slo', 'stability', 'demand', 'cost', 'throughput'}.
    forecast_model_tag : str
        A label for the forecasting model that generated these results
        (e.g. "arima", "prophet").  Stored in the ``forecast_model`` column
        for future cross-run comparison.

    Returns
    -------
    pd.DataFrame
        Sorted by FSS descending.  Columns include all raw metrics,
        sub-scores (score_slo, score_stability, score_demand, score_cost,
        score_throughput), and the composite ``fss``.
    """
    weights = {**DEFAULT_WEIGHTS, **(fss_weights or {})}

    det = load_detailed(results_root)
    if det.empty:
        raise ValueError(
            f"No valid request data found under: {results_root!r}. "
            "Check that the path points to the experiments/results directory."
        )

    cost = load_cost(results_root)

    metrics = _latency_metrics(det)

    if not cost.empty:
        vm = _vm_hours(cost)
        n_win = metrics.set_index(_CFG_KEYS)["n_windows"]
        vm_idx = pd.MultiIndex.from_frame(vm[_CFG_KEYS])
        vm["vm_h_per_window"] = (
            (vm["prod_vm_h"] + vm["spot_vm_h"])
            / n_win.reindex(vm_idx).values.clip(1)
        )
        metrics = metrics.merge(vm, on=_CFG_KEYS, how="left")

    metrics = _compute_fss(metrics, weights)

    # tag for future model-registry comparison
    metrics.insert(0, "forecast_model", forecast_model_tag)

    metrics = metrics.sort_values("fss", ascending=False).reset_index(drop=True)

    return metrics
