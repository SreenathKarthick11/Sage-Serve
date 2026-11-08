"""Latency, completion volume and observed resource metrics for simulator exports.

Raw measurements are primary. rfws is a relative score within a shared model,
trace and window inventory; it is neither forecast accuracy nor an absolute SLO.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .loader import DETAIL_COLUMNS, InputError, Results, _parse_application, load_results

SLO_TTFT = 10.0
SLO_TBT = 0.20
DEFAULT_WEIGHTS = {"slo": .35, "stability": .25, "demand": .20, "cost": .10, "throughput": .10}
_CFG_KEYS = ["experiment_id", "model"]
SCORING_METHOD = "relative_model_trace_window_minmax_v2"
UPSTREAM_LIMITATIONS = [
    "The simulator may omit applications created after initial setup from detailed exports.",
    "The simulator does not flush the final partial window; exported completions are not all input requests.",
    "Arbiter memory_util may be attributed to the triggering model; it is a snapshot diagnostic, not verified per-model utilization.",
    "Detailed exports contain no request IDs, arrival/completion timestamps, per-request SLA or per-token gaps.",
    "A forecast_model tag is user supplied and does not verify which forecasts the simulator used.",
]


@dataclass
class Report:
    metrics: pd.DataFrame
    windows: pd.DataFrame
    metadata: dict


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def normalize_weights(overrides=None):
    supplied = overrides or {}
    unknown = set(supplied) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown rfws weights: {sorted(unknown)}")
    weights = {**DEFAULT_WEIGHTS, **supplied}
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"rfws weight {name} must be finite and nonnegative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one rfws weight must be positive")
    return {key: value / total for key, value in weights.items()}


def _cv(values):
    values = values.dropna()
    if len(values) < 2 or values.mean() <= 0:
        return np.nan
    return float(values.std(ddof=1) / values.mean())


def _fraction_above(values, limit):
    values = values.dropna()
    return float(values.gt(limit).mean()) if len(values) else np.nan


def _quantile(values, quantile):
    return float(values.quantile(quantile)) if values.notna().any() else np.nan


def _window_metrics(results: Results):
    rows = []
    det = results.detailed
    counts = det.groupby(["experiment_id", "model", "window"]).size().to_dict() if not det.empty else {}
    for exp in results.experiments:
        for model in sorted(exp.model_regions):
            apps = {a for a, (_, m) in exp.applications.items() if m == model}
            regions_with_apps = {r for r, m in exp.applications.values() if m == model}
            for window in sorted(exp.windows):
                expected = {(a, window) for a in apps}
                unexported = {a for a in exp.unexported_applications.get(window, set())
                              if _parse_application(a, exp)[1] == model}
                complete = (bool(apps) and expected <= exp.detail_files and
                            not (expected & exp.invalid_files) and
                            exp.model_regions[model] <= regions_with_apps and not unexported and not exp.memory_audit_failed)
                if not complete:
                    results.diagnostics.add("incomplete_request_export", exp.path,
                                            f"Model {model}, window {window}: missing/invalid application exports or measurements.")
                observed = counts.get((exp.meta["experiment_id"], model, window), 0)
                start, end = exp.bounds.get(window, (np.nan, np.nan))
                rows.append({**exp.meta, "model": model, "window": window,
                             "observed_requests": observed,
                             "completed_requests": observed if complete else np.nan,
                             "request_data_complete": complete,
                             "n_unexported_live_applications": len(unexported),
                             "start_seconds": start, "end_seconds": end,
                             "duration_seconds": end-start})
    return pd.DataFrame(rows)


def _latency_row(g, ttft_threshold, tbt_threshold):
    row = {"n_requests": len(g)}
    for column in DETAIL_COLUMNS:
        row[f"n_valid_{column}"] = int(g[column].notna().sum())
    for source, prefix, quantiles in [
        ("response_times", "e2e", [.5, .95, .99]),
        ("queue_times", "queue", [.5, .95, .99]),
        ("ttft_times", "ttft", [.5, .95, .99]),
        ("tbt_times", "avg_tbt", [.5, .95, .99]),
    ]:
        for q in quantiles:
            row[f"{prefix}_p{int(q*100)}"] = _quantile(g[source], q)
    row["cv_response"] = _cv(g.response_times)
    row["prompt_size_cv"] = _cv(g.prompt_sizes)
    row["prompt_size_mean"] = float(g.prompt_sizes.mean())
    row["output_tokens_mean"] = float(g.token_sizes.mean())
    row["queue_fraction_mean"] = float((g.queue_times / g.response_times.replace(0, np.nan)).mean())
    row["viol_ttft"] = _fraction_above(g.ttft_times, ttft_threshold)
    row["viol_avg_tbt"] = _fraction_above(g.tbt_times, tbt_threshold)
    pairs = g[["ttft_times", "tbt_times"]].dropna()
    row["n_valid_benchmark_pairs"] = len(pairs)
    row["benchmark_joint_pass_fraction"] = (
        float((pairs.ttft_times.le(ttft_threshold) & pairs.tbt_times.le(tbt_threshold)).mean())
        if len(pairs) else np.nan)
    row["ttft_threshold_seconds"] = ttft_threshold
    row["avg_tbt_threshold_seconds"] = tbt_threshold
    row["latency_sample_status"] = ("no_samples" if len(g) == 0 else "single_sample" if len(g) == 1
                                    else "small_sample" if len(g) < 100 else "ok")
    # Backward-compatible aliases; their precise meanings are documented.
    row["tbt_p99"] = row["avg_tbt_p99"]
    row["viol_tbt"] = row["viol_avg_tbt"]
    row["demand_cv"] = row["prompt_size_cv"]
    row["demand_mean"] = row["prompt_size_mean"]
    return row


def _resource_row(cost, exp, model):
    """Integrate state(t) until the next observation, across file boundaries.

    Without supplied boundaries, only [first snapshot, last snapshot] is
    observed. Never invent a median tail or extrapolate a single snapshot.
    """
    known_bounds = bool(exp.windows) and set(exp.bounds) == exp.windows
    duration = sum(end-start for start, end in exp.bounds.values()) if known_bounds else np.nan
    totals = np.zeros(2)
    covered = 0.0
    n_intervals = 0
    snapshots = 0
    all_valid = True
    memories = []
    for region in sorted(exp.model_regions[model]):
        g = cost[(cost.model == model) & (cost.region == region)].copy() if not cost.empty else pd.DataFrame()
        if g.empty:
            all_valid = False
            continue
        snapshots += len(g)
        memories.extend(g.memory_util.dropna().tolist())
        # Stable ordering preserves the final state when several updates share a time.
        g = g.sort_values(["timestamp", "window"], kind="stable")
        all_valid &= g[["timestamp", "prod", "spot"]].notna().all().all()
        g = g[g.timestamp.notna()].drop_duplicates("timestamp", keep="last")
        times = g.timestamp.to_numpy()
        states = g[["prod", "spot"]].to_numpy()
        window_ids = g.window.to_numpy()
        intervals = list(exp.bounds.values()) if known_bounds else (
            [(times[0], times[-1])] if len(times) >= 2 else [])
        for i, (start, state) in enumerate(zip(times, states)):
            end = times[i+1] if i+1 < len(times) else (max(b[1] for b in intervals) if known_bounds else start)
            if not np.isfinite(state).all() or end <= start:
                continue
            # A missing/corrupt intervening export cannot establish a continuous state.
            last_window = int(window_ids[i+1]) if i+1 < len(times) else max(exp.windows)
            paths = [exp.path / "arbiters" / region / f"{w}.csv"
                     for w in range(int(window_ids[i]), last_window+1)]
            if not all(p.exists() and p not in exp.invalid_cost_files for p in paths):
                all_valid = False
                continue
            for left, right in intervals:
                dt = max(0., min(end, right)-max(start, left))
                if dt:
                    totals += state * dt / 3600.
                    covered += dt
                    n_intervals += 1
    expected_seconds = duration * len(exp.model_regions[model])
    coverage = covered / expected_seconds if np.isfinite(expected_seconds) and expected_seconds > 0 else np.nan
    complete = bool(known_bounds and np.isclose(coverage, 1., rtol=1e-8) and all_valid)
    status = ("complete" if complete else "missing" if not snapshots else
              "insufficient_snapshots" if not n_intervals else
              "partial" if known_bounds else "observed_intervals_only")
    return {"prod_vm_h": float(totals[0]) if n_intervals else np.nan,
            "spot_vm_h": float(totals[1]) if n_intervals else np.nan,
            "resource_observed_region_seconds": covered,
            "resource_time_coverage": coverage,
            "resource_coverage_status": status, "resource_snapshots": snapshots,
            "memory_util_snapshot_mean": float(np.mean(memories)) if memories else np.nan,
            "mem_util_mean": float(np.mean(memories)) if memories else np.nan}


def _minmax(s):
    valid = s.dropna()
    result = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.empty:
        return result
    lo, hi = valid.min(), valid.max()
    result.loc[valid.index] = .5 if hi == lo else (valid-lo)/(hi-lo)
    return result


def _compute_rfws(df, weights):
    """Score eligible peers together; return rows plus explicit references."""
    out = df.copy()
    out["score_slo"] = 1-(out.viol_ttft+out.viol_avg_tbt)/2
    out["score_stability"] = 1/(1+out.cv_response)
    out["score_demand"] = 1/(1+out.prompt_size_cv)
    for col in ("score_cost", "score_throughput", "rfws"):
        out[col] = np.nan
    for name in weights:
        out[f"contribution_{name}"] = np.nan
        out[f"weight_{name}"] = weights[name]
    out["scoring_method"] = SCORING_METHOD
    out["rfws_cohort_id"] = ""
    out["rfws_status"] = "insufficient_comparable_peers"
    references = []
    for (model, trace, signature), group in out.groupby(["model", "trace", "window_signature"], dropna=False):
        eligible = pd.Series(True, index=group.index)
        reasons = {idx: [] for idx in group.index}
        for idx, row in group.iterrows():
            if not row.request_data_complete:
                reasons[idx].append("incomplete_request_data")
            if row.input_error_count:
                reasons[idx].append("partial_input_errors")
            if row.n_requests < 2:
                reasons[idx].append("insufficient_samples")
            if weights["cost"] and row.resource_coverage_status != "complete":
                reasons[idx].append("incomplete_resource_duration")
            for name, column in [("slo", "score_slo"), ("stability", "score_stability"),
                                 ("demand", "score_demand"), ("throughput", "requests_per_window")]:
                if weights[name] and not np.isfinite(row[column]):
                    reasons[idx].append(f"unknown_{name}")
            if reasons[idx]:
                eligible.loc[idx] = False
                out.loc[idx, "rfws_status"] = ";".join(reasons[idx])
        peers = group[eligible]
        reference = {"method": SCORING_METHOD, "model": model, "trace": trace, "window_signature": signature,
                     "thresholds_seconds": {"ttft": group.ttft_threshold_seconds.iloc[0],
                                            "request_average_tbt": group.avg_tbt_threshold_seconds.iloc[0]},
                     "weights": weights,
                     "members": peers[["experiment_id", "viol_ttft", "viol_avg_tbt", "cv_response",
                                      "prompt_size_cv", "vm_h_per_window", "requests_per_window"]].to_dict("records"),
                     "normalization": {}}
        inputs = {"slo": peers.score_slo, "stability": peers.score_stability,
                  "demand": peers.score_demand, "cost": peers.vm_h_per_window,
                  "throughput": peers.requests_per_window}
        for name, values in inputs.items():
            reference["normalization"][name] = {
                "min": float(values.min()) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None}
        # JSON null, not nonstandard NaN, keeps metadata portable.
        reference = _json_safe(reference)
        cohort_id = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()[:16]
        out.loc[group.index, "rfws_cohort_id"] = cohort_id
        reference["cohort_id"] = cohort_id
        references.append(reference)
        if len(peers) < 2:
            for idx in group.index:
                if out.loc[idx, "rfws_status"] != "insufficient_comparable_peers":
                    out.loc[idx, "rfws_status"] += ";insufficient_comparable_peers"
            continue
        total = pd.Series(0., index=peers.index)
        for name, values in inputs.items():
            norm = _minmax(values)
            if name == "cost":
                norm = 1-norm
                out.loc[peers.index, "score_cost"] = norm
            if name == "throughput":
                out.loc[peers.index, "score_throughput"] = norm
            contribution = norm*weights[name]*100 if weights[name] else pd.Series(0., index=peers.index)
            out.loc[peers.index, f"contribution_{name}"] = contribution
            total += contribution
        out.loc[peers.index, "rfws"] = total
        out.loc[peers.index, "rfws_status"] = "relative_to_cohort"
    return out, references


def build_report(results_root, rfws_weights=None, forecast_model_tag="baseline", *,
                 allow_partial=False, window_boundaries=None,
                 ttft_threshold=SLO_TTFT, tbt_threshold=SLO_TBT):
    weights = normalize_weights(rfws_weights)
    for name, value in (("TTFT", ttft_threshold), ("average TBT", tbt_threshold)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} threshold must be finite and positive")
    results = load_results(results_root, allow_partial=allow_partial, window_boundaries=window_boundaries)
    if not any(exp.detail_files for exp in results.experiments):
        raise InputError("No readable detailed request exports were found")
    windows = _window_metrics(results)
    if windows.empty:
        raise InputError("No model/window inventory was found")
    rows = []
    for exp in results.experiments:
        det = results.detailed
        selected = det[det.experiment_id == exp.meta["experiment_id"]] if not det.empty else pd.DataFrame(columns=DETAIL_COLUMNS+["model"])
        cost = results.cost
        selected_cost = cost[cost.experiment_id == exp.meta["experiment_id"]] if not cost.empty else cost
        for model in sorted(exp.model_regions):
            g = selected[selected.model == model]
            w = windows[(windows.experiment_id == exp.meta["experiment_id"]) & (windows.model == model)]
            row = {**exp.meta, "model": model, **_latency_row(g, ttft_threshold, tbt_threshold)}
            row["input_error_count"] = sum(i["severity"] == "error" and
                                           (i["path"] == str(exp.path) or i["path"].startswith(str(exp.path)+"/"))
                                           for i in results.diagnostics.issues)
            row["n_windows"] = len(w)
            row["requests_per_window"] = len(g)/len(w) if len(w) else np.nan
            row["n_quiet_windows"] = int(w.completed_requests.eq(0).sum())
            row["n_observed_empty_windows"] = int(w.observed_requests.eq(0).sum())
            row["n_unexported_live_applications"] = len({a for names in exp.unexported_applications.values()
                                                        for a in names if _parse_application(a, exp)[1] == model})
            row["request_data_complete"] = bool(w.request_data_complete.all()) and exp.windows == set(range(max(exp.windows)+1))
            row["completion_window_cv"] = _cv(w.completed_requests) if row["request_data_complete"] else np.nan
            row["window_signature"] = json.dumps([[int(x.window), x.start_seconds if pd.notna(x.start_seconds) else None,
                                                  x.end_seconds if pd.notna(x.end_seconds) else None]
                                                 for x in w.itertuples()], separators=(",", ":"))
            row["export_duration_seconds"] = float(w.duration_seconds.sum()) if w.duration_seconds.notna().all() else np.nan
            row["completed_requests_per_second"] = (len(g)/row["export_duration_seconds"]
                                                    if row["export_duration_seconds"] > 0 and row["request_data_complete"] else np.nan)
            row.update(_resource_row(selected_cost, exp, model))
            hours = row["prod_vm_h"] + row["spot_vm_h"]
            row["vm_h_per_window"] = hours/len(w) if len(w) and row["resource_coverage_status"] == "complete" else np.nan
            row["vm_h_per_1000_completions"] = (hours/len(g)*1000 if len(g) and
                                               row["resource_coverage_status"] == "complete" and row["request_data_complete"] else np.nan)
            rows.append(row)
    metrics, references = _compute_rfws(pd.DataFrame(rows), weights)
    metrics.insert(0, "forecast_model", forecast_model_tag)
    metrics = metrics.sort_values(["rfws", "experiment_id", "model"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
    metadata = {"schema_version": 2, "scoring_method": SCORING_METHOD, "normalized_weights": weights,
                "benchmark_thresholds_seconds": {"ttft": ttft_threshold, "request_average_tbt": tbt_threshold},
                "forecast_model_label": forecast_model_tag, "forecast_provenance_verified": False,
                "allow_partial": allow_partial, "diagnostics": results.diagnostics.issues,
                "input_files": results.diagnostics.files,
                "experiments": [{**e.meta, "configuration": e.config, "windows": sorted(e.windows),
                                 "unexported_live_applications": {str(w): sorted(names) for w, names in e.unexported_applications.items()},
                                 "explicit_boundaries": {str(w): list(b) for w, b in e.bounds.items()}}
                                for e in results.experiments],
                "rfws_cohorts": references, "upstream_limitations": UPSTREAM_LIMITATIONS}
    if window_boundaries:
        from pathlib import Path
        path = Path(window_boundaries).resolve()
        metadata["boundary_provenance"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return Report(metrics, windows, metadata)


def compute_metrics(results_root, rfws_weights=None, forecast_model_tag="baseline", **kwargs):
    """Return one row per experiment/model; metadata is available in df.attrs.

    Use build_report for the companion per-window table and structured diagnostics.
    """
    report = build_report(results_root, rfws_weights, forecast_model_tag, **kwargs)
    for issue in report.metadata["diagnostics"]:
        import warnings
        warnings.warn(f"{issue['code']}: {issue['path']}: {issue['message']}", RuntimeWarning, stacklevel=2)
    report.metrics.attrs["metadata"] = report.metadata
    return report.metrics
