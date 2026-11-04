"""Optional evaluation of minute-level prompt-token forecasts against a trace.

This evaluates explicitly supplied artifacts, independently of simulator results.
It cannot prove a simulator used them or that predictions were out of sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

_FILE_RE = re.compile(r"forecast_(.+)_region(\d+)\.csv")
BUCKET_SECONDS = 60


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_forecasts(trace_path, forecast_dir, *, observation_end, evaluation_start=3600):
    """Return (metrics, aligned buckets, provenance) over [start, end).

    Bounds must be complete minute boundaries, explicitly supplied by the caller.
    Default start excludes the forecaster's first-hour warm-up. Region means
    FIRST-PRIORITY SOURCE region, matching forecasting.arima_forecaster.build_series;
    it is not necessarily the region where the simulator eventually served a row.
    """
    if (not np.isfinite([observation_end, evaluation_start]).all() or
            not 0 <= evaluation_start < observation_end or
            observation_end % 60 or evaluation_start % 60):
        raise ValueError("Evaluation start/end must be finite minute boundaries with 0 <= start < end")
    trace_path, forecast_dir = Path(trace_path), Path(forecast_dir)
    trace = pd.read_csv(trace_path, dtype={"regions": "string", "model_type": "string"})
    required = {"arrival_timestamp", "prompt_size", "model_type", "regions"}
    if not required <= set(trace):
        raise ValueError(f"Trace missing columns: {sorted(required-set(trace))}")
    for column in ("arrival_timestamp", "prompt_size"):
        trace[column] = pd.to_numeric(trace[column], errors="raise")
        if trace[column].isna().any() or not np.isfinite(trace[column]).all() or trace[column].lt(0).any():
            raise ValueError(f"Trace has invalid {column}")
    if trace.model_type.isna().any():
        raise ValueError("Trace has missing model_type")
    regions = trace.regions.str.zfill(3)
    if not regions.isin(["012", "021", "102", "120", "201", "210"]).all():
        raise ValueError("Trace regions must encode a permutation of 012")
    trace["source_region"] = regions.str[0].astype(int)
    trace["minute"] = (trace.arrival_timestamp//60).astype(int)
    end_minute, start_minute = int(observation_end//60), int(evaluation_start//60)
    within = trace[trace.arrival_timestamp.lt(observation_end)]
    observed = within.groupby(["model_type", "source_region", "minute"]).prompt_size.sum()
    metrics, aligned, files = [], [], []
    forecast_paths = {}
    for path in sorted(forecast_dir.glob("forecast_*_region*.csv")):
        match = _FILE_RE.fullmatch(path.name)
        if match:
            pair = (match[1], int(match[2]))
            if pair in forecast_paths or pair[1] not in (0, 1, 2):
                raise ValueError(f"Duplicate forecast series or invalid source region: {path}")
            forecast_paths[pair] = path
    if not forecast_paths:
        raise ValueError(f"No forecast_<model>_region<id>.csv files under {forecast_dir}")
    combinations = set(forecast_paths) | {(str(m), int(r)) for m, r, _ in observed.index}
    for model, region in sorted(combinations):
        path = forecast_paths.get((model, region))
        pred = pd.read_csv(path) if path else pd.DataFrame(columns=["Time", "Predicted"])
        if not {"Time", "Predicted"} <= set(pred):
            raise ValueError(f"{path}: required columns Time, Predicted")
        for column in ("Time", "Predicted"):
            pred[column] = pd.to_numeric(pred[column], errors="raise")
            if pred[column].isna().any() or not np.isfinite(pred[column]).all() or pred[column].lt(0).any():
                raise ValueError(f"{path}: invalid {column}")
        if pred.Time.mod(60).ne(0).any() or pred.Time.duplicated().any():
            raise ValueError(f"{path}: Time must contain unique minute boundaries in seconds")
        actual = pd.Series([float(observed.get((model, region, minute), 0.))
                            for minute in range(end_minute)], index=np.arange(end_minute)*60)
        table = pd.DataFrame({"Time": np.arange(start_minute, end_minute)*60})
        table["actual"] = table.Time.map(actual)
        table["persistence"] = table.Time.map(actual.shift(1))
        table = table.merge(pred[["Time", "Predicted"]], on="Time", how="left", validate="one_to_one")
        valid = table.Predicted.notna()
        scored = table[valid]
        error = scored.Predicted-scored.actual
        common = scored[scored.persistence.notna()]
        baseline_mae = float((common.persistence-common.actual).abs().mean())
        common_mae = float((common.Predicted-common.actual).abs().mean())
        total = float(scored.actual.sum())
        row = {"model": model, "source_region": region, "n_expected_buckets": len(table),
               "n_scored_buckets": len(scored), "prediction_coverage": len(scored)/len(table),
               "mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean())),
               "bias": float(error.mean()), "wape": float(error.abs().sum()/total) if total > 0 else np.nan,
               "persistence_mae_on_common_buckets": baseline_mae,
               "model_mae_on_common_buckets": common_mae,
               "n_common_baseline_buckets": len(common),
               "mae_skill_vs_persistence": 1-common_mae/baseline_mae if baseline_mae > 0 else np.nan}
        metrics.append(row)
        table["model"], table["source_region"] = model, region
        aligned.append(table)
        if path:
            files.append({"path": str(path.resolve()), "sha256": _hash(path)})
    metadata = {"trace": {"path": str(trace_path.resolve()), "sha256": _hash(trace_path)},
                "forecast_files": files, "evaluation_start_seconds": evaluation_start,
                "observation_end_seconds": observation_end, "bucket_seconds": 60,
                "target": "sum of prompt_size for all trace rows, by first-priority source region",
                "scoring_interval": "[evaluation_start_seconds, observation_end_seconds)",
                "rows_at_or_after_observation_end": int(trace.arrival_timestamp.ge(observation_end).sum()),
                "simulation_usage_verified": False, "out_of_sample_verified": False,
                "limitations": ["Prediction files have no forecast-origin timestamps; absence of look-ahead leakage cannot be established.",
                                "Missing prediction buckets remain unknown; future buckets outside the observation interval are not scored.",
                                "This is source-region demand, not routed or completed demand.",
                                "All trace rows contribute, matching the current ARIMA target; prod/dev and batches are not filtered."]}
    manifest = forecast_dir/"manifest.json"
    if manifest.exists():
        metadata["forecast_manifest"] = {"path": str(manifest.resolve()), "sha256": _hash(manifest)}
    return pd.DataFrame(metrics), pd.concat(aligned, ignore_index=True), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--forecasts", required=True)
    parser.add_argument("--observation-end", required=True, type=float)
    parser.add_argument("--evaluation-start", type=float, default=3600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        metrics, buckets, metadata = evaluate_forecasts(args.trace, args.forecasts,
                                                       observation_end=args.observation_end,
                                                       evaluation_start=args.evaluation_start)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out, index=False)
        buckets.to_csv(out.with_name(out.stem+"_buckets.csv"), index=False)
        out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
