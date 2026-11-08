"""Prepare, replay and analyze an isolated workload-shift experiment.

Preparation/analysis never run a simulator or fit a model. Only the explicit
`run` command starts four shifted replays using the frozen control forecasts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from inference.run_rfws_matrix import (CASES, REPORTS, REPO, check_path_lengths,
                                       now, run_case, save_json, sha256, verify_inputs, within)

ROOT = REPORTS / "drift"
END = 3900


def study_path(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,12}", name):
        raise ValueError("Use a study name with 1-12 letters/digits/underscore/hyphen.")
    return within(ROOT / name, ROOT)


def production(trace):
    return trace[(trace.workload_type == "prod") & (trace.batch_id == -1)].copy()


def read_trace(path):
    frame = pd.read_csv(path, dtype={"regions": "string", "model_type": "string"})
    frame["source_region"] = frame.regions.str.zfill(3).str[0].astype(int)
    return frame


def verify_control(manifest):
    baseline = Path(manifest["baseline_directory"])
    if sha256(baseline / "manifest.json") != manifest["baseline_manifest_sha256"]:
        raise ValueError("Baseline evidence changed since preparation")
    if sha256(baseline.with_suffix(".csv")) != manifest["baseline_report_sha256"]:
        raise ValueError("Baseline report changed since preparation")
    evidence = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    for case in evidence["cases"]:
        inventory = baseline / "logs" / f"{case['case']}_exports.json"
        for item in json.loads(inventory.read_text(encoding="utf-8")):
            path = within(Path(case["output_directory"]) / item["path"], case["output_directory"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Baseline export changed: {path}")
    return baseline


def psi(reference, current, *, min_samples=50):
    """Fixed reference bins, explicit overflow bins, additive 0.5 count smoothing."""
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("PSI observations must be finite")
    if len(ref) < min_samples or len(cur) < min_samples:
        return np.nan
    margin = max(1e-6, abs(float(ref.max())) * 1e-9)
    edges = np.unique(np.r_[-np.inf, ref.min() - margin,
                            np.quantile(ref, np.linspace(0, 1, 11))[1:-1],
                            ref.max() + margin, np.inf])
    p, q = (np.histogram(values, bins=edges)[0].astype(float) + .5 for values in (ref, cur))
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def workload_reports(study, manifest):
    """Measure source-region inputs, independent of incomplete resource exports."""
    onset, window = manifest["onset_seconds"], manifest["window_seconds"]
    control = production(read_trace(study / "i" / "control.csv"))
    shifted = production(read_trace(study / "i" / "shifted.csv"))
    control = control[control.arrival_timestamp < END]
    shifted = shifted[shifted.arrival_timestamp < END]
    scores, demand = [], []
    for model in "ABCD":
        for region in range(3):
            ref = control[(control.model_type == model) & (control.source_region == region)
                          & (control.arrival_timestamp < onset)]
            forecast = pd.read_csv(study / "i" / "f" / f"forecast_{model}_region{region}.csv")
            pred = forecast.set_index("Time").Predicted
            for scenario, trace in (("Control", control), ("Shifted", shifted)):
                series = trace[(trace.model_type == model) & (trace.source_region == region)]
                for start in range(0, END, window):
                    stop = min(start + window, END)
                    current = series[(series.arrival_timestamp >= start) & (series.arrival_timestamp < stop)]
                    for feature in ("prompt_size", "token_size"):
                        # Do not compare a reference subset with its own containing reference interval.
                        value = psi(ref[feature], current[feature]) if start >= onset and stop-start == window else np.nan
                        status = ("Reference period" if start < onset else "Partial window" if stop-start < window
                                  else "Insufficient samples" if not np.isfinite(value)
                                  else "High shift" if value >= .25 else "Moderate shift" if value >= .1 else "Low shift")
                        scores.append(dict(scenario=scenario, model=model, region=region, feature=feature,
                                           start_seconds=start, end_seconds=stop, psi=value, status=status,
                                           reference_samples=len(ref), current_samples=len(current)))
                counts = series.groupby((series.arrival_timestamp // 60).astype(int)).prompt_size.sum()
                errors = []
                for minute in range(END // 60):
                    time = minute * 60
                    actual, predicted = float(counts.get(minute, 0.)), float(pred.get(time, np.nan))
                    error = predicted - actual
                    errors.append(abs(error))
                    recent = errors[-5:]
                    demand.append(dict(scenario=scenario, model=model, region=region, time_seconds=time,
                                       actual=actual, predicted=predicted, error=error,
                                       rolling_mae=float(np.mean(recent)) if len(recent) == 5 and np.isfinite(recent).all() else np.nan))
    pd.DataFrame(scores).to_csv(study / "psi.csv", index=False)
    pd.DataFrame(demand).to_csv(study / "demand.csv", index=False)


def prepare(args):
    study, baseline = study_path(args.name), within(REPORTS / args.baseline, REPORTS)
    if study.exists():
        raise ValueError("Drift study already exists; use a fresh name. Nothing is overwritten.")
    if baseline.is_relative_to(ROOT):
        raise ValueError("Choose a normal matrix study as control, not a drift study.")
    original = json.loads((baseline / "prepared_inputs.json").read_text(encoding="utf-8-sig"))
    control_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8-sig"))
    if control_manifest.get("status") != "complete" or control_manifest.get("duration_seconds") != END:
        raise ValueError("Control must be a complete 3900-second four-policy matrix.")
    cases = {(c["level"], c["sts"], c["lts"]) for c in control_manifest["cases"] if c["status"] == "complete"}
    if cases != {(c.level, c.sts, c.lts) for c in CASES}:
        raise ValueError("Control must contain all four matched level-2 policies.")
    verify_inputs(original)
    if not math.isfinite(args.factor) or args.factor <= 1:
        raise ValueError("Prompt multiplier must be finite and greater than one.")
    if not 0 < args.onset < END or args.window < 60 or args.window % 60 or args.onset % args.window:
        raise ValueError("Use a minute-aligned window and onset divisible by it, inside the replay duration.")
    check_path_lengths(study / "s", "d000000000000")
    trace = read_trace(original["evaluation_trace"]).drop(columns="source_region")
    regions = trace.regions.str.zfill(3).str[0].astype(int)
    affected = ((trace.model_type == args.model) & (regions == args.region)
                & (trace.workload_type == "prod") & (trace.batch_id == -1)
                & (trace.arrival_timestamp >= args.onset) & (trace.arrival_timestamp < END))
    if not affected.any():
        raise ValueError("No replayed requests match the requested shift.")
    changed = trace.copy()
    changed.loc[affected, "prompt_size"] = np.ceil(changed.loc[affected, "prompt_size"] * args.factor).astype("int64")
    if not changed.loc[~affected].equals(trace.loc[~affected]):
        raise ValueError("An untargeted row changed")
    from etl_module.validation import validate_raw_trace
    validate_raw_trace(changed)
    (study / "i" / "f").mkdir(parents=True)
    shutil.copy2(original["evaluation_trace"], study / "i" / "control.csv")
    changed.to_csv(study / "i" / "shifted.csv", index=False)
    for path in Path(original["forecast_directory"]).glob("forecast_*.csv"):
        shutil.copy2(path, study / "i" / "f" / path.name)
    frozen = [study / "i" / "control.csv", study / "i" / "shifted.csv", *sorted((study / "i" / "f").glob("*.csv"))]
    if len(frozen) != 14:
        raise ValueError("Expected twelve frozen forecast files")
    manifest = dict(created_at_utc=now(), status="prepared", baseline=args.baseline,
                    baseline_directory=str(baseline), baseline_manifest_sha256=sha256(baseline / "manifest.json"),
                    baseline_report_sha256=sha256(REPORTS / f"{args.baseline}.csv"),
                    target_model=args.model, target_region=args.region, prompt_multiplier=args.factor,
                    onset_seconds=args.onset, window_seconds=args.window, duration_seconds=END,
                    affected_rows=int(affected.sum()), reference_interval=[0, args.onset],
                    input_sha256={str(p): sha256(p) for p in frozen}, cases=[],
                    psi_method="Reference quantile bins + overflow bins; 0.5 count smoothing; minimum 50 samples; full windows only",
                    interpretation="PSI is workload distribution shift, not proof of concept drift or SLA failure.")
    save_json(study / "manifest.json", manifest)
    workload_reports(study, manifest)
    print(f"Prepared {study}; changed {affected.sum()} rows. No simulation or forecasting fit was run.")
    return study


def analyze(study, manifest):
    from inference.metrics import build_report, _json_safe
    verify_inputs(manifest)
    baseline = verify_control(manifest)
    if len(manifest["cases"]) != len(CASES) or any(c["status"] != "complete" for c in manifest["cases"]):
        raise ValueError("Complete all four shifted simulations before impact analysis")
    frames, diagnostics = [], {}
    for scenario, root in (("Control", baseline / "r"), ("Shifted", study / "s" / "r")):
        report = build_report(root, rfws_weights={"cost": 0}, forecast_model_tag="frozen-drift-control")
        frame = report.metrics.copy()
        frame["scenario"] = scenario
        frames.append(frame)
        diagnostics[scenario] = dict(diagnostics=report.metadata["diagnostics"],
                                     upstream_limitations=report.metadata["upstream_limitations"])
    combined = pd.concat(frames, ignore_index=True)
    for scenario, group in combined.groupby("scenario"):
        if len(group) != 16 or set(group.model) != set("ABCD"):
            raise ValueError(f"Incomplete {scenario} model/policy report")
        for _, model_rows in group.groupby("model"):
            flags = set(zip(model_rows.scaling_level.astype(int),
                            model_rows.sts.astype(str).str.lower().eq("true"),
                            model_rows.lts.astype(str).str.lower().eq("true")))
            if flags != {(c.level, c.sts, c.lts) for c in CASES}:
                raise ValueError(f"Missing/duplicate {scenario} policy rows")
    # Nested outputs only: no normal report CSV, six-plot set or allocation files.
    combined.to_csv(study / "impact.csv", index=False)
    save_json(study / "impact.metadata.json", _json_safe(diagnostics))
    workload_reports(study, manifest)
    manifest.update(status="complete", finished_at_utc=now())
    save_json(study / "manifest.json", manifest)
    print(f"Drift analysis: {study}. Normal reports were not modified.")


def run(study, manifest, sim_python):
    if manifest["status"] != "prepared":
        raise ValueError("Only a fresh prepared study can run; failures preserve evidence. Use a fresh name after fixing the cause.")
    sim_python = Path(sim_python).resolve(strict=True)
    verify_inputs(manifest)
    verify_control(manifest)
    (study / "s").mkdir()
    prepared = dict(evaluation_trace=str(study / "i" / "shifted.csv"),
                    forecast_directory=str(study / "i" / "f"),
                    trace_label="d" + sha256(study / "i" / "shifted.csv")[:12],
                    input_sha256=manifest["input_sha256"])
    manifest["status"] = "running"
    save_json(study / "manifest.json", manifest)
    try:
        for index, case in enumerate(CASES, 1):
            print(f"[{index}/{len(CASES)}] Shifted replay {case.name}", flush=True)
            manifest["cases"].append(run_case(case, study / "s", prepared, sim_python))
            save_json(study / "manifest.json", manifest)
        analyze(study, manifest)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="failed", error=str(exc) or type(exc).__name__)
        save_json(study / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Copy/shift traces and measure workload drift; no simulator")
    prep.add_argument("--baseline", default="s01")
    prep.add_argument("--name", default="d01")
    prep.add_argument("--model", choices=list("ABCD"), default="A")
    prep.add_argument("--region", type=int, choices=range(3), default=0)
    prep.add_argument("--factor", type=float, default=4.)
    prep.add_argument("--onset", type=int, default=1800)
    prep.add_argument("--window", type=int, default=900)
    execute = commands.add_parser("run", help="Explicitly start four shifted simulator replays, then analyze")
    execute.add_argument("--name", default="d01")
    execute.add_argument("--sim-python", default=str(REPO / "SageServe-original" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
    again = commands.add_parser("analyze", help="Reanalyze completed exports; no simulator")
    again.add_argument("--name", default="d01")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            study = study_path(args.name)
            manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
            run(study, manifest, args.sim_python) if args.command == "run" else analyze(study, manifest)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {str(exc) or type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
