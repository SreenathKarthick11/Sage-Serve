"""Run four scaling-level-2 policy cases, then analyze their actual saved exports.

Nothing runs on import. Invoke with the MLOps interpreter; subprocesses use the
separate simulator interpreter. Generated files stay inside inference/reports.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPORTS = HERE / "reports"
MLOPS = HERE.parent
if str(MLOPS) not in sys.path:
    sys.path.insert(0, str(MLOPS))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def within(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path is outside {root}: {resolved}")
    return resolved


@dataclass(frozen=True)
class Case:
    level: int
    sts: bool
    lts: bool

    @property
    def name(self):
        return f"l{self.level}_s{int(self.sts)}_t{int(self.lts)}"


# Hold scaling level fixed so this small study isolates STS, LTS and their interaction.
# Level 0 disables resource transfers and can hide the effect of either policy.
CASES = tuple(Case(2, sts, lts)
              for sts, lts in ((False, False), (True, False), (False, True), (True, True)))


def case_directory(study, trace_label, case):
    return (Path(study) / "r" / "0" / trace_label / f"scaling_level_{case.level}"
            / f"lts_{case.lts}_sts{case.sts}" / "arbiter_basic_arbiter")


def check_path_lengths(study, trace_label):
    # Check before launching any case. Leave room for nested exports on Windows.
    for case in CASES:
        sample = case_directory(study, trace_label, case) / "detailed" / "centralus_A_999_alloc" / "999.csv"
        if os.name == "nt" and len(str(sample)) >= 240:
            raise ValueError("Output paths are too long. Use a shorter --name or checkout path.")


def archive_current(reports=REPORTS):
    """Archive only known older matrix outputs, preserving data and a move ledger."""
    reports = Path(reports).resolve()
    if not reports.exists():
        return None
    targets = sorted(p for p in reports.iterdir()
                     if p.name == "r4" or p.name.startswith(("rfws_matrix_", "fss_matrix_")))
    if not targets:
        return None
    # Validate every source and destination before moving anything; never follow links.
    for path in targets:
        if path.is_symlink():
            raise ValueError(f"Refusing to archive a symbolic link: {path}")
        within(path, reports)
    archive = reports / "_archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    within(archive, reports)
    destinations = [(path, within(archive / path.name, reports)) for path in targets]
    archive.mkdir(parents=True)
    ledger = {"created_at_utc": now(), "moves": []}
    for source, destination in destinations:
        source.rename(destination)
        ledger["moves"].append({"from": str(source), "to": str(destination)})
        save_json(archive / "archive_manifest.json", ledger)
    return archive


def prepare_inputs(source, study):
    """Fit on historical traffic only, then freeze predictions for held-out replay."""
    import numpy as np
    import pandas as pd
    import yaml
    from etl_module.validation import validate_raw_trace
    from forecasting.arima_forecaster import ARIMAForecaster

    trace = pd.read_csv(source, dtype={"regions": "string", "model_type": "string"})
    history = trace[trace.arrival_timestamp < 600].copy()
    evaluation = trace[trace.arrival_timestamp >= 600].copy()
    evaluation["arrival_timestamp"] -= 600
    historical = history[(history.workload_type == "prod") & (history.batch_id == -1)]
    fed = evaluation[(evaluation.workload_type == "prod") & (evaluation.batch_id == -1)]
    if historical.empty or fed.empty or fed.arrival_timestamp.max() <= 3900:
        raise ValueError("Trace needs history before 600 s and production arrivals beyond original 4500 s.")
    if set(fed.model_type) != {"A", "B", "C", "D"}:
        raise ValueError("Evaluation traffic must include models A-D.")
    validate_raw_trace(history)
    validate_raw_trace(evaluation)
    inputs = Path(study) / "i"
    forecasts = inputs / "f"
    forecasts.mkdir(parents=True)
    history_path, evaluation_path = inputs / "history.csv", inputs / "evaluation.csv"
    history.to_csv(history_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    config = {"model_info": {"type": "arima", "description": "Frozen historical matrix forecast"},
              "parameters": {"order": {"p": 1, "d": 1, "q": 1}, "train_window_mins": 10,
                             "future_horizon": 180, "bucket_seconds": 60}}
    config_path = inputs / "arima.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    forecaster = ARIMAForecaster(str(config_path))
    regions = historical.regions.str.zfill(3).str[0].astype(int)
    minutes = (historical.arrival_timestamp // 60).astype(int)
    for model in "ABCD":
        for region in range(3):
            subset = historical[(historical.model_type == model) & (regions == region)]
            series = subset.groupby(minutes.loc[subset.index]).prompt_size.sum()
            values = series.reindex(range(10), fill_value=0).to_numpy(dtype=float)
            predictions = forecaster.predict(values)
            pd.DataFrame({"Time": np.arange(180) * 60, "Predicted": predictions}).to_csv(
                forecasts / f"forecast_{model}_region{region}.csv", index=False)
    frozen = [Path(source).resolve(), history_path, evaluation_path, config_path,
              *sorted(forecasts.glob("forecast_*.csv"))]
    manifest = {"source_trace": str(Path(source).resolve()),
                "evaluation_trace": str(evaluation_path), "forecast_directory": str(forecasts),
                "trace_label": "h" + sha256(evaluation_path)[:12],
                "input_sha256": {str(path): sha256(path) for path in frozen},
                "history_interval_original_seconds": [0, 600],
                "evaluation_origin_original_seconds": 600,
                "forecast_target": "historical non-batched production prompt tokens per minute by source region",
                "forecast_accuracy_validated": False, "forecaster": forecaster.get_metadata()}
    save_json(Path(study) / "prepared_inputs.json", manifest)
    return manifest


def verify_inputs(prepared):
    for filename, expected in prepared["input_sha256"].items():
        if sha256(filename) != expected:
            raise ValueError(f"Input changed during study: {filename}")


def simulator_arguments(case, directory, prepared):
    args = [f"trace.path={Path(prepared['evaluation_trace']).as_posix()}",
            f"trace.filename={prepared['trace_label']}", f"output_dir={directory.as_posix()}",
            "controller=us3-new", "global_router=probabilistic", "global_router.no_reroute=False",
            "seed=0", "feed_async=False", "siloed=False", "end_time=3900",
            f"scaling_level={case.level}", "scaling_interval=-1",
            f"short_term_scaling={case.sts}", f"long_term_scaling={case.lts}",
            f"global_arbiter.arima_traces={Path(prepared['forecast_directory']).as_posix()}",
            f"global_arbiter.ilp_output_dir={directory.as_posix()}/ilp",
            "global_arbiter.long_term_scaling_interval=3600",
            "global_arbiter.post_processing_strategy=immediate", "global_arbiter.max_time=3900"]
    args.extend(f"controller.regions.{region}.model_endpoints.{model}.instance_count=8"
                for region in range(3) for model in range(4))
    return args


def validate_saved_config(directory, prepared, case):
    import yaml
    config_path = directory / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    expected = {"scaling_level": case.level, "short_term_scaling": case.sts,
                "long_term_scaling": case.lts, "seed": 0, "end_time": 3900,
                "feed_async": False, "siloed": False, "scaling_interval": -1}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Saved configuration mismatch: {key} in {directory}")
    if Path(config["trace"]["path"]).resolve() != Path(prepared["evaluation_trace"]).resolve():
        raise ValueError("Simulator used a different evaluation trace.")
    if config["global_router"]["no_reroute"] is not False:
        raise ValueError("Saved routing setting differs from the matrix control.")
    regions = config["controller"]["regions"]
    if len(regions) != 3 or any(
        r["arbiter"] != "basic_arbiter" or len(r["model_endpoints"]) != 4
        or any(e["instance_count"] != 8 for e in r["model_endpoints"]) for r in regions
    ):
        raise ValueError("Saved initial allocation or regional arbiter differs from the control.")
    if case.lts and Path(config["global_arbiter"]["arima_traces"]).resolve() != Path(prepared["forecast_directory"]).resolve():
        raise ValueError("LTS used a different forecast directory.")
    return sha256(config_path)


def run_case(case, study, prepared, sim_python):
    verify_inputs(prepared)
    directory = case_directory(study, prepared["trace_label"], case)
    if directory.exists():
        raise ValueError(f"Refusing to reuse case directory: {directory}")
    logs = Path(study) / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{case.name}.log"
    evidence_path = logs / f"{case.name}.json"
    command = [str(sim_python), "-X", "utf8", "run.py", *simulator_arguments(case, directory, prepared)]
    evidence = {"case": case.name, "level": case.level, "sts": case.sts, "lts": case.lts,
                "started_at_utc": now(), "command": command, "cwd": str(REPO / "SageServe-original"),
                "output_directory": str(directory), "log": str(log_path), "status": "running"}
    save_json(evidence_path, evidence)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=REPO / "SageServe-original", stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        evidence["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"Simulator exited {completed.returncode}; read {log_path}")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"TraceSimulator completed at ([0-9.eE+-]+)", text)
        if not matches or not math.isfinite(float(matches[-1])) or float(matches[-1]) < 3900:
            raise RuntimeError(f"No completion marker reaching 3900 s; read {log_path}")
        evidence["simulation_completed_seconds"] = float(matches[-1])
        evidence["config_sha256"] = validate_saved_config(directory, prepared, case)
        details = [p for p in (directory / "detailed").glob("*/*.csv")
                   if not p.parent.name.endswith("_alloc")]
        if not details:
            raise RuntimeError(f"No detailed request exports: {directory}")
        inventory = [{"path": p.relative_to(directory).as_posix(), "bytes": p.stat().st_size,
                      "sha256": sha256(p)} for p in sorted(directory.rglob("*.csv"))]
        save_json(logs / f"{case.name}_exports.json", inventory)
        evidence.update(status="complete", detailed_export_files=len(details),
                        missing_forecast_warning="No ARIMA forecast available" in text,
                        log_sha256=sha256(log_path))
        if evidence["missing_forecast_warning"]:
            raise RuntimeError("Planner reported missing forecast windows; inspect the log.")
        verify_inputs(prepared)
        return evidence
    except (Exception, KeyboardInterrupt) as exc:
        evidence.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        evidence["finished_at_utc"] = now()
        save_json(evidence_path, evidence)


def analyze(study, report_path):
    import pandas as pd
    from inference.run_inference import main as inference_main
    from inference.matrix_plots import generate_matrix_scores

    code = inference_main(["--results", str(Path(study) / "r"), "--out", str(report_path),
                           "--plots-dir", str(report_path.with_name(report_path.stem + "_plots")),
                           "--forecast-model", "frozen-historical-arima-matrix",
                           "--cost-weight", "0", "--ttft-threshold", "10", "--tbt-threshold", "0.2"])
    if code == 1:
        return code
    metrics = pd.read_csv(report_path)
    eligible = metrics.rfws_status.eq("relative_to_cohort") & metrics.rfws.notna()
    cohorts = metrics.groupby("rfws_cohort_id").size()
    expected_models = {"A", "B", "C", "D"}
    expected_cases = {(case.level, case.sts, case.lts) for case in CASES}
    case_membership = all(
        set(zip(group.scaling_level, group.sts, group.lts)) == expected_cases
        for _, group in metrics.groupby("model")
    )
    complete = (set(metrics.model) == expected_models and case_membership
                and len(metrics) == len(expected_models) * len(CASES) and eligible.all()
                and len(cohorts) == len(expected_models) and cohorts.eq(len(CASES)).all())
    # Keep all four models visible, with four policy comparisons in each panel.
    plots_dir = report_path.with_name(report_path.stem + "_plots")
    generate_matrix_scores(metrics, plots_dir)
    metadata_path = report_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata["matrix_plot_layout"] = f"one panel per model, {len(CASES)} configurations per panel"
    metadata["matrix_evidence"] = str(Path(study) / "manifest.json")
    from inference.allocation import export_allocations
    metadata["allocation_evidence"] = str(export_allocations(report_path))
    save_json(metadata_path, metadata)
    inputs = ["viol_ttft", "viol_avg_tbt", "cv_response", "prompt_size_cv", "requests_per_window"]
    variation = metrics.groupby("model")[inputs].nunique(dropna=False)
    variation.to_csv(Path(study) / "metric_variation.csv")
    if variation.eq(1).all().all():
        print("All enabled scoring inputs tie within each model. Equal RFWS values are a measured result.")
    if not complete:
        print("Matrix exports are incomplete or some RFWS rows are ineligible. Inspect report statuses.", file=sys.stderr)
    return 0 if code == 0 and complete else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    default_sim = REPO / "SageServe-original" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    p.add_argument("--sim-python", type=Path, default=default_sim)
    p.add_argument("--trace", type=Path, help="Validated static trace; default: latest clean artifact.")
    p.add_argument("--name", default="m" + datetime.now(timezone.utc).strftime("%m%d%H%M%S"),
                   help="Fresh short study name (1-18 letters/digits/underscore/hyphen).")
    p.add_argument("--archive-current", action="store_true",
                   help="Archive old rfws_matrix_*/fss_matrix_* outputs and r4 before starting.")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = None
    study = None
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,18}", args.name):
            raise ValueError("Use a short --name with letters/digits/underscore/hyphen only.")
        sim_python = args.sim_python.resolve()
        if not sim_python.is_file():
            raise FileNotFoundError(f"Simulator Python is missing: {sim_python}")
        source = args.trace
        if source is None:
            clean = Path(os.environ.get("MLOPS_DATA_ROOT", REPO / "data")) / "static" / "clean"
            pointer = json.loads((clean / "latest.json").read_text(encoding="utf-8-sig"))
            source = clean / pointer["version"] / "validated_trace.csv"
        source = source.resolve(strict=True)
        study = within(REPORTS / args.name, REPORTS)
        report_path = REPORTS / f"{args.name}.csv"
        if study.exists() or report_path.exists() or (REPORTS / f"{args.name}_plots").exists():
            raise ValueError("Study/report already exists. Use a fresh --name; nothing is overwritten.")
        check_path_lengths(study, "h000000000000")
        if args.archive_current:
            # Do not archive a source trace supplied from an older study.
            if source.is_relative_to(REPORTS):
                raise ValueError("With --archive-current, use a validated trace outside reports.")
            archived = archive_current(REPORTS)
            if archived:
                print(f"Archived older outputs to {archived}", flush=True)
        study.mkdir(parents=True)
        manifest = {"created_at_utc": now(), "status": "preparing", "source_trace": str(source),
                    "sim_python": str(sim_python), "mlops_python": sys.executable,
                    "cost_weight": 0, "initial_instances_per_model_region": 8,
                    "duration_seconds": 3900, "planned_cases": [case.__dict__ for case in CASES],
                    "cases": []}
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        manifest["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
        save_json(study / "manifest.json", manifest)
        prepared = prepare_inputs(source, study)
        manifest["prepared_inputs"] = str(study / "prepared_inputs.json")
        manifest["status"] = "running"
        for index, case in enumerate(CASES, 1):
            print(f"[{index}/{len(CASES)}] Starting {case.name}; log: {study / 'logs' / (case.name + '.log')}", flush=True)
            evidence = run_case(case, study, prepared, sim_python)
            manifest["cases"].append(evidence)
            save_json(study / "manifest.json", manifest)
            print(f"[{index}/{len(CASES)}] Completed {case.name}", flush=True)
        manifest["status"] = "analyzing"
        save_json(study / "manifest.json", manifest)
        code = analyze(study, report_path)
        manifest.update(status="complete" if code == 0 else "partial", report=str(report_path),
                        finished_at_utc=now())
        save_json(study / "manifest.json", manifest)
        print(f"Report: {report_path}\nRFWS plots: {REPORTS / (args.name + '_plots')}")
        print("Select this CSV in the Results page. No expected winning policy or score is imposed.")
        return code
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc) or type(exc).__name__
        if manifest is not None:
            manifest.update(status="failed", error=message, finished_at_utc=now())
            save_json(study / "manifest.json", manifest)
        print(f"ERROR: {message}", file=sys.stderr)
        print("Evidence is preserved. Fix the cause and use a fresh study name.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
