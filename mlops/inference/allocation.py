"""Extract allocation evidence from saved exports; never run the simulator.

Usage: python mlops/inference/allocation.py --report mlops/inference/reports/s01.csv
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml


KEYS = ["experiment_id", "case", "model", "region"]
DECISIONS = KEYS + ["time_seconds", "current_instances", "requested_change", "target_instances", "source"]
OBSERVATIONS = KEYS + ["time_seconds", "production_instances", "spot_instances", "source"]
COVERAGE = KEYS + ["snapshot_times", "first_seconds", "last_seconds", "status"]
COMPARISON = KEYS + ["decision_seconds", "requested_change", "target_instances", "before_seconds",
                     "before_instances", "after_seconds", "after_instances", "observed_change", "status"]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flag(value):
    return str(value).lower() == "true"


def compare_around_hour(decisions, observations, coverage, hour=3600):
    """Use the first LTS decision at/after one hour; never reuse pre-decision snapshots as after."""
    rows = []
    for _, item in coverage.iterrows():
        base = {k: item[k] for k in KEYS}
        ds, obs = decisions, observations
        for key in KEYS:
            ds, obs = ds[ds[key] == item[key]], obs[obs[key] == item[key]]
        ds = ds[ds.time_seconds >= hour].sort_values("time_seconds")
        row = {**base, "status": "No LTS decision recorded at/after one hour"}
        if not ds.empty:
            decision = ds.iloc[0]
            t = decision.time_seconds
            row.update(decision_seconds=t, requested_change=decision.requested_change,
                       target_instances=decision.target_instances)
            before = obs[obs.time_seconds < t].sort_values("time_seconds").tail(1)
            after = obs[obs.time_seconds > t].sort_values("time_seconds").head(1)
            for label, frame in (("before", before), ("after", after)):
                if not frame.empty:
                    row[label + "_seconds"] = frame.iloc[0].time_seconds
                    row[label + "_instances"] = frame.iloc[0].production_instances
            row["status"] = "Insufficient observations"
            if not before.empty and not after.empty:
                row["observed_change"] = row["after_instances"] - row["before_instances"]
                row["status"] = "Observed difference; attribution to LTS unverified"
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON)


def export_allocations(report):
    report = Path(report).resolve(strict=True)
    metrics = pd.read_csv(report)
    required = {"experiment_id", "source_path", "model", "sts", "lts", "scaling_level"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Report lacks allocation identity columns: {sorted(required - set(metrics.columns))}")
    out = report.with_suffix("") / "allocation"
    # Restrict generated files to inference/reports (including existing-study refreshes).
    reports_root = Path(__file__).resolve().parent / "reports"
    if not out.resolve().is_relative_to(reports_root.resolve()):
        raise ValueError("Allocation outputs must stay inside inference/reports.")
    decisions, observations, coverage, issues, sources = [], [], [], [], {}
    configured_ends = {}
    initial_allocations = []

    def record(path):
        sources[str(path)] = _hash(path)

    for exp_id, group in metrics.groupby("experiment_id", sort=False):
        first = group.iloc[0]
        leaf = Path(first.source_path)
        level, sts, lts = int(first.scaling_level), _flag(first.sts), _flag(first.lts)
        case_name = f"l{level}_s{int(sts)}_t{int(lts)}"
        title = {(False, False): "Baseline", (True, False): "STS only",
                 (False, True): "LTS only", (True, True): "STS + LTS"}[(sts, lts)]
        title += f" (L{level})"
        base = {"experiment_id": exp_id, "case": title}
        config_path = leaf / ".hydra" / "config.yaml"
        region_map, model_order = {}, []
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
            end = config.get("end_time")
            if isinstance(end, (int, float)) and not isinstance(end, bool) and np.isfinite(end) and end > 0:
                configured_ends[str(exp_id)] = float(end)
            regions = config["controller"]["regions"]
            region_map = {int(r["region_id"]): r["region_name"] for r in regions}
            model_order = sorted({e["model_name"] for r in regions for e in r["model_endpoints"]})
            for region in regions:
                for endpoint in region["model_endpoints"]:
                    count = endpoint.get("instance_count")
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        initial_allocations.append({**base, "model": endpoint["model_name"],
                                                    "region": region["region_name"], "instances": count})
            record(config_path)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            issues.append(f"{exp_id}: configuration unavailable; decision mapping skipped: {exc}")

        log_path = report.with_suffix("") / "logs" / f"{case_name}.log"
        if lts and log_path.is_file() and region_map and model_order:
            record(log_path)
            pending = None
            decision_count = len(decisions)
            # literal_eval only accepts literals, never executable log content.
            for line_number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                match = re.search(r"(Current allocation|ILP forecast) at time ([\d.eE+\-]+):\s*(\[.*\])\s*$", line)
                if not match:
                    if "Current allocation at time" in line or "ILP forecast at time" in line:
                        issues.append(f"{log_path}:{line_number}: Unrecognized allocation log format")
                    continue
                kind, raw_time, raw = match.groups()
                try:
                    t, matrix = float(raw_time), np.asarray(ast.literal_eval(raw), dtype=float)
                    shape = (len(region_map), len(model_order), 1)
                    if matrix.shape != shape or not np.isfinite(matrix).all() or not np.equal(matrix, np.floor(matrix)).all() or not np.isfinite(t) or t < 0:
                        raise ValueError("Unexpected allocation dimensions or values")
                    if kind == "Current allocation":
                        pending = (t, matrix, line_number)
                        continue
                    if pending is None or pending[0] != t:
                        raise ValueError("No matching current allocation for decision")
                    current = pending[1]
                    if (current < 0).any() or (current + matrix < 0).any():
                        raise ValueError("Negative current or target allocation")
                    for ri, region_id in enumerate(sorted(region_map)):
                        for mi, model in enumerate(model_order):
                            n, delta = int(current[ri, mi, 0]), int(matrix[ri, mi, 0])
                            decisions.append({**base, "model": model, "region": region_map[region_id],
                                              "time_seconds": t, "current_instances": n,
                                              "requested_change": delta, "target_instances": n + delta,
                                              "source": f"{log_path}:{pending[2]},{line_number}"})
                    pending = None
                except (ValueError, SyntaxError, TypeError) as exc:
                    issues.append(f"{log_path}:{line_number}: {exc}")
            if len(decisions) == decision_count:
                issues.append(f"{exp_id}: No valid paired LTS decisions extracted")
        elif lts:
            issues.append(f"{exp_id}: LTS decisions unavailable (missing log or configuration mapping)")

        frames = []
        for path in sorted(leaf.glob("arbiters/*/*.csv"), key=lambda p: (p.parent.name, int(p.stem) if p.stem.isdigit() else -1)):
            record(path)
            try:
                frame = pd.read_csv(path)
                if not {"timestamp", "model", "prod", "spot"}.issubset(frame.columns):
                    raise ValueError("Missing resource columns")
                for col in ("timestamp", "prod", "spot"):
                    frame[col] = pd.to_numeric(frame[col], errors="raise")
                numeric = frame[["timestamp", "prod", "spot"]].to_numpy(dtype=float)
                if not np.isfinite(numeric).all() or (numeric < 0).any() or not np.equal(numeric[:, 1:], np.floor(numeric[:, 1:])).all():
                    raise ValueError("Invalid resource measurements")
                frame = frame.rename(columns={"timestamp": "time_seconds", "prod": "production_instances", "spot": "spot_instances"})
                frame["region"], frame["source"] = path.parent.name, str(path)
                frames.append(frame)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                issues.append(f"{path}: {exc}")
        observed = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OBSERVATIONS)
        # Several updates share a timestamp; retain the last exported row, not their sum.
        observed = observed.drop_duplicates(["model", "region", "time_seconds"], keep="last")
        for _, row in observed.iterrows():
            observations.append({**base, **{k: row[k] for k in OBSERVATIONS if k not in base}})
        region_names = sorted(set(region_map.values()) | set(observed.region))
        for model in sorted(group.model.unique()):
            for region in region_names:
                times = observed[(observed.model == model) & (observed.region == region)].time_seconds
                count = len(times)
                coverage.append({**base, "model": model, "region": region, "snapshot_times": count,
                                 "first_seconds": times.min(), "last_seconds": times.max(),
                                 "status": "No snapshots" if not count else "Single timestamp" if count == 1 else "Observed snapshots only"})

    ds, obs, cov = (pd.DataFrame(rows, columns=cols) for rows, cols in
                    ((decisions, DECISIONS), (observations, OBSERVATIONS), (coverage, COVERAGE)))
    comparison = compare_around_hour(ds, obs, cov)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("allocation_decisions", ds), ("allocation_observations", obs),
                        ("allocation_coverage", cov), ("allocation_hour_comparison", comparison)):
        frame.to_csv(out / f"{name}.csv", index=False)
    metadata = {"schema_version": 1, "report": str(report), "report_sha256": _hash(report),
                "configured_end_seconds": configured_ends,
                "initial_allocations": initial_allocations,
                "source_sha256": sources, "issues": issues,
                "limitations": ["Counts are model instances, not individual GPUs.",
                                "Targets are optimizer requests, not confirmed allocations.",
                                "Only saved snapshots are observations; no interpolation or end-of-run extrapolation.",
                                "Same-time resource rows retain the last exported state; intermediate changes are not shown.",
                                "Before/after differences can include STS and are not proof of LTS causality."]}
    (out / "allocation.metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    print(export_allocations(parser.parse_args().report))
