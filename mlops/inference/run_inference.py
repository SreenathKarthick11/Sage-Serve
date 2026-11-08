#!/usr/bin/env python3
"""Compute inference diagnostics without confusing a label with forecast evidence.

Example (from the repository root):
  python mlops/inference/run_inference.py --results experiments/results \
      --out mlops/inference/reports/baseline.csv

Exit codes: 0 = outputs requested were written; 1 = failed; 2 = partial output
(incomplete request exports, explicitly skipped inputs, or failed plots).
Unknown duration or unavailable relative rfws are reported statuses, not crashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from inference.metrics import DEFAULT_WEIGHTS, _json_safe, build_report


def _build_parser():
    parser = argparse.ArgumentParser(description="Analyze exported simulation measurements; rfws is cohort-relative.")
    parser.add_argument("--results", default=str(_HERE.parents[1] / "experiments" / "results"))
    parser.add_argument("--out", default=str(_HERE / "inference_report.csv"))
    parser.add_argument("--plots-dir", default=str(_HERE / "plots"))
    parser.add_argument("--forecast-model", default="baseline", help="Unverified label only; does not run or evaluate a forecaster.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Report usable data with explicit diagnostics; partial output exits 2.")
    parser.add_argument("--window-boundaries", help="Optional CSV: experiment_id,window,start_seconds,end_seconds. See INFERENCE_REPORT.md.")
    parser.add_argument("--ttft-threshold", type=float, default=10.)
    parser.add_argument("--tbt-threshold", type=float, default=.2, help="Threshold for REQUEST-AVERAGE TBT, in seconds.")
    for name, value in DEFAULT_WEIGHTS.items():
        parser.add_argument(f"--{name}-weight", type=float, default=value,
                            help="Nonnegative relative weight; all weights are normalized to sum to one.")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    out = Path(args.out)
    metadata_path = out.with_suffix(".metadata.json")
    windows_path = out.with_name(out.stem + "_windows.csv")
    try:
        report = build_report(
            args.results, {key: getattr(args, f"{key}_weight") for key in DEFAULT_WEIGHTS},
            args.forecast_model, allow_partial=args.allow_partial,
            window_boundaries=args.window_boundaries,
            ttft_threshold=args.ttft_threshold, tbt_threshold=args.tbt_threshold)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metadata = report.metadata
    metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["results_root"] = str(Path(args.results).resolve())
    metadata["outputs"] = {"report": str(out.resolve()), "windows": str(windows_path.resolve()),
                           "metadata": str(metadata_path.resolve()), "plots": []}
    partial = (any(i["severity"] == "error" for i in metadata["diagnostics"]) or
               not report.metrics.request_data_complete.all())
    for issue in metadata["diagnostics"]:
        print(f"{issue['severity'].upper()} [{issue['code']}] {issue['path']}: {issue['message']}", file=sys.stderr)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        report.metrics.to_csv(out, index=False)
        report.windows.to_csv(windows_path, index=False)
        metadata["plot_status"] = "skipped" if args.no_plots else "pending"
        # Write provenance even if plotting later fails.
        metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n")
        if not args.no_plots:
            try:
                from inference.plots import generate_all
                paths = generate_all(report.metrics, args.plots_dir, windows=report.windows)
                metadata["outputs"]["plots"] = [str(p.resolve()) for p in paths]
                metadata["plot_status"] = "complete"
            except Exception as exc:
                partial = True
                metadata["plot_status"] = "failed"
                metadata["plot_error"] = str(exc)
                print(f"ERROR: report saved, but plotting failed: {exc}", file=sys.stderr)
        metadata["output_status"] = "partial" if partial else "complete"
        metadata["output_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (out, windows_path)}
        metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        print(f"ERROR writing report outputs: {exc}", file=sys.stderr)
        return 1

    columns = ["experiment_id", "model", "n_requests", "n_windows", "n_quiet_windows",
               "e2e_p99", "viol_ttft", "viol_avg_tbt", "resource_coverage_status", "rfws", "rfws_status"]
    print(report.metrics[columns].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nNormalized weights: {metadata['normalized_weights']}")
    print(f"Saved report: {out}\nSaved window measurements: {windows_path}\nSaved provenance: {metadata_path}")
    print("rfws values are comparable only within the same rfws_cohort_id; missing scores remain unknown.")
    print("See metadata.upstream_limitations for simulator export limitations.")
    return 2 if partial else 0


if __name__ == "__main__":
    sys.exit(main())
