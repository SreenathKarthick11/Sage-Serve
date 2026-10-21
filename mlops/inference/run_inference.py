#!/usr/bin/env python3
"""
run_inference.py  –  CLI entry-point for SageServe simulation inference.

Usage
-----
  # Run from the repo root using the mlops venv:
  mlops/.venv/bin/python mlops/inference/run_inference.py

  # Tag the forecasting model that produced these results:
  mlops/.venv/bin/python mlops/inference/run_inference.py --forecast-model arima

  # When you run a second simulation with a different forecasting model,
  # pass a different tag and a different --out file, then compare FSS values:
  mlops/.venv/bin/python mlops/inference/run_inference.py \\
      --forecast-model prophet \\
      --out mlops/inference/inference_prophet.csv

  # Adjust FSS weights:
  mlops/.venv/bin/python mlops/inference/run_inference.py --slo-weight 0.50

Options
-------
  --results         Path to experiments/results (default: auto-detected)
  --out             Output CSV path  (default: mlops/inference/inference_report.csv)
  --plots-dir       Directory for PNG plots (default: mlops/inference/plots/)
  --forecast-model  Tag for the forecasting model that generated these results
  --no-plots        Skip plot generation

Output
------
  * inference_report.csv   – full metrics + FSS + sub-scores for every
                             (model, config) combination.
  * plots/*.png            – five diagnostic charts.
  * Console summary        – FSS table printed to stdout.

Exit codes: 0 = success, 1 = error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── importable regardless of cwd ────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent
_MLOPS = _HERE.parent
if str(_MLOPS) not in sys.path:
    sys.path.insert(0, str(_MLOPS))

from inference.metrics import compute_metrics, DEFAULT_WEIGHTS
from inference.plots   import generate_all


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    repo_root       = _HERE.parents[1]
    default_results = repo_root / "experiments" / "results"
    default_out     = _HERE / "inference_report.csv"
    default_plots   = _HERE / "plots"

    p = argparse.ArgumentParser(
        description=(
            "Compute FSS and latency/cost metrics from SageServe simulation "
            "results.  FSS (0-100) is the single discrete indicator you can "
            "use to compare forecasting models across simulation runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", default=str(default_results),
                   help="Path to the experiments/results directory.")
    p.add_argument("--out", default=str(default_out),
                   help="Output CSV path for the inference report.")
    p.add_argument("--plots-dir", default=str(default_plots),
                   help="Directory to write PNG plots into.")
    p.add_argument("--forecast-model", default="baseline",
                   help="Label for the forecasting model that generated these "
                        "simulation results (e.g. arima, prophet, lstm). "
                        "Stored in the report for cross-run comparison.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation.")

    # FSS weight overrides
    for key, default in DEFAULT_WEIGHTS.items():
        p.add_argument(
            f"--{key}-weight", type=float, default=default,
            help=f"FSS weight for the '{key}' sub-score.",
        )
    return p


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 72


def _banner(msg: str) -> None:
    print(f"\n{_SEP}\n  {msg}\n{_SEP}")


_DISPLAY_COLS = [
    "forecast_model", "model", "trace", "lts", "sts", "arbiter",
    "n_requests", "requests_per_window",
    "e2e_p99", "ttft_p99", "tbt_p99",
    "viol_ttft", "viol_tbt",
    "cv_response", "demand_cv",
    "vm_h_per_window",
    "score_slo", "score_stability", "score_demand", "score_cost", "score_throughput",
    "fss",
]


def _print_table(df: pd.DataFrame) -> None:
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    with pd.option_context(
        "display.max_columns", None,
        "display.width",       200,
        "display.float_format", "{:.3f}".format,
    ):
        print(df[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_parser().parse_args()

    results_root = Path(args.results)
    out_path     = Path(args.out)
    plots_dir    = Path(args.plots_dir)

    if not results_root.exists():
        print(f"ERROR: results directory not found: {results_root}", file=sys.stderr)
        return 1

    weights = {k: getattr(args, f"{k}_weight") for k in DEFAULT_WEIGHTS}

    _banner("SageServe Simulation Inference")
    print(f"  Results root    : {results_root}")
    print(f"  Forecast model  : {args.forecast_model}")
    print(f"  Output CSV      : {out_path}")
    print(f"  Plots dir       : {plots_dir}")
    print(f"  FSS weights     : {weights}")

    # ── compute ──────────────────────────────────────────────────────────────
    _banner("Step 1/2  –  Computing metrics")
    try:
        df = compute_metrics(
            str(results_root),
            fss_weights=weights,
            forecast_model_tag=args.forecast_model,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  Found {len(df)} (model, config) combinations.")

    # ── display ──────────────────────────────────────────────────────────────
    _banner("Inference Report  (sorted by FSS, highest first)")
    _print_table(df)

    # ── top pick ─────────────────────────────────────────────────────────────
    top = df.iloc[0]
    _banner("Highest FSS")
    print(
        f"  model={top['model']}  trace={top['trace']}  "
        f"lts={top['lts']}  arbiter={top['arbiter']}"
    )
    print(f"  FSS = {top['fss']:.2f} / 100")
    print()
    print("  Sub-scores:")
    for col, label in [
        ("score_slo",        "SLO compliance   "),
        ("score_stability",  "Latency stability"),
        ("score_demand",     "Demand regularity"),
        ("score_cost",       "Cost efficiency  "),
        ("score_throughput", "Throughput signal"),
    ]:
        print(f"    {label}: {top.get(col, float('nan')):.3f}")

    # ── save CSV ─────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _banner(f"Saved report: {out_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        _banner("Step 2/2  –  Generating plots")
        try:
            from inference.plots import generate_all
            generate_all(df, plots_dir)
        except Exception as exc:
            print(f"WARNING: plot generation failed: {exc}", file=sys.stderr)

    print(f"\nDone.  FSS range: {df['fss'].min():.1f} – {df['fss'].max():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
