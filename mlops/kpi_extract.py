"""
kpi_extract.py - turn a SageServe `results/` tree into one tidy KPI table.

Usage:  python kpi_extract.py experiments/results  kpis.csv

One row per (run, trace, lts, sts, arbiter, window, region, model).
Path layout assumed (taken from your tree):
  results/<run>/<trace>/.../lts_<bool>_sts<bool>/arbiter_<name>/
      summary/<window>.csv        <- latency stats per application_id
      schedulers/<region>_<model>_<i>.csv   <- one file per instance ever created
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CFG = re.compile(
    r"results/(?P<run>\d+)/(?P<trace>[^/]+)/.*?/"
    r"lts_(?P<lts>True|False)_sts(?P<sts>True|False)/arbiter_(?P<arbiter>[^/]+)/"
)

KEEP = [
    "response_times_mean", "response_times_p50", "response_times_p90", "response_times_p99",
    "ttft_times_mean", "ttft_times_p99",
    "tbt_times_mean", "tbt_times_p99",
    "queue_times_mean", "queue_times_p99", "queue_times_max",
    "prompt_sizes_mean", "token_sizes_mean",
]

# ---- PLACEHOLDER SLOs: replace with the targets you actually want to defend ----
SLO = {"ttft_times_p99": 10.0, "tbt_times_p99": 0.20}  # seconds


def instance_counts(arbiter_dir: Path) -> Counter:
    """Cost proxy: number of scheduler (instance) files per (region, model)."""
    c = Counter()
    for f in (arbiter_dir / "schedulers").glob("*.csv"):
        region, model, _ = f.stem.split("_")
        c[(region, model)] += 1
    return c


def load_summary(path: Path, arbiter_dir: Path, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    # -1 is the "no data" sentinel -> drop, don't average it
    df = df[df["response_times_mean"] != -1].copy()
    parts = df["application_id"].str.split("_", expand=True)
    df["region"], df["model"], df["app_idx"] = parts[0], parts[1], parts[2].astype(int)
    df["window"] = int(path.stem)
    counts = instance_counts(arbiter_dir)
    df["instances"] = [counts.get((r, m), np.nan) for r, m in zip(df["region"], df["model"])]
    for k, v in cfg.items():
        df[k] = v
    for col, limit in SLO.items():
        df[f"viol_{col}"] = df[col] > limit
    return df[["run", "trace", "lts", "sts", "arbiter", "window", "region", "model",
               "app_idx", "instances", *KEEP, *[f"viol_{c}" for c in SLO]]]


def main(root: str, out: str) -> None:
    frames = []
    for p in Path(root).rglob("summary/*.csv"):
        m = CFG.search(p.as_posix())
        if not m:
            continue
        arbiter_dir = p.parent.parent
        frames.append(load_summary(p, arbiter_dir, m.groupdict()))
    if not frames:
        sys.exit("no summary files matched - check the path layout")
    kpis = pd.concat(frames, ignore_index=True)
    kpis.to_csv(out, index=False)

    # quick comparison across policy configs (unweighted - see note in chat on request counts)
    g = kpis.groupby(["trace", "lts", "sts", "arbiter", "model"])
    print(g.agg(e2e_p99=("response_times_p99", "mean"),
                ttft_p99=("ttft_times_p99", "mean"),
                instances=("instances", "sum"),
                ttft_viol=("viol_ttft_times_p99", "mean")).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])