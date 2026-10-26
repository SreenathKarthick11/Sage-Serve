"""
loader.py  –  Reads SageServe simulation results from experiments/results.

Two public functions:
  load_summary(results_root)  ->  DataFrame of per-application summary stats
  load_detailed(results_root) ->  DataFrame of per-request measurements
  load_cost(results_root)     ->  DataFrame of VM scaling decisions (cost proxy)

All three attach run-metadata columns so you can group/filter by trace, lts,
sts, arbiter etc.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ── path pattern that extracts experiment metadata from the directory tree ──
_CFG_RE = re.compile(
    r"results/(?P<run>\d+)/(?P<trace>[^/]+)/.*?"
    r"/lts_(?P<lts>True|False)_sts(?P<sts>True|False)"
    r"/arbiter_(?P<arbiter>[^/]+)$"
)

_META_KEYS = ["run", "trace", "lts", "sts", "arbiter"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_run_dirs(results_root: str | Path):
    """Yield (run_dir, meta_dict) for every valid experiment leaf."""
    root = Path(results_root)
    for d in sorted(root.rglob("arbiter_*")):
        if not d.is_dir():
            continue
        if not (d / "detailed").is_dir():
            continue
        m = _CFG_RE.search(d.as_posix())
        if m:
            yield d, m.groupdict()


def _available_windows(run_dir: Path) -> list[int]:
    summary_dir = run_dir / "summary"
    if not summary_dir.is_dir():
        return []
    return sorted(int(p.stem) for p in summary_dir.glob("*.csv"))


def _attach_meta(df: pd.DataFrame, meta: dict, window: int | None = None) -> pd.DataFrame:
    for k in _META_KEYS:
        df[k] = meta[k]
    if window is not None:
        df["window"] = window
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_summary(results_root: str | Path) -> pd.DataFrame:
    """
    Load per-application summary stats from summary/<window>.csv files.

    Each row describes one application_id (e.g. westus_A_0) in one window.
    Rows with sentinel value -1 (inactive applications) are dropped.

    Returns
    -------
    pd.DataFrame with columns:
        application_id, window, <latency stats...>, run, trace, lts, sts, arbiter
    """
    frames: list[pd.DataFrame] = []

    for run_dir, meta in _iter_run_dirs(results_root):
        for w in _available_windows(run_dir):
            path = run_dir / "summary" / f"{w}.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            # Drop inactive applications (sentinel -1 in response_times_mean)
            if "response_times_mean" in df.columns:
                df = df[df["response_times_mean"] >= 0].copy()

            if df.empty:
                continue

            _attach_meta(df, meta, window=w)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_detailed(results_root: str | Path) -> pd.DataFrame:
    """
    Load per-request measurements from detailed/<app>/<window>.csv files.

    Each row is one request.  The application directory name is split into
    region and model for convenience.

    Returns
    -------
    pd.DataFrame with columns:
        region, model, application_id, window,
        response_times, queue_times, ttft_times, tbt_times,
        nth_token_overheads, prompt_sizes, token_sizes,
        run, trace, lts, sts, arbiter
    """
    frames: list[pd.DataFrame] = []

    for run_dir, meta in _iter_run_dirs(results_root):
        detail_root = run_dir / "detailed"
        if not detail_root.is_dir():
            continue

        for app_dir in sorted(detail_root.iterdir()):
            if not app_dir.is_dir():
                continue

            # Parse region_model_replica  e.g.  centralus_A_0
            parts = app_dir.name.split("_")
            if len(parts) < 3:
                continue
            region = parts[0]
            model = parts[1]
            app_id = app_dir.name

            for csv_file in sorted(app_dir.glob("*.csv")):
                try:
                    w = int(csv_file.stem)
                except ValueError:
                    continue

                try:
                    df = pd.read_csv(csv_file)
                except Exception:
                    continue

                if df.empty or "response_times" not in df.columns:
                    continue

                # Drop sentinel rows
                df = df[df["response_times"] >= 0].copy()
                if df.empty:
                    continue

                df["region"] = region
                df["model"] = model
                df["application_id"] = app_id
                df["window"] = w
                _attach_meta(df, meta)
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_cost(results_root: str | Path) -> pd.DataFrame:
    """
    Load VM scaling decisions from arbiters/<region>/<window>.csv.

    Returns
    -------
    pd.DataFrame with columns:
        region, model, timestamp, prod, spot, memory_util, window,
        run, trace, lts, sts, arbiter
    """
    frames: list[pd.DataFrame] = []

    for run_dir, meta in _iter_run_dirs(results_root):
        arbiter_root = run_dir / "arbiters"
        if not arbiter_root.is_dir():
            continue

        for region_dir in sorted(arbiter_root.iterdir()):
            if not region_dir.is_dir():
                continue
            region = region_dir.name

            for csv_file in sorted(region_dir.glob("*.csv")):
                try:
                    w = int(csv_file.stem)
                except ValueError:
                    continue

                try:
                    df = pd.read_csv(csv_file)
                except Exception:
                    continue

                if df.empty:
                    continue

                df["region"] = region
                df["window"] = w
                _attach_meta(df, meta)
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
