"""Headless diagnostic plots. Unknown measurements are never drawn as zero."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
FIGURE_SIZE = (12, 6)
FIGURE_DPI = 130


def _row_label(row):
    return (f"{row['model']} | run {row.get('run', '?')} | "
            f"{str(row.get('experiment_id', '?'))[:8]}\n"
            f"LTS={row.get('lts', '?')} STS={row.get('sts', '?')} "
            f"async={row.get('feed_async', '?')} scale={row.get('scaling_level', '?')}")


def _labels(df):
    return [_row_label(row) for _, row in df.iterrows()]


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Explicit margins reserve label/legend space without changing the canvas.
        with matplotlib.rc_context({"savefig.bbox": None}):
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches=None)
    finally:
        plt.close(fig)
    return path


def _horizontal(df):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=.34, right=.97, bottom=.20, top=.90)
    y = np.arange(len(df))
    ax.set_yticks(y, _labels(df), fontsize=max(6, min(8, 40 / max(1, len(df)))))
    ax.set_ylim(-.65, len(df)-.35)
    return fig, ax, y


def _unknown(ax, positions, values, label="unknown"):
    for position, value in zip(positions, values):
        if not np.isfinite(value):
            ax.text(.01, position, label, transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8, color="#666")


def plot_rfws_bar(df, out_dir):
    data = df.sort_values("rfws", na_position="first").reset_index(drop=True)
    fig, ax, y = _horizontal(data)
    valid = data.rfws.notna().to_numpy()
    ax.barh(y[valid], data.rfws[valid], color="#4C72B0")
    for position, (_, row) in zip(y, data.iterrows()):
        text = (f"{row.rfws:.1f}" if pd.notna(row.rfws) else
                textwrap.fill(f"unavailable: {row.rfws_status}", width=58))
        ax.text(row.rfws+1 if pd.notna(row.rfws) else 1, position, text, va="center", fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Relative rfws (0–100); compare only within the same cohort")
    ax.set_title("Relative scores by experiment and model")
    return _save(fig, out_dir, "rfws_bar.png")


def plot_sub_scores(df, out_dir):
    """Stack actual normalized, weighted contributions; their sum equals rfws."""
    fig, ax, y = _horizontal(df)
    valid = df.rfws.notna().to_numpy()
    left = np.zeros(int(valid.sum()))
    names = [("slo", "Benchmark compliance"), ("stability", "Response stability"),
             ("demand", "Prompt-size regularity"), ("cost", "Instance-hour proxy"),
             ("throughput", "Completions per window")]
    for color, (name, label) in zip(_COLORS, names):
        values = df.loc[valid, f"contribution_{name}"].to_numpy()
        ax.barh(y[valid], values, left=left, color=color, label=label)
        left += values
    _unknown(ax, y, df.rfws, "rfws unavailable; see report status")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Weighted contribution in rfws points (sum = rfws)")
    ax.set_title("What contributes to the relative score")
    handles = [Patch(facecolor=color, label=label) for color, (_, label) in zip(_COLORS, names)]
    fig.legend(handles=handles, fontsize=7, loc="lower center",
               bbox_to_anchor=(.65, .02), ncol=2)
    return _save(fig, out_dir, "sub_scores.png")


def plot_latency_pct(df, out_dir):
    fig, ax, y = _horizontal(df)
    for offset, column, label, color in zip([-.23, 0, .23], ["e2e_p50", "e2e_p95", "e2e_p99"],
                                           ["p50", "p95", "p99"], _COLORS):
        values = df[column].to_numpy()
        valid = np.isfinite(values)
        ax.barh(y[valid]+offset, values[valid], height=.22, label=label, color=color)
    _unknown(ax, y, df.e2e_p50)
    ax.set_xlabel("End-to-end response time (seconds)")
    ax.set_title("Pooled request latency percentiles")
    ax.legend()
    return _save(fig, out_dir, "latency_pct.png")


def plot_slo_violations(df, out_dir):
    fig, ax, y = _horizontal(df)
    ttft = df.ttft_threshold_seconds.iloc[0]
    tbt = df.avg_tbt_threshold_seconds.iloc[0]
    for column, label, marker, color, offset in [
        ("viol_ttft", f"TTFT > {ttft:g} s", "o", _COLORS[3], -.1),
        ("viol_avg_tbt", f"Request-average TBT > {tbt:g} s", "s", _COLORS[1], .1),
    ]:
        values = df[column].to_numpy()*100
        valid = np.isfinite(values)
        ax.scatter(values[valid], y[valid]+offset, label=label, marker=marker, color=color)
        _unknown(ax, y+offset, values)
    ax.set_xlim(-1, 101)
    ax.set_xlabel("Threshold violation fraction among valid measurements (%)")
    ax.set_title("Fixed benchmark thresholds; original request SLAs are unavailable")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "slo_violations.png")


def plot_cv_scatter(df, out_dir):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=.11, right=.91, bottom=.20, top=.90)
    valid = df.cv_response.notna() & df.prompt_size_cv.notna()
    scored = valid & df.rfws.notna()
    if scored.any():
        points = ax.scatter(df.loc[scored, "cv_response"], df.loc[scored, "prompt_size_cv"],
                            c=df.loc[scored, "rfws"], vmin=0, vmax=100, cmap="viridis", s=70)
        fig.colorbar(points, ax=ax, label="Relative rfws (same cohort only)")
    unscored = valid & df.rfws.isna()
    if unscored.any():
        ax.scatter(df.loc[unscored, "cv_response"], df.loc[unscored, "prompt_size_cv"],
                   c="#777", s=70, label="rfws unavailable")
        ax.legend()
    for _, row in df[valid].iterrows():
        ax.annotate(f"{row.model} | {row.experiment_id[:8]}", (row.cv_response, row.prompt_size_cv),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    if not valid.all():
        ax.text(.02, .98, f"{int((~valid).sum())} rows omitted: insufficient measurements",
                transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("Response-time CV")
    ax.set_ylabel("Per-request prompt-size CV")
    ax.set_title("Variability of latency and request sizes (not forecast accuracy)")
    ax.grid(alpha=.2)
    return _save(fig, out_dir, "cv_scatter.png")


def plot_resource_usage(df, out_dir):
    fig, ax, y = _horizontal(df)
    prod, spot = df.prod_vm_h.to_numpy(), df.spot_vm_h.to_numpy()
    valid = np.isfinite(prod) & np.isfinite(spot)
    ax.barh(y[valid], prod[valid], color=_COLORS[0], label="Production")
    ax.barh(y[valid], spot[valid], left=prod[valid], color=_COLORS[1], label="Spot")
    _unknown(ax, y, prod+spot)
    for position, (_, row) in zip(y, df.iterrows()):
        if pd.notna(row.prod_vm_h):
            ax.annotate(row.resource_coverage_status, (row.prod_vm_h+row.spot_vm_h, position),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
    ax.margins(x=.4)
    ax.set_xlabel("Observed instance-hours (resource proxy; no prices applied)")
    ax.set_title("Resource accounting coverage; partial intervals are not total run usage")
    ax.legend()
    return _save(fig, out_dir, "resource_usage.png")


def generate_all(df, out_dir, *, windows=None):
    if df.empty:
        raise ValueError("Cannot plot an empty inference report")
    functions = [plot_rfws_bar, plot_sub_scores, plot_latency_pct, plot_slo_violations,
                 plot_cv_scatter, plot_resource_usage]
    paths = [function(df, out_dir) for function in functions]
    return paths
