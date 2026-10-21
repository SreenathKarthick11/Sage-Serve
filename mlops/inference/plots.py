"""
plots.py  –  Visualisation for SageServe inference metrics.

All functions accept the DataFrame returned by ``compute_metrics`` and write
PNG files to a given output directory.

Generated plots
---------------
fss_bar.png         FSS per (model, config) as a horizontal bar chart.
                    This is the primary "at-a-glance" indicator.

sub_scores.png      Stacked bar showing each FSS sub-score contribution
                    (slo / stability / demand / cost / throughput) per row.

latency_pct.png     Grouped bar of e2e_p50 / p95 / p99 across models.

slo_violations.png  Dot-plot of viol_ttft and viol_tbt per (model, config).

cv_scatter.png      Scatter of cv_response vs demand_cv (coloured by FSS).
                    Points close to the origin are the easiest to forecast.

All functions return the Path to the saved file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

plt.rcParams.update({
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        10,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_label(row: pd.Series) -> str:
    lts = "LTS" if str(row.get("lts", "")) == "True" else "noLTS"
    return f"{row['model']} ({lts})"


def _labels(df: pd.DataFrame) -> list[str]:
    return [_row_label(r) for _, r in df.iterrows()]


def _fss_colour(fss: float) -> str:
    """Continuous green→red gradient mapped to FSS [0,100]."""
    r = 1.0 - fss / 100.0
    g = fss / 100.0
    return (r * 0.85, g * 0.75, 0.15)


# ---------------------------------------------------------------------------
# Plot 1 – FSS bar chart
# ---------------------------------------------------------------------------

def plot_fss_bar(df: pd.DataFrame, out_dir: Path) -> Path:
    """Horizontal bar chart of FSS per row, sorted highest→lowest."""
    data = df.sort_values("fss", ascending=True).reset_index(drop=True)
    labels = _labels(data)
    fss = data["fss"].values

    colours = [_fss_colour(v) for v in fss]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(data))))

    bars = ax.barh(labels, fss, color=colours, edgecolor="white", height=0.6)

    for bar, val in zip(bars, fss):
        ax.text(
            val + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center", ha="left", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Forecast Suitability Score (FSS)  [0 – 100]")
    ax.set_title(
        "FSS per Model / Config\n"
        "Higher = more forecastable, stable, SLO-compliant",
        fontsize=11,
    )
    ax.axvline(100, color="#aaa", lw=0.8, ls="--")

    fig.tight_layout()
    path = out_dir / "fss_bar.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2 – FSS sub-score stacked bar
# ---------------------------------------------------------------------------

_SUB_SCORE_COLS = [
    ("score_slo",        "SLO compliance",    "#55A868"),
    ("score_stability",  "Latency stability", "#4C72B0"),
    ("score_demand",     "Demand regularity", "#8172B3"),
    ("score_cost",       "Cost efficiency",   "#DD8452"),
    ("score_throughput", "Throughput signal", "#C44E52"),
]


def plot_sub_scores(df: pd.DataFrame, out_dir: Path) -> Path:
    """Stacked horizontal bar showing each sub-score's raw (un-weighted) value."""
    data = df.sort_values("fss", ascending=True).reset_index(drop=True)
    labels = _labels(data)
    n = len(data)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.55 * n)))

    lefts = np.zeros(n)
    for col, label, colour in _SUB_SCORE_COLS:
        if col not in data.columns:
            continue
        vals = data[col].fillna(0).values
        ax.barh(labels, vals, left=lefts, color=colour,
                edgecolor="white", height=0.6, label=label)
        lefts += vals

    ax.set_xlim(0, len(_SUB_SCORE_COLS) + 0.5)
    ax.set_xlabel("Sum of sub-scores  (each sub-score in [0, 1])")
    ax.set_title("FSS Sub-score Breakdown per Model / Config", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    fig.tight_layout()
    path = out_dir / "sub_scores.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3 – Latency percentiles
# ---------------------------------------------------------------------------

def plot_latency_pct(df: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar chart of e2e p50 / p95 / p99 per row."""
    labels = _labels(df)
    n = len(df)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    for i, (col, label, colour) in enumerate([
        ("e2e_p50", "p50", "#4C72B0"),
        ("e2e_p95", "p95", "#DD8452"),
        ("e2e_p99", "p99", "#C44E52"),
    ]):
        vals = df[col].fillna(0).values
        ax.bar(x + (i - 1) * width, vals, width, label=label,
               color=colour, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Response time  (seconds)")
    ax.set_title("End-to-End Response Time Percentiles", fontsize=11)
    ax.legend()

    fig.tight_layout()
    path = out_dir / "latency_pct.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 4 – SLO violations dot-plot
# ---------------------------------------------------------------------------

def plot_slo_violations(df: pd.DataFrame, out_dir: Path) -> Path:
    """Dot-plot: viol_ttft and viol_tbt per row."""
    labels = _labels(df)
    n = len(df)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * n)))

    for col, label, colour, marker in [
        ("viol_ttft", "TTFT > 10 s", "#C44E52", "o"),
        ("viol_tbt",  "TBT  > 0.2 s", "#DD8452", "s"),
    ]:
        vals = df[col].fillna(0).values * 100   # convert to %
        ax.scatter(vals, y, label=label, color=colour,
                   marker=marker, s=80, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("SLO violation rate  (%)")
    ax.set_title("SLO Violations per Model / Config\n(lower is better)", fontsize=11)
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.legend(loc="lower right")
    ax.grid(axis="x", ls="--", alpha=0.4)

    fig.tight_layout()
    path = out_dir / "slo_violations.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 5 – CV scatter
# ---------------------------------------------------------------------------

def plot_cv_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    cv_response vs demand_cv scatter, coloured by FSS.
    Points near the origin are the easiest to forecast.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    fss_vals = df["fss"].values
    scatter = ax.scatter(
        df["cv_response"], df["demand_cv"],
        c=fss_vals, cmap="RdYlGn", vmin=0, vmax=100,
        s=120, edgecolors="grey", linewidths=0.5, zorder=3,
    )

    for _, row in df.iterrows():
        ax.annotate(
            _row_label(row),
            (row["cv_response"], row["demand_cv"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color="#333",
        )

    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("FSS", rotation=270, labelpad=12)

    ax.set_xlabel("CV of response time  (lower = more stable)")
    ax.set_ylabel("CV of prompt-token demand  (lower = more regular)")
    ax.set_title(
        "Forecastability Space\n"
        "Bottom-left = easiest to forecast  |  colour = FSS",
        fontsize=11,
    )
    ax.grid(ls="--", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "cv_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------

def generate_all(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """
    Generate all five plots and return a list of saved file paths.

    Parameters
    ----------
    df      : DataFrame from compute_metrics()
    out_dir : Directory to save PNG files into (created if missing).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated = []
    for fn in [
        plot_fss_bar,
        plot_sub_scores,
        plot_latency_pct,
        plot_slo_violations,
        plot_cv_scatter,
    ]:
        path = fn(df, out)
        generated.append(path)
        print(f"  Saved plot: {path}")

    return generated
