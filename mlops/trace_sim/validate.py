"""Check a generated trace against the paper's characteristics and plot it (compare with Figs. 3-6)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TIERS = ["IW-F", "IW-N", "NIW"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def analyze(path: Path, plot: Path | None = None):
    df = pd.read_csv(path, usecols=["arrival_timestamp", "tier", "region", "model_type", "ttft",
                                    "e2e", "sla_met"])
    df["hour"] = (df.arrival_timestamp // 3600).astype(int)
    h0, h1 = df.hour.min(), df.hour.max()
    span_h = h1 - h0 + 1
    print(f"\n{len(df):,} requests over {span_h} simulated hours ({span_h / 24:.1f} days)\n")

    share = df.tier.value_counts(normalize=True)
    print("Tier share   (paper: IW-F + IW-N ~ 72%)")
    for t in TIERS:
        print(f"  {t:5s} {share.get(t, 0):6.1%}")
    print(f"  IW total {share.get('IW-F', 0) + share.get('IW-N', 0):.1%}\n")

    hourly = (df.groupby(["hour", "tier"]).size().unstack(fill_value=0)
              .reindex(range(h0, h1 + 1), fill_value=0))
    hourly["dow"] = (hourly.index // 24) % 7
    print("Periodicity  (lag-24h autocorrelation; paper: IW-F 0.7-0.95, IW-N 0.3-0.8, NIW 0-0.29)")
    for t in TIERS:
        ac = hourly[t].autocorr(lag=24) if span_h > 48 else float("nan")
        wk = hourly.loc[hourly.dow < 5, t].mean()
        we = hourly.loc[hourly.dow >= 5, t].mean()
        ratio = we / wk if wk and not pd.isna(we) else float("nan")
        print(f"  {t:5s} ACF24 = {ac:5.2f}   weekend/weekday mean = {ratio:5.2f}")

    print("\nRegion share (paper: East > Central > West)")
    print((df.region.value_counts(normalize=True)).round(3).to_string())
    print("\nIW-F share by model within each region")
    print(df[df.tier == "IW-F"].groupby("region").model_type.value_counts(normalize=True).round(2).unstack().to_string())

    iw = df[df.tier != "NIW"]
    print(f"\nSLA met: " + ", ".join(f"{t} {df[df.tier == t].sla_met.mean():.1%}" for t in TIERS))
    print(f"IW-F TTFT p50/p95: {df[df.tier == 'IW-F'].ttft.quantile(.5):.2f}s / "
          f"{df[df.tier == 'IW-F'].ttft.quantile(.95):.2f}s")

    # ---- plots ----
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    x = hourly.index / 24
    for t in TIERS:
        ax[0].plot(x, hourly[t] / 3600, label=t)
    ax[0].set_ylabel("requests / sim-sec"); ax[0].set_title("Hourly request rate by tier (cf. Fig. 3c)"); ax[0].legend()
    f = df[df.tier == "IW-F"].groupby(["hour", "region"]).size().unstack(fill_value=0).reindex(range(h0, h1 + 1), fill_value=0)
    for r in f.columns:
        ax[1].plot(f.index / 24, f[r] / 3600, label=r)
    ax[1].set_ylabel("requests / sim-sec"); ax[1].set_title("IW-F by region (cf. Fig. 4a-c)"); ax[1].legend()
    p95 = iw[iw.tier == "IW-F"].groupby("hour").ttft.quantile(.95).reindex(range(h0, h1 + 1))
    ax[2].plot(p95.index / 24, p95.values, color="tab:red")
    ax[2].set_ylabel("IW-F TTFT p95 (s)"); ax[2].set_title("Latency under load / bursts (cf. Fig. 6d)")
    ax[2].set_xlabel("simulated day (0 = Monday)")
    for a_ in ax:
        a_.set_xticks(range(int(x.min()), int(x.max()) + 2))
        a_.grid(alpha=.3)
    plot = plot or Path(path).with_suffix(".png")
    fig.tight_layout(); fig.savefig(plot, dpi=120)
    print(f"\nplot saved: {plot}")
