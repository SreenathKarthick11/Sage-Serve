"""Dedicated drift experiment view; never mixed into normal inference reports."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from core import REPORT_DIR

REGIONS = {0: "westus", 1: "centralus", 2: "eastus"}
COLORS = alt.Scale(domain=["Control", "Shifted"], range=["#2563eb", "#dc2626"])


def _show(chart):
    st.altair_chart(chart.configure(background="white").configure_view(fill="white").configure_axis(
        labelColor="#475569", titleColor="#334155", gridColor="#e5e7eb").configure_title(
        color="#0f172a", anchor="start"), width="stretch", theme=None)


def render_drift():
    st.subheader("Controlled workload drift")
    root = REPORT_DIR / "drift"
    studies = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []
    if not studies:
        st.info("No drift study prepared yet. See mlops/inference/DRIFT_EXPERIMENT.md.")
        return
    selected = st.selectbox("Drift study", studies, format_func=lambda p: p.parent.name, key="drift-study")
    study = selected.parent
    try:
        manifest = json.loads(selected.read_text(encoding="utf-8"))
        scores = pd.read_csv(study / "psi.csv")
        demand = pd.read_csv(study / "demand.csv")
    except (OSError, ValueError) as exc:
        st.warning(f"Drift evidence unavailable: {exc}")
        return
    onset = manifest["onset_seconds"] / 60
    region = int(manifest["target_region"])
    st.caption(f"Model {manifest['target_model']} in {REGIONS[manifest['target_region']]}: prompt sizes multiplied "
               f"by {manifest['prompt_multiplier']:g} from minute {onset:g}. "
               "Other requests and the pre-shift ARIMA forecast are unchanged. Dashed vertical lines mark the injected shift.")
    scores = scores[scores.region == region].copy()
    demand = demand[demand.region == region].copy()
    scores["minutes"] = scores.end_seconds / 60
    demand["minutes"] = demand.time_seconds / 60
    tabs = st.tabs(["Workload shift (PSI)", "Forecast quality"])
    with tabs[0]:
        feature = "prompt_size"
        st.caption(f"Fixed reference: unmodified control traffic in minutes 0–{onset:g}. "
                   f"Non-batched production requests are compared in full {manifest['window_seconds'] / 60:g}-minute windows. "
                   "0.10/0.25 are investigation thresholds, not proof of concept drift or serving failure.")
        for model in [manifest["target_model"]]:
            rows = scores[(scores.model == model) & (scores.feature == feature)].dropna(subset=["psi"])
            if rows.empty:
                st.markdown(f"**Model {model}**")
                st.info("Insufficient samples for this comparison.")
                continue
            chart = alt.Chart(rows).mark_line(point=True).encode(
                x=alt.X("minutes:Q", title="Window ending at simulated minute"),
                y=alt.Y("psi:Q", title="PSI", scale=alt.Scale(zero=True)),
                color=alt.Color("scenario:N", scale=COLORS, title="Trace"),
                tooltip=["scenario:N", "start_seconds:Q", "end_seconds:Q", "psi:Q", "status:N",
                         "reference_samples:Q", "current_samples:Q"])
            thresholds = alt.Chart(pd.DataFrame({"threshold": [.1, .25]})).mark_rule(
                strokeDash=[4, 4], color="#d97706").encode(y="threshold:Q")
            marker = alt.Chart(pd.DataFrame({"onset": [onset]})).mark_rule(
                color="#64748b", strokeDash=[2, 3]).encode(x="onset:Q")
            _show((chart + thresholds + marker).properties(title=f"Model {model}", height=240))
    with tabs[1]:
        metric = st.selectbox("Forecast view", ["Demand versus ARIMA forecast", "Rolling MAE (5 minutes)"],
                              key=f"drift-forecast-{study.name}")
        st.caption("Demand is production prompt tokens per minute in the first-priority source region. "
                   "Forecasts were fitted before replay and are not updated after the shift. "
                   "Forecast errors indicate prediction quality, not proved concept drift.")
        for model in [manifest["target_model"]]:
            rows = demand[demand.model == model]
            if metric.startswith("Demand"):
                actuals = rows[["minutes", "scenario", "actual"]].rename(columns={"scenario": "series", "actual": "value"})
                frozen = rows[rows.scenario == "Control"][["minutes", "predicted"]].rename(columns={"predicted": "value"})
                frozen["series"] = "ARIMA forecast (before shift)"
                plotted = pd.concat([actuals, frozen], ignore_index=True)
                color = alt.Color("series:N", scale=alt.Scale(domain=["Control", "Shifted", "ARIMA forecast (before shift)"],
                                                              range=["#2563eb", "#dc2626", "#059669"]))
            else:
                plotted = rows.rename(columns={"scenario": "series", "rolling_mae": "value"}).dropna(subset=["value"])
                color = alt.Color("series:N", scale=COLORS)
            chart = alt.Chart(plotted).mark_line().encode(
                x=alt.X("minutes:Q", title="Simulated minute (bucket start)"),
                y=alt.Y("value:Q", title="Prompt tokens per minute", scale=alt.Scale(zero=True)),
                color=color, tooltip=["series:N", "minutes:Q", "value:Q"])
            marker = alt.Chart(pd.DataFrame({"onset": [onset]})).mark_rule(
                color="#64748b", strokeDash=[2, 3]).encode(x="onset:Q")
            _show((chart + marker).properties(title=f"Model {model}", height=240))
