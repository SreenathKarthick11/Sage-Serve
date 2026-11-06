"""Overview page: what SageServe MLOps is and how to use this console."""
from __future__ import annotations

import streamlit as st

from core import REGISTRY_DIR, artifact_versions, read_json, registry_models
from ui import go, hero_svg


def render() -> None:
    st.markdown(
        '<div class="sg-hero"><div><h1>Serve LLMs with capacity that sees the load coming</h1>'
        "<p>SageServe is a forecast-aware autoscaler for large language model inference. "
        "This console is its MLOps layer: it produces versioned traffic data, trains and governs "
        "the forecasting models, replays traffic through the SageServe simulator, and turns the "
        "outcome into latency and SLA reports.</p></div>"
        f"<div>{hero_svg()}</div></div>",
        unsafe_allow_html=True,
    )

    # live status
    counts = {s: sum(len(artifact_versions(src, s)) for src in ("static", "streaming"))
              for s in ("raw", "clean", "forecasts")}
    active = read_json(REGISTRY_DIR / "active_model.json")
    c = st.columns(5)
    c[0].metric("Raw traces", counts["raw"])
    c[1].metric("Clean traces", counts["clean"])
    c[2].metric("Forecasts", counts["forecasts"])
    c[3].metric("Registered models", len(registry_models()))
    c[4].metric("Production model", f"{active.get('active_model_type','?')}" if active else "None")
    if not any(counts.values()):
        st.info("Nothing here yet. Start by generating a trace; the first pipeline run creates raw data, a clean copy and a forecast.")
        st.button("Run your first trace", type="primary", on_click=go, args=("Data & traces",))

    st.subheader("The workflow")
    st.caption("Work left to right. Each step reads what the previous one produced.")
    steps = [
        ("Data & traces", "Generate or stream request traces, clean them, and forecast load per minute."),
        ("Forecast models", "Register ARIMA or HMM forecasters, document them, and promote one to Production."),
        ("Simulator", "Replay a trace through SageServe with short-term and long-term scaling."),
        ("Results", "Turn simulator output into latency and SLA reports with diagnostic plots."),
    ]
    cols = st.columns(4)
    for i, (col, (page, text)) in enumerate(zip(cols, steps), start=1):
        with col.container(border=True):
            st.markdown(f'<div class="sg-num">{i}</div>', unsafe_allow_html=True)
            st.markdown(f"**{page}**")
            st.caption(text)
            st.button(f"Open", key=f"open-{page}", on_click=go, args=(page,), use_container_width=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Why forecasts matter")
        st.markdown(
            "LLM traffic rises and falls through the day, and starting a model instance takes time. "
            "If capacity only reacts after load arrives, requests wait and latency targets are missed; "
            "if capacity is always sized for the peak, GPUs sit idle.\n\n"
            "SageServe pairs two layers:\n"
            "- **Long-term scaling** uses a forecast of upcoming load to plan capacity ahead of time.\n"
            "- **Short-term scaling** reacts to what is happening now and absorbs forecast misses.\n\n"
            "That is why this console treats the forecaster as a governed, versioned model: "
            "a better forecast should lead to better scaling decisions, and the Results page lets you check that."
        )
    with right:
        st.subheader("Glossary")
        with st.expander("Trace, clean trace, forecast"):
            st.markdown("A **trace** is a log of LLM requests (arrival time, prompt size, …). "
                        "**ETL** cleans it. A **forecast** predicts request load per minute from the clean trace.")
        with st.expander("Static vs streaming"):
            st.markdown("**Static** traces are generated in one go from a request count and seed. "
                        "**Streaming** traces are produced over wall-clock time, sped up by a factor you choose.")
        with st.expander("Registry stages"):
            st.markdown("A model version moves through **Staging**, **Production** and **Archived**. "
                        "Production is gated: it needs measured **MAE ≤ 15** or **F1 ≥ 0.80**.")
        with st.expander("Report metrics"):
            st.markdown("**e2e_p99**: 99th-percentile end-to-end latency. "
                        "**viol_ttft** / **viol_avg_tbt**: violations of the time-to-first-token and time-between-tokens targets. "
                        "**fss**: the summary score computed by the inference report (see its provenance panel).")
        with st.expander("Lineage"):
            st.markdown("Every artifact records its parents, so you can trace a forecast back to the clean and raw data it came from.")
