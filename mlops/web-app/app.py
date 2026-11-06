"""SageServe MLOps console.   Run:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MLOps for SageServe", layout="wide")

import overview, views  # noqa: E402
from core import show_last_run  # noqa: E402
from ui import apply_theme, render_sidebar  # noqa: E402

apply_theme()
page = render_sidebar()

{
    "Overview": overview.render,
    "Data & traces": views.render_traces,
    "Forecast models": views.render_models,
    "Simulator": views.render_simulator,
    "Results": views.render_results,
}[page]()

show_last_run()