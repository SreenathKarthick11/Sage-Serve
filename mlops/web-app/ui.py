"""Theme (forest palette), sidebar navigation and small HTML helpers."""
from __future__ import annotations

import html

import streamlit as st

from core import (DATA_ROOT, MLOPS_PYTHON, REGISTRY_DIR, SIM_PYTHON, artifact_versions,
                  read_json, registry_models)

NAV = ["Overview", "Data & traces", "Forecast models", "Simulator", "Results"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--forest:#12372A;--moss:#436850;--sage:#ADBC9F;--cream:#FBFADA;--paper:#FFFEF2;--line:#d6dbb8;--muted:#4f6a58;}
html,body,[data-testid="stAppViewContainer"],.stMarkdown,button,input,textarea,select{font-family:'DM Sans',system-ui,sans-serif;}
code,pre,[data-testid="stCode"] *{font-family:'JetBrains Mono',ui-monospace,monospace !important;}
[data-testid="stAppViewContainer"]{background:var(--cream);color:var(--forest);}
[data-testid="stHeader"]{background:rgba(251,250,218,.85);backdrop-filter:blur(6px);}
.block-container{max-width:1180px;padding-top:1.6rem;padding-bottom:4rem;}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',sans-serif;color:var(--forest);letter-spacing:-.015em;font-weight:600;}
h2{font-size:1.5rem;} h3{font-size:1.15rem;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--forest);}
[data-testid="stSidebar"] *{color:var(--cream);}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{padding-bottom:0;}
.sg-brand{font-family:'Bricolage Grotesque',sans-serif;font-size:1.45rem;font-weight:700;margin:.2rem 0 0;}
.sg-brand small{display:block;font-family:'DM Sans';font-size:.8rem;font-weight:400;color:var(--sage);margin-top:.1rem;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.15rem;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.5rem .75rem;border-radius:8px;width:100%;cursor:pointer;border-left:3px solid transparent;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(173,188,159,.14);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:rgba(173,188,159,.22);border-left-color:var(--cream);font-weight:600;}
[data-testid="stSidebar"] hr{border-color:rgba(173,188,159,.3);}
.sg-side{display:flex;justify-content:space-between;align-items:center;padding:.2rem 0;font-size:.86rem;}
.sg-side span:first-child{color:var(--sage);}
[data-testid="stSidebar"] button{background:transparent;border:1px solid var(--sage);color:var(--cream);}

/* panels, metrics */
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line) !important;background:var(--paper);border-radius:14px;}
[data-testid="stMetric"]{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--moss);border-radius:10px;padding:.7rem .95rem;}
[data-testid="stMetricLabel"] p{color:var(--muted);font-size:.82rem;}
[data-testid="stMetricValue"]{font-family:'Bricolage Grotesque';font-weight:600;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:10px;background:var(--paper);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px;}

/* inputs */
div[data-baseweb="input"] > div,div[data-baseweb="select"] > div,div[data-baseweb="textarea"]{background:#fffff6 !important;border-color:var(--line) !important;}
div[data-testid="stButton"] > button{border-radius:9px;border:1px solid var(--moss);color:var(--forest);background:transparent;font-weight:500;}
div[data-testid="stButton"] > button:hover{background:rgba(67,104,80,.1);border-color:var(--forest);color:var(--forest);}
div[data-testid="stButton"] > button[kind="primary"]{background:var(--forest);color:var(--cream);border-color:var(--forest);}
div[data-testid="stButton"] > button[kind="primary"]:hover{background:var(--moss);border-color:var(--moss);color:var(--cream);}
button:focus-visible{outline:2px solid var(--moss) !important;outline-offset:2px;}
[data-testid="stTabs"] [role="tablist"]{gap:1.2rem;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--moss);font-weight:600;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:var(--moss);}

/* hero */
.sg-hero{background:var(--forest);color:var(--cream);border-radius:18px;padding:2rem 2.2rem;display:grid;grid-template-columns:1.1fr 1fr;gap:1.5rem;align-items:center;margin-bottom:1.2rem;}
.sg-hero h1{color:var(--cream);font-size:2.3rem;line-height:1.1;margin:0 0 .7rem;}
.sg-hero p{color:#dfe5c6;font-size:1.02rem;line-height:1.55;margin:0;max-width:34rem;}
.sg-hero svg{width:100%;height:auto;}
.sg-page{margin:0 0 1rem;}
.sg-page h1{font-size:2rem;margin:0;} .sg-page p{color:var(--muted);margin:.25rem 0 0;max-width:46rem;}
@media (max-width:800px){.sg-hero{grid-template-columns:1fr;padding:1.4rem;}}

/* pills, flow */
.sg-pill{display:inline-block;padding:.1rem .6rem;border-radius:999px;font-size:.78rem;font-weight:500;border:1px solid var(--sage);color:var(--sage);}
.sg-pill.good{background:var(--sage);color:var(--forest);border-color:var(--sage);}
.sg-pill.warn{color:#f3d58a;border-color:#f3d58a;}
.sg-pill.bad{color:#f4a39b;border-color:#f4a39b;}
.sg-pill.light{color:var(--moss);border-color:var(--moss);}
.sg-flow{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;margin:.2rem 0 .6rem;}
.sg-step{padding:.3rem .8rem;border-radius:8px;border:1px dashed var(--sage);color:var(--muted);font-size:.88rem;}
.sg-step.on{border:1px solid var(--forest);background:var(--forest);color:var(--cream);}
.sg-arrow{color:var(--sage);}
.sg-mode{border-left:4px solid var(--moss);background:rgba(173,188,159,.22);padding:.6rem .9rem;border-radius:0 10px 10px 0;margin:.4rem 0 .8rem;}
.sg-num{font-family:'Bricolage Grotesque';font-size:1.8rem;font-weight:700;color:var(--moss);line-height:1;}
@media (prefers-reduced-motion:reduce){*{transition:none !important;animation:none !important;}}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="sg-page"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def pill(text, kind: str = "") -> str:
    return f'<span class="sg-pill {kind}">{html.escape(str(text))}</span>'


def flow(active: list[str], steps=("ingest", "etl", "forecast")) -> None:
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="sg-step {"on" if s in active else ""}">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="sg-arrow">›</span>')
    st.markdown(f'<div class="sg-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def go(page: str) -> None:
    """on_click callback: switch page."""
    st.session_state["nav"] = page


def render_sidebar() -> str:
    st.session_state.setdefault("nav", "Overview")
    with st.sidebar:
        st.markdown('<div class="sg-brand">SageServe<small>MLOps console</small></div>', unsafe_allow_html=True)
        st.write("")
        page = st.radio("Navigate", NAV, key="nav", label_visibility="collapsed")
        st.divider()
        def row(label, value):
            st.markdown(f'<div class="sg-side"><span>{html.escape(label)}</span>{value}</div>', unsafe_allow_html=True)
        row("MLOps env", pill("ready", "good") if MLOPS_PYTHON.is_file() else pill("missing", "bad"))
        row("Simulator env", pill("ready", "good") if SIM_PYTHON.is_file() else pill("missing", "bad"))
        active = read_json(REGISTRY_DIR / "active_model.json")
        row("Production", pill(f"{active.get('active_model_type','?')} / {active.get('active_version','?')}", "good")
            if active else pill("none", "warn"))
        row("Models", pill(len(registry_models())))
        st.caption(f"Data: `{DATA_ROOT}`")
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    return page


def hero_svg() -> str:
    """Demand (curve) vs capacity (steps): the idea behind forecast-aware scaling."""
    return (
        '<svg viewBox="0 0 560 220" role="img" aria-label="Demand curve with stepped capacity following it">'
        '<g stroke="#436850" stroke-width="1" opacity=".6"><line x1="0" y1="200" x2="560" y2="200"/></g>'
        '<path d="M0,170 H70 V140 H150 V90 H230 V120 H310 V80 H390 V40 H470 V60 H560" fill="none" stroke="#FBFADA" stroke-width="3"/>'
        '<path d="M0,150 C60,140 100,70 160,80 S260,150 320,110 S440,25 560,50" fill="none" stroke="#ADBC9F" stroke-width="3" stroke-dasharray="2 7" stroke-linecap="round"/>'
        '<text x="8" y="214" fill="#ADBC9F" font-size="12" font-family="DM Sans">time</text>'
        '<text x="400" y="26" fill="#FBFADA" font-size="12" font-family="DM Sans">GPU capacity (steps)</text>'
        '<text x="250" y="168" fill="#ADBC9F" font-size="12" font-family="DM Sans">request load (dotted)</text></svg>'
    )
