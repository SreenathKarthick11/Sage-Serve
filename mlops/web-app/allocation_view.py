"""Plot configured starts, known allocation states and requested/observed deltas."""
import hashlib
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def allocation_timeline(decisions, observations, initial, region):
    """Use one region's saved states without requiring synchronized regional samples."""
    initial = initial[initial.region == region]
    decisions = decisions[decisions.region == region]
    observations = observations[observations.region == region]
    keys = ["experiment_id", "case", "model", "region"]
    events = []
    for identity, starts in initial.groupby(keys):
        base = dict(zip(keys, identity))
        events.append({**base, "time_seconds": 0., "instances": starts.instances.sum(),
                       "requested_change": float("nan"), "target_instances": float("nan"),
                       "evidence": "Configured initial state", "event_order": 0})
        for source, field, kind, order in ((decisions, "current_instances", "Before optimizer decision", 1),
                                            (observations, "production_instances", "Exported snapshot", 2)):
            selected = source
            for key, value in base.items():
                selected = selected[selected[key] == value]
            for time, group in selected.groupby("time_seconds", sort=True):
                if group.region.duplicated().any():
                    continue
                event = {**base, "time_seconds": time, "instances": group[field].sum(),
                         "requested_change": float("nan"), "target_instances": float("nan"),
                         "evidence": kind, "event_order": order}
                if kind == "Before optimizer decision":
                    event.update(requested_change=group.requested_change.sum(),
                                 target_instances=group.target_instances.sum())
                events.append(event)
    timeline = pd.DataFrame(events)
    if timeline.empty:
        return timeline
    timeline = timeline.sort_values(keys + ["time_seconds", "event_order"], kind="stable")
    timeline["observed_change"] = timeline.groupby(keys).instances.diff()
    timeline.loc[timeline.evidence.eq("Configured initial state"), "observed_change"] = 0.
    timeline["time_minutes"] = timeline.time_seconds / 60
    timeline["lts_change"] = timeline.apply(
        lambda row: "Not applicable (LTS disabled)" if row.case.startswith(("Baseline", "STS only"))
        else "No decision at this sample" if pd.isna(row.requested_change)
        else f"{row.requested_change:+g}", axis=1)
    return timeline


def render_allocation(report):
    folder = Path(report).with_suffix("") / "allocation"
    st.subheader("Instance counts and allocation changes by region")
    if not folder.is_dir():
        st.info("No allocation evidence extracted for this report yet.")
        return
    try:
        metadata = json.loads((folder / "allocation.metadata.json").read_text(encoding="utf-8"))
        if metadata["report_sha256"] != hashlib.sha256(Path(report).read_bytes()).hexdigest():
            st.warning("This report changed after extraction. Extract its allocation evidence again.")
            return
        decisions = pd.read_csv(folder / "allocation_decisions.csv")
        observations = pd.read_csv(folder / "allocation_observations.csv")
        initial = pd.DataFrame(metadata.get("initial_allocations", []))
        if initial.empty:
            st.info("Extract allocation evidence again to include configured counts at time zero.")
            return
        regions = sorted(initial.region.unique())
        region = st.selectbox("Allocation region", regions,
                              index=regions.index("westus") if "westus" in regions else 0,
                              key=f"allocation-region-{Path(report).stem}")
        timeline = allocation_timeline(decisions, observations, initial, region)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.warning(f"Could not read allocation evidence: {exc}")
        return
    if timeline.empty:
        st.info("No allocation evidence is available for this report.")
        return
    policies = sorted(timeline.case.unique())
    configured_end = max(metadata.get("configured_end_seconds", {}).values(), default=0.)
    end_minutes = max(1., configured_end / 60, timeline.time_minutes.max())
    maximum = max(1., timeline.instances.max())
    models = sorted(timeline.model.unique())
    for start in range(0, len(models), 2):
        columns = st.columns(2)
        for column, model in zip(columns, models[start:start + 2]):
            with column:
                samples = timeline[timeline.model == model]
                samples = samples.assign(point_order=range(len(samples)))
                base = alt.Chart(samples).encode(
                    x=alt.X("time_minutes:Q", title="Simulated time (minutes)",
                            scale=alt.Scale(domain=[0, end_minutes], nice=False),
                            axis=alt.Axis(values=list(range(0, int(end_minutes), 10)) + [end_minutes])),
                    y=alt.Y("instances:Q", title="Instances",
                            scale=alt.Scale(domain=[0, maximum * 1.15], nice=False),
                            axis=alt.Axis(tickCount=3, format="d")),
                    color=alt.Color("case:N", legend=None,
                                    scale=alt.Scale(domain=policies,
                                                    range=["#2563eb", "#d97706", "#059669", "#9333ea"])),
                    detail="experiment_id:N",
                    order="point_order:Q")
                # Separate policy lanes prevent equal measurements from hiding each other.
                # Dashed lines join known states only; never extend to the configured end.
                lines = base.mark_line(strokeWidth=2, strokeDash=[5, 3], opacity=.65)
                points = base.mark_point(size=45, filled=True).encode(
                    tooltip=[alt.Tooltip("case:N", title="Policy"),
                             alt.Tooltip("region:N", title="Region"),
                             alt.Tooltip("time_seconds:Q", title="At simulated second"),
                             alt.Tooltip("evidence:N", title="Count source"),
                             alt.Tooltip("instances:Q", title="Known instance count"),
                             alt.Tooltip("lts_change:N", title="LTS requested change"),
                             alt.Tooltip("target_instances:Q", title="Requested target"),
                             alt.Tooltip("observed_change:Q", title="Count change since previous known state")])
                chart = (lines + points).properties(height=65).facet(
                    row=alt.Row("case:N", sort=policies, title=None,
                                header=alt.Header(labelAngle=0, labelOrient="top", labelAlign="left",
                                                  labelFontSize=11, labelPadding=4)),
                    spacing=10, title=f"Model {model} - {region}").configure(
                        background="white").configure_view(
                        fill="white", stroke="#e5e7eb").configure_axis(
                        labelColor="#475569", titleColor="#334155", gridColor="#e5e7eb").configure_header(
                        labelColor="#334155").configure_title(color="#0f172a", anchor="start")
                st.altair_chart(chart, width="stretch", theme=None)
