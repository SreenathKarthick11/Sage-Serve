"""Data & traces, Forecast models, Simulator and Results pages."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
import yaml

from core import (MLOPS_DIR, MLOPS_PYTHON, REGISTRY_DIR, REPO_ROOT, REPORT_DIR, RESULTS_ROOT,
                  SIM_DIR, SIM_PYTHON, artifact_versions, last_job, lineage_records, read_json,
                  registry_models, run_command, version_labels)
from ui import flow, go, page_header, pill

MOSS, SAGE = "#436850", "#ADBC9F"


# ═══════════════════════════ Data & traces ═══════════════════════════
STAGE_OPTIONS = {
    "Ingest → ETL → forecast": ("ingest,etl,forecast", "Full run: new trace, cleaned, then forecast."),
    "Ingest only": ("ingest", "Only produce a new raw trace."),
    "ETL + forecast latest raw": ("etl,forecast", "Clean the latest raw trace and forecast it."),
    "Forecast latest clean": ("forecast", "Forecast an existing clean trace."),
}


def _select_artifact(source: str, stage: str, key: str):
    versions = artifact_versions(source, stage)
    if not versions:
        return None
    labels = version_labels(versions)
    selected = st.selectbox(f"{stage.title()} version", labels, key=key)
    _, path, manifest = versions[labels.index(selected)]
    return path, manifest


def _show_artifact(source: str, stage: str, key: str) -> None:
    selected = _select_artifact(source, stage, key)
    if selected is None:
        st.info(f"No {source} {stage} data yet. Run the pipeline above to create it.")
        return
    artifact_dir, manifest = selected
    details = manifest.get("details", {})
    m = st.columns(4)
    m[0].metric("Version", manifest.get("version", artifact_dir.name))
    m[1].metric("Rows", details.get("rows", details.get("rows_out", details.get("rows_in", "—"))))
    m[2].metric("Status", details.get("status", "available"))
    m[3].metric("Created", str(manifest.get("created_at", "—"))[:19])
    st.caption(f"Artifact directory: `{artifact_dir}`")

    csv_files = sorted(artifact_dir.glob("*.csv"))
    if not csv_files:
        st.info("This artifact has no CSV data files.")
        st.json({"parents": manifest.get("parents", []), "params": manifest.get("params", {}), "details": details})
        return
    selected_csv = st.selectbox("Data file", csv_files, format_func=lambda p: p.name, key=f"{key}-file")
    try:
        data = pd.read_csv(selected_csv, nrows=100_000, dtype={"regions": "string"})
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read `{selected_csv.name}`: {exc}")
        return

    if stage == "forecasts" and "Predicted" in data.columns:
        chart = data[["Time", "Predicted"]].set_index("Time") if "Time" in data.columns else data[["Predicted"]]
        st.line_chart(chart, height=260, color=MOSS)
        values = pd.to_numeric(data["Predicted"], errors="coerce").dropna()
        if not values.empty:
            a, b, c = st.columns(3)
            a.metric("Forecast points", f"{len(values):,}")
            b.metric("Mean predicted load", f"{values.mean():,.1f}")
            c.metric("Peak predicted load", f"{values.max():,.1f}")
    elif {"arrival_timestamp", "prompt_size"} <= set(data.columns):
        data["arrival_timestamp"] = pd.to_numeric(data["arrival_timestamp"], errors="coerce")
        data["prompt_size"] = pd.to_numeric(data["prompt_size"], errors="coerce")
        sampled = data.dropna(subset=["arrival_timestamp", "prompt_size"]).copy()
        if not sampled.empty:
            sampled["minute"] = (sampled["arrival_timestamp"] // 60).astype("int64")
            activity = sampled.groupby("minute").agg(requests=("prompt_size", "size"), mean_prompt=("prompt_size", "mean"))
            l, r = st.columns(2)
            l.caption("Requests per minute")
            l.area_chart(activity[["requests"]], height=220, color=MOSS)
            r.caption("Mean prompt size per minute")
            r.line_chart(activity[["mean_prompt"]], height=220, color="#12372A")
            st.caption("Charts use up to the first 100,000 rows of the selected CSV.")

    st.markdown("**Data preview**")
    st.dataframe(data.head(100), width="stretch", hide_index=True)
    with st.expander("Artifact details"):
        st.json({"id": manifest.get("id"), "parents": manifest.get("parents", []),
                 "parameters": manifest.get("params", {}), "details": details, "files": manifest.get("files", {})})


def render_traces() -> None:
    page_header("Data & traces", "Create a request trace, clean it, forecast its load, then inspect every artifact and where it came from.")
    with st.container(border=True):
        st.subheader("Run a trace")
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            source = st.radio("Trace type", ["static", "streaming"], horizontal=True)
            names = list(STAGE_OPTIONS)
            stages_choice = st.radio("Stages to run", names, captions=[STAGE_OPTIONS[n][1] for n in names])
            stages = STAGE_OPTIONS[stages_choice][0]
        with right:
            flow(stages.split(","))
            if source == "static":
                c1, c2 = st.columns(2)
                num_requests = c1.number_input("Number of requests", min_value=1, value=5000, step=1000)
                seed = c2.number_input("Trace seed", min_value=0, value=42, step=1)
                run_args = ["--num-requests", str(num_requests), "--seed", str(seed)]
            else:
                c1, c2, c3 = st.columns(3)
                duration = c1.number_input("Duration (wall-clock s)", min_value=1, value=600, step=60)
                speedup = c2.number_input("Simulation speedup", min_value=0.1, value=300.0, step=10.0)
                rate_scale = c3.number_input("Traffic rate scale", min_value=0.0001, value=0.002, format="%.4f")
                seed = st.number_input("Trace seed", min_value=0, value=42, step=1)
                run_args = ["--duration", str(duration), "--speedup", str(speedup),
                            "--rate-scale", str(rate_scale), "--seed", str(seed)]
            if stages == "forecast":
                clean_versions = artifact_versions(source, "clean")
                if clean_versions:
                    options = ["latest"] + [v for v, _, _ in clean_versions]
                    input_version = st.selectbox("Clean data version", list(dict.fromkeys(options)))
                else:
                    input_version = "latest"
                    st.warning("No clean trace yet. Choose “Ingest → ETL → forecast” first.")
                run_args.extend(["--input-version", input_version])
        if st.button("Run trace pipeline", type="primary"):
            run_command([str(MLOPS_PYTHON), "-m", "pipeline.run_pipeline", "--source", source,
                         "--stages", stages, *run_args], MLOPS_DIR, f"{source.title()} trace pipeline")

    with st.container(border=True):
        st.subheader("Browse data")
        c1, c2 = st.columns(2)
        data_source = c1.radio("Data source", ["static", "streaming"], horizontal=True, key="data_source")
        data_stage = c2.radio("Artifact stage", ["raw", "clean", "forecasts"], horizontal=True, key="data_stage")
        _show_artifact(data_source, data_stage, "data-artifact")

    with st.container(border=True):
        st.subheader("Lineage")
        _lineage(data_source)


def _lineage(data_source: str) -> None:
    records = lineage_records()
    source_records = [r for r in records if r.get("source") == data_source and r.get("id")]
    source_records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    if not source_records:
        st.info(f"No {data_source} lineage yet. It appears after the first pipeline run.")
        return
    by_id = {r.get("id"): r for r in records if r.get("id")}
    chosen = st.selectbox("Artifact", list(dict.fromkeys(r["id"] for r in source_records)), key="lineage-artifact")
    chain, seen, pending = [], set(), [chosen]
    while pending:
        node = pending.pop(0)
        if node in seen:
            continue
        seen.add(node)
        rec = by_id.get(node)
        chain.append({"Artifact": node, "Created": rec.get("created_at", "") if rec else "",
                      "Parent(s)": ", ".join(rec.get("parents", [])) if rec else "Not in ledger",
                      "Stage": rec.get("stage", "") if rec else ""})
        if rec:
            pending.extend(rec.get("parents", []))
    st.dataframe(pd.DataFrame(chain), width="stretch", hide_index=True)
    rec = by_id.get(chosen, {})
    with st.expander("Parameters and artifact details"):
        st.json({"parameters": rec.get("params", {}), "details": rec.get("details", {}), "files": rec.get("files", {})})


# ═══════════════════════════ Forecast models ═══════════════════════════
def render_models() -> None:
    page_header("Forecast models", "Register the models that predict load, document them, and decide which one is in Production.")
    active = read_json(REGISTRY_DIR / "active_model.json")
    if active:
        st.success(f"Production model: **{active.get('active_model_type', 'Unknown')} / {active.get('active_version', 'Unknown')}**")
    else:
        st.info("No model is in Production yet. Register one below, then promote it.")

    with st.container(border=True):
        _register()
    _versions()


def _register() -> None:
    st.subheader("Register a model")
    model_type = st.segmented_control("Model type", ["arima", "hmm"], default="arima", key="model-type") or "arima"
    description = st.text_input("Model description", f"{model_type.upper()} workload forecaster")
    c1, c2, c3 = st.columns(3)
    parameters = {
        "train_window_mins": int(c1.number_input("Training window (minutes)", min_value=1, value=60)),
        "future_horizon": int(c2.number_input("Forecast horizon (minutes)", min_value=1, value=60)),
        "bucket_seconds": int(c3.number_input("Bucket size (seconds)", min_value=1, value=60)),
    }
    uploaded, train_from = None, ""
    if model_type == "arima":
        p, d, q = st.columns(3)
        parameters["order"] = {
            "p": int(p.number_input("AR order (p)", min_value=0, value=1)),
            "d": int(d.number_input("Difference order (d)", min_value=0, value=1)),
            "q": int(q.number_input("MA order (q)", min_value=0, value=1)),
        }
        st.caption("ARIMA is fitted to each workload series when you run a forecast; it has no saved weights file.")
    else:
        parameters["n_states"] = int(st.number_input("HMM states", min_value=2, value=3))
        clean_sources = {f"{s} / {v}": path for s in ("static", "streaming")
                         for v, path, _ in artifact_versions(s, "clean")}
        mode = st.radio("HMM weights", ["Train from a clean trace", "Upload an existing weights file"], horizontal=True)
        if mode == "Train from a clean trace":
            if clean_sources:
                train_from = str(clean_sources[st.selectbox("Training trace", list(clean_sources))])
            else:
                st.warning("Run ETL to create a clean trace before training an HMM.")
        else:
            uploaded = st.file_uploader("Weights file (.pkl)", type=["pkl"])

    model_config = {"model_info": {"type": model_type, "description": description}, "parameters": parameters}
    with st.expander("Config file preview"):
        st.code(yaml.safe_dump(model_config, sort_keys=False), language="yaml")
    metrics_input = st.text_input("Measured governance metrics", placeholder="mae=12.4 or f1=0.84",
                                  help="Enter real measured metrics only. Production requires MAE <= 15 or F1 >= 0.80.")
    can_register = model_type != "hmm" or bool(train_from or uploaded)
    label = ("Train and register HMM" if model_type == "hmm" and train_from
             else "Register HMM with weights" if model_type == "hmm" else "Register ARIMA")
    if st.button(label, type="primary", disabled=not can_register):
        metrics = [i.strip() for i in metrics_input.replace(",", " ").split() if i.strip()]
        tmp = Path(tempfile.mkdtemp(prefix="register_"))
        config_path = tmp / "config.yaml"
        config_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
        command = [str(MLOPS_PYTHON), "-m", "model_registry.run", "register", "--name", model_type, "--config", str(config_path)]
        if metrics:
            command.extend(["--metrics", *metrics])
        if train_from:
            command.extend(["--train-from", train_from])
        elif uploaded:
            weights_file = tmp / Path(uploaded.name).name
            weights_file.write_bytes(uploaded.getvalue())
            command.extend(["--weights", str(weights_file)])
        run_command(command, MLOPS_DIR, f"Register {model_type.upper()} model",
                    extra={"cleanup_dir": str(tmp)})

def _versions() -> None:
    models = registry_models()
    with st.container(border=True):
        st.subheader("Registered versions")
        if not models:
            st.info("No model versions registered yet.")
            return
        st.dataframe(pd.DataFrame([{
            "Model": m["name"], "Version": m["version"], "Stage": m["manifest"].get("stage", "None"),
            "Model card": "Ready" if m["has_card"] else "Missing",
            "Metrics": json.dumps(m["manifest"].get("metrics", {}))} for m in models]),
            width="stretch", hide_index=True)
        choices = {f"{m['name']} / {m['version']}": m for m in models}
        label = st.selectbox("Manage version", list(choices))
        selected = choices[label]
        with st.expander("Registered config and manifest"):
            cfg = selected["path"] / "config.yaml"
            if cfg.is_file():
                st.code(cfg.read_text(encoding="utf-8"), language="yaml")
            st.json(selected["manifest"])
        card_tab, promote_tab = st.tabs(["Model card", "Promote"])
        with card_tab:
            c1, c2 = st.columns(2)
            desc = c1.text_input("Card description", "Workload forecasting model", key="card-description")
            use = c2.text_input("Intended use", "Forecast per-minute SageServe request workload.", key="card-intended-use")
            lim = st.text_input("Limitations", "Forecast quality depends on trace history and model parameters.", key="card-limitations")
            if st.button("Save model card"):
                run_command([str(MLOPS_PYTHON), "-m", "model_registry.run", "card", "--name", selected["name"],
                             "--version", selected["version"], "--description", desc,
                             "--intended-use", use, "--limitations", lim], MLOPS_DIR, "Save model card")
        with promote_tab:
            stage = st.segmented_control("Promote to stage", ["Staging", "Production", "Archived"],
                                         default="Staging", key="promote-stage") or "Staging"
            if st.button("Promote model", type="primary"):
                run_command([str(MLOPS_PYTHON), "-m", "model_registry.run", "promote", "--name", selected["name"],
                             "--version", selected["version"], "--stage", stage],
                            MLOPS_DIR, f"Promote {label} to {stage}")


# ═══════════════════════════ Simulator ═══════════════════════════
MODES = {
    (True, True): ("Full SageServe", "Capacity is planned from the forecast and corrected live."),
    (True, False): ("Short-term only", "Live reaction to load, with no forecast-based planning."),
    (False, True): ("Long-term only", "Forecast-based planning, with no live correction."),
    (False, False): ("Scaling off", "Fixed capacity: use this as the baseline to compare against."),
}


def _use_run(path: str) -> None:
    st.session_state["inference_results_dir"] = path
    st.session_state["inference-results-dir"] = path
    st.session_state["nav"] = "Results"


def render_simulator() -> None:
    page_header("Simulator", "Replay a trace through SageServe and see how scaling behaves under that traffic.")
    with st.container(border=True):
        st.subheader("Inputs")
        src = st.radio("Trace source", ["static", "streaming"], horizontal=True, key="sim-source")
        raws, fcs = artifact_versions(src, "raw"), artifact_versions(src, "forecasts")
        if not raws or not fcs:
            st.warning(f"The {src} source needs a raw trace and a forecast.")
            st.button("Go to Data & traces", on_click=go, args=("Data & traces",))
        else:
            _sim_form(src, raws, fcs)
    _sim_results()


def _scaling_card(col, title: str, text: str, key: str) -> bool:
    with col.container(border=True):
        top, tog = st.columns([3, 1])
        top.markdown(f"**{title}**")
        with tog:
            on = st.toggle(title, value=True, key=key, label_visibility="collapsed")
        st.caption(text)
    return on


def _sim_form(src, raws, fcs) -> None:
    rl, fl = version_labels(raws), version_labels(fcs)
    c1, c2, c3 = st.columns([1, 1, 0.6])
    raw_label = c1.selectbox("Raw trace version", rl)
    fc_label = c2.selectbox("Forecast version", fl)
    seed = c3.number_input("Simulator seed", min_value=0, value=0, step=1)
    raw_path = raws[rl.index(raw_label)][1] / "trace.csv"
    forecast_dir = fcs[fl.index(fc_label)][1]

    st.markdown("**Scaling**")
    a, b = st.columns(2)
    short_term = _scaling_card(a, "Short-term scaling", "Reacts to current load by adding or removing instances.", "sim-sts")
    long_term = _scaling_card(b, "Long-term scaling", "Plans capacity ahead using the selected forecast.", "sim-lts")
    name, note = MODES[(short_term, long_term)]
    st.markdown(f'<div class="sg-mode"><b>{name}.</b> {note}</div>', unsafe_allow_html=True)

    if not st.button("Run SageServe", type="primary"):
        return
    if not raw_path.is_file():
        st.error(f"Raw trace file is missing: `{raw_path}`")
        return
    if not SIM_PYTHON.is_file():
        st.error(f"Simulator Python environment not found: `{SIM_PYTHON}`")
        return
    trace_name = f"webui_{src}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        controller = yaml.safe_load((SIM_DIR / "configs" / "controller" / "us3-new.yaml").read_text(encoding="utf-8"))
        arbiter = controller["regions"][0]["arbiter"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        st.error(f"Could not determine simulator output configuration: {exc}")
        return
    out = (RESULTS_ROOT / str(seed) / trace_name / "feed_async_True" / "feed_async_granularity_1"
           / "scaling_level_2" / "scaling_interval_-1" / f"lts_{long_term}_sts{short_term}" / f"arbiter_{arbiter}")
    command = [str(SIM_PYTHON), "run.py", f"trace.path={raw_path}", f"trace.filename={trace_name}",
               f"global_arbiter.arima_traces={forecast_dir}", f"short_term_scaling={str(short_term).lower()}",
               f"long_term_scaling={str(long_term).lower()}", f"seed={seed}", f"hydra.run.dir={out}"]
    run_command(command, SIM_DIR, "SageServe simulator", extra={"output_dir": str(out)})


def _sim_results() -> None:
    last = last_job() or {}
    out = last.get("output_dir")
    if not (out and last.get("returncode") == 0 and Path(out).is_dir()):
        return
    path = Path(out)
    with st.container(border=True):
        st.subheader("Latest run")
        files = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
        c1, c2 = st.columns([1, 3])
        c1.metric("Output files", len(files))
        c2.success(f"Saved to `{path}`")
        with st.expander("Simulator output files"):
            st.code("\n".join(files[:200]) or "No files found.")
        st.button("Analyze this run in Results", type="primary", on_click=_use_run, args=(str(path),))


# ═══════════════════════════ Results ═══════════════════════════
def render_results() -> None:
    page_header("Results", "Analyze saved simulator output into latency and SLA reports. This does not run any LLM requests.")
    with st.container(border=True):
        st.subheader("Generate a report")
        if "inference-results-dir" not in st.session_state:
            st.session_state["inference-results-dir"] = st.session_state.get("inference_results_dir", str(RESULTS_ROOT))
        results_dir = st.text_input("Simulator results directory", key="inference-results-dir")
        c1, c2 = st.columns(2)
        report_name = c1.text_input("Report name", "sage_serve_run")
        forecast_model = c2.text_input("Forecast model label", "unverified",
                                       help="Descriptive only. It does not verify which forecast the simulator used.")
        allow_partial = st.toggle("Allow partial input", value=False)
        safe = bool(report_name.strip()) and Path(report_name).name == report_name and report_name not in {".", ".."}
        if not safe:
            st.warning("Report name must be a plain file name without slashes.")
        if st.button("Generate report", type="primary", disabled=not safe):
            command = [str(MLOPS_PYTHON), str(MLOPS_DIR / "inference" / "run_inference.py"),
                       "--results", str(Path(results_dir).expanduser().resolve()),
                       "--forecast-model", forecast_model,
                       "--out", str(REPORT_DIR / f"{report_name}.csv"),
                       "--plots-dir", str(REPORT_DIR / f"{report_name}_plots")]
            if allow_partial:
                command.append("--allow-partial")
            run_command(command, REPO_ROOT, "Inference report")
    _reports()


def _coverage_data(data: pd.DataFrame) -> pd.DataFrame:
    """Coverage for the selected report; unknown and zero-sample rates stay blank."""
    columns = {
        "experiment_id": "Experiment",
        "model": "Model",
        "n_requests": "Observed requests",
        "n_valid_response_times": "Valid response times",
        "response_coverage": "Response coverage (%)",
        "n_valid_ttft_times": "Valid TTFT measurements",
        "ttft_coverage": "TTFT coverage (%)",
        "n_valid_tbt_times": "Valid average-TBT measurements",
        "tbt_coverage": "Average-TBT coverage (%)",
        "n_windows": "Exported windows",
        "n_unexported_live_applications": "Missing application exports",
        "request_data_complete": "Request exports complete",
        "resource_coverage_status": "Resource coverage",
        "export_duration_seconds": "Known export duration (s)",
    }
    coverage = data.reindex(columns=list(columns)).copy()
    observed = pd.to_numeric(coverage["n_requests"], errors="coerce")
    denominator = observed.where(observed > 0)
    for source, target in (
        ("n_valid_response_times", "response_coverage"),
        ("n_valid_ttft_times", "ttft_coverage"),
        ("n_valid_tbt_times", "tbt_coverage"),
    ):
        coverage[target] = pd.to_numeric(coverage[source], errors="coerce") / denominator * 100
    return coverage.rename(columns=columns)


def _reports() -> None:
    reports = (sorted((p for p in REPORT_DIR.glob("*.csv") if not p.stem.endswith("_windows")),
                      key=lambda p: p.stat().st_mtime, reverse=True) if REPORT_DIR.is_dir() else [])
    with st.container(border=True):
        st.subheader("Reports")
        if not reports:
            st.info("No reports yet. Generate one above.")
            return
        report = st.selectbox("Report to view", reports, format_func=lambda p: p.name)
        try:
            data = pd.read_csv(report)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not load report `{report}`: {exc}")
            return
        st.dataframe(data, width="stretch", hide_index=True)
        meta = read_json(report.with_suffix(".metadata.json"))
        if meta:
            diag = meta.get("diagnostics", [])
            if diag:
                with st.expander(f"Input diagnostics ({len(diag)})"):
                    st.dataframe(pd.DataFrame(diag), width="stretch", hide_index=True)
            with st.expander("Report provenance"):
                st.json({k: meta.get(k) for k in ("output_status", "normalized_weights", "upstream_limitations", "outputs")})
        numeric = [c for c in ("e2e_p99", "viol_ttft", "viol_avg_tbt", "fss")
                   if c in data.columns and pd.to_numeric(data[c], errors="coerce").notna().any()]
        if numeric and not data.empty:
            metric = st.segmented_control("Plot metric", numeric, default=numeric[0], key="plot-metric") or numeric[0]
            label_col = "model" if "model" in data.columns else data.columns[0]
            chart = data[[label_col, metric]].copy()
            if "experiment_id" in data.columns:
                chart[label_col] = chart[label_col].astype(str) + " · " + data["experiment_id"].astype(str).str[:8]
            chart[metric] = pd.to_numeric(chart[metric], errors="coerce")
            st.bar_chart(chart.set_index(label_col), height=260, color=MOSS)
        plots_dir = REPORT_DIR / f"{report.stem}_plots"
        # Old report folders can still contain the retired coverage chart.
        plots = (sorted(p for p in plots_dir.glob("*.png") if p.name != "data_coverage.png")
                 if plots_dir.is_dir() else [])
        st.subheader("Data coverage")
        st.dataframe(
            _coverage_data(data), width="stretch", hide_index=True,
            column_config={name: st.column_config.NumberColumn(format="%.1f") for name in (
                "Response coverage (%)", "TTFT coverage (%)", "Average-TBT coverage (%)")},
        )
        st.caption("Coverage percentages use observed request rows, not all input arrivals. "
                   "Blank values are unavailable, including percentages with no request samples. "
                   "Complete request exports do not establish coverage of the entire simulation.")
        if plots:
            st.subheader(f"Diagnostic plots ({len(plots)})")
            for i in range(0, len(plots), 2):
                cols = st.columns(2)
                for off, plot in enumerate(plots[i:i + 2]):
                    with cols[off]:
                        st.image(str(plot), caption=plot.stem.replace("_", " ").title())
        st.caption(f"CSV: `{report}` · Plots: `{plots_dir}`")
