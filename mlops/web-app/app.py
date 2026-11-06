from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import pandas as pd
import streamlit as st
import yaml


st.set_page_config(page_title="SageServe MLOps", page_icon="S", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[2]
MLOPS_DIR = REPO_ROOT / "mlops"
SIM_DIR = REPO_ROOT / "SageServe-original"
DATA_ROOT = Path(os.environ.get("MLOPS_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()
REGISTRY_DIR = Path(
    os.environ.get("MODEL_REGISTRY_DIR", REPO_ROOT / "data" / "model_registry")
).expanduser().resolve()
MLOPS_PYTHON = MLOPS_DIR / ".venv" / "bin" / "python"
SIM_PYTHON = SIM_DIR / ".venv" / "bin" / "python"
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 70% 0%, #102745 0%, #0a1422 42%, #050a12 100%);
        color: #e6edf7;
    }
    [data-testid="stHeader"] {background: rgba(5, 10, 18, .8);}
    [data-testid="stSidebar"] {background: #070d17;}
    .block-container {max-width: 1440px; padding-top: 1.8rem; padding-bottom: 3rem;}
    h1, h2, h3 {color: #e9f2ff;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #102641, #0b1524);
        border: 1px solid #1f3a5a; border-radius: 10px; padding: .8rem 1rem;
    }
    [data-testid="stMetricLabel"] p {color: #9fb6cf;}
    [data-testid="stTabs"] button[role="tab"] {color: #a9bfd7;}
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #72b7ff; border-bottom-color: #3b9cff;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: #1976d2; border-color: #2589e8; color: white;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #2589e8; border-color: #59a8f3;
    }
    div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] > div {
        background-color: #0c1726; color: #e6edf7; border-color: #29425f;
    }
    [data-testid="stDataFrame"] {border: 1px solid #1c3048; border-radius: 8px;}
    code {color: #8bc7ff;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not read `{path}`: {exc}")
        return None
    return value if isinstance(value, dict) else None


def artifact_versions(source: str, stage: str) -> list[tuple[str, Path, dict]]:
    stage_dir = DATA_ROOT / source / stage
    pointer = read_json(stage_dir / "latest.json")
    latest = str(pointer.get("version")) if pointer and pointer.get("version") else None
    versions = []
    if not stage_dir.is_dir():
        return versions
    for path in stage_dir.iterdir():
        if not path.is_dir() or path.name.startswith(".tmp_"):
            continue
        manifest = read_json(path / "manifest.json")
        if manifest is not None:
            versions.append((path.name, path, manifest))
    versions.sort(key=lambda item: (item[0] != latest, item[2].get("created_at", "")), reverse=False)
    return versions


def select_artifact(source: str, stage: str, key: str) -> tuple[Path, dict] | None:
    versions = artifact_versions(source, stage)
    if not versions:
        return None
    labels = [
        f"{'Latest · ' if index == 0 else ''}{version}"
        for index, (version, _, _) in enumerate(versions)
    ]
    selected = st.selectbox(f"{stage.title()} version", labels, key=key)
    index = labels.index(selected)
    _, path, manifest = versions[index]
    return path, manifest


def lineage_records() -> list[dict]:
    path = DATA_ROOT / "lineage" / "lineage.jsonl"
    if not path.is_file():
        return []
    records = []
    try:
        with path.open(encoding="utf-8") as ledger:
            for line_number, line in enumerate(ledger, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    st.warning(f"Invalid lineage record on line {line_number}: {exc}")
    except OSError as exc:
        st.warning(f"Could not read lineage ledger `{path}`: {exc}")
    return records


def registry_models() -> list[dict]:
    models_dir = REGISTRY_DIR / "models"
    models = []
    if not models_dir.is_dir():
        return models
    for manifest_path in sorted(models_dir.glob("*/*/manifest.json")):
        manifest = read_json(manifest_path)
        if manifest is not None:
            models.append(
                {
                    "name": manifest_path.parent.parent.name,
                    "version": manifest_path.parent.name,
                    "path": manifest_path.parent,
                    "manifest": manifest,
                    "has_card": (manifest_path.parent / "model_card.json").is_file(),
                }
            )
    return models


def run_command(command: list[str], cwd: Path, label: str, extra: dict | None = None) -> None:
    started = time.monotonic()
    with st.spinner(f"Running {label.lower()}…"):
        try:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
            record = {
                "label": label,
                "command": command,
                "cwd": str(cwd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed": time.monotonic() - started,
                **(extra or {}),
            }
        except OSError as exc:
            record = {
                "label": label,
                "command": command,
                "cwd": str(cwd),
                "returncode": 127,
                "stdout": "",
                "stderr": f"Could not start command: {exc}",
                "elapsed": time.monotonic() - started,
                **(extra or {}),
            }
    st.session_state["last_run"] = record
    st.rerun()


def show_last_run() -> None:
    result = st.session_state.get("last_run")
    if not result:
        return
    if result["returncode"] == 0:
        st.success(f"{result['label']} completed in {result['elapsed']:.1f}s.")
    elif result["returncode"] == 2 and result["label"] == "Inference report":
        st.warning("Inference report saved with partial data or plot diagnostics (exit code 2).")
    else:
        st.error(f"{result['label']} failed with exit code {result['returncode']}.")
    with st.expander("Run output", expanded=result["returncode"] not in (0, 2)):
        st.code(" ".join(result["command"]), language="bash")
        st.caption(f"Working directory: `{result['cwd']}`")
        if result["stdout"]:
            st.markdown("**Output**")
            try:
                st.json(json.loads(result["stdout"]))
            except json.JSONDecodeError:
                st.code(result["stdout"], language="text")
        if result["stderr"]:
            st.markdown("**Errors and diagnostics**")
            st.code(result["stderr"], language="text")
    if result.get("output_dir"):
        st.caption(f"Simulator output: `{result['output_dir']}`")


def show_artifact_data(source: str, stage: str, key: str) -> None:
    selected = select_artifact(source, stage, key)
    if selected is None:
        st.info(f"No {source} {stage} artifacts yet.")
        return
    artifact_dir, manifest = selected
    details = manifest.get("details", {})
    metric_cols = st.columns(4)
    metric_cols[0].metric("Version", manifest.get("version", artifact_dir.name))
    metric_cols[1].metric(
        "Rows",
        details.get("rows", details.get("rows_out", details.get("rows_in", "—"))),
    )
    metric_cols[2].metric("Status", details.get("status", "available"))
    metric_cols[3].metric("Created", str(manifest.get("created_at", "—"))[:19])
    st.caption(f"Artifact directory: `{artifact_dir}`")
    csv_files = sorted(artifact_dir.glob("*.csv"))
    if not csv_files:
        st.info("This artifact has no CSV data files.")
        st.json({"parents": manifest.get("parents", []), "params": manifest.get("params", {}), "details": details})
        return

    selected_csv = st.selectbox(
        "Data file", csv_files, format_func=lambda path: path.name, key=f"{key}-file"
    )
    try:
        data = pd.read_csv(selected_csv, nrows=100_000, dtype={"regions": "string"})
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read `{selected_csv.name}`: {exc}")
        return

    if stage == "forecasts" and "Predicted" in data.columns:
        chart = data[["Time", "Predicted"]].set_index("Time") if "Time" in data.columns else data[["Predicted"]]
        st.line_chart(chart, height=260)
        values = pd.to_numeric(data["Predicted"], errors="coerce").dropna()
        if not values.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Forecast points", f"{len(values):,}")
            m2.metric("Mean predicted load", f"{values.mean():,.1f}")
            m3.metric("Peak predicted load", f"{values.max():,.1f}")
    elif {"arrival_timestamp", "prompt_size"} <= set(data.columns):
        data["arrival_timestamp"] = pd.to_numeric(data["arrival_timestamp"], errors="coerce")
        data["prompt_size"] = pd.to_numeric(data["prompt_size"], errors="coerce")
        sampled = data.dropna(subset=["arrival_timestamp", "prompt_size"]).copy()
        if not sampled.empty:
            sampled["minute"] = (sampled["arrival_timestamp"] // 60).astype("int64")
            activity = sampled.groupby("minute").agg(
                requests=("prompt_size", "size"), mean_prompt=("prompt_size", "mean")
            )
            st.line_chart(activity, height=260)
            st.caption("Trace chart uses up to the first 100,000 rows of the selected CSV.")

    st.markdown("**Data preview**")
    st.dataframe(data.head(100), width="stretch", hide_index=True)
    with st.expander("Artifact details"):
        st.json(
            {
                "id": manifest.get("id"),
                "parents": manifest.get("parents", []),
                "parameters": manifest.get("params", {}),
                "details": details,
                "files": manifest.get("files", {}),
            }
        )


st.title("SageServe MLOps")
st.caption("Run traces, inspect artifacts, manage forecasting models, simulate, and review inference.")
show_last_run()

trace_tab, registry_tab, simulator_tab, inference_tab = st.tabs(
    ["Traces & data", "Model registry", "SageServe simulator", "Inference plots"]
)

with trace_tab:
    st.header("Run a trace")
    source = st.radio("Trace type", ["static", "streaming"], horizontal=True)
    stages_choice = st.selectbox(
        "Stages to run",
        ["Ingest → ETL → forecast", "Ingest only", "ETL + forecast latest raw", "Forecast latest clean"],
    )
    stages = {
        "Ingest → ETL → forecast": "ingest,etl,forecast",
        "Ingest only": "ingest",
        "ETL + forecast latest raw": "etl,forecast",
        "Forecast latest clean": "forecast",
    }[stages_choice]
    if source == "static":
        c1, c2 = st.columns(2)
        num_requests = c1.number_input("Number of requests", min_value=1, value=5000, step=1000)
        seed = c2.number_input("Trace seed", min_value=0, value=42, step=1)
        run_args = ["--num-requests", str(num_requests), "--seed", str(seed)]
    else:
        c1, c2, c3 = st.columns(3)
        duration = c1.number_input("Duration (wall-clock seconds)", min_value=1, value=600, step=60)
        speedup = c2.number_input("Simulation speedup", min_value=0.1, value=300.0, step=10.0)
        rate_scale = c3.number_input("Traffic rate scale", min_value=0.0001, value=0.002, format="%.4f")
        seed = st.number_input("Trace seed", min_value=0, value=42, step=1)
        run_args = [
            "--duration", str(duration), "--speedup", str(speedup),
            "--rate-scale", str(rate_scale), "--seed", str(seed),
        ]
    if stages == "forecast":
        clean_versions = artifact_versions(source, "clean")
        if clean_versions:
            version_options = ["latest"] + [version for version, _, _ in clean_versions]
            input_version = st.selectbox("Clean artifact version", list(dict.fromkeys(version_options)))
        else:
            input_version = "latest"
            st.warning("No clean trace exists; forecast will fail until ETL has completed.")
        run_args.extend(["--input-version", input_version])

    if st.button("Run trace pipeline", type="primary"):
        command = [
            str(MLOPS_PYTHON), "-m", "pipeline.run_pipeline",
            "--source", source, "--stages", stages, *run_args,
        ]
        run_command(command, MLOPS_DIR, f"{source.title()} trace pipeline")

    st.divider()
    st.header("Current data")
    data_source = st.radio("Data source", ["static", "streaming"], horizontal=True, key="data_source")
    data_stage = st.radio("Artifact stage", ["raw", "clean", "forecasts"], horizontal=True, key="data_stage")
    show_artifact_data(data_source, data_stage, "data-artifact")

    st.divider()
    st.header("Lineage")
    records = lineage_records()
    source_records = [
        record for record in records
        if record.get("source") == data_source and record.get("id")
    ]
    source_records.sort(key=lambda record: record.get("created_at", ""), reverse=True)
    if not source_records:
        st.info(f"No {data_source} lineage records yet.")
    else:
        record_by_id = {record.get("id"): record for record in records if record.get("id")}
        artifact_ids = list(dict.fromkeys(record["id"] for record in source_records))
        chosen_id = st.selectbox("Artifact", artifact_ids, key="lineage-artifact")
        chain = []
        seen = set()
        pending = [chosen_id]
        while pending:
            node = pending.pop(0)
            if node in seen:
                continue
            seen.add(node)
            record = record_by_id.get(node)
            chain.append(
                {
                    "Artifact": node,
                    "Created": record.get("created_at", "") if record else "",
                    "Parent(s)": ", ".join(record.get("parents", [])) if record else "Not in ledger",
                    "Stage": record.get("stage", "") if record else "",
                }
            )
            if record:
                pending.extend(record.get("parents", []))
        st.dataframe(pd.DataFrame(chain), width="stretch", hide_index=True)
        selected_record = record_by_id.get(chosen_id, {})
        with st.expander("Parameters and artifact details"):
            st.json(
                {
                    "parameters": selected_record.get("params", {}),
                    "details": selected_record.get("details", {}),
                    "files": selected_record.get("files", {}),
                }
            )

with registry_tab:
    st.header("Model registry")
    active = read_json(REGISTRY_DIR / "active_model.json")
    if active:
        active_type = active.get("active_model_type", "Unknown")
        active_version = active.get("active_version", "Unknown")
        st.success(f"Production model: **{active_type} / {active_version}**")
    else:
        st.info("No active Production model.")

    st.subheader("Configure and register a model")
    model_type = st.selectbox("Model type", ["arima", "hmm"])
    description = st.text_input("Model description", f"{model_type.upper()} workload forecaster")
    col1, col2, col3 = st.columns(3)
    train_window = col1.number_input("Training window (minutes)", min_value=1, value=60)
    horizon = col2.number_input("Forecast horizon (minutes)", min_value=1, value=60)
    bucket_seconds = col3.number_input("Bucket size (seconds)", min_value=1, value=60)

    parameters = {
        "train_window_mins": int(train_window),
        "future_horizon": int(horizon),
        "bucket_seconds": int(bucket_seconds),
    }
    if model_type == "arima":
        p_col, d_col, q_col = st.columns(3)
        parameters["order"] = {
            "p": int(p_col.number_input("AR order (p)", min_value=0, value=1)),
            "d": int(d_col.number_input("Difference order (d)", min_value=0, value=1)),
            "q": int(q_col.number_input("MA order (q)", min_value=0, value=1)),
        }
        st.caption("ARIMA is fitted to each workload series when you run a forecast; it has no saved weights file.")
    else:
        parameters["n_states"] = int(st.number_input("HMM states", min_value=2, value=3))
        clean_sources = {
            f"{source} / {version}": path
            for source in ("static", "streaming")
            for version, path, _ in artifact_versions(source, "clean")
        }
        weight_mode = st.radio(
            "HMM weights",
            ["Train from a clean trace", "Use an existing weights file"],
            horizontal=True,
        )
        weights_path = ""
        train_from = ""
        if weight_mode == "Train from a clean trace":
            if clean_sources:
                training_choice = st.selectbox("Training trace", list(clean_sources))
                train_from = str(clean_sources[training_choice])
            else:
                st.warning("Run ETL to create a clean trace before training an HMM.")
        else:
            weights_path = st.text_input("Weights file", placeholder="/path/to/transition_matrix.pkl")

    model_config = {
        "model_info": {"type": model_type, "description": description},
        "parameters": parameters,
    }
    with st.expander("Config file preview", expanded=False):
        st.code(yaml.safe_dump(model_config, sort_keys=False), language="yaml")

    metrics_input = st.text_input(
        "Measured governance metrics",
        placeholder="mae=12.4 or f1=0.84",
        help="Enter real measured metrics only. Production requires MAE <= 15 or F1 >= 0.80.",
    )
    can_register = model_type != "hmm" or bool(train_from or weights_path.strip())
    register_label = (
        "Train and register HMM" if model_type == "hmm" and train_from
        else "Register HMM with weights" if model_type == "hmm"
        else "Register ARIMA"
    )
    if st.button(register_label, type="primary", disabled=not can_register):
        metrics = [item.strip() for item in metrics_input.replace(",", " ").split() if item.strip()]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False) as config_file:
            yaml.safe_dump(model_config, config_file, sort_keys=False)
            config_path = Path(config_file.name)
        command = [
            str(MLOPS_PYTHON), "-m", "model_registry.run",
            "register", "--name", model_type, "--config", str(config_path),
        ]
        if metrics:
            command.extend(["--metrics", *metrics])
        if train_from:
            command.extend(["--train-from", train_from])
        elif weights_path.strip():
            command.extend(["--weights", str(Path(weights_path).expanduser().resolve())])
        try:
            run_command(command, MLOPS_DIR, f"Register {model_type.upper()} model")
        finally:
            config_path.unlink(missing_ok=True)

    models = registry_models()
    st.divider()
    st.subheader("Registered versions")
    if not models:
        st.info("No model versions registered.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": model["name"],
                        "Version": model["version"],
                        "Stage": model["manifest"].get("stage", "None"),
                        "Model card": "Ready" if model["has_card"] else "Missing",
                        "Metrics": json.dumps(model["manifest"].get("metrics", {})),
                    }
                    for model in models
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        choices = {f"{model['name']} / {model['version']}": model for model in models}
        selected_label = st.selectbox("Manage version", list(choices))
        selected = choices[selected_label]
        with st.expander("Registered config and manifest"):
            config_path = selected["path"] / "config.yaml"
            if config_path.is_file():
                st.code(config_path.read_text(encoding="utf-8"), language="yaml")
            st.json(selected["manifest"])

        st.markdown("**Model card**")
        card_col1, card_col2 = st.columns(2)
        card_description = card_col1.text_input(
            "Card description",
            "Workload forecasting model",
            key="card-description",
        )
        intended_use = card_col2.text_input(
            "Intended use",
            "Forecast per-minute SageServe request workload.",
            key="card-intended-use",
        )
        limitations = st.text_input(
            "Limitations",
            "Forecast quality depends on trace history and model parameters.",
            key="card-limitations",
        )
        if st.button("Save model card"):
            run_command(
                [
                    str(MLOPS_PYTHON), "-m", "model_registry.run", "card",
                    "--name", selected["name"], "--version", selected["version"],
                    "--description", card_description, "--intended-use", intended_use,
                    "--limitations", limitations,
                ],
                MLOPS_DIR,
                "Save model card",
            )

        stage_col, action_col = st.columns([1, 1])
        target_stage = stage_col.selectbox("Promote to stage", ["Staging", "Production", "Archived"])
        if action_col.button("Promote model", type="primary"):
            run_command(
                [
                    str(MLOPS_PYTHON), "-m", "model_registry.run", "promote",
                    "--name", selected["name"], "--version", selected["version"],
                    "--stage", target_stage,
                ],
                MLOPS_DIR,
                f"Promote {selected_label} to {target_stage}",
            )

with simulator_tab:
    st.header("Run SageServe simulator")
    sim_source = st.radio("Trace source", ["static", "streaming"], horizontal=True, key="sim-source")
    raw_versions = artifact_versions(sim_source, "raw")
    forecast_versions = artifact_versions(sim_source, "forecasts")
    if not raw_versions or not forecast_versions:
        st.warning("This source needs both a raw trace and a forecast artifact. Run the pipeline first.")
    else:
        raw_labels = [f"{'Latest · ' if i == 0 else ''}{version}" for i, (version, _, _) in enumerate(raw_versions)]
        forecast_labels = [
            f"{'Latest · ' if i == 0 else ''}{version}"
            for i, (version, _, _) in enumerate(forecast_versions)
        ]
        sim_col1, sim_col2 = st.columns(2)
        raw_label = sim_col1.selectbox("Raw trace version", raw_labels)
        forecast_label = sim_col2.selectbox("Forecast version", forecast_labels)
        raw_path = raw_versions[raw_labels.index(raw_label)][1] / "trace.csv"
        forecast_dir = forecast_versions[forecast_labels.index(forecast_label)][1]

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        sim_seed = sim_col1.number_input("Simulator seed", min_value=0, value=0, step=1)
        short_term = sim_col2.checkbox("Short-term scaling", value=True)
        long_term = sim_col3.checkbox("Long-term scaling", value=True)
        if short_term or long_term:
            st.caption("Scaling is enabled with the selected source's matching forecast directory.")

        if st.button("Run SageServe", type="primary"):
            if not raw_path.is_file():
                st.error(f"Raw trace file is missing: `{raw_path}`")
            elif not SIM_PYTHON.is_file():
                st.error(f"Simulator Python environment not found: `{SIM_PYTHON}`")
            else:
                trace_name = f"webui_{sim_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                controller_config = SIM_DIR / "configs" / "controller" / "us3-new.yaml"
                try:
                    controller = yaml.safe_load(controller_config.read_text(encoding="utf-8"))
                    arbiter = controller["regions"][0]["arbiter"]
                except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
                    st.error(f"Could not determine simulator output configuration: {exc}")
                else:
                    output_dir = (
                        RESULTS_ROOT / str(sim_seed) / trace_name
                        / "feed_async_True" / "feed_async_granularity_1"
                        / "scaling_level_2" / "scaling_interval_-1"
                        / f"lts_{long_term}_sts{short_term}" / f"arbiter_{arbiter}"
                    )
                    command = [
                        str(SIM_PYTHON), "run.py",
                        f"trace.path={raw_path}",
                        f"trace.filename={trace_name}",
                        f"global_arbiter.arima_traces={forecast_dir}",
                        f"short_term_scaling={str(short_term).lower()}",
                        f"long_term_scaling={str(long_term).lower()}",
                        f"seed={sim_seed}",
                        f"hydra.run.dir={output_dir}",
                    ]
                    run_command(
                        command,
                        SIM_DIR,
                        "SageServe simulator",
                        extra={"output_dir": str(output_dir)},
                    )

    last_result = st.session_state.get("last_run", {})
    last_output = last_result.get("output_dir")
    if last_output and last_result.get("returncode") == 0:
        output_path = Path(last_output)
        if output_path.is_dir():
            st.success(f"Simulator results saved to `{output_path}`.")
            output_files = sorted(
                path.relative_to(output_path).as_posix()
                for path in output_path.rglob("*")
                if path.is_file()
            )
            st.write(f"{len(output_files)} output files")
            with st.expander("Simulator output files"):
                st.code("\n".join(output_files[:200]) or "No files found.")
            if st.button("Use this run for inference"):
                st.session_state["inference_results_dir"] = str(output_path)
                st.session_state["inference-results-dir"] = str(output_path)
                st.rerun()

with inference_tab:
    st.header("Inference plots")
    st.caption("Analyze saved simulator exports; inference analysis does not execute LLM requests.")
    if "inference-results-dir" not in st.session_state:
        st.session_state["inference-results-dir"] = st.session_state.get(
            "inference_results_dir", str(RESULTS_ROOT)
        )
    results_dir = st.text_input("Simulator results directory", key="inference-results-dir")
    c1, c2 = st.columns(2)
    report_name = c1.text_input("Report name", "sage_serve_run")
    forecast_model = c2.text_input("Forecast model label", "unverified")
    st.caption("This is a descriptive report label; it does not verify which forecast the simulator used.")
    allow_partial = st.checkbox("Allow partial input", value=False)
    safe_report_name = (
        bool(report_name.strip())
        and Path(report_name).name == report_name
        and report_name not in {".", ".."}
    )
    if st.button("Generate inference report", type="primary", disabled=not safe_report_name):
        report_dir = MLOPS_DIR / "inference" / "reports"
        report_csv = report_dir / f"{report_name}.csv"
        plots_dir = report_dir / f"{report_name}_plots"
        command = [
            str(MLOPS_PYTHON),
            str(MLOPS_DIR / "inference" / "run_inference.py"),
            "--results", str(Path(results_dir).expanduser().resolve()),
            "--forecast-model", forecast_model,
            "--out", str(report_csv),
            "--plots-dir", str(plots_dir),
        ]
        if allow_partial:
            command.append("--allow-partial")
        run_command(command, REPO_ROOT, "Inference report")

    report_dir = MLOPS_DIR / "inference" / "reports"
    reports = (
        sorted(
            (path for path in report_dir.glob("*.csv") if not path.stem.endswith("_windows")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if report_dir.is_dir()
        else []
    )
    if not reports:
        st.info("No inference reports yet.")
    else:
        report = st.selectbox("Report to view", reports, format_func=lambda path: path.name)
        try:
            report_data = pd.read_csv(report)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not load report `{report}`: {exc}")
        else:
            st.dataframe(report_data, width="stretch", hide_index=True)
            metadata = read_json(report.with_suffix(".metadata.json"))
            if metadata:
                diagnostics = metadata.get("diagnostics", [])
                if diagnostics:
                    with st.expander(f"Input diagnostics ({len(diagnostics)})"):
                        st.dataframe(pd.DataFrame(diagnostics), width="stretch", hide_index=True)
                with st.expander("Report provenance"):
                    st.json(
                        {
                            "output_status": metadata.get("output_status"),
                            "normalized_weights": metadata.get("normalized_weights"),
                            "upstream_limitations": metadata.get("upstream_limitations"),
                            "outputs": metadata.get("outputs"),
                        }
                    )

            numeric_metrics = [
                column for column in ("e2e_p99", "viol_ttft", "viol_avg_tbt", "fss")
                if column in report_data.columns and pd.to_numeric(report_data[column], errors="coerce").notna().any()
            ]
            if numeric_metrics and not report_data.empty:
                metric_name = st.selectbox("Plot metric", numeric_metrics)
                label_column = "model" if "model" in report_data.columns else report_data.columns[0]
                chart_data = report_data[[label_column, metric_name]].copy()
                if "experiment_id" in report_data.columns:
                    chart_data[label_column] = (
                        chart_data[label_column].astype(str)
                        + " · " + report_data["experiment_id"].astype(str).str[:8]
                    )
                chart_data[metric_name] = pd.to_numeric(chart_data[metric_name], errors="coerce")
                st.bar_chart(chart_data.set_index(label_column), height=260)

            plots_dir = report_dir / f"{report.stem}_plots"
            plots = sorted(plots_dir.glob("*.png")) if plots_dir.is_dir() else []
            if plots:
                st.subheader(f"Diagnostic plots ({len(plots)})")
                for index in range(0, len(plots), 2):
                    plot_cols = st.columns(2)
                    for offset, plot in enumerate(plots[index:index + 2]):
                        with plot_cols[offset]:
                            st.image(str(plot), caption=plot.stem.replace("_", " ").title())
            st.caption(f"CSV: `{report}` · Plots: `{plots_dir}`")
