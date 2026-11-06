"""Settings, artifact/registry readers and the subprocess runner (no UI styling here)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import streamlit as st

# ── settings (keep this folder where your original script lived) ────────────
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
REPORT_DIR = MLOPS_DIR / "inference" / "reports"


# ── readers ─────────────────────────────────────────────────────────────────
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
    """Versions for a source/stage; the one in latest.json first, then newest first."""
    stage_dir = DATA_ROOT / source / stage
    pointer = read_json(stage_dir / "latest.json")
    latest = str(pointer.get("version")) if pointer and pointer.get("version") else None
    versions: list[tuple[str, Path, dict]] = []
    if not stage_dir.is_dir():
        return versions
    for path in stage_dir.iterdir():
        if not path.is_dir() or path.name.startswith(".tmp_"):
            continue
        manifest = read_json(path / "manifest.json")
        if manifest is not None:
            versions.append((path.name, path, manifest))
    versions.sort(key=lambda item: item[2].get("created_at", ""), reverse=True)
    versions.sort(key=lambda item: item[0] != latest)
    return versions


def version_labels(versions: list[tuple[str, Path, dict]]) -> list[str]:
    return [f"{'Latest · ' if i == 0 else ''}{v}" for i, (v, _, _) in enumerate(versions)]


def lineage_records() -> list[dict]:
    path = DATA_ROOT / "lineage" / "lineage.jsonl"
    if not path.is_file():
        return []
    records: list[dict] = []
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
    models: list[dict] = []
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


# ── subprocess runner ───────────────────────────────────────────────────────
def run_command(command: list[str], cwd: Path, label: str, extra: dict | None = None) -> None:
    started = time.monotonic()
    with st.spinner(f"Running {label.lower()}…"):
        try:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
            record = {
                "label": label, "command": command, "cwd": str(cwd),
                "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                "elapsed": time.monotonic() - started, **(extra or {}),
            }
        except OSError as exc:
            record = {
                "label": label, "command": command, "cwd": str(cwd), "returncode": 127,
                "stdout": "", "stderr": f"Could not start command: {exc}",
                "elapsed": time.monotonic() - started, **(extra or {}),
            }
    st.session_state["last_run"] = record
    st.rerun()


def show_last_run() -> None:
    result = st.session_state.get("last_run")
    if not result:
        return
    if result["returncode"] == 0:
        st.success(f"{result['label']} finished in {result['elapsed']:.1f}s.")
    elif result["returncode"] == 2 and result["label"] == "Inference report":
        st.warning("Inference report saved with partial data or plot diagnostics (exit code 2).")
    else:
        st.error(f"{result['label']} failed (exit code {result['returncode']}). Open the run output for details.")
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
