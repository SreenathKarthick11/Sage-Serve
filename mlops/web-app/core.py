"""Settings, artifact/registry readers and the subprocess runner (no UI styling here)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import shutil
import threading
import html
import signal
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


# ── subprocess runner (background thread, survives page changes) ────────────
@st.cache_resource
def _jobs() -> dict:
    return {"current": None}


def last_job() -> dict | None:
    return _jobs()["current"]


def _worker(job: dict) -> None:
    # No Streamlit calls in here: this runs outside the script thread.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.Popen(job["command"], cwd=job["cwd"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                                start_new_session=True)  # own process group, so Stop can kill children too
        job["proc"] = proc
        if job.get("stopped"):  # Stop was clicked before the process existed
            _kill(proc)
        for line in proc.stdout:
            job["lines"].append(line)
        job["returncode"] = proc.wait()
    except OSError as exc:
        job["lines"].append(f"Could not start command: {exc}\n")
        job["returncode"] = 127
    finally:
        job["elapsed"] = time.monotonic() - job["started"]
        if job.get("cleanup_dir"):
            shutil.rmtree(job["cleanup_dir"], ignore_errors=True)
        job["running"] = False


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop_job() -> None:
    job = last_job()
    if job and job["running"]:
        job["stopped"] = True
        job["lines"].append("\n[stopped by user]\n")
        if job.get("proc"):
            _kill(job["proc"])


def run_command(command: list[str], cwd: Path, label: str, extra: dict | None = None) -> None:
    current = last_job()
    if current and current["running"]:
        st.warning(f"“{current['label']}” is still running. Wait for it to finish or stop it.")
        if extra and extra.get("cleanup_dir"):
            shutil.rmtree(extra["cleanup_dir"], ignore_errors=True)
        return
    job = {"label": label, "command": command, "cwd": str(cwd), "returncode": None,
           "lines": [], "running": True, "stopped": False, "proc": None,
           "started": time.monotonic(), "elapsed": 0.0, **(extra or {})}
    _jobs()["current"] = job
    threading.Thread(target=_worker, args=(job,), daemon=True).start()
    st.rerun()

_PILL_CSS = """
<style>
.sg-note{position:fixed;top:.7rem;right:4.5rem;z-index:1000002;display:flex;align-items:center;gap:.55rem;
  color:#FBFADA;padding:.4rem .9rem;border-radius:999px;font-size:.85rem;
  box-shadow:0 2px 10px rgba(0,0,0,.25);}
.sg-note.run{background:#12372A;}
.sg-note.ok{background:#2e7d4f;}
.sg-note.warn{background:#b7791f;}
.sg-note.bad{background:#b3261e;}
.sg-note.fade{animation:sg-fade 8s forwards;}
@keyframes sg-fade{0%,80%{opacity:1;}100%{opacity:0;visibility:hidden;}}
.sg-spin{width:14px;height:14px;border:2px solid #ADBC9F;border-top-color:transparent;border-radius:50%;
  animation:sg-rot .8s linear infinite;}
@keyframes sg-rot{to{transform:rotate(360deg);}}
</style>
"""


def _notify(kind: str, text: str, spinner: bool = False, fade: bool = False) -> None:
    spin = '<span class="sg-spin"></span>' if spinner else ""
    st.markdown(f'{_PILL_CSS}<div class="sg-note {kind}{" fade" if fade else ""}">{spin}{html.escape(text)}</div>',
                unsafe_allow_html=True)


def _terminal() -> None:
    job = last_job()
    if job is None:
        return
    if job["running"]:
        st.session_state["term_was_running"] = True
        _notify("run", f"{job['label']} is running…", spinner=True)
        c1, c2 = st.columns([5, 1])
        c1.caption(f"⏳ {job['label']} running… {time.monotonic() - job['started']:.0f}s")
        c2.button("⏹ Stop", on_click=stop_job, use_container_width=True, key="stop-job")
    else:
        if st.session_state.pop("term_was_running", False):
            st.rerun()  # full rerun so the page picks up the finished state
        rc = job["returncode"]
        fresh = not job.get("seen")  # show the header notification once per finished job
        job["seen"] = True
        if job.get("stopped"):
            st.warning(f"{job['label']} was stopped.")
            if fresh:
                _notify("warn", f"{job['label']} stopped", fade=True)
        elif rc == 0:
            st.success(f"{job['label']} finished in {job['elapsed']:.1f}s.")
            if fresh:
                _notify("ok", f"✓ {job['label']} completed · exit code 0", fade=True)
        elif rc == 2 and job["label"] == "Inference report":
            st.warning("Inference report saved with partial data or plot diagnostics (exit code 2).")
            if fresh:
                _notify("warn", f"{job['label']} completed with warnings · exit code 2", fade=True)
        else:
            st.error(f"{job['label']} failed (exit code {rc}).")
            if fresh:
                _notify("bad", f"✕ {job['label']} failed · exit code {rc}", fade=True)
    st.code("$ " + " ".join(job["command"]) + "\n" + "".join(job["lines"][-500:]), language="text")
    if job.get("output_dir"):
        st.caption(f"Simulator output: `{job['output_dir']}`")


def show_last_run() -> None:
    job = last_job()
    if job is None:
        return
    st.divider()
    st.subheader("Terminal")
    st.fragment(run_every=1 if job["running"] else None)(_terminal)()