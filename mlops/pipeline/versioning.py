"""Immutable, content-addressed versions of pipeline artifacts.

Every artifact is a directory   data/<source>/<stage>/<version>/   holding its files plus
manifest.json.  The version id is a pure function of the inputs:

    raw      : id = SHA-256(file content)[:12]
    clean,
    forecasts: id = SHA-256({parents, params})[:12]

Same inputs + same parameters  ->  same id  ->  the stage is skipped.
The git commit is stored in the manifest only (not in the id), so unrelated commits
do not create "new" versions of identical data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .paths import DATA_ROOT, PROJECT_ROOT, SOURCES, STAGES

MANIFEST_NAME = "manifest.json"
LINEAGE_PATH = DATA_ROOT / "lineage" / "lineage.jsonl"


def _check(source: str, stage: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")


# ---------------------------------------------------------------- hashing / ids
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_id(parents: list[str], params: dict) -> str:
    blob = json.dumps({"parents": sorted(parents), "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def artifact_id(source: str, stage: str, version: str) -> str:
    return f"{source}/{stage}/{version}"


# ---------------------------------------------------------------- locations
def stage_dir(source: str, stage: str) -> Path:
    _check(source, stage)
    return DATA_ROOT / source / stage


def version_dir(source: str, stage: str, version: str) -> Path:
    return stage_dir(source, stage) / version


def version_exists(source: str, stage: str, version: str) -> bool:
    return (version_dir(source, stage, version) / MANIFEST_NAME).exists()


def latest_version(source: str, stage: str) -> str:
    p = stage_dir(source, stage) / "latest.json"
    if not p.exists():
        raise FileNotFoundError(f"No versions yet for {source}/{stage} (missing {p})")
    return json.loads(p.read_text())["version"]


def set_latest(source: str, stage: str, version: str) -> None:
    d = stage_dir(source, stage)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".latest_{uuid.uuid4().hex[:8]}"
    tmp.write_text(json.dumps({"version": version}))
    os.replace(tmp, d / "latest.json")                       # atomic pointer swap


def resolve_version(source: str, stage: str, version: str = "latest") -> Path:
    """Return the directory of a version ('latest' or an explicit id)."""
    if version == "latest":
        version = latest_version(source, stage)
    path = version_dir(source, stage, version)
    if not (path / MANIFEST_NAME).exists():
        raise FileNotFoundError(f"Version not found: {artifact_id(source, stage, version)} ({path})")
    return path


def new_tmp_dir(source: str, stage: str) -> Path:
    """Scratch directory INSIDE the stage dir, so the final rename is atomic."""
    d = stage_dir(source, stage) / f".tmp_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------- publishing
def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def finalize(tmp_dir: Path, source: str, stage: str, version_id: str,
             parents: list[str], params: dict, details: dict | None = None) -> Path:
    """Seal tmp_dir as version `version_id`: hash files, write manifest, rename, update
    latest pointer, append to the lineage ledger.  If the version already exists, the
    scratch directory is discarded and only the `latest` pointer is moved."""
    final = version_dir(source, stage, version_id)
    if (final / MANIFEST_NAME).exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        set_latest(source, stage, version_id)
        return final

    files = {p.name: sha256_file(p) for p in sorted(tmp_dir.iterdir())
             if p.is_file() and p.name != MANIFEST_NAME}
    manifest = {
        "id": artifact_id(source, stage, version_id),
        "source": source,
        "stage": stage,
        "version": version_id,
        "parents": sorted(parents),
        "params": params,
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "details": details or {},
    }
    (tmp_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str))
    tmp_dir.rename(final)
    set_latest(source, stage, version_id)

    record = {**manifest, "path": final.relative_to(DATA_ROOT).as_posix()}
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LINEAGE_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return final
