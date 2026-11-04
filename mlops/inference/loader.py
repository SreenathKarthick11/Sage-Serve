"""Read simulator exports without merging experiments or hiding bad inputs.

load_results retains configuration, file provenance, quiet windows and diagnostics.
YAML is read as data: Hydra resolvers and Python are never executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

_LTS_RE = re.compile(r"lts_(True|False)_sts(True|False)")
DETAIL_COLUMNS = ["response_times", "queue_times", "ttft_times", "tbt_times",
                  "prompt_sizes", "token_sizes"]
COST_COLUMNS = ["timestamp", "model", "prod", "spot", "memory_util"]


class InputError(ValueError):
    """An export cannot be interpreted reliably."""


@dataclass
class Diagnostics:
    allow_partial: bool = False
    issues: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)

    def add(self, code, path, message, *, error=False):
        self.issues.append({"severity": "error" if error else "warning",
                            "code": code, "path": str(path), "message": message})
        if error and not self.allow_partial:
            raise InputError(f"{path}: {message} (use --allow-partial to report usable data)")

    def emit(self):
        for issue in self.issues:
            warnings.warn(f"{issue['code']}: {issue['path']}: {issue['message']}",
                          RuntimeWarning, stacklevel=3)


@dataclass
class Experiment:
    path: Path
    meta: dict
    config: dict
    windows: set[int] = field(default_factory=set)
    applications: dict[str, tuple[str, str]] = field(default_factory=dict)
    model_regions: dict[str, set[str]] = field(default_factory=dict)
    detail_files: set[tuple[str, int]] = field(default_factory=set)
    invalid_files: set[tuple[str, int]] = field(default_factory=set)
    invalid_cost_files: set[Path] = field(default_factory=set)
    unexported_applications: dict[int, set[str]] = field(default_factory=dict)
    memory_audit_failed: bool = False
    bounds: dict[int, tuple[float, float]] = field(default_factory=dict)

    def add_model(self, region, model):
        self.model_regions.setdefault(model, set()).add(region)


@dataclass
class Results:
    experiments: list[Experiment]
    detailed: pd.DataFrame
    cost: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: Diagnostics


def _window_files(folder, diagnostics):
    paths = []
    seen = set()
    for path in sorted(folder.glob("*.csv")):
        if not path.stem.isdigit():
            diagnostics.add("ignored_filename", path, "Expected a nonnegative numeric window filename.")
            continue
        window = int(path.stem)
        if window in seen:
            diagnostics.add("duplicate_window", path, "Multiple filenames represent the same numeric window.", error=True)
            continue
        seen.add(window)
        paths.append((window, path))
    return sorted(paths)


def _discover(root, diagnostics):
    if not root.is_dir():
        raise InputError(f"Results directory does not exist: {root}")
    candidates = {p for p in root.rglob("arbiter_*") if p.is_dir()}
    if root.name.startswith("arbiter_"):
        candidates.add(root)
    experiments = []
    for path in sorted(candidates):
        if not any((path / name).is_dir() for name in ("detailed", "summary", "arbiters")):
            continue
        parts = path.parts
        lts_index = next((i for i in range(len(parts)-2, -1, -1)
                          if _LTS_RE.fullmatch(parts[i])), None)
        if lts_index is None:
            diagnostics.add("unrecognized_experiment", path, "Missing lts_<bool>_sts<bool> directory.", error=True)
            continue
        run_index = next((i for i in range(lts_index-2, -1, -1) if parts[i].isdigit()), None)
        if run_index is None:
            diagnostics.add("unrecognized_experiment", path, "Cannot identify numeric run and trace directories.", error=True)
            continue
        config_path = path / ".hydra" / "config.yaml"
        config, config_hash = {}, ""
        if config_path.exists():
            try:
                raw = config_path.read_bytes()
                config_hash = hashlib.sha256(raw).hexdigest()
                config = YAML(typ="safe").load(raw) or {}
                if not isinstance(config, dict):
                    raise ValueError("Expected a YAML mapping")
                controller = config.get("controller", {})
                if not isinstance(controller, dict) or not isinstance(controller.get("regions", []), list):
                    raise ValueError("controller.regions must be a list of region mappings")
                for region in controller.get("regions", []):
                    if not isinstance(region, dict) or "region_name" not in region or not isinstance(region.get("model_endpoints", []), list):
                        raise ValueError("Each region needs region_name and a model_endpoints list")
                    if any(not isinstance(e, dict) or "model_name" not in e for e in region.get("model_endpoints", [])):
                        raise ValueError("Each model endpoint needs model_name")
            except Exception as exc:
                diagnostics.add("invalid_config", config_path, str(exc), error=True)
                config = {}
        match = _LTS_RE.fullmatch(parts[lts_index])
        relative = path.relative_to(root).as_posix()
        identity = relative if relative != "." else "/".join(parts[run_index:])
        meta = {"experiment_id": hashlib.sha256((identity + config_hash).encode()).hexdigest()[:16],
                "run": parts[run_index], "trace": parts[run_index+1],
                "lts": match[1], "sts": match[2], "arbiter": path.name[len("arbiter_"):],
                "source_path": str(path), "relative_path": identity,
                "config_sha256": config_hash, "config_path": str(config_path) if config_path.exists() else ""}
        for setting in ("feed_async", "feed_async_granularity", "scaling_level", "scaling_interval"):
            value = next((s[len(setting)+1:] for s in parts[run_index+2:lts_index]
                          if s.startswith(setting + "_")), None)
            if setting == "feed_async":
                value = next((s[len(setting)+1:] for s in parts[run_index+2:lts_index]
                              if s in ("feed_async_True", "feed_async_False")), None)
            meta[setting] = value if value is not None else str(config.get(setting, "unknown"))
        exp = Experiment(path, meta, config)
        for region in config.get("controller", {}).get("regions", []):
            for endpoint in region.get("model_endpoints", []):
                exp.add_model(str(region["region_name"]), str(endpoint["model_name"]))
        experiments.append(exp)
    if not experiments:
        raise InputError(f"No simulator experiment directories found under {root}")
    return experiments


def _parse_application(name, exp):
    prefix, replica = name.rsplit("_", 1)
    if not replica.isdigit():
        raise ValueError(f"Invalid application suffix in {name!r}")
    known_regions = {r for regions in exp.model_regions.values() for r in regions}
    for region in sorted(known_regions, key=len, reverse=True):
        if prefix.startswith(region + "_"):
            return region, prefix[len(region)+1:]
    # Split once: model names may themselves contain underscores.
    region, model = prefix.split("_", 1)
    return region, model


def _read(path, required, diagnostics):
    try:
        raw = path.read_bytes()
        diagnostics.files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                  "bytes": len(raw)})
        df = pd.read_csv(io.BytesIO(raw))
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        return df
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        diagnostics.add("invalid_csv", path, str(exc), error=True)
        return None


def _numeric(df, columns, path, diagnostics, *, nonnegative=(), integers=()):
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").astype(float)
        invalid = df[column].notna() & (values.isna() | ~np.isfinite(values))
        if column in nonnegative:
            invalid |= values.lt(0)
        if column in integers:
            invalid |= values.notna() & values.mod(1).ne(0)
        if invalid.any():
            diagnostics.add("invalid_values", path,
                            f"{int(invalid.sum())} invalid values in {column}; unknown in partial mode.", error=True)
            values.loc[invalid] = np.nan
        if values.isna().any():
            diagnostics.add("missing_values", path, f"{int(values.isna().sum())} unknown values in {column}.")
        df[column] = values


def _attach(df, exp, window, path):
    for key, value in exp.meta.items():
        df[key] = value
    df["window"], df["source_file"] = window, str(path)
    return df


def _audit_memory_exports(exp, diagnostics):
    """Find live applications omitted by the upstream detailed exporter.

    Memory files can be large: inspect identifiers in chunks, without retaining
    per-instance samples or treating sample timestamps as export boundaries.
    """
    for window, path in _window_files(exp.path / "memory", diagnostics):
        exp.windows.add(window)
        live = set()
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            diagnostics.files.append({"path": str(path), "sha256": digest.hexdigest(),
                                      "bytes": path.stat().st_size})
            for chunk in pd.read_csv(path, usecols=["instance"], chunksize=100000):
                if chunk.instance.isna().any():
                    raise ValueError("Missing instance identifier")
                live.update(str(name).rsplit("_", 1)[0] for name in chunk.instance.unique())
            missing = set()
            for application in live:
                region, model = _parse_application(application, exp)
                exp.add_model(region, model)
                if (application, window) not in exp.detail_files:
                    missing.add(application)
            if missing:
                exp.unexported_applications[window] = missing
                diagnostics.add("unexported_live_applications", path,
                                f"{len(missing)} live applications lack detailed measurements: {', '.join(sorted(missing))}")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            exp.memory_audit_failed = True
            diagnostics.add("invalid_memory_export", path, str(exc), error=True)


def _load_boundaries(path, experiments):
    table = pd.read_csv(path)
    required = {"window", "start_seconds", "end_seconds"}
    if not required <= set(table):
        raise InputError(f"{path}: required boundary columns: {sorted(required)}")
    if "experiment_id" not in table:
        if len(experiments) != 1:
            raise InputError("Boundary CSV needs experiment_id when loading multiple experiments")
        table["experiment_id"] = experiments[0].meta["experiment_id"]
    known = {e.meta["experiment_id"]: e for e in experiments}
    for row in table.to_dict("records"):
        if str(row["experiment_id"]) not in known:
            raise InputError(f"Unknown boundary experiment_id: {row['experiment_id']}")
        exp = known[str(row["experiment_id"])]
        w, start, end = (float(row[c]) for c in ("window", "start_seconds", "end_seconds"))
        if not np.isfinite([w, start, end]).all() or w < 0 or w != int(w) or not 0 <= start < end:
            raise InputError(f"Invalid window boundary: {row}")
        w = int(w)
        if w not in exp.windows or w in exp.bounds:
            raise InputError(f"Unknown or duplicate boundary window {w} for {exp.path}")
        exp.bounds[w] = (start, end)
    for exp in experiments:
        previous_end = -1.0
        for w in sorted(exp.bounds):
            start, end = exp.bounds[w]
            if start < previous_end:
                raise InputError(f"Overlapping or unordered window boundaries in {exp.path}")
            previous_end = end


def load_results(results_root, *, allow_partial=False, window_boundaries=None):
    """Load once, preserving file hashes, window inventory and diagnostics.

    Invalid files/values fail by default. Partial mode skips invalid files and
    converts invalid cells to NaN. Missing cells remain unknown in either mode.
    """
    diagnostics = Diagnostics(allow_partial)
    experiments = _discover(Path(results_root).resolve(), diagnostics)
    detail, cost, summaries = [], [], []
    for exp in experiments:
        for w, path in _window_files(exp.path / "summary", diagnostics):
            exp.windows.add(w)
            df = _read(path, ["application_id", "response_times_mean"], diagnostics)
            if df is None:
                continue
            _numeric(df, ["response_times_mean"], path, diagnostics)
            for name in df.application_id:
                try:
                    region, model = _parse_application(str(name), exp)
                    exp.add_model(region, model)
                    exp.applications[str(name)] = (region, model)
                except ValueError as exc:
                    diagnostics.add("invalid_application", path, str(exc), error=True)
            summaries.append(_attach(df, exp, w, path))
        for app in sorted((exp.path / "detailed").glob("*")):
            if not app.is_dir():
                continue
            try:
                region, model = _parse_application(app.name, exp)
            except ValueError as exc:
                diagnostics.add("invalid_application", app, str(exc), error=True)
                continue
            exp.applications[app.name] = (region, model)
            exp.add_model(region, model)
            for w, path in _window_files(app, diagnostics):
                exp.windows.add(w)
                df = _read(path, DETAIL_COLUMNS, diagnostics)
                if df is None:
                    exp.invalid_files.add((app.name, w))
                    continue
                exp.detail_files.add((app.name, w))
                _numeric(df, DETAIL_COLUMNS, path, diagnostics, nonnegative=DETAIL_COLUMNS,
                         integers=("prompt_sizes", "token_sizes"))
                invalid = (df.queue_times > df.ttft_times + 1e-6) | (df.ttft_times > df.response_times + 1e-6)
                invalid |= df.token_sizes.eq(0)
                expected_tbt = (df.response_times - df.ttft_times) / df.token_sizes.replace(0, np.nan)
                valid_pair = expected_tbt.notna() & df.tbt_times.notna()
                invalid |= valid_pair & ~np.isclose(df.tbt_times, expected_tbt, rtol=1e-5, atol=1e-6)
                if invalid.any():
                    diagnostics.add("invalid_timing", path,
                                    f"{int(invalid.sum())} rows violate queue <= TTFT <= response, positive token count, or the exported TBT formula.", error=True)
                    df.loc[invalid, DETAIL_COLUMNS] = np.nan
                if df[DETAIL_COLUMNS].isna().any().any():
                    exp.invalid_files.add((app.name, w))
                df["region"], df["model"], df["application_id"] = region, model, app.name
                detail.append(_attach(df, exp, w, path))
        for region_dir in sorted((exp.path / "arbiters").glob("*")):
            if not region_dir.is_dir():
                continue
            for w, path in _window_files(region_dir, diagnostics):
                exp.windows.add(w)
                df = _read(path, COST_COLUMNS, diagnostics)
                if df is None:
                    exp.invalid_cost_files.add(path)
                    continue
                _numeric(df, ["timestamp", "prod", "spot", "memory_util"], path, diagnostics,
                         nonnegative=("timestamp", "prod", "spot", "memory_util"), integers=("prod", "spot"))
                if df[["timestamp", "model", "prod", "spot"]].isna().any().any():
                    exp.invalid_cost_files.add(path)
                if df.model.isna().any():
                    diagnostics.add("missing_model", path, "Arbiter rows have no model identifier.", error=True)
                    df = df[df.model.notna()].copy()
                df["model"] = df.model.astype(str)
                for model in df.model.unique():
                    exp.add_model(region_dir.name, model)
                df["region"] = region_dir.name
                cost.append(_attach(df, exp, w, path))
        _audit_memory_exports(exp, diagnostics)
        if exp.windows and exp.windows != set(range(max(exp.windows)+1)):
            diagnostics.add("missing_windows", exp.path, "Numbered exports have gaps; unexported intervals cannot be assumed quiet.")
    if window_boundaries is not None:
        _load_boundaries(window_boundaries, experiments)
        for exp in experiments:
            for frame in cost:
                if frame.empty or frame.experiment_id.iloc[0] != exp.meta["experiment_id"]:
                    continue
                for window, (start, end) in exp.bounds.items():
                    times = frame.loc[frame.window.eq(window), "timestamp"].dropna()
                    if times.lt(start-1e-6).any() or times.gt(end+1e-6).any():
                        raise InputError(f"{exp.path}: capacity snapshots fall outside supplied boundaries for window {window}")
    for exp in experiments:
        if len(exp.bounds) != len(exp.windows):
            diagnostics.add("unknown_duration", exp.path,
                            "Exact export boundaries are unavailable; window numbers and configured end_time are not durations.")
    concat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return Results(experiments, concat(detail), concat(cost), concat(summaries), diagnostics)


def _public_frame(results_root, name, **kwargs):
    result = load_results(results_root, **kwargs)
    result.diagnostics.emit()
    df = getattr(result, name)
    df.attrs["diagnostics"] = result.diagnostics.issues
    return df


def load_detailed(results_root, **kwargs):
    return _public_frame(results_root, "detailed", **kwargs)


def load_cost(results_root, **kwargs):
    return _public_frame(results_root, "cost", **kwargs)


def load_summary(results_root, **kwargs):
    """Return active summaries; inventory still includes inactive exports."""
    df = _public_frame(results_root, "summary", **kwargs)
    return df[df.response_times_mean >= 0].copy() if not df.empty else df
