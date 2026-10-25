from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.versioning import (
    artifact_id,
    derive_id,
    finalize,
    new_tmp_dir,
    set_latest,
    version_dir,
    version_exists,
)

from .validation import validate_raw_trace

# Bump CLEANING_VERSION whenever clean_trace() or the validation suite changes behaviour.
# It is part of the clean-stage id, so a logic change yields a new version of the same raw input.
CLEANING_VERSION = "1"
CLEAN_PARAMS = {
    "cleaning_version": CLEANING_VERSION,
    "validation_suite": "sageserve_raw_trace",
    "regions_zfill": 3,
}

RAW_FILE = "trace.csv"
CLEAN_FILE = "validated_trace.csv"


def load_trace(trace_path: Path) -> pd.DataFrame:
    """
    Load a SageServe workload trace.

    `regions` is explicitly read as a string so values such as
    `012` retain their leading zero.
    """

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path.resolve()}")

    df = pd.read_csv(
        trace_path,
        dtype={
            "regions": "string",
            "model_type": "string",
        },
    )

    return df


def clean_trace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize values without silently dropping bad records.

    Data-quality problems are left in the DataFrame so that the
    validation stage can report them.
    """

    df = df.copy()

    # Normalize string fields.
    df["model_type"] = (df["model_type"].astype("string").str.strip())
    df["regions"] = (df["regions"].astype("string").str.strip().str.zfill(3))

    # Normalize numeric fields.
    numeric_columns = [
        "request_id",
        "batch_id",
        "client_tenant",
        "request_type",
        "sla",
        "utility",
        "application_id",
        "arrival_timestamp",
        "batch_size",
        "prompt_size",
        "token_size",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column],errors="coerce")

    return df


def run_etl(input_dir: Path, source: str) -> Path:
    """
    Execute the ETL pipeline on one raw version.

    input_dir : data/<source>/raw/<version>/   (contains trace.csv + manifest.json)
    returns   : data/<source>/clean/<version>/ (contains validated_trace.csv + manifest.json)

    The ETL preserves the original SageServe trace schema. The clean version id is
    derived from (raw artifact id, CLEAN_PARAMS), so re-running on the same raw version
    with the same logic is skipped.
    """

    raw_id = artifact_id(source, "raw", input_dir.name)
    version_id = derive_id([raw_id], CLEAN_PARAMS)

    if version_exists(source, "clean", version_id):
        print(f"Clean version {version_id} already exists for {raw_id}; skipping.")
        set_latest(source, "clean", version_id)
        return version_dir(source, "clean", version_id)

    trace_path = input_dir / RAW_FILE
    print(f"Loading trace: {trace_path}")
    df = load_trace(trace_path)
    rows_in = len(df)
    print(f"Raw rows: {rows_in}")

    df = clean_trace(df)
    result = validate_raw_trace(df)          # raises ValueError on failure

    tmp_dir = new_tmp_dir(source, "clean")
    try:
        df.to_csv(tmp_dir / CLEAN_FILE, index=False)
        details = {
            "rows_in": rows_in,
            "rows_out": len(df),
            "validation": {"suite": CLEAN_PARAMS["validation_suite"], "success": bool(result.success)},
        }
        final = finalize(tmp_dir, source, "clean", version_id, parents=[raw_id],
                         params=CLEAN_PARAMS, details=details)
    except BaseException:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(f"Validated forecasting input written to: {final / CLEAN_FILE}")
    return final
