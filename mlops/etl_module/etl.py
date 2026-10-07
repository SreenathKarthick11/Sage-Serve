from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import validate_raw_trace


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

def run_etl( trace_path: Path, output_path: Path ) -> pd.DataFrame:
    """
    Execute the ETL pipeline.

    The ETL preserves the original SageServe trace schema.
    """

    print(f"Loading trace: {trace_path}")
    df = load_trace(trace_path)
    print(f"Raw rows: {len(df)}")

    df = clean_trace(df)
    validate_raw_trace(df)

    output_path.parent.mkdir(parents=True,exist_ok=True,)
    df.to_csv(output_path,index=False)

    print(f"Validated forecasting input written to: "f"{output_path}")

    return df