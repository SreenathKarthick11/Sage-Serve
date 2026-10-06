from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import (
    VALID_REGION_ENCODINGS,
    validate_forecasting_dataset,
    validate_raw_trace,
)


BUCKET_SECONDS = 60


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


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fields required for workload aggregation.

    The first digit of the `regions` priority encoding is treated
    as the source region.
    """

    df = df.copy()

    if not df["regions"].isin(
        VALID_REGION_ENCODINGS
    ).all():
        invalid = sorted(
            df.loc[
                ~df["regions"].isin(
                    VALID_REGION_ENCODINGS
                ),
                "regions",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Invalid region encodings: {invalid}"
        )

    df["region_id"] = (
        df["regions"]
        .str[0]
        .astype(int)
    )

    df["minute"] = (
        df["arrival_timestamp"]
        // BUCKET_SECONDS
    ).astype("Int64")

    return df


def aggregate_trace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate workload into one-minute model/region buckets.

    The prompt_size signal matches the demand signal currently
    used by the SageServe ARIMA forecasting implementation.
    """

    aggregated = (
        df.groupby(
            [
                "minute",
                "model_type",
                "region_id",
            ],
            as_index=False,
        )
        .agg(
            prompt_size=(
                "prompt_size",
                "sum",
            ),
            token_size=(
                "token_size",
                "sum",
            ),
            request_count=(
                "request_id",
                "count",
            ),
        )
        .sort_values(
            [
                "minute",
                "model_type",
                "region_id",
            ]
        )
        .reset_index(drop=True)
    )

    return aggregated


def run_etl(
    trace_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Execute the complete ETL pipeline.
    """

    print(f"Loading trace: {trace_path}")

    df = load_trace(trace_path)

    print(f"Raw rows: {len(df)}")

    # Normalize values first.
    df = clean_trace(df)

    # Validate the cleaned raw trace before aggregation.
    validate_raw_trace(df)

    # Add forecasting-specific fields.
    df = add_derived_columns(df)

    # Aggregate workload.
    forecasting_df = aggregate_trace(df)

    print(
        f"Forecasting rows: "
        f"{len(forecasting_df)}"
    )

    # Validate the final forecasting dataset.
    validate_forecasting_dataset(
        forecasting_df
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasting_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Forecasting dataset written to: "
        f"{output_path}"
    )

    return forecasting_df