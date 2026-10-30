"""ARIMA forecaster. Reads a versioned clean trace, writes a versioned set of forecasts.

    python -m forecasting.arima_forecaster --source streaming                 # latest clean version
    python -m forecasting.arima_forecaster --source static --input-version <id>

Output: data/<source>/forecasts/<version>/forecast_<model>_region<r>.csv  (+ manifest.json)
The forecast version id is derived from (clean artifact id, FORECAST_PARAMS).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from pipeline.paths import SOURCES
from pipeline.versioning import (
    artifact_id,
    derive_id,
    finalize,
    new_tmp_dir,
    resolve_version,
    set_latest,
    version_dir,
    version_exists,
)

CLEAN_FILE = "validated_trace.csv"

ARIMA_ORDER = (1, 1, 1)

BUCKET_SECONDS = 60
TRAIN_WINDOW = 60

# Forecast 61 future minutes so that the long-term
# 20-60 minute look-ahead is fully covered.
FUTURE_HORIZON = 61

# Regions configured in SageServe.
REGIONS = [0, 1, 2]

# Everything that influences the numbers; part of the forecast version id.
FORECAST_PARAMS = {
    "arima_order": list(ARIMA_ORDER),
    "bucket_seconds": BUCKET_SECONDS,
    "train_window": TRAIN_WINDOW,
    "future_horizon": FUTURE_HORIZON,
    "regions": REGIONS,
    "enforce_stationarity": False,
    "enforce_invertibility": False,
}


def load_trace(trace_path: Path):
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Trace not found: {trace_path.resolve()}"
        )

    # `regions` as string keeps leading zeros ("012"), instead of relying on zfill later.
    df = pd.read_csv(trace_path, dtype={"regions": "string"})

    required = [
        "arrival_timestamp",
        "model_type",
        "regions",
        "prompt_size",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def get_source_region(region_code):
    """
    Extract the first-priority region from the three-region
    priority encoding.

    Examples:
        012 -> 0
        021 -> 0
        102 -> 1
        120 -> 1
        201 -> 2
        210 -> 2
    """

    value = str(region_code).zfill(3)

    return int(value[0])


def build_series(df, model_name, region_id):
    """
    Build one-minute prompt-token demand for a model and region.

    The current SageServe ARIMA checking code adds prompt_size for
    every request that reaches the model endpoint, so this signal
    uses all rows in the trace rather than filtering by workload_type.
    """

    subset = df[
        (df["model_type"] == model_name)
        & (
            df["regions"].apply(get_source_region)
            == region_id
        )
    ].copy()

    if subset.empty:
        return pd.Series(dtype=float)

    subset["minute"] = (
        subset["arrival_timestamp"]
        // BUCKET_SECONDS
    )

    tokens_per_minute = (
        subset
        .groupby("minute")["prompt_size"]
        .sum()
        .sort_index()
    )

    full_index = pd.RangeIndex(
        tokens_per_minute.index.min(),
        tokens_per_minute.index.max() + 1,
        name="minute",
    )

    tokens_per_minute = (
        tokens_per_minute
        .reindex(full_index, fill_value=0)
        .astype(float)
    )

    return tokens_per_minute


def fit_arima(series):
    """
    Fit the configured ARIMA model.
    """

    model = ARIMA(
        series.astype(float),
        order=ARIMA_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model.fit()


def generate_forecast(series):
    """
    Generate one-step-ahead forecasts over the observed trace,
    followed by FUTURE_HORIZON recursive forecasts.

    A full 60-minute ARIMA history is available starting at
    target minute 60.

    To avoid refitting thousands of ARIMA models, parameters are
    refitted every TRAIN_WINDOW minutes. Between refits, the fitted
    model is updated with newly observed values.
    """

    series = series.astype(float)

    last_observed_minute = int(series.index.max())

    forecasts = {}

    fitted = None

    # ---------------------------------------------------------
    # Warm-up + walk-forward forecasting
    # ---------------------------------------------------------

    for target_minute in range(
        1,
        last_observed_minute + 1
    ):

        history_end = target_minute

        history_start = max(
            0,
            history_end - TRAIN_WINDOW
        )

        history = series.loc[
            history_start:history_end - 1
        ]

        # -----------------------------------------------------
        # Warm-up period (< 60 minutes)
        # -----------------------------------------------------

        if target_minute < TRAIN_WINDOW:

            if len(history) == 0:
                prediction = 0.0
            else:
                # Simple causal fallback until enough history
                # exists for ARIMA.
                prediction = float(history.mean())

        else:

            # -------------------------------------------------
            # Refit once per 60 minutes
            # -------------------------------------------------

            if (
                fitted is None
                or target_minute % TRAIN_WINDOW == 0
            ):

                history = series.loc[
                    target_minute - TRAIN_WINDOW:
                    target_minute - 1
                ]

                fitted = fit_arima(history)

            # -------------------------------------------------
            # Predict next minute
            # -------------------------------------------------

            prediction = float(
                fitted.forecast(steps=1).iloc[0]
            )

            prediction = max(
                0.0,
                prediction,
            )

        forecasts[target_minute] = prediction

        # -----------------------------------------------------
        # Update model with the newly observed value
        # -----------------------------------------------------

        if fitted is not None:

            actual_value = float(
                series.loc[target_minute]
            )

            fitted = fitted.append(
                [actual_value],
                refit=False,
            )

    # ---------------------------------------------------------
    # Future forecast after the trace ends
    # ---------------------------------------------------------

    final_history = series.iloc[-TRAIN_WINDOW:]

    if len(final_history) >= TRAIN_WINDOW:

        fitted = fit_arima(
            final_history
        )

        future_predictions = (
            fitted
            .forecast(
                steps=FUTURE_HORIZON
            )
        )

        first_future_minute = (
            last_observed_minute + 1
        )

        for offset, prediction in enumerate(
            future_predictions,
            start=0,
        ):

            minute = (
                first_future_minute
                + offset
            )

            forecasts[minute] = max(
                0.0,
                float(prediction),
            )

    forecast_series = pd.Series(
        forecasts,
        dtype=float,
    )

    forecast_series.index.name = "minute"

    return forecast_series


def generate_zero_forecast(
    last_observed_minute
):
    """
    Used when a model/region has no workload.
    """

    future_minutes = np.arange(
        1,
        last_observed_minute
        + FUTURE_HORIZON
        + 1,
    )

    return pd.DataFrame({
        "Time":
            future_minutes
            * BUCKET_SECONDS,

        "Predicted": 0.0,
    })


def run_forecast(clean_dir: Path, source: str) -> Path:
    """Generate forecasts for every (model, region) from one clean version.

    clean_dir : data/<source>/clean/<version>/
    returns   : data/<source>/forecasts/<version>/
    """

    clean_id = artifact_id(source, "clean", clean_dir.name)
    version_id = derive_id([clean_id], FORECAST_PARAMS)

    if version_exists(source, "forecasts", version_id):
        print(f"Forecast version {version_id} already exists for {clean_id}; skipping.")
        set_latest(source, "forecasts", version_id)
        return version_dir(source, "forecasts", version_id)

    df = load_trace(clean_dir / CLEAN_FILE)

    models = sorted(
        df["model_type"].unique()
    )

    last_observed_minute = int(
        df["arrival_timestamp"].max()
        // BUCKET_SECONDS
    )

    print("Models:", models)
    print("Regions:", REGIONS)
    print(
        "Last observed minute:",
        last_observed_minute,
    )

    print(
        "Last observed timestamp:",
        df["arrival_timestamp"].max(),
    )

    tmp_dir = new_tmp_dir(source, "forecasts")
    try:
        n_files = 0
        for model_name in models:

            for region_id in REGIONS:

                print()
                print(
                    f"Generating forecast: "
                    f"model={model_name}, "
                    f"region={region_id}"
                )

                series = build_series(
                    df,
                    model_name,
                    region_id,
                )

                if series.empty:

                    print(
                        "  No workload found. "
                        "Using zero forecast."
                    )

                    forecast = (
                        generate_zero_forecast(
                            last_observed_minute
                        )
                    )

                else:

                    print(
                        "  Historical observations:",
                        len(series),
                    )

                    print(
                        "  Total prompt tokens:",
                        int(series.sum()),
                    )

                    forecast_series = (
                        generate_forecast(series)
                    )

                    forecast = pd.DataFrame({
                        "Time":
                            forecast_series.index
                            .astype(int)
                            * BUCKET_SECONDS,

                        "Predicted":
                            forecast_series.values,
                    })

                output_path = (
                    tmp_dir
                    / (
                        f"forecast_"
                        f"{model_name}_"
                        f"region{region_id}.csv"
                    )
                )

                forecast.to_csv(
                    output_path,
                    index=False,
                )
                n_files += 1

                print(
                    f"  Saved {output_path.name} "
                    f"({len(forecast)} rows)"
                )

        details = {
            "n_forecast_files": n_files,
            "models": [str(m) for m in models],
            "last_observed_minute": last_observed_minute,
        }
        final = finalize(tmp_dir, source, "forecasts", version_id, parents=[clean_id],
                         params=FORECAST_PARAMS, details=details)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print()
    print(f"Forecast generation completed -> {final}")
    return final


def main():
    parser = argparse.ArgumentParser(description="Generate versioned ARIMA forecasts from a clean trace version.")
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--input-version", default="latest",
                        help="clean version id, or 'latest' (default: %(default)s)")
    args = parser.parse_args()

    clean_dir = resolve_version(args.source, "clean", args.input_version)
    run_forecast(clean_dir, args.source)


if __name__ == "__main__":
    main()
