from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


TRACE_PATH = Path("traces/random_trace.csv")
OUTPUT_DIR = Path("traces/forecasts")

ARIMA_ORDER = (1, 1, 1)
BUCKET_SECONDS = 60
FORECAST_HORIZON = 60

# The simulator is configured with these regions.
REGIONS = [0, 1, 2]


def load_trace():
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Trace not found: {TRACE_PATH.resolve()}")

    df = pd.read_csv(TRACE_PATH)

    required = ["arrival_timestamp", "model_type", "regions", "prompt_size"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_source_region(region_code):
    """
    Extract the source region from the current synthetic trace encoding.

    For example:
        12  -> region 1
        21  -> region 2
        102 -> region 1
        120 -> region 1
        201 -> region 2
        210 -> region 2
    """
    value = str(region_code)
    return int(value[0])


def build_series(df, model_name, region_id):
    subset = df[
        (df["model_type"] == model_name)
        & (df["regions"].apply(get_source_region) == region_id)
    ].copy()

    # No observations for this model/region.
    if subset.empty:
        return pd.Series(dtype=float)

    subset["minute"] = subset["arrival_timestamp"] // BUCKET_SECONDS

    tokens_per_minute = (
        subset.groupby("minute")["prompt_size"]
        .sum()
        .sort_index()
    )

    full_index = pd.RangeIndex(
        tokens_per_minute.index.min(),
        tokens_per_minute.index.max() + 1,
        name="minute",
    )

    tokens_per_minute = tokens_per_minute.reindex(
        full_index,
        fill_value=0,
    )

    # prompt tokens/minute -> input TPS
    return tokens_per_minute / BUCKET_SECONDS


def generate_zero_forecast(last_observed_minute):
    """
    Generate a zero-demand forecast for a model/region
    which has no observations in the trace.
    """
    future_minutes = np.arange(
        last_observed_minute + 1,
        last_observed_minute + 1 + FORECAST_HORIZON,
    )

    return pd.DataFrame({
        "Time": future_minutes * BUCKET_SECONDS,
        "Predicted": 0.0,
    })


def generate_forecast(series):
    if len(series) < 10:
        raise ValueError(
            f"Not enough observations for ARIMA: {len(series)}"
        )

    model = ARIMA(
        series.astype(float),
        order=ARIMA_ORDER,
    )

    fitted = model.fit()

    predictions = fitted.forecast(
        steps=FORECAST_HORIZON
    )

    predictions = np.maximum(
        np.asarray(predictions, dtype=float),
        0,
    )

    first_future_minute = int(series.index[-1]) + 1

    future_minutes = np.arange(
        first_future_minute,
        first_future_minute + FORECAST_HORIZON,
    )

    return pd.DataFrame({
        "Time": future_minutes * BUCKET_SECONDS,
        "Predicted": predictions,
    })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_trace()

    models = sorted(
        df["model_type"].unique()
    )

    # Always use the regions configured by the simulator,
    # not only the regions that happen to occur in this trace.
    regions = REGIONS

    last_observed_minute = int(
        df["arrival_timestamp"].max() // BUCKET_SECONDS
    )

    print("Models:", models)
    print("Regions:", regions)
    print("Last observed minute:", last_observed_minute)

    for model_name in models:
        for region_id in regions:
            print(
                f"Generating forecast: "
                f"model={model_name}, region={region_id}"
            )

            series = build_series(
                df,
                model_name,
                region_id,
            )

            if series.empty:
                print(
                    f"  No workload found. "
                    f"Generating zero forecast."
                )

                forecast = generate_zero_forecast(
                    last_observed_minute
                )

            else:
                print(
                    f"  Historical observations: "
                    f"{len(series)}"
                )

                forecast = generate_forecast(
                    series
                )

            output_path = (
                OUTPUT_DIR
                / f"forecast_{model_name}_region{region_id}.csv"
            )

            forecast.to_csv(
                output_path,
                index=False,
            )

            print(
                f"  Saved {output_path} "
                f"({len(forecast)} rows)"
            )

    print("\nForecast generation completed.")


if __name__ == "__main__":
    main()