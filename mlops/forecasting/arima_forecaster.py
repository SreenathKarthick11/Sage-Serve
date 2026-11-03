"""ARIMA forecaster and versioned forecast-stage orchestration.

The ARIMAForecaster class implements the BaseForecaster contract and predicts
one time series. run_forecast() is pipeline code: it reads a clean trace,
discovers model/region combinations, invokes the forecaster, and publishes
versioned CSV artifacts.

The pipeline should load the active forecaster through ForecasterFactory and
pass that instance to run_forecast().
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.arima.model import ARIMA

from forecasting.base import BaseForecaster
from pipeline.versioning import (
    artifact_id,
    derive_id,
    finalize,
    new_tmp_dir,
    set_latest,
    version_dir,
    version_exists,
)

CLEAN_FILE = "validated_trace.csv"
FORECAST_STAGE = "forecasts"
FORECAST_IMPLEMENTATION = "arima-wrapper-v1"


class ARIMAForecaster(BaseForecaster):
    """ARIMA implementation of the common BaseForecaster interface.

    One instance forecasts one input time series per predict() call.
    It does not read trace CSVs, discover model/region pairs, or publish
    versioned artifacts; those tasks belong to run_forecast().
    """

    def __init__(
        self,
        config_path: str,
        weights_path: Union[str, None] = None,
    ) -> None:
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(
                f"ARIMA config file not found: {config_file.resolve()}"
            )

        with config_file.open("r", encoding="utf-8") as f:
            loaded_config = yaml.safe_load(f) or {}

        if not isinstance(loaded_config, dict):
            raise ValueError("ARIMA config must contain a YAML mapping.")

        self.config: Dict[str, Any] = loaded_config
        self.model_info: Dict[str, Any] = self.config.get("model_info", {})
        self.parameters: Dict[str, Any] = self.config.get("parameters", {})

        if not isinstance(self.model_info, dict):
            raise ValueError("'model_info' in ARIMA config must be a mapping.")
        if not isinstance(self.parameters, dict):
            raise ValueError("'parameters' in ARIMA config must be a mapping.")

        order_dict = self.parameters.get(
            "order", {"p": 1, "d": 1, "q": 1}
        )
        if not isinstance(order_dict, dict):
            raise ValueError("'parameters.order' must be a mapping.")

        try:
            self.order: Tuple[int, int, int] = (
                int(order_dict["p"]),
                int(order_dict["d"]),
                int(order_dict["q"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "'parameters.order' must define integer p, d and q values."
            ) from exc

        if any(v < 0 for v in self.order):
            raise ValueError("ARIMA order values p, d and q must be non-negative.")

        self.train_window = self._positive_int("train_window_mins", 60)
        self.future_horizon = self._positive_int("future_horizon", 60)
        self.bucket_seconds = self._positive_int("bucket_seconds", 60)
        self.enforce_stationarity = self._boolean_parameter(
            "enforce_stationarity", False
        )
        self.enforce_invertibility = self._boolean_parameter(
            "enforce_invertibility", False
        )

        self.config_path = str(config_file)
        # Accepted for BaseForecaster interface compatibility. ARIMA has no
        # separate weights file; it estimates parameters from predict() input.
        self.weights_path = weights_path

    def _positive_int(self, key: str, default: int) -> int:
        value = self.parameters.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"'parameters.{key}' must be a positive integer.")
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'parameters.{key}' must be a positive integer."
            ) from exc
        if value <= 0:
            raise ValueError(f"'parameters.{key}' must be a positive integer.")
        return value

    def _boolean_parameter(self, key: str, default: bool) -> bool:
        value = self.parameters.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"'parameters.{key}' must be true or false.")
        return value

    def predict(
        self,
        input_data: Union[list[float], np.ndarray],
    ) -> np.ndarray:
        """Forecast the next future_horizon buckets from one workload series.

        input_data must be chronological, oldest to newest. If it contains
        more than train_window observations, only the most recent window is
        used. The returned array contains future predictions only.
        """
        values = np.asarray(input_data, dtype=float).reshape(-1)

        if not np.all(np.isfinite(values)):
            raise ValueError("input_data contains NaN or infinite values.")
        if np.any(values < 0):
            raise ValueError("Workload observations must be non-negative.")

        if values.size == 0 or np.all(values == 0):
            return np.zeros(self.future_horizon, dtype=float)

        history = values[-self.train_window:]

        # Very short histories are not reliable ARIMA training inputs.
        # Use a simple, explicit fallback rather than failing the whole stage.
        minimum_history = max(5, sum(self.order) + 2)
        if history.size < minimum_history:
            return np.full(
                self.future_horizon,
                float(np.mean(history)),
                dtype=float,
            )

        try:
            fitted = ARIMA(
                history,
                order=self.order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            ).fit()

            predictions = np.asarray(
                fitted.forecast(steps=self.future_horizon),
                dtype=float,
            )
            predictions = np.nan_to_num(
                predictions, nan=0.0, posinf=0.0, neginf=0.0
            )
            return np.maximum(predictions, 0.0)

        except Exception:
            # Keep serving robust if statsmodels cannot fit a particular
            # short/degenerate series. This is a mean-baseline fallback.
            return np.full(
                self.future_horizon,
                float(np.mean(history)),
                dtype=float,
            )

    def get_metadata(self) -> Dict[str, Any]:
        """Return stable model/config metadata for the registry and lineage."""
        return {
            "model_type": "ARIMA",
            "order": list(self.order),
            "train_window": self.train_window,
            "future_horizon": self.future_horizon,
            "bucket_seconds": self.bucket_seconds,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
            "model_info": self.model_info,
        }


def _source_region(region_code: Any) -> int:
    """Extract the first-priority region from a three-digit region encoding."""
    if pd.isna(region_code):
        raise ValueError("Encountered a missing value in the 'regions' column.")
    value = str(region_code).strip().zfill(3)
    if not value.isdigit() or len(value) != 3:
        raise ValueError(f"Invalid region code: {region_code!r}")
    return int(value[0])


def _load_clean_trace(clean_dir: Path) -> pd.DataFrame:
    trace_path = clean_dir / CLEAN_FILE
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"Clean trace not found: {trace_path.resolve()}"
        )

    df = pd.read_csv(trace_path, dtype={"regions": "string"})
    required = [
        "arrival_timestamp",
        "model_type",
        "regions",
        "prompt_size",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Clean trace is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"Clean trace contains no rows: {trace_path}")

    df = df.copy()
    df["arrival_timestamp"] = pd.to_numeric(
        df["arrival_timestamp"], errors="raise"
    )
    df["prompt_size"] = pd.to_numeric(df["prompt_size"], errors="raise")
    if df[["arrival_timestamp", "prompt_size"]].isna().any().any():
        raise ValueError("Clean trace contains null timestamps or prompt sizes.")
    if (df["arrival_timestamp"] < 0).any():
        raise ValueError("arrival_timestamp values must be non-negative.")
    if (df["prompt_size"] < 0).any():
        raise ValueError("prompt_size values must be non-negative.")
    if df["model_type"].isna().any():
        raise ValueError("model_type contains missing values.")

    df["source_region"] = df["regions"].map(_source_region)
    return df


def _safe_filename_part(value: Any) -> str:
    """Make model names safe for filenames while retaining readable names."""
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "unnamed"


def run_forecast(
    clean_dir: Path,
    source: str,
    forecaster: BaseForecaster,
) -> Path:
    """Forecast every discovered model/source-region pair and publish an artifact.

    The input forecaster is normally obtained from ForecasterFactory.
    The clean trace is expected at clean_dir / validated_trace.csv.
    """
    clean_dir = Path(clean_dir)
    df = _load_clean_trace(clean_dir)

    # Discover entities from the trace; no model names or region IDs are fixed.
    models = sorted(df["model_type"].dropna().unique().tolist(), key=str)
    regions = sorted(df["source_region"].dropna().unique().tolist())

    if not models or not regions:
        raise ValueError("Could not discover any model names or source regions.")

    metadata = forecaster.get_metadata()
    if not isinstance(metadata, dict):
        raise TypeError("forecaster.get_metadata() must return a dictionary.")

    # These attributes are part of the ARIMAForecaster implementation.
    # A different forecaster may expose its own window/horizon contract.
    train_window = int(getattr(forecaster, "train_window"))
    future_horizon = int(getattr(forecaster, "future_horizon"))
    bucket_seconds = int(getattr(forecaster, "bucket_seconds"))

    df["minute"] = (
        df["arrival_timestamp"] // bucket_seconds
    ).astype(int)

    last_observed_minute = int(df["minute"].max())
    first_input_minute = max(0, last_observed_minute - train_window + 1)
    input_minutes = pd.RangeIndex(
        first_input_minute,
        last_observed_minute + 1,
        name="minute",
    )

    clean_id = artifact_id(source, "clean", clean_dir.name)
    params = {
        "forecaster_implementation": FORECAST_IMPLEMENTATION,
        "forecaster_metadata": metadata,
        "train_window": train_window,
        "future_horizon": future_horizon,
        "bucket_seconds": bucket_seconds,
        "workload_metric": "sum(prompt_size) per minute",
        "models": [str(model) for model in models],
        "source_regions": [int(region) for region in regions],
    }
    forecast_version = derive_id([clean_id], params)

    if version_exists(source, FORECAST_STAGE, forecast_version):
        print(
            f"Forecast version {forecast_version} already exists; "
            "reusing it."
        )
        set_latest(source, FORECAST_STAGE, forecast_version)
        return version_dir(source, FORECAST_STAGE, forecast_version)

    tmp_dir = new_tmp_dir(source, FORECAST_STAGE)
    try:
        n_files = 0

        # Cartesian product is intentional: a model-region pair with no
        # requests in the input window receives an all-zero workload series.
        for model_name in models:
            for region_id in regions:
                subset = df[
                    (df["model_type"] == model_name)
                    & (df["source_region"] == region_id)
                ]

                per_minute = (
                    subset.groupby("minute")["prompt_size"]
                    .sum()
                    .reindex(input_minutes, fill_value=0.0)
                    .astype(float)
                )

                predictions = np.asarray(
                    forecaster.predict(per_minute.to_numpy()),
                    dtype=float,
                ).reshape(-1)

                if predictions.size != future_horizon:
                    raise ValueError(
                        f"{type(forecaster).__name__}.predict() returned "
                        f"{predictions.size} values; expected {future_horizon}."
                    )
                if not np.all(np.isfinite(predictions)):
                    raise ValueError("Forecaster returned NaN or infinite predictions.")

                future_minutes = np.arange(
                    last_observed_minute + 1,
                    last_observed_minute + 1 + future_horizon,
                )
                forecast_df = pd.DataFrame(
                    {
                        "Time": future_minutes * bucket_seconds,
                        "Predicted": np.maximum(predictions, 0.0),
                    }
                )

                filename = (
                    f"forecast_{_safe_filename_part(model_name)}"
                    f"_region{region_id}.csv"
                )
                forecast_df.to_csv(tmp_dir / filename, index=False)
                n_files += 1

                print(
                    f"Saved {filename}: model={model_name}, "
                    f"region={region_id}, rows={len(forecast_df)}"
                )

        details = {
            "status": "complete",
            "n_forecast_files": n_files,
            "models": [str(model) for model in models],
            "source_regions": [int(region) for region in regions],
            "last_observed_minute": last_observed_minute,
        }

        final_dir = finalize(
            tmp_dir,
            source,
            FORECAST_STAGE,
            forecast_version,
            parents=[clean_id],
            params=params,
            details=details,
        )
        print(f"Forecast generation completed: {final_dir}")
        return final_dir

    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
