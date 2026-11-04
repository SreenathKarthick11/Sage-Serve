"""ARIMA forecaster.

ARIMAForecaster implements the BaseForecaster contract. predict() forecasts one
time series; run_forecast() reads the clean trace, forecasts every
model/source-region pair, and publishes versioned CSV artifacts.

The pipeline loads the active forecaster through ForecasterFactory and calls
forecaster.run_forecast(); it never needs to know which model is active.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.arima.model import ARIMA

from forecasting.base import BaseForecaster
from forecasting.trace_io import load_clean_trace, safe_filename_part
from pipeline.versioning import (
    artifact_id,
    derive_id,
    finalize,
    new_tmp_dir,
    set_latest,
    version_dir,
    version_exists,
)

FORECAST_STAGE = "forecasts"
FORECAST_IMPLEMENTATION = "arima-wrapper-v1"


class ARIMAForecaster(BaseForecaster):
    """ARIMA implementation of the common BaseForecaster interface."""

    REQUIRED_PARAMS = ["order", "train_window_mins", "future_horizon", "bucket_seconds"]
    WEIGHTS_FILE = None

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

        order_dict = self.parameters.get("order", {"p": 1, "d": 1, "q": 1})
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
        self.enforce_stationarity = self._boolean_parameter("enforce_stationarity", False)
        self.enforce_invertibility = self._boolean_parameter("enforce_invertibility", False)

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

        minimum_history = max(5, sum(self.order) + 2)
        if history.size < minimum_history:
            return np.full(self.future_horizon, float(np.mean(history)), dtype=float)

        try:
            fitted = ARIMA(
                history,
                order=self.order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            ).fit()

            predictions = np.asarray(
                fitted.forecast(steps=self.future_horizon), dtype=float
            )
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
            return np.maximum(predictions, 0.0)

        except Exception:
            # Mean-baseline fallback for short/degenerate series.
            return np.full(self.future_horizon, float(np.mean(history)), dtype=float)

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

    def run_forecast(self, clean_dir, source: str) -> Path:
        """Forecast every model/source-region pair and publish a versioned artifact."""
        clean_dir = Path(clean_dir)
        df = load_clean_trace(clean_dir)

        models = sorted(df["model_type"].dropna().unique().tolist(), key=str)
        regions = sorted(df["source_region"].dropna().unique().tolist())
        if not models or not regions:
            raise ValueError("Could not discover any model names or source regions.")

        df["minute"] = (df["arrival_timestamp"] // self.bucket_seconds).astype(int)
        last_min = int(df["minute"].max())
        input_minutes = pd.RangeIndex(
            max(0, last_min - self.train_window + 1), last_min + 1, name="minute"
        )

        clean_id = artifact_id(source, "clean", clean_dir.name)
        params = {
            "forecaster_implementation": FORECAST_IMPLEMENTATION,
            "forecaster_metadata": self.get_metadata(),
            "train_window": self.train_window,
            "future_horizon": self.future_horizon,
            "bucket_seconds": self.bucket_seconds,
            "workload_metric": "sum(prompt_size) per minute",
            "models": [str(m) for m in models],
            "source_regions": [int(r) for r in regions],
        }
        version = derive_id([clean_id], params)

        if version_exists(source, FORECAST_STAGE, version):
            print(f"Forecast version {version} already exists; reusing it.")
            set_latest(source, FORECAST_STAGE, version)
            return version_dir(source, FORECAST_STAGE, version)

        tmp_dir = new_tmp_dir(source, FORECAST_STAGE)
        try:
            n_files = 0
            # Cartesian product is intentional: a model-region pair with no
            # requests in the window gets an all-zero workload series.
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
                        self.predict(per_minute.to_numpy()), dtype=float
                    ).reshape(-1)

                    if predictions.size != self.future_horizon:
                        raise ValueError(
                            f"predict() returned {predictions.size} values; "
                            f"expected {self.future_horizon}."
                        )
                    if not np.all(np.isfinite(predictions)):
                        raise ValueError("Forecaster returned NaN or infinite predictions.")

                    future_minutes = np.arange(
                        last_min + 1, last_min + 1 + self.future_horizon
                    )
                    forecast_df = pd.DataFrame(
                        {
                            "Time": future_minutes * self.bucket_seconds,
                            "Predicted": np.maximum(predictions, 0.0),
                        }
                    )

                    filename = (
                        f"forecast_{safe_filename_part(model_name)}"
                        f"_region{region_id}.csv"
                    )
                    forecast_df.to_csv(tmp_dir / filename, index=False)
                    n_files += 1
                    print(
                        f"Saved {filename}: model={model_name}, "
                        f"region={region_id}, rows={len(forecast_df)}"
                    )

            final_dir = finalize(
                tmp_dir,
                source,
                FORECAST_STAGE,
                version,
                parents=[clean_id],
                params=params,
                details={
                    "status": "complete",
                    "n_forecast_files": n_files,
                    "models": [str(m) for m in models],
                    "source_regions": [int(r) for r in regions],
                    "last_observed_minute": last_min,
                },
            )
            print(f"Forecast generation completed: {final_dir}")
            return final_dir

        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise