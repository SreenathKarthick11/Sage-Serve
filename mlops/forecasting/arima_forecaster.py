"""
mlops/forecasting/arima_forecaster.py

ARIMA implementation of the BaseForecaster interface.
Preserves full walk-forward trace generation and zero-forecast fallback
as expected by SageServe.
"""

import os
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yaml

from mlops.forecasting.base import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA Forecaster implementing walk-forward and future recursive forecasting
    for SageServe workload prediction.
    """

    def __init__(self, config_path: str, weights_path: Union[str, None] = None) -> None:
        if not os.path.exists(config_path):     # basic check for existence of config file
            raise FileNotFoundError(f"ARIMA config file not found at: {config_path}")

        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.model_info = self.config.get("model_info", {})
        self.parameters = self.config.get("parameters", {})

        # Hyperparameters from config.yaml
        order_dict = self.parameters.get("order", {"p": 1, "d": 1, "q": 1})     # these are specific to ARIMA
        self.order = (order_dict["p"], order_dict["d"], order_dict["q"])
        self.train_window = self.parameters.get("train_window_mins", 60)    # default 60 for 60 minutes
        self.future_horizon = self.parameters.get("future_horizon", 61)
        self.bucket_seconds = self.parameters.get("bucket_seconds", 60)
        self.config_path = config_path

    def _fit_arima_internal(self, history_series: pd.Series):
        """Helper to fit statsmodels ARIMA on a history window. This window not necessarily 60 minutes"""
        try:
            model = ARIMA(history_series, order=self.order)
            return model.fit()
        except Exception:
            return None

    def generate_zero_forecast(self, last_observed_minute: int) -> pd.DataFrame:
        """
        Generates a zero-filled forecast DataFrame when no workload is found.
        Matches original SageServe specification.
        """
        future_minutes = np.arange(
            1,
            last_observed_minute + self.future_horizon + 1
        )
        
        return pd.DataFrame({
            "Time": future_minutes * self.bucket_seconds,
            "Predicted": 0.0
        })

    def predict(
        self, 
        input_data: Union[List[float], np.ndarray, pd.Series],
        return_dataframe: bool = False,
        last_observed_minute: Union[int, None] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Generates forecast predictions.

        Args:
            input_data: Time series workload data.
            return_dataframe (bool): If True, returns DataFrame with ['Time', 'Predicted']
                                     columns for saving directly to CSV.
            last_observed_minute (int, optional): Explicit last observed minute for zero/trace forecasts.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Forecasted values.
        """
        # for uniformity convert to pd Series
        if isinstance(input_data, pd.Series):
            series = input_data.astype(float).copy()
        else:
            series = pd.Series(np.asarray(input_data, dtype=float))

        # Handle empty series / zero forecast case
        if series.empty:
            obs_min = last_observed_minute if last_observed_minute is not None else 60
            zero_df = self.generate_zero_forecast(obs_min)
            return zero_df if return_dataframe else zero_df["Predicted"].values

        # Case 1: Real-time single 60-min window input (returns array)
        if len(series) <= self.train_window and not return_dataframe:
            fitted = self._fit_arima_internal(series)
            if fitted is not None:
                try:
                    raw_preds = fitted.forecast(steps=self.future_horizon)
                    return np.maximum(0.0, np.asarray(raw_preds, dtype=np.float32))
                except Exception:
                    pass
            # case when fitted is none, means corrupted data or insufficient points available
            mean_val = max(0.0, float(series.mean())) if len(series) > 0 else 0.0
            return np.full((self.future_horizon,), fill_value=mean_val, dtype=np.float32)

        # Case 2: Walk-forward trace execution (original generate_forecast logic)
        obs_min = last_observed_minute if last_observed_minute is not None else int(series.index.max())
        forecasts = {}
        fitted = None

        # 1. Warm-up + Walk-forward loop over observed trace
        for target_minute in range(1, obs_min + 1):
            history_start = max(0, target_minute - self.train_window)
            history = series.loc[history_start : target_minute - 1]

            if target_minute < self.train_window:   # case when not enough data points there
                prediction = 0.0 if len(history) == 0 else float(history.mean())
            else:
                if fitted is None or target_minute % self.train_window == 0:
                    # fit on the window only
                    history = series.loc[target_minute - self.train_window : target_minute - 1]
                    fitted = self._fit_arima_internal(history)

                if fitted is not None:
                    try:
                        pred_val = float(fitted.forecast(steps=1).iloc[0])
                        prediction = max(0.0, pred_val)
                    except Exception:
                        prediction = max(0.0, float(history.mean()))
                else:
                    prediction = max(0.0, float(history.mean()))

            forecasts[target_minute] = prediction

            if fitted is not None and target_minute in series.index:
                actual_value = float(series.loc[target_minute])
                try:
                    fitted = fitted.append([actual_value], refit=False)
                except Exception:
                    pass

        # 2. Recursive future forecasting past trace end
        final_history = series.iloc[-self.train_window :]

        if len(final_history) >= self.train_window:

            fitted_final = self._fit_arima_internal(final_history)

            if fitted_final is not None:
                try:
                    future_preds = fitted_final.forecast(steps=self.future_horizon)
                    for offset, pred in enumerate(future_preds):
                        forecasts[obs_min + 1 + offset] = max(0.0, float(pred))

                except Exception:
                    last_val = max(0.0, float(final_history.iloc[-1]))
                    for offset in range(self.future_horizon):
                        forecasts[obs_min + 1 + offset] = last_val
                        
            else:
                last_val = max(0.0, float(final_history.iloc[-1]))
                for offset in range(self.future_horizon):
                    forecasts[obs_min + 1 + offset] = last_val

        # Construct final output Series
        forecast_series = pd.Series(forecasts, dtype=float)
        forecast_series.index.name = "minute"

        if return_dataframe:
            return pd.DataFrame({
                "Time": forecast_series.index.astype(int) * self.bucket_seconds,
                "Predicted": forecast_series.values
            })

        return forecast_series.values

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_info.get("type", "arima"),
            "version": self.model_info.get("version", "v1"),
            "order": self.order,
            "train_window": self.train_window,
            "future_horizon": self.future_horizon,
            "config_path": self.config_path,
        }