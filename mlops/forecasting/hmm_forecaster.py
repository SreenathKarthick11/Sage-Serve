"""
mlops/forecasting/hmm_forecaster.py

Hidden Markov Model (HMM) implementation of the BaseForecaster interface.
Loads pre-trained transition/emission matrices from the Model Registry and
generates time-series forecasts for the SageServe pipeline.
"""

import os
import pickle
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import yaml

from mlops.forecasting.base import BaseForecaster


class HMMForecaster(BaseForecaster):
    """
    Forecaster using a Hidden Markov Model (HMM) to predict future workload.
    """

    def __init__(self, config_path: str, weights_path: Union[str, None] = None) -> None:
        if not os.path.exists(config_path):     # check path
            raise FileNotFoundError(f"HMM config file not found at: {config_path}")

        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.model_info = self.config.get("model_info", {})
        self.parameters = self.config.get("parameters", {})

        self.future_horizon = self.parameters.get("future_horizon", 61)
        self.bucket_seconds = self.parameters.get("bucket_seconds", 60)
        self.config_path = config_path

        # Resolve weights path
        if weights_path is None:    # search for the transition/emission matrix weights
            weights_filename = self.parameters.get("weights_file", "transition_matrix.pkl")
            model_dir = os.path.dirname(config_path)
            weights_path = os.path.join(model_dir, weights_filename)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"HMM weights file not found at: {weights_path}")

        # Load pre-trained HMM artifact
        with open(weights_path, "rb") as f:
            self.hmm_artifact = pickle.load(f)      # it should be stored in pickle file

        self.weights_path = weights_path

    def generate_zero_forecast(self, last_observed_minute: int) -> pd.DataFrame:
        """Generates a zero-filled forecast DataFrame when no workload is found."""

        future_minutes = np.arange(1, last_observed_minute + self.future_horizon + 1)
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
        Generates a forecast based on input data and the pre-trained HMM artifact.

        Args:
            input_data: Time series workload data.
            return_dataframe (bool): If True, returns DataFrame with ['Time', 'Predicted']
                                     columns for saving directly to CSV.
            last_observed_minute (int, optional): Explicit last observed minute for zero/trace forecasts.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Forecasted values.
        """
        # convert all to pd series
        if isinstance(input_data, pd.Series):
            series = input_data.astype(float).copy()
        else:
            series = pd.Series(np.asarray(input_data, dtype=float))

        # Handle empty series / zero forecast case
        if series.empty:
            obs_min = last_observed_minute if last_observed_minute is not None else 60
            zero_df = self.generate_zero_forecast(obs_min)
            return zero_df if return_dataframe else zero_df["Predicted"].values

        data = series.values
        horizon = self.future_horizon

        # Infer future state projection using HMM transition matrix
        if isinstance(self.hmm_artifact, dict) and "transition_matrix" in self.hmm_artifact:
            trans_matrix = self.hmm_artifact["transition_matrix"]
            means = self.hmm_artifact.get("means", [data[-1]])

            current_state = int(np.argmin([abs(data[-1] - m) for m in means]))
            predictions = []

            for _ in range(horizon):
                current_state = int(np.argmax(trans_matrix[current_state]))
                predictions.append(means[current_state])

            forecast_array = np.maximum(0.0, np.array(predictions, dtype=np.float32))

        elif hasattr(self.hmm_artifact, "predict"):
            obs = data.reshape(-1, 1)
            hidden_states = self.hmm_artifact.predict(obs)
            last_state = hidden_states[-1]

            trans_matrix = self.hmm_artifact.transmat_
            means = self.hmm_artifact.means_.flatten()

            predictions = []
            curr_s = last_state
            for _ in range(horizon):
                curr_s = int(np.argmax(trans_matrix[curr_s]))
                predictions.append(means[curr_s])

            forecast_array = np.maximum(0.0, np.array(predictions, dtype=np.float32))

        else:
            # Fallback naive projection if custom artifact structure is passed
            last_val = max(0.0, float(data[-1]))
            forecast_array = np.full(shape=(horizon,), fill_value=last_val, dtype=np.float32)

        if return_dataframe:
            obs_min = last_observed_minute if last_observed_minute is not None else int(series.index.max())
            future_minutes = np.arange(obs_min + 1, obs_min + horizon + 1)
            return pd.DataFrame({
                "Time": future_minutes * self.bucket_seconds,
                "Predicted": forecast_array
            })

        return forecast_array

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_info.get("type", "hmm"),
            "version": self.model_info.get("version", "v1"),
            "description": self.model_info.get("description", "Hidden Markov Model Forecaster"),
            "weights_path": self.weights_path,
            "parameters": self.parameters,
        }