"""
mlops/forecasting/base.py

This is where we make the abstraction that all forecasting models in the SageServe
MLOps pipeline must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np


class BaseForecaster(ABC):
    """
    Abstract Base Class for all forecasting models (ARIMA, HMM, NN etc.).

    Makes unified interface for loading configurations/weights and
    executing predictions for downstream consumption by the SageServe LP solver.
    """

    @abstractmethod
    def __init__(self, config_path: str, weights_path: Union[str, None] = None) -> None:
        """
        Initializes the forecaster.

        Args:
            config_path (str): Path to the model's config.yaml file in the model_registry.
            weights_path (str, optional): Path to model weights or pre-trained artifacts
                                         (e.g., .pt or .pkl), if applicable (for NN or HMM models).
        """
        pass

    @abstractmethod
    def predict(self, input_data: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Generates a forecast for the next time window based on the input data.

        Args:
            input_data (Union[List[float], np.ndarray]): Time series data for the lookback window
                                                         (e.g., last 60 minutes of QPS/TPS).

        Returns:
            np.ndarray: Predicted values for the next time window (e.g., 60 predicted data points).
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata about the loaded forecaster model.

        Returns:
            Dict[str, Any]: Dictionary containing model information (type, version, parameters, etc.).
        """
        pass

    @abstractmethod
    def run_forecast(self, clean_dir, source: str):
        """Read the clean trace in clean_dir, forecast, write forecast CSVs,
        and return the published forecast directory."""
        pass

    @classmethod
    def validate_config(cls, config: dict, name: str) -> None:
        """Check a user-supplied config against this model's requirements."""
        info = config.get("model_info")
        params = config.get("parameters")
        if not isinstance(info, dict) or not isinstance(params, dict):
            raise ValueError("config needs 'model_info' and 'parameters' mappings.")
        if str(info.get("type", "")).lower() != name:
            raise ValueError(
                f"config model_info.type must be '{name}', got {info.get('type')!r}."
            )
        missing = [k for k in cls.REQUIRED_PARAMS if k not in params]
        if missing:
            raise ValueError(f"config parameters missing: {missing}")

    @classmethod
    def train(cls, config: dict, clean_dir):
        """Return the weights artifact to pickle. Only needed if WEIGHTS_FILE is set."""
        raise NotImplementedError(f"{cls.__name__} cannot be trained from a trace.")