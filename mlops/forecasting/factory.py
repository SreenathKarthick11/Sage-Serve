"""
mlops/forecasting/factory.py

Factory class responsible for dynamically reading the active model pointer 
from the Model Registry and instantiating the corresponding forecaster.
"""

import json
import os
from typing import Optional

from mlops.forecasting.arima_forecaster import ARIMAForecaster
from mlops.forecasting.base import BaseForecaster
from mlops.forecasting.hmm_forecaster import HMMForecaster
# from mlops.forecasting.nn_forecaster import NNForecaster

REGISTERED_MODELS = ["nn", "hmm", "arima"]      # should be read from a config or found out, no hard code

class ForecasterFactory:
    """
    Factory for loading and instantiating forecasting models based on
    Model Registry configurations.
    """

    @staticmethod
    def load_active_forecaster(
        registry_dir: str = "mlops/model_registry",
        override_type: Optional[str] = None,
        override_version: Optional[str] = None,
    ) -> BaseForecaster:
        """
        Reads the active model pointer and returns the corresponding forecaster instance.

        Args:
            registry_dir (str): Base path to the model registry directory.
            override_type (str, optional): Explicitly request a model type (e.g., 'hmm').
            override_version (str, optional): Explicitly request a version (e.g., 'v2').

        Returns:
            BaseForecaster: An instantiated subclass implementing BaseForecaster.
        """
        pointer_path = os.path.join(registry_dir, "active_model.json")

        if not os.path.exists(pointer_path) and (not override_type or not override_version):
            raise FileNotFoundError(
                f"Model registry pointer not found at: {pointer_path}. \
                Ensure active_model.json exists or provide override arguments."
            )

        # Determine model_type and version
        if override_type and override_version:
            model_type = override_type.lower()
            version = override_version.lower()
        else:
            with open(pointer_path, "r") as f:
                active_info = json.load(f)

            # fallback to arima v1 if this ever fails
            model_type = active_info.get("active_model_type", "arima").lower()
            version = active_info.get("active_version", "v1").lower()

        # Build path to model artifact directory
        model_dir = os.path.join(registry_dir, "models", model_type, version)
        config_path = os.path.join(model_dir, "config.yaml")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        # Create instance and return based on model_type
        if model_type == "arima":
            return ARIMAForecaster(config_path=config_path)

        elif model_type == "hmm":
            weights_path = os.path.join(model_dir, "transition_matrix.pkl")
            return HMMForecaster(config_path=config_path, weights_path=weights_path)

        # elif model_type == "nn":
        #     weights_path = os.path.join(model_dir, "model.pt")
        #     return NNForecaster(config_path=config_path, weights_path=weights_path)

        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Expected one of {REGISTERED_MODELS}."
            )