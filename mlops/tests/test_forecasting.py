"""
    Tests for the forecasting module

    To run test :
        python -m unittest tests.test_forecasting -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

from forecasting.base import BaseForecaster
from forecasting.factory import ForecasterFactory
from forecasting.arima_forecaster import ARIMAForecaster
from forecasting.hmm_forecaster import HMMForecaster


class DummyForecaster(BaseForecaster):
    """Dummy subclass to test BaseForecaster validation and abstract contracts."""
    REQUIRED_PARAMS = ["train_window_mins", "future_horizon"]
    WEIGHTS_FILE = None

    def __init__(self, config_path: str, weights_path: str | None = None) -> None:
        pass

    def predict(self, input_data):
        return np.zeros(10)

    def get_metadata(self) -> dict:
        return {"model_type": "dummy"}

    def run_forecast(self, clean_dir, source: str):
        pass


class BaseForecasterValidationTests(unittest.TestCase):
    def test_validate_config_success(self):
        config = {
            "model_info": {"type": "dummy"},
            "parameters": {"train_window_mins": 60, "future_horizon": 60}
        }
        # Should not raise
        DummyForecaster.validate_config(config, "dummy")

    def test_validate_config_wrong_type_fails(self):
        config = {
            "model_info": {"type": "arima"},
            "parameters": {"train_window_mins": 60, "future_horizon": 60}
        }
        with self.assertRaises(ValueError):
            DummyForecaster.validate_config(config, "dummy")

    def test_validate_config_missing_parameters_fails(self):
        config = {
            "model_info": {"type": "dummy"},
            "parameters": {"train_window_mins": 60}  # missing future_horizon
        }
        with self.assertRaises(ValueError):
            DummyForecaster.validate_config(config, "dummy")


class ARIMAForecasterUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp_dir.name) / "arima_config.yaml"

        config_data = {
            "model_info": {"type": "arima", "version": "v1"},
            "parameters": {
                "order": {"p": 1, "d": 1, "q": 1},
                "train_window_mins": 30,
                "future_horizon": 10,
                "bucket_seconds": 60,
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init_loads_parameters_correctly(self):
        forecaster = ARIMAForecaster(str(self.config_path))
        self.assertEqual(forecaster.order, (1, 1, 1))
        self.assertEqual(forecaster.train_window, 30)
        self.assertEqual(forecaster.future_horizon, 10)
        self.assertEqual(forecaster.bucket_seconds, 60)

    def test_predict_empty_or_zero_input_returns_zeros(self):
        forecaster = ARIMAForecaster(str(self.config_path))
        preds = forecaster.predict([0.0, 0.0, 0.0])
        self.assertEqual(len(preds), 10)
        np.testing.assert_array_equal(preds, np.zeros(10))

    def test_predict_returns_correct_horizon_length(self):
        forecaster = ARIMAForecaster(str(self.config_path))
        # Provide short input history that triggers fallback mean prediction
        history = [10.0, 20.0, 15.0]
        preds = forecaster.predict(history)
        self.assertEqual(preds.shape, (10,))
        self.assertTrue(np.all(preds >= 0.0))

    def test_get_metadata_contains_expected_keys(self):
        forecaster = ARIMAForecaster(str(self.config_path))
        metadata = forecaster.get_metadata()
        self.assertEqual(metadata["model_type"], "ARIMA")
        self.assertEqual(metadata["future_horizon"], 10)
        self.assertEqual(metadata["train_window"], 30)


class ForecasterFactoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self.tmp_dir.name)

        # Setup mock model registry structure
        model_dir = self.registry_dir / "models" / "arima" / "v1"
        model_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = model_dir / "config.yaml"
        config_data = {
            "model_info": {"type": "arima"},
            "parameters": {
                "order": {"p": 1, "d": 1, "q": 1},
                "train_window_mins": 30,
                "future_horizon": 5,
                "bucket_seconds": 60,
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config_data, f)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_active_forecaster_with_overrides(self):
        forecaster = ForecasterFactory.load_active_forecaster(
            registry_dir=str(self.registry_dir),
            override_type="arima",
            override_version="v1"
        )
        self.assertIsInstance(forecaster, ARIMAForecaster)
        self.assertEqual(forecaster.future_horizon, 5)

    def test_load_active_forecaster_raises_not_found_when_missing(self):
        with self.assertRaises(FileNotFoundError):
            ForecasterFactory.load_active_forecaster(
                registry_dir=str(self.registry_dir / "nonexistent")
            )

    def test_load_active_forecaster_raises_value_error_on_unknown_type(self):
        # Create pointer directory for unknown model type
        bad_dir = self.registry_dir / "models" / "unknown_model" / "v1"
        bad_dir.mkdir(parents=True, exist_ok=True)
        with open(bad_dir / "config.yaml", "w") as f:
            yaml.dump({"model_info": {"type": "unknown_model"}, "parameters": {}}, f)

        with self.assertRaises(ValueError):
            ForecasterFactory.load_active_forecaster(
                registry_dir=str(self.registry_dir),
                override_type="unknown_model",
                override_version="v1"
            )


if __name__ == "__main__":
    unittest.main()