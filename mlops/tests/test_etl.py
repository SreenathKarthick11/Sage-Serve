"""
    Tests for the ETL module
    (etl_module.etl, etl_module.validation, etl_module.run_etl).

    TO test run :
    python -m unittest tests.test_etl -v
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from pipeline.paths import SOURCES
from etl_module import etl, run_etl as cli, validation

RAW_COLUMNS = validation.RAW_COLUMNS
VALID_MODELS = validation.VALID_MODELS
VALID_REGION_ENCODINGS = validation.VALID_REGION_ENCODINGS
validate_raw_trace = validation.validate_raw_trace


def make_trace(n: int = 4) -> pd.DataFrame:
    """Generate a valid sample raw trace DataFrame for testing."""
    return pd.DataFrame({
        "request_id": [float(i) for i in range(n)],
        "batch_id": [float(i) for i in range(n)],
        "client_tenant": [0.0] * n,
        "request_type": [0.0] * n,
        "scenario": ["default"] * n,
        "sla": [1.0] * n,
        "utility": [1.0] * n,
        "regions": pd.array([VALID_REGION_ENCODINGS[i % len(VALID_REGION_ENCODINGS)] for i in range(n)], dtype="string"),
        "model_type": pd.array([VALID_MODELS[i % len(VALID_MODELS)] for i in range(n)], dtype="string"),
        "workload_type": ["interactive"] * n,
        "application_id": [0.0] * n,
        "arrival_timestamp": [float(i * 10) for i in range(n)],
        "batch_size": [1.0] * n,
        "prompt_size": [100.0 + i for i in range(n)],
        "token_size": [50.0] * n,
    })[RAW_COLUMNS]


def quiet():
    """Swallow stdout print statements during validation runs."""
    return contextlib.redirect_stdout(io.StringIO())


class _TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


# Load Trace Tests

class LoadTraceTests(_TempDirTestCase):
    def test_missing_file_raises_file_not_found(self):
        missing = self.root / "nonexistent.csv"
        with self.assertRaises(FileNotFoundError):
            etl.load_trace(missing)

    def test_loads_correct_columns_and_types(self):
        path = self.root / "trace.csv"
        make_trace(3).to_csv(path, index=False)
        df = etl.load_trace(path)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), RAW_COLUMNS)
        self.assertEqual(str(df["regions"].dtype), "string")
        self.assertEqual(str(df["model_type"].dtype), "string")


# Clean Trace Tests

class CleanTraceTests(unittest.TestCase):
    def test_strips_whitespace_and_zero_pads_regions(self):
        df = make_trace(2)
        df["model_type"] = pd.array([" A ", "B"], dtype="string")
        df["regions"] = pd.array(["12", "120"], dtype="string")

        cleaned = etl.clean_trace(df)
        self.assertEqual(cleaned.loc[0, "model_type"], "A")
        self.assertEqual(cleaned.loc[0, "regions"], "012")  # padded from "12"
        self.assertEqual(cleaned.loc[1, "regions"], "120")

    def test_coerces_invalid_numbers_to_nan(self):
        df = make_trace(2)
        df["prompt_size"] = pd.Series(["100", "invalid"], dtype="object")

        cleaned = etl.clean_trace(df)
        self.assertEqual(cleaned.loc[0, "prompt_size"], 100.0)
        self.assertTrue(pd.isna(cleaned.loc[1, "prompt_size"]))

    def test_does_not_mutate_original_dataframe(self):
        df = make_trace(2)
        before = df.copy(deep=True)
        etl.clean_trace(df)
        pd.testing.assert_frame_equal(df, before)


# Validation Tests
class ValidationTests(unittest.TestCase):
    def test_valid_trace_passes(self):
        df = make_trace(4)
        with quiet():
            result = validate_raw_trace(df)
        self.assertTrue(result.success)

    def test_missing_required_column_fails(self):
        df = make_trace(4).drop(columns=["prompt_size"])
        with quiet(), self.assertRaises(ValueError):
            validate_raw_trace(df)

    def test_invalid_model_type_fails(self):
        df = make_trace(4)
        df.loc[0, "model_type"] = "INVALID_MODEL"
        with quiet(), self.assertRaises(ValueError):
            validate_raw_trace(df)

    def test_negative_timestamp_fails(self):
        df = make_trace(4)
        df.loc[0, "arrival_timestamp"] = -10.0
        with quiet(), self.assertRaises(ValueError):
            validate_raw_trace(df)


# (run_etl) Tests

class RunEtlTests(_TempDirTestCase):
    SOURCE = "static"
    RAW_VERSION = "raw123"
    CLEAN_VERSION = "clean456"

    def setUp(self):
        super().setUp()
        self.raw_dir = self.root / "raw" / self.RAW_VERSION
        self.raw_dir.mkdir(parents=True)
        self.out_root = self.root / "out"

        # Create the scratch directory on disk so pandas can write to it
        scratch_dir = self.root / "tmp_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # Mock pipeline versioning functions to keep unit tests local and fast
        self.artifact_id = MagicMock(return_value="static/raw/raw123")
        self.derive_id = MagicMock(return_value=self.CLEAN_VERSION)
        self.version_exists = MagicMock(return_value=False)
        self.set_latest = MagicMock()
        self.version_dir = MagicMock(side_effect=lambda s, st, v: self.out_root / s / st / v)
        self.new_tmp_dir = MagicMock(return_value=self.root / "tmp_scratch")
        self.finalize = MagicMock(side_effect=lambda tmp, s, st, v, parents, params, details: self.out_root / s / st / v)

        patcher = patch.multiple(
            etl,
            artifact_id=self.artifact_id,
            derive_id=self.derive_id,
            version_exists=self.version_exists,
            set_latest=self.set_latest,
            version_dir=self.version_dir,
            new_tmp_dir=self.new_tmp_dir,
            finalize=self.finalize,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_run_etl_happy_path(self):
        trace_path = self.raw_dir / "trace.csv"
        make_trace(4).to_csv(trace_path, index=False)

        with quiet():
            out_dir = etl.run_etl(self.raw_dir, self.SOURCE)

        self.assertEqual(out_dir, self.out_root / self.SOURCE / "clean" / self.CLEAN_VERSION)
        self.finalize.assert_called_once()

    def test_run_etl_validation_failure_raises(self):
        trace_path = self.raw_dir / "trace.csv"
        bad_df = make_trace(4)
        bad_df.loc[0, "model_type"] = "INVALID"
        bad_df.to_csv(trace_path, index=False)

        with quiet(), self.assertRaises(ValueError):
            etl.run_etl(self.raw_dir, self.SOURCE)
        self.finalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()