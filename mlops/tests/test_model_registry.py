"""
    Tests for the model registry module (model_registry.scripts).

    To run test :
        python -m unittest tests.test_model_registry -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_registry import scripts
from model_registry.scripts import GovernanceError


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.registry_dir = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_register_model_creates_version_and_manifest(self):
        # Create a temporary file to act as a model artifact/weight
        weight_file = Path(self.registry_dir) / "dummy_weights.pkl"
        weight_file.write_bytes(b"fake model weights content")

        manifest = scripts.register_model(
            name="arima",
            model_path=str(weight_file),
            metrics={"mae": 10.0},
            registry_dir=self.registry_dir
        )

        self.assertEqual(manifest["version_id"], "v1")
        self.assertEqual(manifest["name"], "arima")
        self.assertEqual(manifest["metrics"]["mae"], 10.0)
        self.assertEqual(manifest["stage"], "None")

        # Check directory layout and file copying
        version_dir = Path(self.registry_dir) / "models" / "arima" / "v1"
        self.assertTrue((version_dir / "manifest.json").exists())
        self.assertTrue((version_dir / "dummy_weights.pkl").exists())

    def test_generate_model_card_validates_fields_and_todos(self):
        weight_file = Path(self.registry_dir) / "dummy.pkl"
        weight_file.write_bytes(b"data")
        scripts.register_model("arima", str(weight_file), {"mae": 10.0}, self.registry_dir)

        card_fields = {
            "model_description": "ARIMA model for workload forecasting",
            "intended_use": "Per-minute prompt forecasting",
            "limitations": "Performs poorly on bursty spikes"
        }

        # Valid model card generation
        card = scripts.generate_model_card("arima", "v1", card_fields, self.registry_dir)
        self.assertEqual(card["version_id"], "v1")
        self.assertTrue((Path(self.registry_dir) / "models" / "arima" / "v1" / "model_card.json").exists())

        # Invalid model card containing "TODO" placeholder should raise ValueError
        bad_fields = card_fields.copy()
        bad_fields["limitations"] = "TODO: fill this out"
        with self.assertRaises(ValueError):
            scripts.generate_model_card("arima", "v1", bad_fields, self.registry_dir)

    def test_promote_model_governance_gates(self):
        weight_file = Path(self.registry_dir) / "dummy.pkl"
        weight_file.write_bytes(b"data")

        # Register a version with failing metrics (MAE > 15.0)
        scripts.register_model("arima", str(weight_file), {"mae": 18.5}, self.registry_dir) # v1
        # Register a version with passing metrics (MAE <= 15.0)
        scripts.register_model("arima", str(weight_file), {"mae": 11.2}, self.registry_dir) # v2

        # 1. Promoting v2 to Production without a model card should fail governance gate
        with self.assertRaises(GovernanceError):
            scripts.promote_model("arima", "v2", "Production", self.registry_dir)

        # Create valid model card for v2
        scripts.generate_model_card("arima", "v2", {
            "model_description": "Good model",
            "intended_use": "Forecasting",
            "limitations": "None"
        }, self.registry_dir)

        # 2. Promoting v2 (passing metrics + card) should succeed
        manifest = scripts.promote_model("arima", "v2", "Production", self.registry_dir)
        self.assertEqual(manifest["stage"], "Production")

        # Verify active_model.json pointer was updated
        active_path = Path(self.registry_dir) / "active_model.json"
        self.assertTrue(active_path.exists())
        active_info = json.loads(active_path.read_text())
        self.assertEqual(active_info["active_version"], "v2")

        # 3. Promoting v1 (failing metrics: MAE 18.5) with a card should still fail governance
        scripts.generate_model_card("arima", "v1", {
            "model_description": "Bad model",
            "intended_use": "Forecasting",
            "limitations": "None"
        }, self.registry_dir)

        with self.assertRaises(GovernanceError):
            scripts.promote_model("arima", "v1", "Production", self.registry_dir)

    def test_get_production_model(self):
        weight_file = Path(self.registry_dir) / "dummy.pkl"
        weight_file.write_bytes(b"data")
        scripts.register_model("arima", str(weight_file), {"mae": 9.0}, self.registry_dir)

        # Should return None if no model is in Production
        self.assertIsNone(scripts.get_production_model("arima", self.registry_dir))

        # Add card and promote to production
        scripts.generate_model_card("arima", "v1", {
            "model_description": "Desc", "intended_use": "Use", "limitations": "None"
        }, self.registry_dir)
        scripts.promote_model("arima", "v1", "Production", self.registry_dir)

        prod = scripts.get_production_model("arima", self.registry_dir)
        self.assertIsNotNone(prod)
        self.assertEqual(prod["version_id"], "v1")
        self.assertEqual(prod["stage"], "Production")


if __name__ == "__main__":
    unittest.main()