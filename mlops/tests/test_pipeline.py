"""
    Tests for the pipeline module (pipeline.versioning, pipeline.lineage).

    To run test :
        python -m unittest tests.test_pipeline -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import versioning, lineage


class PipelineVersioningTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp_dir.name)

        # Isolate versioning and lineage paths to the temporary directory
        self.data_root_patcher = patch("pipeline.versioning.DATA_ROOT", self.data_root)
        self.lineage_path_patcher = patch(
            "pipeline.versioning.LINEAGE_PATH", self.data_root / "lineage" / "lineage.jsonl"
        )
        self.lineage_module_patcher = patch(
            "pipeline.lineage.LINEAGE_PATH", self.data_root / "lineage" / "lineage.jsonl"
        )

        self.data_root_patcher.start()
        self.lineage_path_patcher.start()
        self.lineage_module_patcher.start()

    def tearDown(self):
        self.data_root_patcher.stop()
        self.lineage_path_patcher.stop()
        self.lineage_module_patcher.stop()
        self.tmp_dir.cleanup()

    def test_artifact_id_formatting(self):
        aid = versioning.artifact_id("static", "raw", "abc123456789")
        self.assertEqual(aid, "static/raw/abc123456789")

    def test_derive_id_is_deterministic(self):
        id1 = versioning.derive_id(["static/raw/123"], {"param": "val"})
        id2 = versioning.derive_id(["static/raw/123"], {"param": "val"})
        id3 = versioning.derive_id(["static/raw/123"], {"param": "other"})

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertEqual(len(id1), 12)

    def test_sha256_file_hashing(self):
        f_path = self.data_root / "test.txt"
        f_path.write_text("hello pipeline content")
        h = versioning.sha256_file(f_path)
        self.assertEqual(len(h), 64)

    def test_latest_version_pointer_management(self):
        source, stage = "static", "clean"
        with self.assertRaises(FileNotFoundError):
            versioning.latest_version(source, stage)

        versioning.set_latest(source, stage, "v12345")
        self.assertEqual(versioning.latest_version(source, stage), "v12345")

        versioning.set_latest(source, stage, "v67890")
        self.assertEqual(versioning.latest_version(source, stage), "v67890")


class PipelineLineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp_dir.name)
        self.lineage_file = self.data_root / "lineage" / "lineage.jsonl"
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)

        self.patcher = patch("pipeline.lineage.LINEAGE_PATH", self.lineage_file)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp_dir.cleanup()

    def test_load_empty_lineage(self):
        records = lineage.load()
        self.assertEqual(records, {})

    def test_load_and_resolve_ancestors(self):
        records_data = [
            {"id": "static/raw/r1", "parents": [], "params": {}, "created_at": "2026-01-01T00:00:00"},
            {"id": "static/clean/c1", "parents": ["static/raw/r1"], "params": {}, "created_at": "2026-01-01T01:00:00"}
        ]
        with open(self.lineage_file, "w") as f:
            for r in records_data:
                f.write(json.dumps(r) + "\n")

        loaded = lineage.load()
        self.assertIn("static/raw/r1", loaded)
        self.assertIn("static/clean/c1", loaded)

        anc = lineage.ancestors("static/clean/c1", loaded)
        self.assertEqual(anc, ["static/raw/r1"])


if __name__ == "__main__":
    unittest.main()