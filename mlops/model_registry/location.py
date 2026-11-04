"""Single source of truth for where registry DATA lives (models/ + active_model.json)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = os.environ.get("MODEL_REGISTRY_DIR", str(PROJECT_ROOT / "data" / "model_registry"))