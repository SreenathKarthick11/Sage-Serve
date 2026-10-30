"""
mlops/scripts/model_registry.py

Implements Model Registry lifecycle operations:
1. Registration of a Model (Versioning & File Artifact Packaging)
2. Model Card Generation & Validation
3. Promoted Stage Governance Gates (Staging -> Production -> Archived)
4. Active Production Model Retrieval
"""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Enforced Governance Constants
REQUIRED_CARD_FIELDS = ["model_description", "intended_use", "limitations"]
PRODUCTION_F1_THRESHOLD = 0.80
PRODUCTION_MAE_THRESHOLD = 15.0  # Alternative regression threshold for time-series forecasters


class GovernanceError(Exception):
    """Raised when a promotion is attempted that violates a governance rule."""
    pass


def _now() -> str:
    """Returns ISO-formatted UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _next_version_id(existing_dir: str) -> str:
    """
    Given a directory of existing v1/, v2/, ... subfolders, return the
    next version id string.
    """
    if not os.path.isdir(existing_dir):
        return "v1"
    nums = []
    for name in os.listdir(existing_dir):
        if name.startswith("v") and name[1:].isdigit():
            nums.append(int(name[1:]))
    return f"v{max(nums, default=0) + 1}"


def _model_dir(registry_dir: str, name: str, version_id: str) -> str:
    """Returns absolute/relative directory path for a specific model version."""
    return os.path.join(registry_dir, "models", name, version_id)


def register_model(
    name: str,
    model_path: str,    # this is where the model's weights actually locally live
    metrics: Dict[str, float],
    registry_dir: str = "mlops/model_registry"  # assuming this will be running from PROJECT ROOT
) -> Dict[str, Any]:
    """
    Register a new version of model 'name' in the registry.

    Copies weights/configs from 'model_path' into model_registry/models/<name>/<next_v>/
    and writes an initial manifest.json.
    """
    base_model_dir = os.path.join(registry_dir, "models", name)
    version_id = _next_version_id(base_model_dir)
    target_dir = _model_dir(registry_dir, name, version_id)

    os.makedirs(target_dir, exist_ok=True)

    # Copy model artifacts (e.g. weights, transition matrix, config) into version directory
    # otherwise it will be in whatever random folder it currently is in
    if os.path.isfile(model_path):
        shutil.copy2(model_path, target_dir)

    elif os.path.isdir(model_path):
        # if it is a directory with multiple things, copy the whole thing into the registry's folder
        for item in os.listdir(model_path):
            s = os.path.join(model_path, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # Build and save manifest.json
    manifest = {
        "name": name,
        "version_id": version_id,
        "stage": "None",
        "metrics": metrics,
        "created_at": _now(),
        "history": [
            {
                "from_stage": "None",
                "to_stage": "Registered",
                "at": _now()
            }
        ]
    }

    manifest_path = os.path.join(target_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def generate_model_card(
    name: str,
    version_id: str,
    card_fields: Dict[str, str],
    registry_dir: str = "mlops/model_registry"
) -> Dict[str, Any]:
    """
    Validate and write a model card for an already-registered model version.

    'card_fields' must contain every key in REQUIRED_CARD_FIELDS mapped to a
    non-empty string without the substring 'TODO'.
    """
    version_dir = _model_dir(registry_dir, name, version_id)
    manifest_path = os.path.join(version_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Model version manifest not found at: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Validate card fields
    for field in REQUIRED_CARD_FIELDS:
        if field not in card_fields:
            raise ValueError(f"Missing required model card field: '{field}'")

        val = card_fields[field]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Model card field '{field}' must be a non-empty string.")

        if "TODO" in val:
            raise ValueError(f"Model card field '{field}' contains uncompleted 'TODO' placeholder.")

    # Construct model card 
    model_card = {
        "name": name,
        "version_id": version_id,
        "card_fields": card_fields,
        "metrics": manifest.get("metrics", {}),
        "created_at": _now()
    }

    card_path = os.path.join(version_dir, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)

    return model_card


def promote_model(
    name: str,
    version_id: str,
    target_stage: str,
    registry_dir: str = "mlops/model_registry"
) -> Dict[str, Any]:
    """
    Move model 'name' version 'version_id' to 'target_stage' ("Staging" or "Production").

    Governance rules enforced for "Production":
      (a) model_card.json must exist.
      (b) metrics["f1"] >= PRODUCTION_F1_THRESHOLD (or metrics["mae"] <= threshold).
    """
    target_stage = target_stage.capitalize()
    if target_stage not in ["Staging", "Production", "Archived"]:
        raise ValueError(f"Invalid target stage: {target_stage}. Must be Staging, Production, or Archived.")

    version_dir = _model_dir(registry_dir, name, version_id)
    manifest_path = os.path.join(version_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest missing for {name} version {version_id}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Governance Check Gate
    if target_stage == "Production":
        card_path = os.path.join(version_dir, "model_card.json")
        if not os.path.exists(card_path):
            raise GovernanceError(
                f"Governance Check Failed: Cannot promote {name}/{version_id} to Production "
                f"because model_card.json does not exist."
            )

        metrics = manifest.get("metrics", {})
        # Evaluates F1 score if provided, or MAE regression metric fallback
        if "f1" in metrics:
            if metrics["f1"] < PRODUCTION_F1_THRESHOLD:
                raise GovernanceError(
                    f"Governance Check Failed: F1 score ({metrics['f1']:.2f}) "
                    f"is below required threshold ({PRODUCTION_F1_THRESHOLD})."
                )
        elif "mae" in metrics:
            if metrics["mae"] > PRODUCTION_MAE_THRESHOLD:
                raise GovernanceError(
                    f"Governance Check Failed: MAE metric ({metrics['mae']:.2f}) "
                    f"exceeds maximum allowed threshold ({PRODUCTION_MAE_THRESHOLD})."
                )
        else:
            raise GovernanceError(
                f"Governance Check Failed: Required evaluation metrics "
                f"are missing from metrics dict."
            )

        # Ensure single active Production version across all versions of model 'name'
        base_models_dir = os.path.join(registry_dir, "models", name)
        if os.path.exists(base_models_dir):
            for v_folder in os.listdir(base_models_dir):
                other_manifest_path = os.path.join(base_models_dir, v_folder, "manifest.json")
                if os.path.exists(other_manifest_path) and v_folder != version_id:
                    with open(other_manifest_path, "r") as f:
                        other_manifest = json.load(f)

                    if other_manifest.get("stage") == "Production":
                        old_stage = other_manifest.get("stage")
                        other_manifest["stage"] = "Archived"
                        other_manifest.setdefault("history", []).append({
                            "from_stage": old_stage,
                            "to_stage": "Archived",
                            "at": _now()
                        })
                        with open(other_manifest_path, "w") as f:
                            json.dump(other_manifest, f, indent=2)

    # Update current version manifest
    old_stage = manifest.get("stage", "None")
    manifest["stage"] = target_stage
    manifest.setdefault("history", []).append({
        "from_stage": old_stage,
        "to_stage": target_stage,
        "at": _now()
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Synchronize top-level active_model.json pointer if promoted to Production
    if target_stage == "Production":
        active_pointer_path = os.path.join(registry_dir, "active_model.json")
        active_payload = {
            "active_model_type": name,
            "active_version": version_id,
            "stage": "Production",
            "updated_at": _now()
        }
        with open(active_pointer_path, "w") as f:
            json.dump(active_payload, f, indent=2)

    return manifest


def get_production_model(name: str, registry_dir: str = "mlops/model_registry") -> Optional[Dict[str, Any]]:
    """
    Scan model directory `name` and return the manifest dictionary of the
    version whose stage == "Production". Return None if no version is in Production.
    """
    base_models_dir = os.path.join(registry_dir, "models", name)
    if not os.path.exists(base_models_dir):
        return None

    for v_folder in os.listdir(base_models_dir):
        manifest_path = os.path.join(base_models_dir, v_folder, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if manifest.get("stage") == "Production":
                return manifest

    return None