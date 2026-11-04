"""CLI for the model registry.

Run from the project root (so the default registry path resolves):

    python -m mlops.model_registry.run register --name arima --path ./my_model_dir --metrics mae=9.2
    python -m mlops.model_registry.run card --name arima --version v3 \
        --description "ARIMA(1,1,1) workload forecaster" \
        --intended-use "Per-minute prompt-size forecasting" \
        --limitations "Poor on bursty traffic"
    python -m mlops.model_registry.run promote --name arima --version v3 --stage Production
    python -m mlops.model_registry.run active
    python -m mlops.model_registry.run list
    python -m mlops.model_registry.run production --name arima
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import scripts

DEFAULT_REGISTRY = str(Path(__file__).resolve().parent)   # works from any cwd


def _parse_metrics(pairs: list[str]) -> dict[str, float]:
    metrics = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"metric must be key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        metrics[k.strip()] = float(v)
    return metrics


def _print(obj) -> None:
    print(json.dumps(obj, indent=2))


def cmd_register(a):
    _print(scripts.register_model(a.name, a.path, _parse_metrics(a.metrics), a.registry))


def cmd_card(a):
    fields = {
        "model_description": a.description,
        "intended_use": a.intended_use,
        "limitations": a.limitations,
    }
    _print(scripts.generate_model_card(a.name, a.version, fields, a.registry))


def cmd_promote(a):
    _print(scripts.promote_model(a.name, a.version, a.stage, a.registry))


def cmd_production(a):
    result = scripts.get_production_model(a.name, a.registry)
    if result is None:
        print(f"No Production version for '{a.name}'.")
        sys.exit(1)
    _print(result)


def cmd_active(a):
    path = os.path.join(a.registry, "active_model.json")
    if not os.path.exists(path):
        print("No active_model.json yet.")
        sys.exit(1)
    with open(path) as f:
        _print(json.load(f))


def cmd_list(a):
    models_dir = Path(a.registry) / "models"
    if not models_dir.is_dir():
        print("No models registered.")
        return
    for name_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for ver_dir in sorted(p for p in name_dir.iterdir() if p.is_dir()):
            mf = ver_dir / "manifest.json"
            stage = "?"
            if mf.exists():
                stage = json.loads(mf.read_text()).get("stage", "None")
            card = "card" if (ver_dir / "model_card.json").exists() else "no-card"
            print(f"{name_dir.name}/{ver_dir.name:<5} stage={stage:<11} {card}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Model registry CLI")
    p.add_argument("--registry", default=DEFAULT_REGISTRY, help="registry directory")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("register", help="register a new model version")
    s.add_argument("--name", required=True, help="model type, e.g. arima, hmm")
    s.add_argument("--path", required=True, help="file or dir with config/weights")
    s.add_argument("--metrics", nargs="*", default=[], help="key=value pairs, e.g. mae=9.2 f1=0.85")
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("card", help="generate a model card")
    s.add_argument("--name", required=True)
    s.add_argument("--version", required=True)
    s.add_argument("--description", required=True)
    s.add_argument("--intended-use", required=True)
    s.add_argument("--limitations", required=True)
    s.set_defaults(func=cmd_card)

    s = sub.add_parser("promote", help="promote to Staging/Production/Archived")
    s.add_argument("--name", required=True)
    s.add_argument("--version", required=True)
    s.add_argument("--stage", required=True, choices=["Staging", "Production", "Archived"])
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("production", help="show the Production version of a model")
    s.add_argument("--name", required=True)
    s.set_defaults(func=cmd_production)

    sub.add_parser("active", help="show active_model.json").set_defaults(func=cmd_active)
    sub.add_parser("list", help="list all models, versions, stages").set_defaults(func=cmd_list)

    return p


def main() -> None:
    a = build_parser().parse_args()
    try:
        a.func(a)
    except (scripts.GovernanceError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()