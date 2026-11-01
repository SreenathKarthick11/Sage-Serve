"""Single source of truth for where data lives.

Layout (data sits NEXT TO the repo, as in the original scripts):

    <parent>/mlops/            <- this repository
    <parent>/data/
        streaming/{raw,clean,forecasts}/<version>/...
        static/{raw,clean,forecasts}/<version>/...
        lineage/lineage.jsonl

Set MLOPS_DATA_ROOT to relocate the data directory.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]                       # .../mlops
DATA_ROOT = Path(os.environ.get("MLOPS_DATA_ROOT", PROJECT_ROOT.parent / "data"))

SOURCES = ("streaming", "static")        # streaming = trace_sim server, static = random generator
STAGES = ("raw", "clean", "forecasts")
