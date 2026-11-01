"""Main pipeline:  ingest -> etl -> forecast,  for one source family.

    python -m pipeline.run_pipeline --source static                       # all three stages
    python -m pipeline.run_pipeline --source streaming --duration 600     # simulate 10 min, then ETL + forecast
    python -m pipeline.run_pipeline --source streaming --stages etl,forecast
    python -m pipeline.run_pipeline --source static --stages forecast --input-version <clean id>

--input-version applies to the FIRST stage that is run and is ignored for `ingest`
(default: 'latest').  Stages whose version already exists are skipped (ids are
content/parameter derived), and every new artifact is appended to data/lineage/lineage.jsonl.
"""
from __future__ import annotations

import argparse

from . import lineage
from .paths import SOURCES
from .versioning import artifact_id, resolve_version

ALL_STAGES = ["ingest", "etl", "forecast"]


def run(source: str, stages: list[str], input_version: str = "latest", ingest_args: dict | None = None):
    ingest_args = ingest_args or {}
    raw_dir = clean_dir = forecast_dir = None

    if "ingest" in stages:
        print(f"\n=== [1/3] ingest ({source}) ===")
        if source == "streaming":
            from trace_sim.__main__ import ingest as ingest_streaming
            raw_dir = ingest_streaming(**{k: v for k, v in ingest_args.items()
                                          if k in ("duration", "speedup", "rate_scale", "seed")})
        else:
            from trace_sim.generate_random_trace import ingest as ingest_static
            raw_dir = ingest_static(**{k: v for k, v in ingest_args.items()
                                       if k in ("num_requests", "seed")})

    if "etl" in stages:
        print(f"\n=== [2/3] etl ({source}) ===")
        from etl_module.etl import run_etl
        raw_dir = raw_dir or resolve_version(source, "raw", input_version)
        clean_dir = run_etl(raw_dir, source)

    if "forecast" in stages:
        print(f"\n=== [3/3] forecast ({source}) ===")
        from forecasting.arima_forecaster import run_forecast
        clean_dir = clean_dir or resolve_version(source, "clean", input_version)
        forecast_dir = run_forecast(clean_dir, source)

    last = forecast_dir or clean_dir or raw_dir
    if last is not None:
        stage = last.parent.name
        print("\n=== lineage ===")
        lineage.show(artifact_id(source, stage, last.name))
    return raw_dir, clean_dir, forecast_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help="comma-separated subset of: ingest,etl,forecast (default: all)")
    p.add_argument("--input-version", default="latest",
                   help="version for the first non-ingest stage (default: latest)")
    # streaming ingest
    p.add_argument("--duration", type=float, default=None, help="[streaming] wall-clock seconds (default: until Ctrl+C)")
    p.add_argument("--speedup", type=float, default=60.0, help="[streaming] simulated seconds per wall-clock second")
    p.add_argument("--rate-scale", type=float, default=0.002, help="[streaming] fraction of paper-scale traffic")
    # static ingest
    p.add_argument("--num-requests", type=int, default=5000, help="[static] number of requests")
    p.add_argument("--seed", type=int, default=42, help="seed for either generator")
    a = p.parse_args()

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        p.error(f"unknown stage(s): {bad}; choose from {ALL_STAGES}")

    run(a.source, stages, a.input_version, dict(
        duration=a.duration, speedup=a.speedup, rate_scale=a.rate_scale,
        num_requests=a.num_requests, seed=a.seed))


if __name__ == "__main__":
    main()
