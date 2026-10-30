"""Run ETL on one raw trace version.

    python -m etl_module.run_etl --source streaming                    # latest raw version
    python -m etl_module.run_etl --source static --input-version 3fa9c1d2e7b0
"""
import argparse

from pipeline.paths import SOURCES
from pipeline.versioning import resolve_version

from .etl import run_etl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETL processing on a versioned workload trace.")
    parser.add_argument("--source", required=True, choices=SOURCES,
                        help="which trace family to process")
    parser.add_argument("--input-version", default="latest",
                        help="raw version id, or 'latest' (default: %(default)s)")
    args = parser.parse_args()

    raw_dir = resolve_version(args.source, "raw", args.input_version)
    run_etl(input_dir=raw_dir, source=args.source)


if __name__ == "__main__":
    main()
