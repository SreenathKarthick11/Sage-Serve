import argparse
from pathlib import Path

from .etl import run_etl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "simulated_trace.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "traces" / "validated" / "validated_workload_trace.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETL processing on workload traces.")

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=TRACE_PATH,
        help="Path to input trace CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to save output validated trace CSV file (default: %(default)s)",
    )

    args = parser.parse_args()

    run_etl(trace_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()