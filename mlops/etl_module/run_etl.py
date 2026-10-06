from pathlib import Path

from .etl import run_etl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACE_PATH = (PROJECT_ROOT/ "data"/ "traces"/ "random_trace.csv")

OUTPUT_PATH = (PROJECT_ROOT/ "data"/ "traces"/ "validated"/ "forecasting_input_dataset.csv")


def main() -> None:
    run_etl(trace_path=TRACE_PATH,output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()