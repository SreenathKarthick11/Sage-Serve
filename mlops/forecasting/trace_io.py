from pathlib import Path
import re
from typing import Any
import pandas as pd

REQUIRED_COLS = {"arrival_timestamp", "model_type", "regions", "prompt_size"}


def find_clean_trace(clean_dir: Path) -> Path:
    """Locate the clean trace CSV by its columns, not its filename."""
    clean_dir = Path(clean_dir)
    for p in sorted(clean_dir.glob("*.csv")):
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except Exception:
            continue
        if REQUIRED_COLS <= cols:
            return p
    raise FileNotFoundError(f"No clean trace CSV found in {clean_dir.resolve()}")


def _source_region(region_code: Any) -> int:
    """Extract the first-priority region from a three-digit region encoding."""
    if pd.isna(region_code):
        raise ValueError("Encountered a missing value in the 'regions' column.")
    value = str(region_code).strip().zfill(3)
    if not value.isdigit() or len(value) != 3:
        raise ValueError(f"Invalid region code: {region_code!r}")
    return int(value[0])


def load_clean_trace(clean_dir: Path) -> pd.DataFrame:
    trace_path = find_clean_trace(clean_dir)

    df = pd.read_csv(trace_path, dtype={"regions": "string"})
    missing = [c for c in sorted(REQUIRED_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"Clean trace is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"Clean trace contains no rows: {trace_path}")

    df = df.copy()
    df["arrival_timestamp"] = pd.to_numeric(df["arrival_timestamp"], errors="raise")
    df["prompt_size"] = pd.to_numeric(df["prompt_size"], errors="raise")
    if df[["arrival_timestamp", "prompt_size"]].isna().any().any():
        raise ValueError("Clean trace contains null timestamps or prompt sizes.")
    if (df["arrival_timestamp"] < 0).any():
        raise ValueError("arrival_timestamp values must be non-negative.")
    if (df["prompt_size"] < 0).any():
        raise ValueError("prompt_size values must be non-negative.")
    if df["model_type"].isna().any():
        raise ValueError("model_type contains missing values.")

    df["source_region"] = df["regions"].map(_source_region)
    return df


def safe_filename_part(value: Any) -> str:
    """Make model names safe for filenames while retaining readable names."""
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "unnamed"