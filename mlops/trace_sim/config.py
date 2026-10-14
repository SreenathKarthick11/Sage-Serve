"""Calibration tables (read off the SageServe paper, Figs. 3-6) and runtime settings."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

REGIONS = ("west", "central", "east")
REGION_ID = {r: i for i, r in enumerate(REGIONS)}
MODELS = ("A", "B", "C", "D")
TIERS = ("IW-F", "IW-N", "NIW")

# <repo>/mlops/trace_simulation/config.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "traces" / "simulated_trace.csv"

# SLA (seconds) and utility per tier (paper §2.2): IW-F TTFT<1s, IW-N TTFT<1min, NIW 24h deadline.
TIER_SPEC = {
    "IW-F": {"sla": 1, "utility": 3},
    "IW-N": {"sla": 60, "utility": 2},
    "NIW": {"sla": 24 * 3600, "utility": 1},
}

# Peak requests/sec at *paper scale* per tier x region x model (eyeballed from Fig. 4).
# East is the busiest region; Model A dominates East, Model B dominates Central (IW-F) and West (IW-N);
# NIW is small except Model C in Central/East (feature-testing app).
PEAK_RPS = {
    "IW-F": {
        "west":    {"A": 70,  "B": 40,  "C": 10, "D": 5},
        "central": {"A": 70,  "B": 190, "C": 20, "D": 10},
        "east":    {"A": 400, "B": 60,  "C": 20, "D": 10},
    },
    "IW-N": {
        "west":    {"A": 70,  "B": 100, "C": 30, "D": 10},
        "central": {"A": 50,  "B": 20,  "C": 15, "D": 5},
        "east":    {"A": 200, "B": 90,  "C": 40, "D": 10},
    },
    "NIW": {
        "west":    {"A": 2,  "B": 3,  "C": 5,   "D": 1},
        "central": {"A": 10, "B": 15, "C": 40,  "D": 5},
        "east":    {"A": 30, "B": 40, "C": 100, "D": 10},
    },
}


# Calibration multipliers so the tier mix matches the paper (IW-F largest, IW-F + IW-N ~ 72% of requests).
_TIER_CAL = {"IW-F": 1.0, "IW-N": 0.7, "NIW": 0.8}
PEAK_RPS = {t: {r: {m: v * _TIER_CAL[t] for m, v in ms.items()} for r, ms in rs.items()}
            for t, rs in PEAK_RPS.items()}

@dataclass(frozen=True)
class TierShape:
    diurnal: bool          # has a daily hump?
    floor: float           # off-peak level as a fraction of peak
    weekend: float         # multiplier on the hump (or on the flat level if not diurnal) on Sat/Sun
    weekday_ramp: tuple    # Mon..Fri multipliers (IW-N grows towards Friday)
    ou_theta: float        # mean reversion of slow noise, per sim hour (small => slow drift)
    ou_sigma: float        # stationary std-dev of the log-noise


TIER_SHAPE = {
    # strong, predictable diurnal pattern; weekends nearly silent (periodicity score 0.7-0.95)
    "IW-F": TierShape(True, 0.03, 0.08, (1, 1, 1, 1, 1), 1.0, 0.05),
    # similar shape but higher off-peak/weekend floor, midweek growth, noisier (ps 0.3-0.8)
    "IW-N": TierShape(True, 0.35, 0.10, (0.80, 0.85, 1.0, 1.05, 1.10), 1.0, 0.12),
    # flat, low, non-periodic (ps 0-0.29): slow random drift only
    "NIW": TierShape(False, 1.0, 0.75, (1, 1, 1, 1, 1), 0.05, 0.30),
}
# Extra noise per model for IW tiers: Model A is the most periodic, B/C/D progressively less.
MODEL_NOISE = {"A": 1.0, "B": 3.0, "C": 3.0, "D": 3.5}

# Application mix (Fig. 6a). Names are the paper's anonymised generic labels.
APPLICATIONS = ["RAG", "ContentGeneration", "Workflows", "Chat", "Chat", "Evaluation",
                "ContentGeneration", "Workflows", "Workflows", "ContentGeneration", "Others"]
IW_APP_WEIGHTS = [41.2, 9.3, 8.6, 6.7, 5.3, 3.8, 3.2, 2.4, 1.8, 1.7, 16.2]
NIW_APP_WEIGHTS = [0, 15, 20, 0, 0, 30, 5, 10, 5, 0, 15]   # batch/evaluation flavoured

# Token distributions (Fig. 10): prompts mostly > 1k, outputs mostly < 1k. (log-mean, log-sigma)
PROMPT_LOGN = {"A": (math.log(2200), 0.9), "B": (math.log(1600), 0.9),
               "C": (math.log(1300), 1.0), "D": (math.log(1000), 0.9)}
OUTPUT_LOGN = {"A": (math.log(320), 0.8), "B": (math.log(250), 0.8),
               "C": (math.log(200), 0.9), "D": (math.log(150), 0.8)}
RAG_PROMPT_BOOST = 2.0                                   # RAG requests carry retrieved context
NIW_PROMPT_BOOST = {("central", "C"): 4.0, ("east", "C"): 2.0}   # feature-evaluation app
MAX_PROMPT, MAX_OUTPUT = 32_000, 4_000

# Endpoint latency model (simulated seconds).
PREFILL_TPS = {"A": 8000, "B": 12000, "C": 20000, "D": 30000}   # prompt tokens/s
DECODE_TPS = {"A": 60, "B": 80, "C": 110, "D": 140}             # output tokens/s per request
NETWORK_S = 0.05


def mean_tokens(model: str) -> float:
    pm, ps = PROMPT_LOGN[model]
    om, os_ = OUTPUT_LOGN[model]
    return math.exp(pm + ps ** 2 / 2) * 1.4 + math.exp(om + os_ ** 2 / 2)


@dataclass
class SimConfig:
    output_path: Path = DEFAULT_OUTPUT
    speedup: float = 60.0        # simulated seconds per wall-clock second
    rate_scale: float = 0.002     # fraction of paper-scale RPS (keeps HTTP load manageable)
    seed: int = 42
    start_day: float = 0.0        # 0 = Monday 00:00 of simulated week 0
    bursts_enabled: bool = True
    burst_mean_interval_h: float = 3.0    # mean simulated hours between random IW spikes
    batch_mean_interval_h: float = 2.0    # mean simulated hours between NIW batch drops
    num_tenants: int = 50
    target_peak_util: float = 0.65        # endpoint utilisation at normal peak (sizes capacity)

    def to_env(self) -> dict:
        return {"SIM_OUTPUT": str(self.output_path), "SIM_SPEEDUP": str(self.speedup),
                "SIM_RATE_SCALE": str(self.rate_scale), "SIM_SEED": str(self.seed),
                "SIM_TARGET_UTIL": str(self.target_peak_util)}

    @classmethod
    def from_env(cls) -> "SimConfig":
        g = os.environ.get
        return cls(output_path=Path(g("SIM_OUTPUT", str(DEFAULT_OUTPUT))),
                   speedup=float(g("SIM_SPEEDUP", 60.0)),
                   rate_scale=float(g("SIM_RATE_SCALE", 0.002)),
                   seed=int(g("SIM_SEED", 42)),
                   target_peak_util=float(g("SIM_TARGET_UTIL", 0.65)))


def endpoint_capacity_tokens(cfg: SimConfig, region: str, model: str) -> float:
    """In-flight token capacity of one (region, model) endpoint, auto-sized so that normal peak
    load sits at `target_peak_util`. Bursts then push utilisation towards/over 1 and latency spikes."""
    peak_rps = sum(PEAK_RPS[t][region][model] for t in TIERS) * cfg.rate_scale
    return max(3000.0, peak_rps * mean_tokens(model) * 4.0 / cfg.target_peak_util)
