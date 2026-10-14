"""Time-varying arrival rate lambda(t) per (tier, region, model), plus random bursts."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (MODEL_NOISE, MODELS, PEAK_RPS, REGIONS, TIER_SHAPE, TIERS, SimConfig)


# ---------- diurnal hump -------------------------------------------------------------------
def _raw_bump(h: float) -> float:
    """Two overlapping Gaussians (late morning + mid-afternoon) -> the double-peak workday of Fig. 4."""
    return math.exp(-0.5 * ((h - 10.0) / 1.9) ** 2) + 0.9 * math.exp(-0.5 * ((h - 15.5) / 2.1) ** 2)


_NORM = max(_raw_bump(x / 10) for x in range(240))


def bump(h: float) -> float:
    return _raw_bump(h) / _NORM


def day_hour(t: float):
    """t = simulated seconds since Monday 00:00. Returns (day_of_week 0..6, hour_of_day float)."""
    return int(t // 86400) % 7, (t % 86400) / 3600.0


# ---------- bursts -------------------------------------------------------------------------
@dataclass
class Burst:
    start: float
    duration: float
    peak_mult: float
    tier: str | None          # None = all IW tiers
    region: str | None        # None = every region
    model: str | None         # None = every model

    def factor(self, t: float) -> float:
        x = t - self.start
        if x < 0 or x > self.duration:
            return 1.0
        ramp = 0.1 * self.duration                     # fast ramp-up, exponential decay
        prof = x / ramp if x < ramp else math.exp(-3.0 * (x - ramp) / (self.duration - ramp))
        return 1.0 + (self.peak_mult - 1.0) * prof

    def applies(self, tier: str, region: str, model: str) -> bool:
        return ((self.tier is None and tier != "NIW") or self.tier == tier) and \
               (self.region is None or self.region == region) and \
               (self.model is None or self.model == model)


@dataclass
class BatchEvent:
    time: float
    region: str
    model: str
    size: int


class BurstManager:
    """Poisson process of IW traffic spikes (2-8x, 1-8 sim-minutes) and NIW batch drops."""

    def __init__(self, rng: np.random.Generator, cfg: SimConfig):
        self.rng, self.enabled = rng, cfg.bursts_enabled
        self.burst_gap = cfg.burst_mean_interval_h * 3600
        self.batch_gap = cfg.batch_mean_interval_h * 3600
        # batch sizes shrink with --rate-scale so NIW batches keep their share of the traffic
        self.size_scale = min(1.0, cfg.rate_scale / 0.002)
        self.active: list[Burst] = []
        self.next_burst = self.next_batch = None
        niw = [(r, m, PEAK_RPS["NIW"][r][m]) for r in REGIONS for m in MODELS]
        w = np.array([x[2] for x in niw], float)
        self._niw_pairs, self._niw_p = [(r, m) for r, m, _ in niw], w / w.sum()

    def _init(self, t):
        self.next_burst = t + self.rng.exponential(self.burst_gap)
        self.next_batch = t + self.rng.exponential(self.batch_gap)

    def _random_burst(self, t: float) -> Burst:
        rng = self.rng
        tier = "IW-F" if rng.random() < 0.6 else "IW-N"
        scope = rng.random()
        region = None if scope > 0.85 else REGIONS[rng.integers(3)]
        model = MODELS[rng.integers(4)] if scope < 0.6 else None   # 60% single endpoint, 25% region-wide, 15% global
        mult = float(np.clip(rng.lognormal(math.log(3.5), 0.45), 2.0, 8.0))
        dur = float(math.exp(rng.uniform(math.log(60), math.log(480))))
        return Burst(t, dur, mult, tier, region, model)

    def trigger(self, t: float) -> Burst:
        """Manual burst (SIGUSR1): 8x on all IW traffic of one random region for 5 sim-minutes."""
        b = Burst(t, 300.0, 8.0, None, REGIONS[self.rng.integers(3)], None)
        self.active.append(b)
        return b

    def advance(self, t: float):
        if not self.enabled:
            return
        if self.next_burst is None:
            self._init(t)
        while self.next_burst <= t:
            self.active.append(self._random_burst(self.next_burst))
            self.next_burst += self.rng.exponential(self.burst_gap)
        self.active = [b for b in self.active if t <= b.start + b.duration]

    def multiplier(self, tier: str, region: str, model: str, t: float) -> float:
        m = 1.0
        for b in self.active:
            if b.applies(tier, region, model):
                m *= b.factor(t)
        return m

    def pop_batches(self, t: float) -> list[BatchEvent]:
        if not self.enabled or self.next_batch is None:
            return []
        out = []
        while self.next_batch <= t:
            r, m = self._niw_pairs[self.rng.choice(len(self._niw_pairs), p=self._niw_p)]
            size = int(self.rng.choice([32, 64, 128, 256]) * self.size_scale)
            out.append(BatchEvent(self.next_batch, r, m, max(4, size)))
            self.next_batch += self.rng.exponential(self.batch_gap)
        return out


# ---------- rate model ---------------------------------------------------------------------
class RateModel:
    OU_STEP = 60.0     # slow-noise update interval in simulated seconds

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.bursts = BurstManager(rng, cfg)
        self.x = {(t, r, m): 0.0 for t in TIERS for r in REGIONS for m in MODELS}
        self._last_ou = None
        self._day_noise: dict = {}

    def advance(self, t: float):
        """Move slow noise + burst process forward to simulated time t (call once per tick)."""
        if self._last_ou is None:
            self._last_ou = t
        while t - self._last_ou >= self.OU_STEP:
            self._last_ou += self.OU_STEP
            for (tier, region, model), x in self.x.items():
                sp = TIER_SHAPE[tier]
                theta = sp.ou_theta / 60.0                       # per-minute step
                sigma = sp.ou_sigma * (MODEL_NOISE[model] if tier != "NIW" else 1.0)
                self.x[(tier, region, model)] = x - theta * x + sigma * math.sqrt(2 * theta) * self.rng.normal()
        self.bursts.advance(t)

    def shape(self, tier: str, t: float) -> float:
        dow, h = day_hour(t)
        sp, weekend = TIER_SHAPE[tier], dow >= 5
        if not sp.diurnal:
            return sp.weekend if weekend else 1.0
        key = (tier, int(t // 86400))
        if key not in self._day_noise:                           # day-to-day amplitude wobble
            self._day_noise[key] = math.exp(self.rng.normal(0, 0.05))
        daymult = (sp.weekend if weekend else sp.weekday_ramp[dow]) * self._day_noise[key]
        return sp.floor + (1 - sp.floor) * bump(h) * daymult

    def rate(self, tier: str, region: str, model: str, t: float) -> float:
        """Expected arrivals per *simulated* second."""
        base = PEAK_RPS[tier][region][model] * self.cfg.rate_scale
        return (base * self.shape(tier, t) * math.exp(self.x[(tier, region, model)])
                * self.bursts.multiplier(tier, region, model, t))

    def peak_wall_rps(self) -> float:
        total = sum(PEAK_RPS[t][r][m] for t in TIERS for r in REGIONS for m in MODELS)
        return total * self.cfg.rate_scale * self.cfg.speedup
