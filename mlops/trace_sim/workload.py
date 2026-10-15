"""Workload model: WHEN requests arrive and WHAT each one looks like.

Pure Python + numpy, no I/O. The tables are eyeballed from the SageServe paper
(Figs. 3-6 for the traffic shape, Fig. 10 for token sizes).
Time is *simulated* seconds since Monday 00:00.
"""
import math

import numpy as np

REGIONS = ["west", "central", "east"]      # region ids 0, 1, 2
MODELS = ["A", "B", "C", "D"]

# The 15 columns of the trace CSV, in order.
COLUMNS = ["request_id", "batch_id", "client_tenant", "request_type", "scenario", "sla", "utility",
           "regions", "model_type", "workload_type", "application_id", "arrival_timestamp",
           "batch_size", "prompt_size", "token_size"]

# ---------------------------------------------------------------------------------------------
# 1. Traffic levels
# ---------------------------------------------------------------------------------------------
# Peak requests/sec at paper scale, for models A, B, C, D in each region.
# East is busiest; model A dominates East, B dominates Central (IW-F) and West (IW-N);
# NIW is small, except model C (a feature-testing app).
PEAK_RPS = {
    "IW-F": {"west": [70, 40, 10, 5],      "central": [70, 190, 20, 10], "east": [400, 60, 20, 10]},
    "IW-N": {"west": [49, 70, 21, 7],      "central": [35, 14, 10, 4],   "east": [140, 63, 28, 7]},
    "NIW":  {"west": [1.6, 2.4, 4, 0.8],   "central": [8, 12, 32, 4],    "east": [24, 32, 80, 8]},
}

# IW-F = fast interactive (TTFT < 1 s), IW-N = normal interactive (< 1 min), NIW = batch (24 h).
#   daily   : does it have a daytime hump?
#   floor   : off-peak traffic as a fraction of the daytime peak
#   weekend : how much of the hump is left on Sat/Sun (for NIW: the flat level on the weekend)
#   ramp    : hump height Mon..Fri (IW-N grows towards Friday)
#   noise   : std-dev of the random hour-to-hour wobble
#   day_noise : std-dev of the day-to-day change in hump height (also shifts/reshapes the hump)
TIERS = {
    "IW-F": dict(sla=1,     utility=3, daily=True,  floor=0.03, weekend=0.08, ramp=[1, 1, 1, 1, 1],            noise=0.08, day_noise=0.12),
    "IW-N": dict(sla=60,    utility=2, daily=True,  floor=0.35, weekend=0.10, ramp=[0.8, 0.85, 1, 1.05, 1.1], noise=0.12, day_noise=0.09),
    "NIW":  dict(sla=86400, utility=1, daily=False, floor=1.0,  weekend=0.75, ramp=[1, 1, 1, 1, 1],            noise=0.30, day_noise=0.0),
}

BURST_EVERY_S = 3 * 3600    # on average one IW traffic spike per 3 simulated hours
BATCH_EVERY_S = 2 * 3600    # on average one NIW batch drop per 2 simulated hours


def _hump_raw(h, shift=0.0, balance=1.0):
    # two overlapping peaks (~10:00 and ~15:30) with a dip around lunch.
    # `shift` slides the whole hump (hours); `balance` > 1 makes the morning peak taller than the afternoon one.
    return (balance * math.exp(-0.5 * ((h - shift - 10) / 1.9) ** 2)
            + 0.9 / balance * math.exp(-0.5 * ((h - shift - 15.5) / 2.1) ** 2))


_HUMP_MAX = max(_hump_raw(h / 10) for h in range(240))




# ---------------------------------------------------------------------------------------------
# 2. Request contents
# ---------------------------------------------------------------------------------------------
PROMPT_MEDIAN = {"A": 2200, "B": 1600, "C": 1300, "D": 1000}    # tokens (prompts mostly > 1k)
OUTPUT_MEDIAN = {"A": 320, "B": 250, "C": 200, "D": 150}        # tokens (outputs mostly < 1k)
NIW_PROMPT_BOOST = {("central", "C"): 4, ("east", "C"): 2}      # bulk feature-evaluation prompts

# Application ids (paper Fig. 6a). Id 0 is RAG: 41% of interactive traffic, with long prompts.
_norm = lambda w: np.array(w) / sum(w)
APP_WEIGHTS_IW = _norm([41.2, 9.3, 8.6, 6.7, 5.3, 3.8, 3.2, 2.4, 1.8, 1.7, 16.2])
APP_WEIGHTS_NIW = _norm([0, 15, 20, 0, 0, 30, 5, 10, 5, 0, 15])


class Workload:
    def __init__(self, rate_scale=0.002, seed=42):
        self.scale = rate_scale                 # fraction of paper-scale traffic
        self.rng = np.random.default_rng(seed)
        self.next_request_id = 0
        self.next_batch_id = 0
        self.wobble = {}                        # (tier, region, model, hour) -> random multiplier
        self.day_cache = {}                     # random per-day variation, drawn once per day
        self.bursts = []                        # spikes currently active
        self.events = []                        # messages about new bursts/batches (for printing)
        # NIW batch drops mostly hit the busiest NIW endpoints
        self.niw_targets = [(r, m) for r in REGIONS for m in MODELS]
        w = [PEAK_RPS["NIW"][r][i] for r in REGIONS for i in range(4)]
        self.niw_probs = _norm(w)

    # ---- arrival rate ---------------------------------------------------------------------
    def _day_params(self, tier, region, model, day):
        """Random day-to-day variation so no two days look identical: (height, time shift, balance)."""
        rng, sigma = self.rng, TIERS[tier]["day_noise"]
        if (tier, day) not in self.day_cache:           # a 'busy day' effect shared by all of a tier's streams
            self.day_cache[(tier, day)] = math.exp(rng.normal(0, sigma))
        key = (tier, region, model, day)
        if key not in self.day_cache:
            height = self.day_cache[(tier, day)] * math.exp(rng.normal(0, sigma / 2))
            self.day_cache[key] = (height, rng.normal(0, 0.5), math.exp(rng.normal(0, 0.2)))
        return self.day_cache[key]

    def shape(self, tier, region, model, t):
        """Traffic level at time t as a fraction of the stream's peak (about 0..1)."""
        p = TIERS[tier]
        day, hour = int(t // 86400), (t % 86400) / 3600
        weekend = day % 7 >= 5
        if not p["daily"]:
            return p["weekend"] if weekend else 1.0
        height, shift, balance = self._day_params(tier, region, model, day)
        height *= p["weekend"] if weekend else p["ramp"][day % 7]
        return p["floor"] + (1 - p["floor"]) * height * _hump_raw(hour, shift, balance) / _HUMP_MAX

    def rate(self, tier, region, mi, t):
        """Expected requests per simulated second for one (tier, region, model) stream."""
        model = MODELS[mi]
        hour_key = (tier, region, model, int(t // 3600))
        if hour_key not in self.wobble:
            self.wobble[hour_key] = math.exp(self.rng.normal(0, TIERS[tier]["noise"]))
        r = PEAK_RPS[tier][region][mi] * self.scale * self.shape(tier, region, model, t) * self.wobble[hour_key]
        for b in self.bursts:                   # active spikes multiply the rate, fading out linearly
            if b["tier"] == tier and b["region"] == region and b["model"] in (None, model):
                left = 1 - min(1.0, max(0.0, (t - b["start"]) / b["duration"]))
                r *= 1 + (b["mult"] - 1) * left
        return r

    def _maybe_start_burst(self, t, dt):
        if self.rng.random() < dt / BURST_EVERY_S:
            rng = self.rng
            b = {"start": t, "duration": float(rng.uniform(60, 480)),      # 1-8 simulated minutes
                 "mult": float(rng.uniform(2, 8)),                         # 2x-8x traffic
                 "tier": "IW-F" if rng.random() < 0.6 else "IW-N",
                 "region": REGIONS[rng.integers(3)],
                 "model": MODELS[rng.integers(4)] if rng.random() < 0.7 else None}   # None = all models
            self.bursts.append(b)
            self.events.append(f"burst: {b['mult']:.1f}x {b['tier']} in {b['region']}/{b['model'] or 'all'} "
                               f"for {b['duration'] / 60:.0f} min")

    # ---- generating requests ---------------------------------------------------------------
    def requests_between(self, t0, t1):
        """All requests arriving in simulated time [t0, t1) as (region, model, row) tuples."""
        dt = t1 - t0
        self.bursts = [b for b in self.bursts if t0 < b["start"] + b["duration"]]
        self._maybe_start_burst(t0, dt)

        out = []
        for tier in TIERS:
            for region in REGIONS:
                for mi, model in enumerate(MODELS):
                    n = self.rng.poisson(self.rate(tier, region, mi, (t0 + t1) / 2) * dt)
                    for _ in range(n):
                        t = t0 + self.rng.random() * dt
                        out.append((region, model, self._request(tier, region, model, t)))

        if self.rng.random() < dt / BATCH_EVERY_S:
            out += self._batch(t1)
        return out

    def _batch(self, t):
        """A batch of NIW requests that all arrive at once and share a batch_id."""
        rng = self.rng
        region, model = self.niw_targets[rng.choice(len(self.niw_targets), p=self.niw_probs)]
        size = int(rng.choice([32, 64, 128, 256]) * min(1.0, self.scale / 0.002))   # shrink with scale
        size = max(size, 4)
        batch_id, self.next_batch_id = self.next_batch_id, self.next_batch_id + 1
        workload = "prod" if rng.random() < 0.5 else "dev"
        self.events.append(f"batch: {size} NIW requests -> {region}/{model}")
        return [(region, model, self._request("NIW", region, model, t, batch_id, size, workload))
                for _ in range(size)]

    def _request(self, tier, region, model, t, batch_id=-1, batch_size=1, workload="prod"):
        rng = self.rng
        app = int(rng.choice(11, p=APP_WEIGHTS_NIW if tier == "NIW" else APP_WEIGHTS_IW))
        prompt = rng.lognormal(math.log(PROMPT_MEDIAN[model]), 0.9)
        if app == 0:                                        # RAG carries retrieved context
            prompt *= 2
        if tier == "NIW":
            prompt *= NIW_PROMPT_BOOST.get((region, model), 1)
        output = rng.lognormal(math.log(OUTPUT_MEDIAN[model]), 0.8)

        home = REGIONS.index(region)                        # `regions` = region preference order,
        others = [i for i in range(3) if i != home]         # the region that got the request first
        rng.shuffle(others)

        row = {
            "request_id": self.next_request_id, "batch_id": batch_id,
            "client_tenant": int(rng.integers(50)), "request_type": 2, "scenario": "EnterpriseSydney",
            "sla": TIERS[tier]["sla"], "utility": TIERS[tier]["utility"],
            "regions": "".join(str(i) for i in [home] + others),
            "model_type": model, "workload_type": workload, "application_id": app,
            "arrival_timestamp": round(float(t), 3), "batch_size": batch_size,
            "prompt_size": int(min(max(prompt, 1), 32000)), "token_size": int(min(max(output, 1), 4000)),
        }
        self.next_request_id += 1
        return row