"""Builds request payloads (schema of the original generate_random_trace.py + tier info)."""
from __future__ import annotations

import itertools
import math

import numpy as np

from .config import (APPLICATIONS, IW_APP_WEIGHTS, MAX_OUTPUT, MAX_PROMPT, NIW_APP_WEIGHTS,
                     NIW_PROMPT_BOOST, OUTPUT_LOGN, PROMPT_LOGN, RAG_PROMPT_BOOST, REGION_ID,
                     REGIONS, TIER_SPEC, SimConfig)

_IW_P = np.array(IW_APP_WEIGHTS, float) / sum(IW_APP_WEIGHTS)
_NIW_P = np.array(NIW_APP_WEIGHTS, float) / sum(NIW_APP_WEIGHTS)


class RequestFactory:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self._ids = itertools.count()
        self._batch_ids = itertools.count()

    def _tokens(self, tier, region, model, app):
        pm, ps = PROMPT_LOGN[model]
        om, os_ = OUTPUT_LOGN[model]
        prompt = self.rng.lognormal(pm, ps)
        if APPLICATIONS[app] == "RAG":
            prompt *= RAG_PROMPT_BOOST
        if tier == "NIW":
            prompt *= NIW_PROMPT_BOOST.get((region, model), 1.0)
        out = self.rng.lognormal(om, os_)
        return (int(min(max(prompt, 1), MAX_PROMPT)), int(min(max(out, 1), MAX_OUTPUT)))

    def make(self, tier, region, model, t, batch_id=-1, batch_size=1, workload_type=None) -> dict:
        rng = self.rng
        app = int(rng.choice(len(APPLICATIONS), p=_NIW_P if tier == "NIW" else _IW_P))
        prompt, out = self._tokens(tier, region, model, app)
        others = [REGION_ID[r] for r in REGIONS if r != region]
        rng.shuffle(others)
        spec = TIER_SPEC[tier]
        return {
            "request_id": next(self._ids), "batch_id": batch_id,
            "client_tenant": int(rng.integers(self.cfg.num_tenants)),
            "request_type": 2, "scenario": "EnterpriseSydney",
            "sla": spec["sla"], "utility": spec["utility"],
            "regions": "".join(str(i) for i in [REGION_ID[region], *others]),   # preference order
            "workload_type": workload_type or "prod",
            "application_id": app, "application": APPLICATIONS[app],
            "arrival_timestamp": round(float(t), 3), "batch_size": batch_size,
            "prompt_size": prompt, "token_size": out, "tier": tier,
        }

    def make_batch(self, region, model, t, size) -> list[dict]:
        bid = next(self._batch_ids)
        wt = "prod" if self.rng.random() < 0.5 else "dev"
        return [self.make("NIW", region, model, t, bid, size, wt) for _ in range(size)]
