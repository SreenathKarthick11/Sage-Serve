"""FastAPI app: one route per (region, model) endpoint. Every request is logged to a single CSV.

POST /{region}/{model}/generate   region in west|central|east, model in A|B|C|D
GET  /health, GET /stats
"""
from __future__ import annotations

import asyncio
import csv
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import (DECODE_TPS, MODELS, NETWORK_S, PREFILL_TPS, REGION_ID, REGIONS, SimConfig,
                     endpoint_capacity_tokens)

COLUMNS = ["request_id", "batch_id", "client_tenant", "request_type", "scenario", "sla", "utility",
           "regions", "model_type", "workload_type", "application_id", "arrival_timestamp",
           "batch_size", "prompt_size", "token_size", "region", "tier", "application",
           "ttft", "e2e", "queue_delay", "endpoint_util", "sla_met"]


class GenerateRequest(BaseModel):
    request_id: int
    batch_id: int = -1
    client_tenant: int = 0
    request_type: int = 2
    scenario: str = "EnterpriseSydney"
    sla: float
    utility: int
    regions: str = ""
    workload_type: str = "prod"
    application_id: int = 0
    application: str = ""
    arrival_timestamp: float          # simulated seconds since Monday 00:00
    batch_size: int = 1
    prompt_size: int
    token_size: int
    tier: str


@dataclass
class Endpoint:
    capacity: float          # in-flight token capacity
    inflight: float = 0.0
    served: int = 0


class CsvSink:
    """Single writer task, so rows from concurrent requests never interleave."""

    def __init__(self, path: Path):
        self.path, self.q, self.task = Path(path), asyncio.Queue(), None

    async def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.f = open(self.path, "a", newline="")
        self.w = csv.writer(self.f, lineterminator="\n")
        if new:
            self.w.writerow(COLUMNS)
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        done = False
        while not done:
            batch = [await self.q.get()]
            while not self.q.empty() and len(batch) < 1000:
                batch.append(self.q.get_nowait())
            for row in batch:
                if row is None:
                    done = True
                    break
                self.w.writerow(row)
            self.f.flush()

    async def stop(self):
        await self.q.put(None)
        await self.task
        self.f.close()


def simulate_latency(rng: random.Random, model: str, prompt: int, out: int, util: float):
    """Returns (ttft, e2e, queue_delay) in simulated seconds. Queueing delay blows up as util -> 1."""
    rho = min(util, 0.97)
    queue = (0.15 * rho / (1 - rho) + max(0.0, util - 1.0) * 5.0) * rng.lognormvariate(0, 0.3)
    prefill = prompt / PREFILL_TPS[model] * (1 + 0.5 * min(util, 2.0))
    ttft = NETWORK_S + queue + prefill
    decode = out / DECODE_TPS[model] * (1 + 0.6 * min(util, 1.5)) * rng.lognormvariate(0, 0.2)
    return ttft, ttft + decode, queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = SimConfig.from_env()
    app.state.cfg = cfg
    app.state.rng = random.Random(cfg.seed)
    app.state.eps = {(r, m): Endpoint(endpoint_capacity_tokens(cfg, r, m))
                     for r in REGIONS for m in MODELS}
    app.state.sink = CsvSink(cfg.output_path)
    await app.state.sink.start()
    yield
    await app.state.sink.stop()      # flush everything on shutdown


app = FastAPI(title="SageServe trace simulator", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "output": str(app.state.cfg.output_path)}


@app.get("/stats")
async def stats():
    return {f"{r}/{m}": {"served": e.served, "inflight_tokens": round(e.inflight),
                         "util": round(e.inflight / e.capacity, 3)}
            for (r, m), e in app.state.eps.items()}


@app.post("/{region}/{model}/generate")
async def generate(region: str, model: str, req: GenerateRequest):
    model = model.upper()
    ep = app.state.eps.get((region, model))
    if ep is None:
        raise HTTPException(404, f"unknown endpoint {region}/{model}")
    cfg = app.state.cfg

    util = ep.inflight / ep.capacity
    ttft, e2e, queue = simulate_latency(app.state.rng, model, req.prompt_size, req.token_size, util)
    tokens = req.prompt_size + req.token_size

    ep.inflight += tokens
    try:    # hold the tokens for the (wall-clock) service time so concurrent load builds up realistically
        await asyncio.sleep(min(e2e / cfg.speedup, 60.0))
    finally:
        ep.inflight -= tokens
    ep.served += 1

    sla_met = (e2e <= req.sla) if req.tier == "NIW" else (ttft <= req.sla)
    app.state.sink.q.put_nowait([
        req.request_id, req.batch_id, req.client_tenant, req.request_type, req.scenario, req.sla,
        req.utility, req.regions, model, req.workload_type, req.application_id,
        req.arrival_timestamp, req.batch_size, req.prompt_size, req.token_size, region, req.tier,
        req.application, round(ttft, 4), round(e2e, 4), round(queue, 4), round(util, 4), int(sla_met)])
    return {"request_id": req.request_id, "ttft": ttft, "e2e": e2e, "sla_met": sla_met}
