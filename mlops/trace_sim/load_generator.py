"""Non-homogeneous Poisson load generator driving the FastAPI endpoints on a simulated clock."""
from __future__ import annotations

import asyncio
import time

import httpx
import numpy as np

from .config import MODELS, REGIONS, TIERS, SimConfig
from .rate_model import RateModel, day_hour
from .request_factory import RequestFactory

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class TraceGenerator:
    def __init__(self, cfg: SimConfig, base_url: str = "http://127.0.0.1:8000", tick_s: float = 0.2):
        self.cfg, self.base_url, self.tick = cfg, base_url, tick_s
        self.rng = np.random.default_rng(cfg.seed)
        self.rates = RateModel(cfg, self.rng)
        self.factory = RequestFactory(cfg, self.rng)
        self.sent = self.done = self.errors = 0
        self._tasks: set = set()
        self._manual_burst = False

    def request_burst(self):
        """Thread/signal-safe: next tick injects a manual burst."""
        self._manual_burst = True

    async def _send(self, client, sem, p, delay):
        if delay > 0:
            await asyncio.sleep(delay)
        async with sem:
            try:
                r = await client.post(f"/{p['region']}/{p['model_type']}/generate", json=p)
                r.raise_for_status()
                self.done += 1
            except Exception:
                self.errors += 1

    def _spawn(self, client, sem, region, model, payload, delay):
        payload["region"], payload["model_type"] = region, model
        t = asyncio.create_task(self._send(client, sem, payload, delay))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        self.sent += 1

    def _emit(self, client, sem, t0, t1):
        dt, speed = t1 - t0, self.cfg.speedup
        self.rates.advance(t1)
        if self._manual_burst:
            b = self.rates.bursts.trigger(t1)
            self._manual_burst = False
            print(f"[burst] manual: x{b.peak_mult:.0f} in region '{b.region}'")
        tm = (t0 + t1) / 2
        for tier in TIERS:
            for region in REGIONS:
                for model in MODELS:
                    n = self.rng.poisson(self.rates.rate(tier, region, model, tm) * dt)
                    for _ in range(n):
                        ts = t0 + self.rng.random() * dt
                        p = self.factory.make(tier, region, model, ts)
                        self._spawn(client, sem, region, model, p, (ts - t0) / speed)
        for ev in self.rates.bursts.pop_batches(t1):
            print(f"[burst] NIW batch: {ev.size} requests -> {ev.region}/{ev.model}")
            for p in self.factory.make_batch(ev.region, ev.model, ev.time, ev.size):
                self._spawn(client, sem, ev.region, ev.model, p, 0.0)

    async def run(self, stop: asyncio.Event, duration_s=None, sim_hours=None, max_requests=None):
        """Run until `stop` is set, or wall `duration_s` / simulated `sim_hours` / `max_requests` is hit.
        With none of them set the generator is live and runs until stopped."""
        cfg = self.cfg
        sim0 = cfg.start_day * 86400
        sem = asyncio.Semaphore(2000)
        limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
        async with httpx.AsyncClient(base_url=self.base_url, limits=limits,
                                     timeout=httpx.Timeout(None)) as client:
            w0 = last_log = time.monotonic()
            t_prev, k = sim0, 0
            self.rates.advance(sim0)
            while not stop.is_set():
                k += 1
                delay = w0 + k * self.tick - time.monotonic()
                if delay > 0:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=delay)
                        break
                    except asyncio.TimeoutError:
                        pass
                wall = time.monotonic() - w0
                t = sim0 + wall * cfg.speedup
                if (duration_s and wall >= duration_s) or (sim_hours and t - sim0 >= sim_hours * 3600) \
                        or (max_requests and self.sent >= max_requests):
                    break
                self._emit(client, sem, t_prev, t)
                t_prev = t
                if time.monotonic() - last_log >= 10:
                    last_log = time.monotonic()
                    dow, h = day_hour(t)
                    act = len(self.rates.bursts.active)
                    if len(self._tasks) > 3000:
                        print("[warn] generator is falling behind (many requests in flight): "
                              "lower --speedup or --rate-scale, or the trace's latency dynamics will distort")
                    print(f"[sim {DAYS[dow]} {int(h):02d}:{int(h % 1 * 60):02d}] sent={self.sent} "
                          f"done={self.done} err={self.errors} in_flight={len(self._tasks)} "
                          f"active_bursts={act}")
            if self._tasks:                                  # drain in-flight requests
                await asyncio.wait(self._tasks, timeout=60)
        return {"sent": self.sent, "done": self.done, "errors": self.errors}
