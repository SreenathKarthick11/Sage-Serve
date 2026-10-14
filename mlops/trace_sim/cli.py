"""CLI:  python -m trace_simulation {run,server,generate,validate} ...   (run from mlops/)"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from .config import DEFAULT_OUTPUT, SimConfig


def _common(p: argparse.ArgumentParser):
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV trace path")
    p.add_argument("--speedup", type=float, default=60.0, help="simulated seconds per wall second")
    p.add_argument("--rate-scale", type=float, default=0.002, help="fraction of paper-scale RPS")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-day", type=float, default=0.0, help="0=Mon 00:00 of simulated week")
    p.add_argument("--no-bursts", action="store_true", help="disable random bursts / batch drops")
    p.add_argument("--burst-interval-h", type=float, default=3.0, help="mean sim hours between IW bursts")
    p.add_argument("--batch-interval-h", type=float, default=2.0, help="mean sim hours between NIW batches")


def _stop_args(p: argparse.ArgumentParser):
    p.add_argument("--duration", type=float, default=None,
                   help="stop after this many WALL-clock seconds (default: run until Ctrl+C)")
    p.add_argument("--sim-hours", type=float, default=None, help="stop after this many SIMULATED hours")
    p.add_argument("--max-requests", type=int, default=None, help="stop after this many requests")


def _cfg(a) -> SimConfig:
    return SimConfig(output_path=a.output, speedup=a.speedup, rate_scale=a.rate_scale, seed=a.seed,
                     start_day=a.start_day, bursts_enabled=not a.no_bursts,
                     burst_mean_interval_h=a.burst_interval_h, batch_mean_interval_h=a.batch_interval_h)


def _common_argv(a) -> list[str]:
    argv = ["--output", str(a.output), "--speedup", str(a.speedup), "--rate-scale", str(a.rate_scale),
            "--seed", str(a.seed), "--start-day", str(a.start_day),
            "--burst-interval-h", str(a.burst_interval_h), "--batch-interval-h", str(a.batch_interval_h)]
    return argv + (["--no-bursts"] if a.no_bursts else [])


def cmd_server(a):
    import uvicorn
    os.environ.update(_cfg(a).to_env())
    from .server import app
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


async def _generate(cfg: SimConfig, a) -> dict:
    from .load_generator import TraceGenerator
    gen = TraceGenerator(cfg, f"http://{a.host}:{a.port}")
    print(f"peak load ~{gen.rates.peak_wall_rps():.0f} req/s wall-clock at paper-peak "
          f"(speedup={cfg.speedup:g}, rate_scale={cfg.rate_scale:g}); "
          + ("running LIVE - Ctrl+C to stop, SIGUSR1 for a manual burst" if not (
              a.duration or a.sim_hours or a.max_requests) else "bounded run"))
    stop, loop = asyncio.Event(), asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    if hasattr(signal, "SIGUSR1"):
        loop.add_signal_handler(signal.SIGUSR1, gen.request_burst)
    return await gen.run(stop, a.duration, a.sim_hours, a.max_requests)


def cmd_generate(a):
    print(asyncio.run(_generate(_cfg(a), a)))


def _rotate(path: Path):
    if path.exists() and path.stat().st_size > 0:
        bak = path.with_name(f"{path.stem}.{datetime.now():%Y%m%d-%H%M%S}.bak{path.suffix}")
        path.rename(bak)
        print(f"existing trace moved to {bak.name}")


def cmd_run(a):
    cfg = _cfg(a)
    if not a.append:
        _rotate(cfg.output_path)
    pkg_parent = Path(__file__).resolve().parents[1]
    srv = subprocess.Popen(
        [sys.executable, "-m", __package__, "server", "--host", a.host, "--port", str(a.port),
         *_common_argv(a)], cwd=pkg_parent, start_new_session=True)  # own session: Ctrl+C only hits us
    try:
        for _ in range(100):
            if srv.poll() is not None:
                raise RuntimeError(f"server exited early (code {srv.returncode}); "
                                   f"run `python3 -m {__package__} server` to see the error")
            try:
                if httpx.get(f"http://{a.host}:{a.port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not start")
        print("server up ->", cfg.output_path)
        print(asyncio.run(_generate(cfg, a)))
    finally:
        srv.send_signal(signal.SIGINT)          # graceful: server flushes the CSV on shutdown
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.terminate()
    print("trace written:", cfg.output_path)


def cmd_validate(a):
    from .validate import analyze
    analyze(a.output, a.plot)


def main():
    ap = argparse.ArgumentParser(prog="trace_simulation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, helptxt in [("run", cmd_run, "start server + generator together"),
                              ("server", cmd_server, "start only the FastAPI endpoints"),
                              ("generate", cmd_generate, "start only the load generator")]:
        p = sub.add_parser(name, help=helptxt)
        _common(p)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        if name != "server":
            _stop_args(p)
        if name == "run":
            p.add_argument("--append", action="store_true", help="append to an existing CSV instead of rotating it")
        p.set_defaults(fn=fn)

    p = sub.add_parser("validate", help="check the trace against the paper's characteristics")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--plot", type=Path, default=None, help="PNG path (default: next to the CSV)")
    p.set_defaults(fn=cmd_validate)

    a = ap.parse_args()
    a.fn(a)
