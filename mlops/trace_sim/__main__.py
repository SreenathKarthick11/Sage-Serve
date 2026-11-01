"""Simulate an LLM-serving trace (streamed through the FastAPI model endpoints) and
store it as a new immutable version under  data/streaming/raw/<version>/.

    python -m trace_sim                      # run live until Ctrl+C
    python -m trace_sim --duration 600       # or stop automatically after 600 wall-clock seconds

Simulated time runs `--speedup` times faster than the wall clock, so
    wall-clock duration = simulated seconds / speedup      (one simulated week at speedup 600 = 1008 s)

Each run is written to a scratch directory and sealed at the end (also on Ctrl+C) with a
manifest recording the parameters, the seed and how the run ended.  `--output PATH` skips
versioning and writes a plain CSV (legacy behaviour).
"""
import argparse
import asyncio
import shutil
import signal
import time
from pathlib import Path

import httpx

from pipeline.versioning import finalize, new_tmp_dir, sha256_file

from .server import create_app
from .workload import Workload

TICK = 0.2      # wall-clock seconds between batches of arrivals


def clock(t):
    day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(t // 86400) % 7]
    return f"{day} {int(t % 86400 // 3600):02d}:{int(t % 3600 // 60):02d} (sim hour {int(t // 3600)})"


async def simulate(args):
    """Run the simulation. Returns the version directory (or None when --output is used)."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):         # Ctrl+C = stop cleanly and save the CSV
        loop.add_signal_handler(sig, stop.set)

    versioned = args.output is None
    if versioned:
        tmp_dir = new_tmp_dir("streaming", "raw")
        out_path = tmp_dir / "trace.csv"
    else:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)

    workload = Workload(args.rate_scale, args.seed)
    mode = f"stopping after {args.duration:g}s" if args.duration else "live, press Ctrl+C to stop"
    print(f"writing {out_path} ({mode})")

    sent, t_prev, last_print, start = 0, 0.0, 0.0, time.monotonic()
    reached_duration = False
    try:
        with open(out_path, "w", newline="") as f:
            app = create_app(f)
            # talk to the FastAPI app in-process (no network port needed)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sim") as client:
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), TICK)      # sleep one tick, wake early on Ctrl+C
                    except asyncio.TimeoutError:
                        pass
                    elapsed = time.monotonic() - start
                    if stop.is_set():
                        break
                    if args.duration and elapsed >= args.duration:
                        reached_duration = True
                        break

                    t_now = elapsed * args.speedup                      # simulated seconds since Monday 00:00
                    for region, model, row in workload.requests_between(t_prev, t_now):
                        (await client.post(f"/{region}/{model}/generate", json=row)).raise_for_status()
                        sent += 1
                    t_prev = t_now

                    for event in workload.events:
                        print(f"  [{clock(t_now)}] {event}")
                    workload.events.clear()
                    if elapsed - last_print >= 10:
                        last_print = elapsed
                        print(f"[{clock(t_now)}] {sent:,} requests")
    except BaseException:
        if versioned:
            shutil.rmtree(tmp_dir, ignore_errors=True)       # never publish a crashed run
        raise

    if not versioned:
        print(f"done: {sent:,} requests, {clock(t_prev)} simulated -> {out_path}")
        return None

    params = {
        "duration": args.duration, "speedup": args.speedup,
        "rate_scale": args.rate_scale, "seed": args.seed, "tick": TICK,
    }
    details = {
        "ingest": "fastapi-asgi-inprocess",
        "status": "complete" if reached_duration else "stopped_by_user",
        "rows": sent,
        "simulated_seconds": round(t_prev, 3),
    }
    version_id = sha256_file(out_path)[:12]                  # raw id = content hash
    final = finalize(tmp_dir, "streaming", "raw", version_id, parents=[], params=params, details=details)
    print(f"done: {sent:,} requests, {clock(t_prev)} simulated -> {final}")
    return final


def ingest(duration=None, speedup=60.0, rate_scale=0.002, seed=42, output=None):
    """Programmatic entry point (used by pipeline.run_pipeline)."""
    args = argparse.Namespace(duration=duration, speedup=speedup, rate_scale=rate_scale,
                              seed=seed, output=output)
    return asyncio.run(simulate(args))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=None, help="stop after N wall-clock seconds (default: run until Ctrl+C)")
    p.add_argument("--output", type=Path, default=None, help="plain CSV path, overwritten; skips versioning (default: versioned run)")
    p.add_argument("--speedup", type=float, default=60, help="simulated seconds per wall-clock second")
    p.add_argument("--rate-scale", type=float, default=0.002, help="fraction of the paper's request rate")
    p.add_argument("--seed", type=int, default=42)
    asyncio.run(simulate(p.parse_args()))


if __name__ == "__main__":
    main()
