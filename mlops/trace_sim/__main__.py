"""Simulate an LLM-serving trace and write it to a CSV.

    python -m trace_sim                      # run live until Ctrl+C
    python -m trace_sim --duration 600       # or stop automatically after 600 wall-clock seconds

Simulated time runs `--speedup` times faster than the wall clock, so
    wall-clock duration = simulated seconds / speedup      (one simulated week at speedup 600 = 1008 s)
"""
import argparse
import asyncio
import signal
import time
from pathlib import Path

import httpx

from .server import create_app
from .workload import Workload

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "traces" / "simulated_trace.csv"
TICK = 0.2      # wall-clock seconds between batches of arrivals


def clock(t):
    day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(t // 86400) % 7]
    return f"{day} {int(t % 86400 // 3600):02d}:{int(t % 3600 // 60):02d} (sim hour {int(t // 3600)})"


async def simulate(args):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):         # Ctrl+C = stop cleanly and save the CSV
        loop.add_signal_handler(sig, stop.set)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    workload = Workload(args.rate_scale, args.seed)
    mode = f"stopping after {args.duration:g}s" if args.duration else "live, press Ctrl+C to stop"
    print(f"writing {args.output} ({mode})")

    sent, t_prev, last_print, start = 0, 0.0, 0.0, time.monotonic()
    with open(args.output, "w", newline="") as f:
        app = create_app(f)
        # talk to the FastAPI app in-process (no network port needed)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sim") as client:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), TICK)      # sleep one tick, wake early on Ctrl+C
                except asyncio.TimeoutError:
                    pass
                elapsed = time.monotonic() - start
                if stop.is_set() or (args.duration and elapsed >= args.duration):
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
    print(f"done: {sent:,} requests, {clock(t_prev)} simulated -> {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=None, help="stop after N wall-clock seconds (default: run until Ctrl+C)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV path (overwritten)")
    p.add_argument("--speedup", type=float, default=60, help="simulated seconds per wall-clock second")
    p.add_argument("--rate-scale", type=float, default=0.002, help="fraction of the paper's request rate")
    p.add_argument("--seed", type=int, default=42)
    asyncio.run(simulate(p.parse_args()))


if __name__ == "__main__":
    main()