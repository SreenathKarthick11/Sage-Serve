"""Read the lineage ledger (data/lineage/lineage.jsonl).

    python -m pipeline.lineage list [--source streaming]
    python -m pipeline.lineage show streaming/forecasts/<version>
    python -m pipeline.lineage show streaming/forecasts/latest
"""
from __future__ import annotations

import argparse
import json

from .versioning import LINEAGE_PATH, latest_version


def load() -> dict[str, dict]:
    """artifact id -> record.  Later lines win (the ledger is append-only)."""
    if not LINEAGE_PATH.exists():
        return {}
    records = {}
    with open(LINEAGE_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["id"]] = r
    return records


def resolve(artifact: str) -> str:
    parts = artifact.split("/")
    if len(parts) == 3 and parts[2] == "latest":
        return f"{parts[0]}/{parts[1]}/{latest_version(parts[0], parts[1])}"
    return artifact


def ancestors(artifact: str, records: dict[str, dict] | None = None) -> list[str]:
    """All ancestors of `artifact` (nearest first, no duplicates)."""
    records = records if records is not None else load()
    seen, order, stack = set(), [], [artifact]
    while stack:
        node = stack.pop(0)
        for p in records.get(node, {}).get("parents", []):
            if p not in seen:
                seen.add(p)
                order.append(p)
                stack.append(p)
    return order


def _summary(r: dict) -> str:
    d = r.get("details", {})
    keys = ("status", "rows", "rows_in", "rows_out", "n_forecast_files")
    bits = [f"{k}={d[k]}" for k in keys if k in d]
    return f"[{r['created_at']}, git {r['git_commit']}] " + " ".join(bits)


def show(artifact: str) -> None:
    records = load()
    artifact = resolve(artifact)
    if artifact not in records:
        raise SystemExit(f"Unknown artifact: {artifact}")

    def walk(node: str, depth: int) -> None:
        r = records.get(node)
        pad = "    " * depth
        arrow = "<- " if depth else ""
        if r is None:
            print(f"{pad}{arrow}{node}  (not in ledger)")
            return
        print(f"{pad}{arrow}{node}  {_summary(r)}")
        print(f"{pad}    params: {json.dumps(r['params'], default=str)}")
        for p in r["parents"]:
            walk(p, depth + 1)

    walk(artifact, 0)


def list_all(source: str | None = None) -> None:
    for r in sorted(load().values(), key=lambda r: r["created_at"]):
        if source and r["source"] != source:
            continue
        parents = ", ".join(r["parents"]) or "-"
        print(f"{r['created_at']}  {r['id']:<40}  parents: {parents}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--source", choices=["streaming", "static"])
    sh = sub.add_parser("show")
    sh.add_argument("artifact", help="e.g. streaming/forecasts/<version> or streaming/forecasts/latest")
    a = p.parse_args()
    list_all(a.source) if a.cmd == "list" else show(a.artifact)


if __name__ == "__main__":
    main()
