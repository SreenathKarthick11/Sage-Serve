"""FastAPI 'model endpoints'. Each endpoint receives a request and appends it to the trace CSV."""
import csv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .workload import COLUMNS, MODELS, REGIONS


class TraceRequest(BaseModel):
    """One request = one CSV row (same 15 fields, same order)."""
    request_id: int
    batch_id: int
    client_tenant: int
    request_type: int
    scenario: str
    sla: int
    utility: int
    regions: str
    model_type: str
    workload_type: str
    application_id: int
    arrival_timestamp: float
    batch_size: int
    prompt_size: int
    token_size: int


def create_app(csv_file) -> FastAPI:
    """Build the app; every request it receives is written to `csv_file`."""
    app = FastAPI(title="SageServe trace simulator")
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(COLUMNS)
    count = 0

    # async def => runs on the event loop, so rows can never interleave
    @app.post("/{region}/{model}/generate")
    async def generate(region: str, model: str, req: TraceRequest):
        nonlocal count
        if region not in REGIONS or model not in MODELS:
            raise HTTPException(404, f"no endpoint {region}/{model}")
        writer.writerow([getattr(req, c) for c in COLUMNS])
        count += 1
        if count % 200 == 0:
            csv_file.flush()
        return {"ok": True}

    return app