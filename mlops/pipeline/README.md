# Versioned data pipeline

Run everything from the `mlops/` directory (modules are invoked with `-m`).

    python -m pipeline.run_pipeline --source static                      # random trace -> clean -> forecasts
    python -m pipeline.run_pipeline --source streaming --duration 600    # server simulation -> clean -> forecasts
    python -m pipeline.run_pipeline --source streaming --stages etl,forecast   # reuse latest raw
    python -m pipeline.lineage show streaming/forecasts/latest
    python -m pipeline.lineage list

Single stages: `python -m trace_sim`, `python -m trace_sim.generate_random_trace`,
`python -m etl_module.run_etl --source ...`, `python -m forecasting.arima_forecaster --source ...`.

Data lives in `<parent of mlops>/data` (override with env `MLOPS_DATA_ROOT`):

    data/{streaming,static}/{raw,clean,forecasts}/<version>/   files + manifest.json
    data/{streaming,static}/<stage>/latest.json                pointer to newest version
    data/lineage/lineage.jsonl                                 append-only ledger (one line per artifact)

Version ids: raw = SHA-256(content)[:12]; clean/forecasts = SHA-256(parent ids + params)[:12].
If you change `clean_trace()` / validation logic, bump `CLEANING_VERSION` in `etl_module/etl.py`.
Downstream readers of forecasts should read `data/<source>/forecasts/latest.json`.
