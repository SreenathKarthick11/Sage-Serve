# SageServe — Architecture & UML Reference

## 0. What comes from Splitwise vs. what SageServe adds

SageServe is a **fork** of the [Splitwise simulator](https://github.com/Mutinifni/splitwise-sim). The relationship is not "inspired by" — large parts of the low-level discrete-event engine, request/instance model, and prefill/decode disaggregation logic are **reused code**. SageServe's contribution sits almost entirely in the **autoscaling / multi-region orchestration layer**.

| Layer | Origin | Files |
| --- | --- | --- |
| Discrete-event engine (`clock()`, `schedule_event`) | **Splitwise** (unchanged core) | `simulator.py` |
| Request → Task decomposition, prefill/decode split | **Splitwise** | `request.py`, `task.py`, `flow.py` |
| Instance execution, batching, KV-cache/memory tracking | **Splitwise** | `instance.py`, `server.py`, `processor.py` |
| Analytical performance/power model | **Splitwise** (extended) | `performance_model.py`, `power_model.py` |
| Single-application scheduler (request → instance) | **Splitwise** (baseline), policies extended | `scheduler.py`, `allocator.py` |
| Hydra config-driven object instantiation | **Splitwise** pattern, config trees extended | `initialize.py`, `configs/` |
| **Per-model, per-region reactive arbiters** (memory-utilization scaling, spot-instance reclaim, ARIMA-checked scaling, Chiron-style scaling) | **SageServe (new)** | `arbiter.py` |
| **Global, cross-region, cross-model long-term allocator** (MILP + ARIMA forecast ingestion) | **SageServe (new)** | `global_arbiter.py`, `long_term_allocation.py` |
| **Multi-region topology** (region clusters, region-level routers, cross-region controller) | **SageServe (new)** — Splitwise is single-cluster | `region.py`, `region_cluster.py`, `region_router.py`, `global_router.py`, `controller.py` |
| Forecast-aware vs. reactive scaling experiments, production-trace evaluation | **SageServe (new)** — this is the paper's contribution | `run.py`, `run_lts.py`, `configs/global_arbiter/*` |

**In one sentence:** Splitwise supplies the *simulated GPU cluster that correctly executes LLM requests*; SageServe wraps that engine in *N regions × M models*, and replaces Splitwise's "no-op" router/arbiter stubs with real reactive and forecast-aware autoscaling policies, plus a MILP-based long-term capacity planner, so the paper can compare instance-hours and latency under different scaling strategies on production-like traces.

---

## 1. Module dependency diagram

```mermaid
graph TB
    subgraph Config["Config & Entry Point"]
        A1[configs/config.yaml - Hydra]
        A2[run.py / run_lts.py]
        A3[initialize.py]
    end

    subgraph Engine["Discrete-Event Engine (Splitwise)"]
        B1[simulator.py<br/>clock, schedule_event]
        B2[trace.py<br/>request arrivals]
    end

    subgraph ReqExec["Request Execution (Splitwise)"]
        C1[request.py / task.py / flow.py]
        C2[instance.py]
        C3[server.py / processor.py / node.py]
        C4[performance_model.py / power_model.py]
        C5[model.py / model_repo.py]
    end

    subgraph AppLayer["Per-Application Serving (Splitwise, policies extended)"]
        D1[application.py]
        D2[scheduler.py]
        D3[allocator.py]
    end

    subgraph Auto["Autoscaling — SageServe NEW"]
        E1[arbiter.py<br/>short-term / reactive policies]
        E2[global_arbiter.py<br/>MilpGlobalArbiter]
        E3[long_term_allocation.py<br/>MILP solver]
    end

    subgraph MultiRegion["Multi-Region Orchestration — SageServe NEW"]
        F1[region.py / region_cluster.py]
        F2[region_router.py]
        F3[global_router.py]
        F4[controller.py / orchestrator_repo.py]
        F5[model_endpoint_router.py / model_endpoint_repo.py]
    end

    subgraph Output["Output"]
        G1[metrics.py / utils.py]
        G2[summary.csv / detailed/*.csv / instances/*.csv]
        G3[notebooks/*.ipynb, plotting_scripts/]
    end

    A1 --> A2 --> A3
    A3 --> B1
    A3 --> F4
    B2 --> B1
    B1 --> C1 --> C2 --> C3
    C2 --> C4
    C1 --> C5
    C2 --> D1
    D1 --> D2
    D1 --> D3
    D3 -->|start/stop instances| C2
    F4 --> F1
    F1 --> F2 --> F5
    F3 --> F2
    F1 -->|owns| E1
    F3 -->|owns| E2
    E2 --> E3
    E2 -->|scale commands| E1
    E1 -->|calls| D3
    D2 --> G1
    E1 --> G1
    E2 --> G1
    G1 --> G2 --> G3

    style Auto fill:#2a3a5a,stroke:#7aa2ff,color:#fff
    style MultiRegion fill:#2a3a5a,stroke:#7aa2ff,color:#fff
    style Engine fill:#3a3a3a,stroke:#aaa,color:#fff
    style ReqExec fill:#3a3a3a,stroke:#aaa,color:#fff
    style AppLayer fill:#3a3a3a,stroke:#aaa,color:#fff
```

Blue boxes = SageServe additions. Grey boxes = inherited Splitwise machinery (some lightly extended, e.g. `scheduler.py` policies).

---

## 2. Class UML — Arbiter hierarchy (the reactive scaling policies, `arbiter.py`)

```mermaid
classDiagram
    class Arbiter {
        <<abstract>>
        +cluster
        +overheads
        +applications
        +allocators
        +results
        +add_application()
        +run()
        +allocate()
        +deallocate()
        +scale(model_endpoint)
    }

    class NoOpArbiter {
        Splitwise stub — does nothing
    }

    class BasicArbiter {
        +scaling_in_progress
        +next_scale_time
        +scale_up_from_spot()
        +scale_up_from_spot_other()
        +scale_down_to_spot()
        +spin_up_new_instance()
        +force_scale_up()
        +force_scale_down()
        +scale(model_endpoint)
        Reactive: up if mem_util > 0.5,
        down if mem_util less than 0.25
    }

    class GlobalArbiterAwareShortTermArbiter {
        +changes: dict
        +scale(model_endpoint)
        Consumes a scale-budget
        pushed by the long-term arbiter
        via force_scale_up/down
    }

    class GlobalAribiterMemoryUtilizationScaling {
        +scale_up_logic()
        +scale_down_to_spot()
        +scale(model_endpoint)
        Hybrid: only acts within budget
        AND when mem_util crosses
        threshold; emergency override
        at mem_util > 0.9
    }

    class GlobalArbiterARIMAChecking {
        +arima_forecast: DataFrame
        +threshhold
        +request_arrival_times: Queue
        +request_tokens: Queue
        +tokens_last_minute
        +set_arima_forecast(df)
        +manage_tps(request)
        +ratio_between_arima_and_actual()
        +scale(model_endpoint)
        FORECAST-AWARE: scales up if
        actual/predicted tokens-per-min
        ratio exceeds threshold
    }

    class ChironArbiter {
        +scale(model_endpoint)
        Batch(-d) endpoints: scale on
        pending_time vs ttft_batch.
        Mixed endpoints: scale on
        in-batch percentage (ibp)
    }

    Arbiter <|-- NoOpArbiter
    Arbiter <|-- BasicArbiter
    BasicArbiter <|-- GlobalArbiterAwareShortTermArbiter
    BasicArbiter <|-- GlobalAribiterMemoryUtilizationScaling
    BasicArbiter <|-- ChironArbiter
    GlobalAribiterMemoryUtilizationScaling <|-- GlobalArbiterARIMAChecking
```

**Reading this diagram:** every concrete class overrides `scale()`, which is called periodically per `model_endpoint` by the simulator. `BasicArbiter` is pure local reactive logic (Splitwise-style, but implemented for SageServe's spot/shared-pool scaling model). Everything below `GlobalArbiterAwareShortTermArbiter` in the tree is **forecast-aware**, in that it defers to (or blends with) a `changes` budget pushed down by the global long-term arbiter — `GlobalArbiterARIMAChecking` is the one that additionally does its **own** short-horizon forecast comparison every request.

---

## 3. Class UML — Global (long-term) arbiter and MILP allocator

```mermaid
classDiagram
    class GlobalArbiter {
        <<abstract>>
        +regions: int
        +models: int
        +model_to_idx_mapping
        +long_term_scaling_interval
        +allocation_interval
        +scaling_interval_reached() bool
    }

    class MilpGlobalArbiter {
        +arima_traces: path
        +forecast_df: dict
        +dev_df: dict
        +region_clusters
        +region_routers
        +milp_allocator: MilpLongTermAllocation
        +post_processing_strategy
        +arima_aware_arbiter: bool
        +load_predicted_traces()
        +current_allocations() List
        +get_ilp_forecast(path) List
        +post_process_ilp(forecast) List~Tuple~
        +schedul_scaling_events(changes)
        +scale()
    }

    class MilpLongTermAllocation {
        +models, regions, gpus
        +model_interchange_time
        +model_tps
        +gpu_cost
        +get_ilp_allocations(current, forecast, path) List
        Solves: minimize GPU cost
        subject to forecast token
        throughput per region/model
    }

    GlobalArbiter <|-- MilpGlobalArbiter
    MilpGlobalArbiter --> MilpLongTermAllocation : uses
    MilpGlobalArbiter --> "region N" RegionCluster : reads/commands via .arbiter
    MilpGlobalArbiter ..> ArimaForecastCSV : reads final_{model}_prod_1minute_region{r}_arima.csv

    class RegionCluster {
        +arbiter : Arbiter subclass
        +force_scale_up()
        +force_scale_down()
    }
```

`MilpGlobalArbiter.scale()` is the **forecast-aware long-term loop**: every `long_term_scaling_interval` seconds it reads current allocation + ARIMA forecast → MILP → converts the delta into `force_scale_up`/`force_scale_down` calls on each region's arbiter (from Diagram 2). Note the `post_processing_strategy` knob (`immediate`, `delay_changes`, `keep_maximum_instances`, `keep_minimum_instances`) — this is where a lot of the paper's "how aggressively to trust the forecast" experimentation lives.

---

## 4. Class UML — Core serving objects (mostly Splitwise-inherited)

```mermaid
classDiagram
    class Controller {
        +regions
        +global_router
        +set_global_router()
        +add_region()
    }

    class Region {
        +region_cluster
        +model_endpoint_routers
        +get_model_endpoint(name)
    }

    class RegionCluster {
        +servers
        +arbiter
        +has_spot_instance()
        +has_spot_instance_any()
        +free_processors_and_kill_spot()
        +get_spot_instance_count()
    }

    class ModelEndpointRouter {
        +model_name
        +applications: List~Application~
        +total_instances
        +pending_requests
        +pending_tokens
        +scaling_level
        +get_memory()
        +get_max_memory()
        +add_application()
        +remove_application()
    }

    class Application {
        +application_id
        +instances: List~Instance~
        +scheduler
        +allocator
        +router
        +add_instance()
        +remove_instance()
    }

    class Scheduler {
        <<Splitwise>>
        +schedule(request) Instance
    }

    class Allocator {
        <<Splitwise>>
        +start_spin_up_instance()
        +start_spin_down_instance()
        +start_reclaim_spot_instance()
    }

    class Instance {
        <<Splitwise>>
        +processors
        +memory / max_memory
        +cnt_iw_requests
        +application
        +execute(task)
    }

    class Request {
        <<Splitwise>>
        +prompt_size
        +arrival_time
        +tasks: List~Task~
    }

    class GlobalRouter {
        +regions
        +global_arbiter
        +add_region()
        +add_global_arbiter()
        +route(request) Region
    }

    Controller "1" o-- "many" Region
    Region "1" o-- "1" RegionCluster
    Region "1" o-- "many" ModelEndpointRouter
    RegionCluster "1" o-- "1" Arbiter : short-term policy
    ModelEndpointRouter "1" o-- "many" Application
    Application "1" o-- "many" Instance
    Application "1" o-- "1" Scheduler
    Application "1" o-- "1" Allocator
    Instance --> Request : executes tasks from
    GlobalRouter "1" o-- "many" Region
    GlobalRouter "1" o-- "1" GlobalArbiter
```

---

## 5. Sequence diagram — request lifecycle (Splitwise mechanics, SageServe entry point)

```mermaid
sequenceDiagram
    participant Trace as trace.py
    participant Sim as simulator.py (clock loop)
    participant GR as GlobalRouter
    participant RR as RegionRouter
    participant MER as ModelEndpointRouter
    participant Sch as Scheduler
    participant Inst as Instance
    participant Metrics as metrics.py

    Trace->>Sim: request arrival event @ t
    Sim->>GR: dispatch(request)
    GR->>RR: route to region (probabilistic)
    RR->>MER: route to model endpoint
    MER->>Sch: schedule(request)
    Sch->>Inst: assign to instance (queue if full)
    Inst->>Inst: execute prefill task, then decode task(s)
    Inst-->>MER: update pending_tokens / memory util
    Inst-->>Sim: schedule completion event
    Sim-->>Metrics: log per-request latency (TTFT/TBT/E2E)
    Metrics-->>Metrics: write detailed/{app_id}.csv, summary.csv
```

This part is essentially unmodified Splitwise — SageServe's changes don't touch how one request executes, only how many instances exist to receive it.

---

## 6. Sequence diagram — where SageServe's scaling logic actually fires

```mermaid
sequenceDiagram
    participant Sim as simulator.py
    participant MER as ModelEndpointRouter
    participant STA as Short-Term Arbiter<br/>(arbiter.py, per region)
    participant Cluster as RegionCluster
    participant GA as MilpGlobalArbiter<br/>(global_arbiter.py)
    participant MILP as MilpLongTermAllocation
    participant Forecast as ARIMA forecast CSVs

    Note over Sim,STA: Every request / periodic tick (short_term_scaling=True)
    Sim->>MER: request processed, memory_utilisation updated
    MER->>STA: scale(model_endpoint)
    alt BasicArbiter / reactive
        STA->>STA: check memory_utilisation vs 0.5 / 0.25
    else GlobalArbiterARIMAChecking
        STA->>Forecast: ratio_between_arima_and_actual()
        STA->>STA: compare ratio vs threshold
    end
    STA->>Cluster: has_spot_instance() / free_processors_and_kill_spot()
    STA->>MER: scale_up_from_spot() / scale_down_to_spot()

    Note over GA,MILP: Every long_term_scaling_interval (long_term_scaling=True)
    GA->>Cluster: current_allocations() [read instance counts]
    GA->>Forecast: load 40-min-ahead ARIMA prediction + 12h-lag dev demand
    GA->>MILP: get_ilp_allocations(current, forecast)
    MILP-->>GA: target allocation deltas per (region, model)
    GA->>GA: post_process_ilp() [immediate / delay_changes / keep_max / keep_min]
    GA->>STA: force_scale_up(model_endpoint) / force_scale_down(model_endpoint)
    Note right of STA: sets STA.changes[model] budget,<br/>consumed on next scale() call
```

This is the crux of the "reactive vs. forecast-aware" comparison: with `long_term_scaling=False`, only the top block runs (pure reactive, memory-utilization-triggered or ARIMA-ratio-triggered per request). With `long_term_scaling=True`, the bottom MILP loop also runs on a slower cadence and constrains/guides what the short-term arbiter is allowed to do.

---

## 7. Config → class resolution (Hydra wiring)

```mermaid
graph LR
    CLI["CLI override:<br/>controller.regions.0.arbiter=<br/>global_arbiter_ARIMA_checking"] --> Cfg[configs/arbiter/*.yaml]
    Cfg --> Repo[arbiter_repo.py<br/>name → class map]
    Repo --> Class[GlobalArbiterARIMAChecking<br/>instance, one per region]
    Class --> Region1[Region 0 RegionCluster.arbiter]

    CLI2["CLI override:<br/>global_arbiter.arima_traces=<br/>$PWD/traces/forecasts/"] --> Cfg2[configs/global_arbiter/milp.yaml]
    Cfg2 --> Repo2[global_arbiter_repo resolution]
    Repo2 --> Class2[MilpGlobalArbiter instance<br/>owned by GlobalRouter]
```

To add your own extension (new scheduling policy or forecasting model), you add a class to `arbiter.py` (or a new forecast source consumed in `global_arbiter.py`), register it in `arbiter_repo.py`, add a matching `configs/arbiter/<name>.yaml`, and select it the same way via CLI override — no changes needed to `run.py` or the engine.