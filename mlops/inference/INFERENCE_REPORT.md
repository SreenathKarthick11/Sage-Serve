# Understanding and running inference

This module analyzes **saved simulator measurements**. It does not load an LLM
or send inference requests. Think of it as the analyst reading a restaurant's
receipts and staffing records after an experiment.

The main flow is:

    detailed request CSVs + arbiter capacity CSVs + export inventory
        -> loader.load_results()
        -> metrics.build_report()
        -> report CSV + per-window CSV + provenance JSON
        -> diagnostic PNGs

Summary files identify exported windows and applications; their percentiles
are not averaged together. Latency percentiles come from pooled request rows.
Memory files are streamed to check whether live applications were omitted from
the detailed exports. They are not used to invent export timestamps.

## Run it

From the repository root, using the MLOps environment:

~~~bash
source mlops/mlops_venv/bin/activate

python mlops/inference/run_inference.py \
  --results experiments/results \
  --forecast-model reactive-baseline \
  --out mlops/inference/reports/static_demo.csv \
  --plots-dir mlops/inference/reports/static_demo_plots
~~~

The output paths in this example all stay inside mlops:

- static_demo.csv: one row per experiment and model.
- static_demo_windows.csv: one row per experiment, model and exported window.
- static_demo.metadata.json: complete saved configuration, input file hashes,
  diagnostics, thresholds, scoring references, weights and known limitations.
- static_demo_plots/: six diagnostic charts, each 1560 x 780 pixels.

--forecast-model supplies an **unverified label**. It does not select, run,
or evaluate a forecasting model.

Bad schemas, malformed CSVs, nonnumeric/infinite/negative required measurements,
and inconsistent timing values fail by default with a filename and reason.
Header-only CSVs are valid empty exports. Genuine missing cells remain unknown.
Use --allow-partial to explicitly skip invalid files or retain their usable
cells; every issue is recorded. A skipped application export is not an empty
application.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Requested report outputs were written. Unknown duration or unavailable rfws can still be present. |
| 1 | Input/settings validation or report writing failed. |
| 2 | A usable report was written, but inputs/request exports were incomplete or plot generation failed. |

A detected upstream omission can therefore produce exit code 2 even without
--allow-partial. The report remains useful as a description of the exported
subset. --no-plots avoids importing Matplotlib.

## Experiment identity and loading

A normal experiment ends at:

    <run>/<trace>/feed_async_<bool>/feed_async_granularity_<n>/
    scaling_level_<n>/scaling_interval_<n>/lts_<bool>_sts<bool>/arbiter_<name>/

The parent folder need not be literally named results. You can also point the
loader at a single arbiter directory. A numeric run and the LTS/STS component
must still be present in its ancestry.

experiment_id hashes the experiment's full relative path and saved Hydra
configuration. All configuration levels distinguish experiments. source_path
and relative_path identify the original input, and config_sha256 identifies
the saved .hydra/config.yaml. Full configuration is retained in JSON, including
settings not named in the CSV. IDs should be compared using the same loading
root; changing the scope of that root can change relative identities.

The CSV also retains run, trace, lts, sts, arbiter, feed_async,
feed_async_granularity, scaling_level and scaling_interval. Flags are stored
as their path strings. Model names may contain underscores. Region names with
underscores are resolved using the saved controller configuration; without
that configuration, the first underscore separates region from model.

Repeated runs into the same simulator output directory can overwrite or mix
exports. No analysis module can recover overwritten measurements. Keep each
experiment's source directory distinct.

## Latency, benchmark and volume columns

| Column | Meaning |
|---|---|
| n_requests | Observed request rows, including rows with some unknown measurements in partial input. Not the number of arrivals. |
| n_valid_response_times, n_valid_ttft_times, etc. | Valid sample count for each required measurement. |
| e2e_p50/p95/p99 | Pooled end-to-end response-time percentiles, in seconds. |
| queue_p50/p95/p99 | Pooled queue-time percentiles, in seconds. |
| ttft_p50/p95/p99 | Pooled time-to-first-token percentiles, in seconds. |
| avg_tbt_p50/p95/p99 | Percentiles of each request's average token interval. |
| queue_fraction_mean | Mean of queue_time / response_time for valid, positive-response rows. |
| cv_response | Sample standard deviation / mean of response time. |
| prompt_size_mean, prompt_size_cv | Mean and sample CV of per-request prompt sizes. |
| output_tokens_mean | Mean generated-token count per request. |
| latency_sample_status | no_samples, single_sample, small_sample (<100), or ok. This is a sample-size notice, not statistical certification. |
| viol_ttft, viol_avg_tbt | Fraction of valid measurements above fixed benchmark thresholds. |
| benchmark_joint_pass_fraction | Fraction passing both thresholds among rows with both measurements. |
| n_valid_benchmark_pairs | Denominator for joint compliance. |
| n_windows | Exported window inventory, including valid empty exports. Missing window numbers are diagnosed. |
| requests_per_window | Observed request rows / exported windows; a completion-volume proxy, not an arrival rate. |
| n_quiet_windows | Windows known to have zero completions in complete exports. |
| n_observed_empty_windows | Windows with zero observed rows, which might still have omitted applications. |
| completion_window_cv | CV of complete exported-window completion counts, including zeros. |
| request_data_complete | Whether discovered/expected application exports and required cells are complete. Not a guarantee that the simulator exported every completion. |
| input_error_count | Invalid inputs accepted in partial mode; affected experiments remain ineligible for rfws. |
| n_unexported_live_applications | Distinct live application groups seen in memory logs but absent from relevant detailed exports. |
| export_duration_seconds | Sum of explicitly supplied export durations; otherwise blank. |
| completed_requests_per_second | Completions / explicit duration, only when request coverage and duration are known. |

A percentile computed from a small sample is descriptive, with limited evidence
about the tail. CV uses sample standard deviation (ddof=1) and stays unknown
with fewer than two valid samples or a zero mean. Zero variability is not
substituted for insufficient evidence.

The simulator's TBT is:

    (response_time - time_to_first_token) / token_size

It is a request-level average proxy, **not the largest inter-token gap**.
The benchmark defaults are TTFT >10 seconds and average TBT >0.2 seconds.
Use --ttft-threshold and --tbt-threshold to change them. These are not original
per-request SLAs, which cannot be joined back without request IDs.

Violation fractions use valid measurements only. If some values are missing,
their valid counts and incomplete status remain visible; unknown values are
never counted as successful requests. Request-size CV does not describe
weekday/weekend arrival patterns or establish forecastability.

Compatibility aliases remain: tbt_p99 = avg_tbt_p99, viol_tbt = viol_avg_tbt,
demand_cv = prompt_size_cv, demand_mean = prompt_size_mean.

## Resource accounting and explicit boundaries

For every model and region, capacity snapshots are sorted across the complete
experiment, regardless of CSV partitioning. A recorded capacity state is held
until the next snapshot. For example, two instances at t=0, t=1800 and t=3600
establish **two instance-hours** during the observed hour.

With no explicit boundaries, integration stops at the final snapshot. It
never invents a median-gap tail or extrapolates a lone snapshot. Counts sharing
a timestamp use the last recorded state. Missing/corrupt intervening exports
prevent integrating across that gap.

| Column | Meaning |
|---|---|
| prod_vm_h, spot_vm_h | Observed production/spot instance-hours. May cover only part of the experiment; inspect status. Blank if no interval can be established. |
| resource_coverage_status | complete, partial, observed_intervals_only, insufficient_snapshots, or missing. |
| resource_observed_region_seconds | Sum of covered seconds across expected regions. |
| resource_time_coverage | Covered region-seconds / explicitly known total region-seconds; otherwise blank. |
| resource_snapshots | Number of loaded capacity snapshots. |
| memory_util_snapshot_mean | Event-weighted mean of the upstream memory_util field; not a verified, time-weighted per-model measurement. |
| vm_h_per_window | Instance-hours / exported windows, only with complete resource duration. |
| vm_h_per_1000_completions | Resource proxy per 1,000 exported completions, only with complete request and resource coverage. |

prod_vm_h/spot_vm_h retain their historical names. They are counts of simulated
instances multiplied by duration, not GPU-hours, measured cloud bills, or a
guarantee that different instance types have the same cost.
mem_util_mean remains an alias of memory_util_snapshot_mean.

For precise boundaries, supply --window-boundaries path/to/boundaries.csv:

~~~csv
experiment_id,window,start_seconds,end_seconds
<experiment_id from the report>,0,0,3600
<experiment_id from the report>,1,3600,7216.202420967854
~~~

These numbers illustrate the schema; use independently known export times.
Do not infer boundaries from window_number * 3600, formatted log messages,
or configured end_time. The simulator can save on the next event after an
hour and can stop without flushing its last partial window. The CSV can omit
experiment_id only when loading one experiment. Bounds must be finite,
nonnegative, ordered, nonoverlapping, and refer to actual exported windows.
Partial boundary inventories leave duration unknown.

Known end boundaries permit holding the last recorded state to the end of an
export. An unknown initial state still leaves partial coverage. Even complete
coverage is conditional on the recorded-state model and upstream logging
correctness; input histories cannot prove that every transition was logged.

## What relative rfws means now

Raw metrics are the primary results. rfws is a legacy name for an optional
**relative composite**, not a forecasting accuracy metric or an absolute
quality grade. Schema/scoring version 2 is not comparable to old reports.

Peers share model, trace and window inventory (including supplied boundaries).
Models A and B are not peers simply because both appear in one report.
At least two eligible experiments are required. Incomplete request measurements,
insufficient samples, or unknown enabled components make a row ineligible.
Cost requires complete resource duration. Status explains every missing score.

The components are:

| Component | Raw input | Default relative weight |
|---|---|---|
| slo | 1 - mean(TTFT violation fraction, average-TBT violation fraction) | .35 |
| stability | 1 / (1 + response-time CV) | .25 |
| demand | 1 / (1 + per-request prompt-size CV) | .20 |
| cost | Inverted rank of instance-hours per exported window | .10 |
| throughput | Rank of completions per exported window | .10 |

All components are min-max normalized among eligible peers in the same cohort.
Ties among multiple peers receive 0.5. Missing values remain missing.
Finite nonnegative weights are normalized to sum to one; an all-zero set is
rejected. A component can be disabled with weight zero. For example,
--cost-weight 0 explicitly removes resource cost and renormalizes other weights;
it does not repair incomplete request exports.

contribution_slo, contribution_stability, contribution_demand, contribution_cost
and contribution_throughput are weighted points whose sum equals rfws.
weight_* records the actual normalized weights. score_slo/score_stability/
score_demand are raw diagnostic components; score_cost/score_throughput are
cohort-relative ranks.

rfws_cohort_id fingerprints the eligible member IDs, component measurements,
thresholds, weights and normalization reference. The complete reference is
saved in metadata.rfws_cohorts. Adding a peer or changing evidence can change
relative scores and the cohort ID. Compare rfws only within the **same cohort**.
Do not concatenate independently scored reports and interpret score differences
as evidence that ARIMA is better than Prophet. Use raw metrics for separate
reports, or load aligned experiments together and score them in one cohort.
Matching a trace label still assumes the input workload artifacts are comparable.

## Recommended: four policy experiments per model

Run `run_rfws_matrix.py` once for this small controlled study:

| Case | Scaling level | STS | LTS | Question |
|---|---|---|---|---|
| Baseline | 2 | Off | Off | What happens with neither policy enabled? |
| STS only | 2 | On | Off | What changes when STS is enabled? |
| LTS only | 2 | Off | On | What changes when LTS is enabled? |
| STS + LTS | 2 | On | On | How do the policies behave together? |

This replaces the previous 12-case instructions. Each simulator invocation
includes models A-D, so **four simulator runs produce 16 report rows: four per
model**. They are not 16 separate simulator runs. Scaling levels 0 and 1 are
excluded: holding level 2 fixed isolates the policy flags and allows resource
transfers, which level 0 disables. The smaller study cannot measure the effect
of changing scaling level.

Execute this yourself from the repository root; editing the runner does not
execute the experiments:

~~~powershell
Set-Location D:\Aakash_IIT\MLOps\Sage-Serve
.\mlops\.venv\Scripts\python.exe -X utf8 mlops/inference/run_rfws_matrix.py --name s01
~~~

The runner selects the latest validated static trace and the simulator's
separate virtual environment automatically. For another study, use a fresh
short name such as `s02`; existing names are rejected and are never reused.
Optional `--trace PATH` and `--sim-python PATH` select explicit inputs.

One invocation:

1. Splits history before original t=600 from evaluation traffic, then prepares
   a frozen set of 12 forecast files (three regions times four models) using
   historical traffic only. These are input files, not additional experiments.
2. Runs the four cases sequentially with the same starting allocation of eight
   instances per model/region, seed, routing policy, trace and 3900-second
   simulated duration. This duration reaches the LTS update at t=3600.
3. Checks exit codes, completion markers, saved Hydra configurations, input
   hashes and detailed exports. It saves actual commands, timestamps and export
   hashes; a failure stops the study and preserves its evidence.
4. Analyzes only this study's exports. Checks require all four policy cases for
   each of A-D, 16 eligible rows and four cohorts containing four rows each.
5. Generates six 1560 x 780 PNGs. The two RFWS images show four model panels,
   with four policy bars per panel. The remaining diagnostics use the same
   16 report rows. Select `s01.csv` in the web UI after completion.

Outputs for study NAME, all under `mlops/inference/reports/`:

| Output | Purpose |
|---|---|
| NAME/manifest.json | Planned cases, overall status and per-case evidence. |
| NAME/prepared_inputs.json | Historical/evaluation split and frozen input hashes. |
| NAME/logs/l2_sN_tN.log | Actual simulator stdout and stderr for one case. |
| NAME/logs/l2_sN_tN.json | Command, timestamps, exit code and completion checks. |
| NAME/logs/l2_sN_tN_exports.json | Hashes and sizes of saved exports. |
| NAME/metric_variation.csv | Distinct enabled scoring inputs within each model. |
| NAME.csv, NAME_windows.csv, NAME.metadata.json | Metrics, completions table and provenance. |
| NAME_plots/rfws_bar.png | Four policy scores in each model's panel. |
| NAME_plots/sub_scores.png | Weighted contributions to those scores. |

Plot labels use L for scaling level, S for STS and T for LTS; 0 means off and
1 means on. RFWS comparisons apply within each model's own cohort. Cost weight
is zero because total resource duration is not established by these exports.
Missing resource intervals remain unknown. This study uses a short-history
ARIMA development forecast; it does not establish forecasting accuracy.

Different policies can still yield equal measurements for this workload.
The scorer assigns tied components 0.5, yielding RFWS 50 if every enabled
component ties. The runner does not change thresholds, measurements or scoring
math to force different bars. Compare `metric_variation.csv`, request exports
and per-case logs before interpreting equal scores.

Exit codes: 0 means all four cases and the eligible 16-row analysis completed;
1 means failure; 2 means analysis was saved with partial or ineligible output.
Read the manifest, logs and `rfws_status` when a study fails. The runner does
not resume failed studies: fix the cause and use a fresh short name. It checks
representative nested paths before launch to avoid Windows path-length errors.

The optional `--archive-current` flag only archives older `r4` and
`rfws_matrix_*`/`fss_matrix_*` output families. It does not clear every existing
study. Each new study remains isolated regardless of other saved outputs.

## Allocation evidence and the Results UI

New matrix runs automatically extract allocation evidence after analysis.
For an existing study, run this from the repository root (no simulation or
forecast fitting is performed):

~~~powershell
.\mlops\.venv\Scripts\python.exe -X utf8 mlops/inference/allocation.py --report mlops/inference/reports/s01.csv
~~~

Extraction writes into `reports/s01/allocation/`:

| File | Contents |
|---|---|
| allocation_decisions.csv | Logged current counts, requested changes and calculated targets, per model/region. |
| allocation_observations.csv | Actual saved arbiter production/spot snapshots; last exported row retained at duplicate timestamps. |
| allocation_coverage.csv | Snapshot counts and first/last observation times for every model/region/case. |
| allocation_hour_comparison.csv | Before/after observations around the first LTS decision at or after 3600 seconds. |
| allocation.metadata.json | Source hashes, report hash, extraction issues and interpretation limits. |

Refresh the Results page and select `s01.csv`, then choose an allocation
region (westus by default). Four model charts show that region's production
instance counts. Each model chart has four separate policy lanes on white
backgrounds, so equal counts do not overlap. All regional snapshots are kept;
no synchronization with other regions is required. In s01, initial counts
are eight per model for each region, rather than the 24-instance regional sum.

At time zero, counts come from the saved initial configuration and observed
change is zero. Later points come from arbiter snapshots or current counts
logged immediately before optimizer decisions. Hover shows the region, count,
LTS requested delta/target and count difference since the previous known state.
LTS requested change is Not applicable for baseline and STS-only because LTS
is disabled; an initial state or snapshot without an optimizer decision is
marked No decision at this sample. Zero is shown for a requested delta only
when an actual logged decision requests zero change.

All model charts share time and count scales within the selected region.
Coincident pre-decision and exported states remain separate events. Differences
between known counts can include STS and do not prove execution of an optimizer
request. Baseline has an initial point but no invented later measurements.
Dashed lines connect known samples as visual guides; they do not establish when
changes occurred between observations and never extend beyond the last count.

The allocation section contains four charts and the region selector, without
supporting tables or coverage captions. Existing evidence CSVs remain available
on disk for auditing. Configured initial counts are stored in
allocation.metadata.json; re-extract older studies if initial counts are absent.
Counts are model instances, not individual GPUs. In s01, counts recorded just
before the second LTS decision are available even though post-decision exports
are not. No simulator rerun is needed to switch between saved regions.

This extraction supports single-GPU-type allocation matrices from the current
simulator and uses saved Hydra configuration to map region/model indices.
Missing mappings, malformed decision matrices and invalid resource files are
reported in extraction metadata rather than filled with invented values.

## Plots

All plots use numeric positions and experiment identifiers, so configurations
cannot overlap just because their model/LTS names match. Unknown values are
omitted or explicitly labelled; they are never replaced by apparent zeros.
Every plot uses a 12 x 6 inch canvas at 130 DPI (1560 x 780 pixels). Report row
counts do not change image dimensions. Compact labels use the model, run and
experiment ID; the complete trace/configuration remains in the report table.

| File | Purpose |
|---|---|
| rfws_bar.png | Relative scores or explicit reasons that scores are unavailable. |
| sub_scores.png | Actual weighted rfws contributions, summing to the final score. |
| latency_pct.png | Pooled p50/p95/p99 end-to-end latency. |
| slo_violations.png | Fixed benchmark violation fractions; average TBT is labelled precisely. |
| cv_scatter.png | Response-time variability versus prompt-size variability; no forecastability claim. |
| resource_usage.png | Production/spot instance-hours with resource-coverage status. |

The web UI's Results page displays data coverage as an interactive table for the
selected report, above the diagnostic plots. It includes valid measurement counts
and percentages, exported windows, missing application exports, request-export
completeness, resource coverage and known export duration. Percentages use observed
request rows; zero-sample or unavailable values remain blank. Existing reports
work without regeneration, and legacy `data_coverage.png` files are hidden in the UI.

Completions by window are also displayed as a full-width interactive table, read
from the companion `_windows.csv`. Each row identifies an experiment and export
window, with one completion-count column per model for parallel comparison. Zero
denotes a confirmed quiet window; incomplete or unavailable counts remain blank.
An expandable detail table retains observed subsets from incomplete exports.
Window indexes are not timestamps. A report without its companion window file
displays an availability message rather than inferred counts.

The six plots appear in three two-column rows: score/contributions,
latency/violations, and resources/variability. Legacy `completions_by_window.png`
files are hidden. Regenerate plots from an existing report to apply uniform
dimensions to older images; a simulator rerun is not required.

## Optional forecast error evaluation

This is a separate analysis because completed-request measurements are not the
arrival-demand series a forecaster predicts. Supply the actual input trace and
forecast directory explicitly:

~~~bash
python mlops/inference/forecast_evaluation.py \
  --trace data/static/clean/<version>/validated_trace.csv \
  --forecasts data/static/forecasts/<version> \
  --evaluation-start 3600 \
  --observation-end 4980 \
  --out mlops/inference/reports/forecast_accuracy.csv
~~~

Use an observation end supported by your trace collection interval. The bounds
must be exact minute boundaries; 4980 above is an example. The evaluation
interval is [start, end). Default start excludes the ARIMA implementation's
first-hour warm-up. Future forecasts outside that interval are not scored.

The target matches forecasting/arima_forecaster.py: sum prompt_size in
60-second buckets, per model and **first-priority source region**, using all
trace rows. Quiet minutes receive zero observed demand; absent predictions
remain unknown. This is not demand at the eventual serving region. Configured
evaluation bounds assert which complete minutes were observed.

Outputs include MAE, RMSE, bias, WAPE, prediction coverage and skill relative
to a one-minute persistence baseline on the same available buckets.
WAPE is unknown when total observed demand is zero; skill is unknown when
baseline error is zero. A companion _buckets.csv contains aligned values and
.metadata.json records artifact hashes and evaluation definitions.

These forecast files have no forecast-origin timestamps. This evaluator
cannot prove absence of future-data leakage, or prove a simulation consumed
the supplied artifacts. Those facts remain explicitly unverified. Full
per-request SLA evaluation and worst-token-gap evaluation also require
upstream fields that are not currently exported.

## Known upstream limits

Changes here do not modify SageServe-original:

1. Applications created during scaling can be omitted from detailed exports.
   The memory identifier audit detects omissions visible in memory snapshots;
   a missing snapshot cannot prove that an application never existed.
2. The simulator does not flush final partial windows. Requests beyond the last
   export cannot be reconstructed; n_requests is not total completed workload.
3. Arbiter memory utilization can be assigned to the wrong model. It is retained
   as a snapshot diagnostic and excluded from scoring.

Missing memory files limit omission detection. request_data_complete certifies
the available/discovered export inventory, not the upstream exporter.

## Tests and Python API

The existing inference tests live in mlops/tests/test_inference.py. From the
repository root, activate the MLOps environment and run:

~~~bash
cd mlops
python -B -m unittest tests.test_inference -v
~~~

Tests use small temporary fixtures and do not rerun the simulator.

~~~python
from inference.metrics import build_report, compute_metrics

report = build_report("experiments/results")
report.metrics       # main table
report.windows       # full exported-window inventory and observed subsets
report.metadata      # diagnostics, provenance, normalization and limitations

df = compute_metrics("experiments/results")  # compatibility API; emits warnings
df.attrs["metadata"]
~~~
