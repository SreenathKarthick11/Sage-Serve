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
| 0 | Requested report outputs were written. Unknown duration or unavailable FSS can still be present. |
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
| input_error_count | Invalid inputs accepted in partial mode; affected experiments remain ineligible for FSS. |
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

## What relative FSS means now

Raw metrics are the primary results. FSS is a legacy name for an optional
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
and contribution_throughput are weighted points whose sum equals fss.
weight_* records the actual normalized weights. score_slo/score_stability/
score_demand are raw diagnostic components; score_cost/score_throughput are
cohort-relative ranks.

fss_cohort_id fingerprints the eligible member IDs, component measurements,
thresholds, weights and normalization reference. The complete reference is
saved in metadata.fss_cohorts. Adding a peer or changing evidence can change
relative scores and the cohort ID. Compare FSS only within the **same cohort**.
Do not concatenate independently scored reports and interpret score differences
as evidence that ARIMA is better than Prophet. Use raw metrics for separate
reports, or load aligned experiments together and score them in one cohort.
Matching a trace label still assumes the input workload artifacts are comparable.

## Controlled FSS experiments: run manually in PowerShell

This recipe is an experiment design, not a completed benchmark. Run the commands
yourself, in order, in the same PowerShell terminal. No expected winner or score
is assumed. It uses the existing FSS names; an FWS rename is a separate change.

### Study design

Compare the same validated static trace under three resource-sharing policies:

| Experiment | scaling_level | Short-term scaling | Long-term scaling | Policy |
|---|---|---|---|---|
| Control | 0 | True | False | Scaling decisions run and log state, but resource transfers are disabled. |
| Same-model reclaim | 1 | True | False | Reclaim that model's spot instances; permit supported scale-down to spot. |
| Cross-model reclaim | 2 | True | False | Also allow reclaiming spot capacity held by other models. |

All three keep feed_async=False, scaling_interval=-1, simulator seed=0,
controller=us3-new, siloed=False and the same initial cluster configuration.
The simulator reads one identical clean trace file for all three experiments.
The trace label is identical across peers, and distinct scaling-level paths
prevent overwriting their outputs. No forecasting-model registration or forecast
files are needed because long-term scaling is disabled throughout this study.

The simulation limit is 3900 seconds. The source trace must contain non-batched
production arrivals beyond that limit. The simulator should therefore reach its
first hourly export before stopping. The analysis describes persisted exports,
not all input requests or the unflushed final partial window. Hourly exports
occur on simulation events and are not guaranteed to have identical exact
durations. Matching window indexes meet the current scorer's inventory check;
without independently verified boundaries, interpret this as an exploratory
comparison of exported windows, not a duration-controlled throughput benchmark.

Use cost weight zero for this initial study. The current exports do not establish
complete resource-duration coverage, so keeping the default cost weight would
exclude otherwise usable rows. The remaining normalized weights are compliance
38.89%, stability 27.78%, prompt regularity 22.22%, and completions per window
11.11%. This must be described as **cost-disabled FSS**, not the full five-component
score. Raw observed resource measurements remain in the report for inspection.

Keep the default TTFT/TBT thresholds (10 s and 0.2 s) for all peers. Similar or
identical scores are valid outcomes: low traffic may never require cross-model
reclaim, and several components can tie. Do not change measurements or tune
thresholds after seeing the results simply to produce different bars. The
existing single-run report remains a separate baseline, outside this study.

### 1. Set paths and identify the validated trace

The environments and dependencies must already be installed. Change the first
path to your checkout location. These commands only prepare the study; step 3
starts the simulations. New study outputs stay under mlops/inference/reports.

~~~powershell
Set-Location D:\Aakash_IIT\MLOps\Sage-Serve

$repo = (Get-Location).Path
$mlopsPython = Join-Path $repo 'mlops\.venv\Scripts\python.exe'
$simPython = Join-Path $repo 'SageServe-original\.venv\Scripts\python.exe'

$cleanRoot = Join-Path $repo 'data\static\clean'
$cleanPointer = Get-Content (Join-Path $cleanRoot 'latest.json') -Raw | ConvertFrom-Json
$cleanVersion = $cleanPointer.version
$tracePath = (Resolve-Path (Join-Path $cleanRoot "$cleanVersion\validated_trace.csv")).Path
$tracePath = $tracePath.Replace('\', '/')
$traceHash = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash
$traceLabel = "static_clean_$cleanVersion"

$studyName = 'fss_study_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$studyDir = Join-Path $repo "mlops\inference\reports\$studyName"
if (Test-Path -LiteralPath $studyDir) { throw 'Choose a fresh study name.' }
New-Item -ItemType Directory -Path $studyDir | Out-Null
$resultsRoot = Join-Path $studyDir 'simulator_outputs'

$env:PYTHONUTF8 = '1'
$env:HYDRA_FULL_ERROR = '1'

@'
import sys
import pandas as pd

trace = pd.read_csv(sys.argv[1])
interactive = trace[(trace.workload_type == 'prod') & (trace.batch_id == -1)]
if interactive.empty or interactive.arrival_timestamp.max() <= 3900:
    raise SystemExit('Use a validated trace with non-batched production arrivals beyond 3900 s.')
print('Trace rows:', len(trace))
print('Non-batched production rows:', len(interactive))
print('Last non-batched production arrival:', interactive.arrival_timestamp.max())
'@ | & $mlopsPython -X utf8 - $tracePath
if ($LASTEXITCODE -ne 0) { throw 'Trace preflight failed. Stop here.' }

$studyManifest = [ordered]@{
    study = $studyName
    trace_path = $tracePath
    trace_sha256 = $traceHash
    clean_version = $cleanVersion
    trace_label = $traceLabel
    git_commit = (git rev-parse HEAD)
    simulator_seed = 0
    end_time_seconds = 3900
    scaling_levels = @(0, 1, 2)
    cost_weight = 0
    ttft_threshold_seconds = 10
    average_tbt_threshold_seconds = 0.2
}
$studyManifest | ConvertTo-Json -Depth 5 |
    Set-Content -Encoding utf8 (Join-Path $studyDir 'study_manifest.json')
~~~

Save the displayed study name/path if you close the terminal. The manifest is a
UTF-8 study record; Windows PowerShell can add a BOM, so Python readers should
open it with encoding='utf-8-sig'. Do not change the trace or simulator setup
between the three runs.

### 2. Define a simulator command for one policy

This function records each run's console output, checks the trace hash again,
and refuses to reuse an existing experiment directory. Defining it does not
start a simulation. The numeric run directory and trace/policy components are
kept in the layout expected by the inference loader.

~~~powershell
function Invoke-FssExperiment {
    param([ValidateSet(0, 1, 2)][int]$ScalingLevel)

    if ((Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash -ne $traceHash) {
        throw 'The trace changed. Start a fresh matched study.'
    }

    $relativeOutput = "0/$traceLabel/feed_async_False/feed_async_granularity_1/" +
        "scaling_level_$ScalingLevel/scaling_interval_-1/" +
        'lts_False_stsTrue/arbiter_basic_arbiter'
    $outputDir = (Join-Path $resultsRoot $relativeOutput).Replace('\', '/')
    if (Test-Path -LiteralPath $outputDir) {
        throw "Experiment already exists: $outputDir. Do not overwrite it."
    }
    $consoleLog = Join-Path $studyDir "scaling_level_$($ScalingLevel)_console.log"
    $simArguments = @(
        "trace.path=$tracePath"
        "trace.filename=$traceLabel"
        "output_dir=$outputDir"
        'controller=us3-new'
        'global_router=probabilistic'
        'seed=0'
        'feed_async=False'
        'short_term_scaling=True'
        'long_term_scaling=False'
        "scaling_level=$ScalingLevel"
        'scaling_interval=-1'
        'siloed=False'
        'end_time=3900'
    )

    Push-Location (Join-Path $repo 'SageServe-original')
    try {
        & $simPython -X utf8 run.py @simArguments 2>&1 |
            Tee-Object -FilePath $consoleLog
        $simExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($simExitCode -ne 0) {
        throw "Simulator failed for scaling level $ScalingLevel. Read $consoleLog"
    }
    $detailFiles = @(Get-ChildItem -LiteralPath "$outputDir/detailed" -Filter '*.csv' `
        -File -Recurse -ErrorAction SilentlyContinue)
    if ($detailFiles.Count -eq 0) {
        throw 'No detailed exports were saved. Stop and inspect this run.'
    }
    Write-Output "Saved scaling level $ScalingLevel to $outputDir"
}
~~~

### 3. Run the three experiments yourself

Run each command separately, waiting for completion and resolving any failure
before continuing. The expected policy difference is whether and where resources
can be reclaimed; the measurements determine whether that improves performance.

~~~powershell
# Control: no production/spot resource transfers.
Invoke-FssExperiment -ScalingLevel 0
~~~

~~~powershell
# Reclaim spot instances belonging to the same model.
Invoke-FssExperiment -ScalingLevel 1
~~~

~~~powershell
# Also allow cross-model reclaim of spot capacity.
Invoke-FssExperiment -ScalingLevel 2
~~~

### 4. Audit eligibility before producing comparison charts

Load all three experiments together using this one results root. That preserves
consistent experiment IDs and lets the scorer form one peer group per model.
Do not score each experiment separately and then concatenate its FSS column.

~~~powershell
Set-Location $repo
$inferenceScript = Join-Path $repo 'mlops\inference\run_inference.py'
$eligibilityPath = Join-Path $studyDir 'eligibility.csv'

& $mlopsPython -X utf8 $inferenceScript `
    --results $resultsRoot `
    --out $eligibilityPath `
    --forecast-model 'reactive-policy-comparison' `
    --cost-weight 0 `
    --ttft-threshold 10 `
    --tbt-threshold 0.2 `
    --no-plots
$auditExitCode = $LASTEXITCODE
if ($auditExitCode -ne 0) {
    throw 'Eligibility analysis failed or is partial. Read its diagnostics before continuing.'
}

$eligibility = Import-Csv -LiteralPath $eligibilityPath
$eligibility | Select-Object model, scaling_level, n_requests, n_windows, `
    request_data_complete, resource_coverage_status, fss, fss_status |
    Format-Table -AutoSize

$unscored = @($eligibility | Where-Object { $_.fss_status -ne 'relative_to_cohort' })
if ($unscored.Count -gt 0) {
    throw 'Some rows are ineligible. Inspect eligibility.metadata.json; do not fill missing scores with zero.'
}
$cohorts = @($eligibility | Group-Object fss_cohort_id)
if ($eligibility.Count -ne 12 -or $cohorts.Count -ne 4 -or
    @($cohorts | Where-Object { $_.Count -ne 3 }).Count -gt 0) {
    throw 'Expected four model cohorts with three policy experiments each. Check model and window inventories.'
}
~~~

For this four-model study, successful eligibility means 12 rows with finite FSS,
four cohort IDs, and three policy experiments in each cohort. Complete exports
can still omit the final partial simulation interval. Models with no completed
samples or experiments with detected omitted applications can remain ineligible.
Changing cost weight does not repair those issues. Do not use --allow-partial
as a way to force scores. Preserve failures and investigate their diagnostics.

### 5. Generate the six charts and the UI report

This repeats analysis, not simulation, with the same root, weights and thresholds.
The final CSV is written directly into the web UI's report directory so it appears
in the Results report selector. Study source files and the audit stay in its
subdirectory and do not clutter that selector.

~~~powershell
$reportPath = Join-Path $repo "mlops\inference\reports\$studyName.csv"
$plotsDir = Join-Path $repo "mlops\inference\reports\$($studyName)_plots"

& $mlopsPython -X utf8 $inferenceScript `
    --results $resultsRoot `
    --out $reportPath `
    --plots-dir $plotsDir `
    --forecast-model 'reactive-policy-comparison' `
    --cost-weight 0 `
    --ttft-threshold 10 `
    --tbt-threshold 0.2
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed or is partial. Read the diagnostics.' }

Import-Csv -LiteralPath $reportPath |
    Select-Object model, scaling_level, n_requests, e2e_p99, viol_ttft, `
        viol_avg_tbt, requests_per_window, fss, fss_cohort_id |
    Format-Table -AutoSize

# Only needed if the web UI is not already running.
$env:MLOPS_PYTHON = $mlopsPython
$env:SIM_PYTHON = $simPython
& $mlopsPython -X utf8 -m streamlit run mlops/web-app/app.py
~~~

In Results, select the CSV named after this study. Both the FSS bar and stacked
contribution chart should contain values for eligible rows; the cost contribution
is zero by design. Compare policies within the same model/cohort, even though
all four models are displayed together. Check raw metrics alongside the score.
Identical measurements can legitimately yield identical scores, including 50
when every enabled component ties. This recipe does not establish statistical
significance or general performance across workloads.

### Extending the study

- Repeat with other independently versioned, validated workloads as separate
  studies. Keep each workload identical within its matched policy comparison.
- A full five-component FSS study requires independently verified export
  boundaries and complete recorded resource states for every model and region,
  including the initial state. Supply those through --window-boundaries and
  restore the cost weight only after resource_coverage_status is complete. Do
  not manufacture t=0 snapshots or derive exact boundaries from window indexes.
- A forecast-aware comparison needs forecast files covering the simulator's
  requested planning times. The current end-of-trace forecaster does not provide
  that alignment when replaying the same trace from t=0. Coordinate aligned
  forecasts/exports before adding long-term scaling as another policy; changing
  --forecast-model only changes a label and cannot establish forecast usage.
- Source-code changes to the simulator or forecasting modules are not part of
  this inference study. Keep simulator inputs, console logs, study manifest and
  report metadata with the results so other collaborators can audit the evidence.

## Full factorial study: 12 scaling-level / STS / LTS combinations

This is a separate study from the three-run recipe above. Run the preparation
and simulator commands yourself; these instructions do not represent executed
or verified benchmark results. Use one PowerShell terminal and a fresh matrix
directory. No simulator or forecasting source files need to be edited.

| Case | Scaling level | STS | LTS |
|---|---|---|---|
| E01 | 0 | False | False |
| E02 | 0 | True | False |
| E03 | 0 | False | True |
| E04 | 0 | True | True |
| E05 | 1 | False | False |
| E06 | 1 | True | False |
| E07 | 1 | False | True |
| E08 | 1 | True | True |
| E09 | 2 | False | False |
| E10 | 2 | True | False |
| E11 | 2 | False | True |
| E12 | 2 | True | True |

There are 12 configuration cells, not necessarily 12 different behaviors.
At level 0, both STS and LTS resource-transfer operations are disabled by the
arbiter. With both STS and LTS off, changing the level also does not enable
transfers. Identical outputs from these controls are expected possibilities.

Controls specific to this repository:

- All runs start with eight instances for each of four models in each of three
  regions, using CLI overrides. LTS's start_with_8_each() then makes no initial
  reduction, so it cannot silently introduce a different starting allocation.
- global_router.no_reroute=False is fixed in all runs. Otherwise, the router's
  default enables a different region-selection branch only when LTS is on.
- Keep controller=us3-new and its basic_arbiter for all 12 runs. This tests the
  basic reactive policy alongside the MILP global planner. Forecast-aware
  regional arbiters are additional factors, outside this 12-case matrix.
- Keep feed_async=False, seed=0, scaling_interval=-1, siloed=False and
  end_time=3900 fixed. Use a frozen forecast, with a 3600-second LTS interval.
- The first 600 seconds of the validated source trace form historical training
  data. The remaining trace is held out and rebased so original t=600 becomes
  simulation t=0. Every run replays exactly this same evaluation trace.
- Fit ARIMA on historical non-batched production prompt demand only, matching
  the traffic fed when feed_async=False. This experiment-specific target differs
  from the generic pipeline forecaster's all-row target. Use 60-second buckets,
  ARIMA(1,1,1), ten historical buckets, and 180 forecast buckets. No held-out
  measurements enter fitting. The short history is suitable for an initial
  controlled demonstration, not evidence of a validated forecasting model.
- Forecast CSV Time starts at zero relative to the held-out simulation origin.
  The 180-minute horizon covers every planner look-ahead window through the
  3900-second simulation limit. All LTS cases consume the same 12 forecast files.
- Use cost-disabled FSS for the same export-coverage reasons as the first study.
  Export windows remain exploratory comparisons without verified exact bounds.

### Matrix step 1: prepare held-out inputs and aligned forecasts

This block generates inputs and fits the development forecaster when YOU run it.
It does not register or promote a model, and does not start any simulator run.
All new inputs, forecasts, solver files and logs stay under the inference reports
directory. Change the checkout path on the first line if needed.

~~~powershell
Set-Location D:\Aakash_IIT\MLOps\Sage-Serve
$repo = (Get-Location).Path
$mlopsPython = Join-Path $repo 'mlops\.venv\Scripts\python.exe'
$simPython = Join-Path $repo 'SageServe-original\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:HYDRA_FULL_ERROR = '1'

$cleanRoot = Join-Path $repo 'data\static\clean'
$cleanPointer = Get-Content (Join-Path $cleanRoot 'latest.json') -Raw | ConvertFrom-Json
$sourceTrace = (Resolve-Path (Join-Path $cleanRoot "$($cleanPointer.version)\validated_trace.csv")).Path
$matrixName = 'fss_matrix_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$matrixDir = Join-Path $repo "mlops\inference\reports\$matrixName"
if (Test-Path -LiteralPath $matrixDir) { throw 'Choose a fresh matrix name.' }
New-Item -ItemType Directory -Path $matrixDir | Out-Null
$matrixResults = Join-Path $matrixDir 'simulator_outputs'

@'
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml

repo, source, study = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / 'mlops'))
from etl_module.validation import validate_raw_trace
from forecasting.arima_forecaster import ARIMAForecaster

trace = pd.read_csv(source, dtype={'regions': 'string', 'model_type': 'string'})
history = trace[trace.arrival_timestamp < 600].copy()
evaluation = trace[trace.arrival_timestamp >= 600].copy()
evaluation['arrival_timestamp'] -= 600
fed_history = history[(history.workload_type == 'prod') & (history.batch_id == -1)]
fed_evaluation = evaluation[(evaluation.workload_type == 'prod') & (evaluation.batch_id == -1)]
if fed_history.empty or fed_evaluation.empty or fed_evaluation.arrival_timestamp.max() <= 3900:
    raise SystemExit('Need history before 600 s and production arrivals beyond original 4500 s.')
if set(fed_evaluation.model_type) != {'A', 'B', 'C', 'D'}:
    raise SystemExit('This matrix expects evaluation traffic for all four models A-D.')
validate_raw_trace(history)
validate_raw_trace(evaluation)

inputs = study / 'inputs'
forecasts = inputs / 'forecasts'
forecasts.mkdir(parents=True)
history.to_csv(inputs / 'history.csv', index=False)
eval_path = inputs / 'evaluation.csv'
evaluation.to_csv(eval_path, index=False)
config = {
    'model_info': {'type': 'arima', 'description': 'Frozen historical holdout forecast'},
    'parameters': {'order': {'p': 1, 'd': 1, 'q': 1}, 'train_window_mins': 10,
                   'future_horizon': 180, 'bucket_seconds': 60},
}
config_path = inputs / 'arima.yaml'
config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
forecaster = ARIMAForecaster(str(config_path))
source_regions = fed_history.regions.str.zfill(3).str[0].astype(int)
minutes = (fed_history.arrival_timestamp // 60).astype(int)
for model in ('A', 'B', 'C', 'D'):
    for region in range(3):
        subset = fed_history[(fed_history.model_type == model) & (source_regions == region)]
        series = subset.groupby(minutes.loc[subset.index]).prompt_size.sum()
        series = series.reindex(range(10), fill_value=0).to_numpy(dtype=float)
        predictions = forecaster.predict(series)
        pd.DataFrame({'Time': np.arange(180) * 60, 'Predicted': predictions}).to_csv(
            forecasts / f'forecast_{model}_region{region}.csv', index=False)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    'source_trace': str(source.resolve()), 'source_sha256': sha256(source),
    'evaluation_trace': str(eval_path.resolve()), 'evaluation_sha256': sha256(eval_path),
    'trace_label': 'holdout_' + sha256(eval_path)[:12],
    'forecast_directory': str(forecasts.resolve()),
    'forecast_hashes': {p.name: sha256(p) for p in sorted(forecasts.glob('forecast_*.csv'))},
    'history_interval_original_seconds': [0, 600],
    'evaluation_origin_original_seconds': 600,
    'forecast_time_origin': 'held-out simulation t=0',
    'forecast_target': 'non-batched production prompt tokens per minute, by source region',
    'forecaster': forecaster.get_metadata(),
    'forecast_accuracy_validated': False,
    'initial_instances_per_model_region': 8,
    'controller': 'us3-new', 'regional_arbiter': 'basic_arbiter',
    'no_reroute': False, 'simulator_seed': 0, 'end_time_seconds': 3900,
}
(study / 'prepared_inputs.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print('Prepared held-out trace and 12 frozen forecast files:', inputs)
'@ | & $mlopsPython -X utf8 - $repo $sourceTrace $matrixDir
if ($LASTEXITCODE -ne 0) { throw 'Matrix preparation failed. Stop here.' }

$prepared = Get-Content (Join-Path $matrixDir 'prepared_inputs.json') -Raw | ConvertFrom-Json
$evalTrace = $prepared.evaluation_trace.Replace('\', '/')
$forecastRoot = $prepared.forecast_directory.Replace('\', '/')
$traceLabel = $prepared.trace_label
git rev-parse HEAD | Set-Content (Join-Path $matrixDir 'git_commit.txt')
~~~

This is a frozen forecast made before the evaluation period, not an oracle and
not an online retraining experiment. ARIMA's existing short-series/fitting
fallback remains in effect. A successful preparation step establishes valid
input structure and time alignment; it does not establish forecast accuracy.

### Matrix step 2: define one matched simulator run

Defining this function does not run a simulation. It verifies that the evaluation
trace and all forecast files are unchanged and refuses to overwrite any case.

~~~powershell
function Invoke-FssMatrixCase {
    param(
        [Parameter(Mandatory=$true)][ValidateSet(0, 1, 2)][int]$Level,
        [Parameter(Mandatory=$true)][bool]$STS,
        [Parameter(Mandatory=$true)][bool]$LTS
    )
    if ((Get-FileHash -LiteralPath $evalTrace -Algorithm SHA256).Hash -ne $prepared.evaluation_sha256) {
        throw 'Evaluation trace changed. Start a fresh matrix.'
    }
    foreach ($entry in $prepared.forecast_hashes.PSObject.Properties) {
        $path = Join-Path $forecastRoot $entry.Name
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.Value) {
            throw "Forecast changed: $path"
        }
    }
    $stsText = if ($STS) { 'True' } else { 'False' }
    $ltsText = if ($LTS) { 'True' } else { 'False' }
    $caseName = "level_${Level}_sts${stsText}_lts${ltsText}"
    $relativeOutput = "0/$traceLabel/feed_async_False/feed_async_granularity_1/" +
        "scaling_level_$Level/scaling_interval_-1/" +
        "lts_$($ltsText)_sts$($stsText)/arbiter_basic_arbiter"
    $outputDir = (Join-Path $matrixResults $relativeOutput).Replace('\', '/')
    if (Test-Path -LiteralPath $outputDir) { throw "Case already exists: $caseName" }
    $arguments = @(
        "trace.path=$evalTrace", "trace.filename=$traceLabel", "output_dir=$outputDir",
        'controller=us3-new', 'global_router=probabilistic', 'global_router.no_reroute=False',
        'seed=0', 'feed_async=False', 'siloed=False', 'end_time=3900',
        "scaling_level=$Level", 'scaling_interval=-1',
        "short_term_scaling=$stsText", "long_term_scaling=$ltsText",
        "global_arbiter.arima_traces=$forecastRoot",
        "global_arbiter.ilp_output_dir=$outputDir/ilp",
        'global_arbiter.long_term_scaling_interval=3600',
        'global_arbiter.post_processing_strategy=immediate',
        'global_arbiter.max_time=3900'
    )
    foreach ($region in 0..2) {
        foreach ($endpoint in 0..3) {
            $arguments += "controller.regions.$region.model_endpoints.$endpoint.instance_count=8"
        }
    }
    Push-Location (Join-Path $repo 'SageServe-original')
    try {
        & $simPython -X utf8 run.py @arguments 2>&1 |
            Tee-Object -FilePath (Join-Path $matrixDir "$caseName.log")
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($exitCode -ne 0) { throw "Case failed: $caseName. Inspect its console log." }
    $exports = @(Get-ChildItem -LiteralPath "$outputDir/detailed" -Filter '*.csv' `
        -Recurse -File -ErrorAction SilentlyContinue)
    if ($exports.Count -eq 0) { throw "No saved detailed exports: $caseName" }
    Write-Output "Finished $caseName"
}
~~~

### Matrix step 3: execute the 12 cases

Run each numbered line separately and wait for it to finish. Stop on a failure
and inspect its log. Do not blindly restart failed cases in existing directories;
preserve the evidence and use a fresh matrix for a corrected study.

~~~powershell
# E01-E04: scaling level 0
Invoke-FssMatrixCase -Level 0 -STS $false -LTS $false
Invoke-FssMatrixCase -Level 0 -STS $true  -LTS $false
Invoke-FssMatrixCase -Level 0 -STS $false -LTS $true
Invoke-FssMatrixCase -Level 0 -STS $true  -LTS $true

# E05-E08: scaling level 1
Invoke-FssMatrixCase -Level 1 -STS $false -LTS $false
Invoke-FssMatrixCase -Level 1 -STS $true  -LTS $false
Invoke-FssMatrixCase -Level 1 -STS $false -LTS $true
Invoke-FssMatrixCase -Level 1 -STS $true  -LTS $true

# E09-E12: scaling level 2
Invoke-FssMatrixCase -Level 2 -STS $false -LTS $false
Invoke-FssMatrixCase -Level 2 -STS $true  -LTS $false
Invoke-FssMatrixCase -Level 2 -STS $false -LTS $true
Invoke-FssMatrixCase -Level 2 -STS $true  -LTS $true
~~~

### Matrix step 4: analyze all cases together

Run an eligibility audit first. A full four-model matrix has 48 report rows,
with 12 eligible peers in each of four model cohorts. Missing measurements or
detected omitted applications can prevent this; an executed case is not
automatically eligible for FSS. Keep the same results root for the final report.

~~~powershell
Set-Location $repo
$inferenceScript = Join-Path $repo 'mlops\inference\run_inference.py'
$auditPath = Join-Path $matrixDir 'eligibility.csv'
& $mlopsPython -X utf8 $inferenceScript `
    --results $matrixResults --out $auditPath `
    --forecast-model 'frozen-historical-arima-matrix' `
    --cost-weight 0 --ttft-threshold 10 --tbt-threshold 0.2 --no-plots
if ($LASTEXITCODE -ne 0) { throw 'Matrix audit failed or is partial. Inspect its diagnostics.' }

$audit = Import-Csv -LiteralPath $auditPath
$audit | Select-Object model, scaling_level, sts, lts, n_requests, fss, fss_status |
    Format-Table -AutoSize
$badRows = @($audit | Where-Object { $_.fss_status -ne 'relative_to_cohort' })
$cohorts = @($audit | Group-Object fss_cohort_id)
if ($audit.Count -ne 48 -or $badRows.Count -ne 0 -or $cohorts.Count -ne 4 -or
    @($cohorts | Where-Object { $_.Count -ne 12 }).Count -gt 0) {
    throw 'The complete matched matrix is not eligible yet. Inspect exports and cohort membership.'
}

$reportPath = Join-Path $repo "mlops\inference\reports\$matrixName.csv"
$plotsDir = Join-Path $repo "mlops\inference\reports\$($matrixName)_plots"
& $mlopsPython -X utf8 $inferenceScript `
    --results $matrixResults --out $reportPath --plots-dir $plotsDir `
    --forecast-model 'frozen-historical-arima-matrix' `
    --cost-weight 0 --ttft-threshold 10 --tbt-threshold 0.2
if ($LASTEXITCODE -ne 0) { throw 'Final matrix analysis failed or is partial.' }
~~~

Select the matrix CSV in Results. These are cost-disabled FSS comparisons of
scaling strategies using one historical forecast, not comparisons of forecasting
algorithms. The label does not prove forecast consumption; inspect the saved
Hydra configurations and LTS simulator logs, including any missing-forecast
warnings and planned versus executed allocations. All six charts keep fixed
dimensions; 48 rows can crowd static labels, so use the report table to inspect
individual configurations. Separate reports have different normalization
references and must not be compared as if their FSS values share a cohort.

This is one seed/workload, not 12 independent replications. For stronger evidence,
repeat the whole matched matrix on other workloads/seeds as separately identified
studies and compare raw metrics across repetitions. Do not manufacture missing
measurements, infer exact hourly durations, or force bars by changing thresholds.

## Plots

All plots use numeric positions and experiment identifiers, so configurations
cannot overlap just because their model/LTS names match. Unknown values are
omitted or explicitly labelled; they are never replaced by apparent zeros.
Every plot uses a 12 x 6 inch canvas at 130 DPI (1560 x 780 pixels). Report row
counts do not change image dimensions. Compact labels use the model, run and
experiment ID; the complete trace/configuration remains in the report table.

| File | Purpose |
|---|---|
| fss_bar.png | Relative scores or explicit reasons that scores are unavailable. |
| sub_scores.png | Actual weighted FSS contributions, summing to the final score. |
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

~~~bash
PYTHONPATH=mlops python -B -m unittest discover -s mlops/inference/tests -v
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
