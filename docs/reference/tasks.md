# Task reference

A concise reference for every FLAF task: what it does, what it branches over, and its task-specific
parameters. The **common** parameters (`--version`, `--period`, `--workflow`, `--branches`,
`--test`, …) apply to all of them and are documented in [Command arguments](../workflow/arguments.md).

Production tasks live in `FLAF/AnaProd/tasks.py` (invoke as `FLAF.AnaProd.tasks.<Name>`); analysis
tasks live in `FLAF/Analysis/tasks.py` (invoke as `FLAF.Analysis.tasks.<Name>`). For the order in
which they run, see the [walkthrough](../workflow/walkthrough.md) and
[data flow](../concepts/data-flow.md).

## Production tasks (`AnaProd`)

### `InputFileTask`
Resolves the concrete list of NanoAOD files for the requested datasets and era (from DAS). Runs
locally (it is a `LocalWorkflow`, not submitted to HTCondor) and is cheap. Every downstream task
depends on it, so it runs first.

### `AnaTupleFileTask`
Runs the analysis producer (`AnaProd/anaTupleProducer.py`, inside CMSSW) over input files to create
**anaTuples**. **Branches over input files** (one branch per NanoAOD file) — the workflow you most
often submit to HTCondor.

### `AnaTupleFileListBuilderTask` / `AnaTupleFileListTask`
Helper workflows that assemble the lists of per-file anaTuples to be merged. Normally pulled in
automatically as dependencies of the merge step; you rarely call them directly.

### `AnaTupleMergeTask`
Merges the per-file anaTuples into one anaTuple per dataset (data merged across runs).

- **Parameter:** `--delete-inputs-after-merge` (bool, default `false`) — remove the per-file
  inputs once the merge succeeds, to save space.

## Analysis tasks (`Analysis`)

### `HistTupleProducerTask`
Reads merged anaTuples and computes the analysis **observables** (the configured "payload
producers"), writing **histTuples**.

### `HistFromNtupleProducerTask`
Fills **histograms** of the requested variables from the histTuples, including systematic
variations. **Branches over variables.**

- **Parameters:** `--variables` (string; restrict which variables), `--n-var-batches` (int,
  default `10`; how variables are grouped into branches).

### `HistMergerTask`
Merges the per-piece histograms into per-process histograms ready for plotting and fitting.

- **Parameter:** `--variables` (string; restrict which variables).

### `AnalysisCacheTask`
Pre-computes a per-event payload that later stages reuse — most importantly the **b-tag shape**
weights in HH→bb̄WW. Pulled in automatically when an analysis needs it.

- **Parameter:** `--producer-to-run` (which cached payload producer to run).
- **Caveat:** on a cold cache this can be **time-consuming** (≈ 1 h per branch). Reuse it across
  runs via a [per-task version override](../workflow/arguments.md#per-task-version-overrides).

### `AnalysisCacheAggregationTask`
Aggregates the cached payloads produced by `AnalysisCacheTask` into the form the histogram stages
consume.

- **Parameter:** `--producer-to-aggregate`.

### `HistPlotTask`
Produces the final **plots**. **Branches over variables** (one branch per variable).

- **Parameter:** `--variables` (string; restrict which variables).

## Statistical-inference tasks

The limit/fit tasks (e.g. `PlotResonantLimits`, `PlotPullsAndImpacts`) come from the
`StatInference` and `inference`/`dhi` submodules and run inside CMSSW/Combine. They are
analysis-specific — see each HH analysis's **Statistical inference** page (via
[Analyses](../analyses.md)) and the [walkthrough](../workflow/walkthrough.md#stage-5-statistical-inference).

!!! tip "Discover parameters from the command line"
    `law run <Task> --help` lists every parameter a task accepts, including the ones inherited from
    the base classes.
