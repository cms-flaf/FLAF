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
Resolves the concrete list of NanoAOD files for the requested datasets and era, querying **Rucio**
for the file list and their disk availability. Runs locally (it is a `LocalWorkflow`, not submitted
to HTCondor) and is cheap. Every downstream task depends on it, so it runs first.

Its output also records what each file contains, under `file_info`:

- **`size`** — always, taken from the directory listing that the task performs anyway
  (`gfal-ls --long` for a storage path, the Rucio file list for a DAS dataset).
- **`n_events`** — for datasets discovered through Rucio, from a single **DAS** query per dataset
  (Rucio itself leaves the CMS `events` field empty). One query covers thousands of files in about
  a second.

Both are advisory inputs to job-cost estimation. A missing event count is normal — for HLepRare
skims there is no DAS record — and the cost model falls back to the file size.

### `AnaTupleCostProbeTask`
Times the producer on a short prefix of one file per dataset (`probe_events`, default 5000) and
records the per-event cost. **Branches over datasets.** Runs before `AnaTupleFileTask` and takes a
few minutes; what it buys is job composition based on measurement rather than guesswork, because
per-event cost varies by more than an order of magnitude between datasets and depends on the
analysis selection.

Results live at `<version>/AnaTupleCost/<nano-source>/<dataset>.json` on `fs_anaTuple`, keyed by
version and nano source but **not by era**, so a multi-era production probes each dataset once and
the later eras skip this stage entirely.

A probe that fails is retried once and, if it still fails, writes a result marked not ok and
prints a warning: calibration is an optimisation and never blocks production. That result counts
as the task's output, so to re-probe a dataset after fixing the cause, delete its json. Set
`anaTuple_scheduling.probe_enabled: false` to skip the stage entirely.

### `AnaTupleFileTask`
Runs the analysis producer (`AnaProd/anaTupleProducer.py`, inside CMSSW) over input files to create
**anaTuples**. **Branches over input files** (one branch per NanoAOD file) — the workflow you most
often submit to HTCondor. Branches are grouped into jobs by estimated cost rather than in
fixed-size chunks; see [job composition](../workflow/htcondor.md#how-branches-become-jobs).

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
variations. **Branches over (dataset, file-chunk):** each job reads its chunk of input files
and fills the active variables. Large datasets are parallelized by splitting their files
into chunks.

If the number of histograms booked in one RDataFrame pass — variables × selections ×
(Central + every Up/Down) — exceeds `hist_from_ntuple_max_hists` (default `4000`), the
producer repeats the event loop in batches instead of holding every histogram at once.
That keeps CI (8 GiB) from running out of memory when uncertainties are on. Set the
threshold in `global.yaml` / `user_custom.yaml`, or pass `--max-hists` to the producer
(`0` disables batching). LAW branches stay file-chunks; batching is inside the job.

- **Parameters:** `--variables` (string; restrict which variables), `--n-files-per-job` (int,
  default `20`; input files processed per branch).

### `HistMergerTask`
Merges the per-piece histograms into per-process histograms ready for plotting and fitting.
Each branch (one per variable) merges **all uncertainty sources in a single pass**: every
input file is read once and all histograms are written directly to the final output file.

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

### `PreHistTupleProductionTask`
Runs the **entire AnaTuple + AnalysisCache production** for a version in one command, without
producing histTuples. It shares `HistTupleProducerTask`'s dependency graph but writes only a small
per-branch completion marker, so a single

```sh
law run FLAF.Analysis.tasks.PreHistTupleProductionTask --version <v> --period <era> --workflow local
```

forces every `AnaTupleMergeTask` and `AnalysisCacheTask` (plus their aggregation) to run — handy
to pre-compute and then freeze/share those caches (as `AnaTupleMergeTask` outputs already can be),
instead of submitting each `AnalysisCacheTask --producer-to-run` individually.

### `HistPlotTask`
Produces the final **plots** via the [PlotKit](https://github.com/cms-flaf/PlotKit) submodule
(matplotlib + mplhep by default; optional ROOT + cmsstyle). **Branches over variables** (one branch
per variable).

- **Parameter:** `--variables` (string; restrict which variables).

Plot styling comes from the analysis `config/plot/*.yaml` files (`cms_stacked.yaml`,
`histograms.yaml`, `<era>.yaml`) — unchanged from the legacy renderer. Signal overlays are scaled by
`signal_plot_scale` in `global.yaml`: a fixed factor (e.g. `100`) **or** `bkg` to normalise each
signal's integral to the summed background (shape comparison; the legend then reads
`… (norm. to bkg)`). PlotKit can also render outside FLAF; see its README for the standalone
`python -m PlotKit.cli` entry point.

## Statistical-inference tasks

The limit/fit tasks (e.g. `PlotResonantLimits`, `PlotPullsAndImpacts`) come from the
`StatInference` and `inference`/`dhi` submodules and run inside CMSSW/Combine. They are
analysis-specific — see each HH analysis's **Statistical inference** page (via
[Analyses](../analyses.md)) and the [walkthrough](../workflow/walkthrough.md#stage-5-statistical-inference).

!!! tip "Discover parameters from the command line"
    `law run <Task> --help` lists every parameter a task accepts, including the ones inherited from
    the base classes.
