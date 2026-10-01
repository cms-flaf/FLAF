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
Resolves the concrete list of NanoAOD files of every dataset of the era. **Branches over
datasets** and writes one JSON file list per dataset to the local `data/` area. Runs locally (it
is a `LocalWorkflow` only and forces `--workflow local`) and is cheap. Every downstream task
depends on it, so it runs first.

Where the files come from is decided per dataset:

1. a dataset with its own `fs_nanoAOD` (custom samples, CI inputs) is listed there, in its
   `dirName` (default: the dataset name);
2. otherwise the era's NanoAOD version applies — `nanoAODVersions.data` / `.mc` in the era's
   `global.yaml`, `HLepRare` when unset. For `HLepRare` the dataset's directory is listed on the
   era's `fs_nanoAOD` (the HLepRare skims); for any other version (e.g. `v15`, `22Sep23`) the
   dataset's `nanoAOD.<version>` DAS name is resolved through **Rucio**, which also checks that
   each file is available on disk at some site.

Files that match the dataset's `fileNamePattern` for that NanoAOD version (default: every
`.root` file) are kept. A Rucio
file with no available replica stops the task, unless `ignore_missing_nanoAOD_files` is set for
the dataset's process group, in which case it is listed as inactive and skipped.

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
Runs the analysis producer `AnaProd/anaTupleProducer.py` over one input file and fuses its
outputs (`AnaProd/FuseAnaTuples.py`) into one per-file **anaTuple** plus a JSON report, under
`<version>/AnaTuples_split/<era>/<dataset>/` on `fs_anaTuple`. **Branches over input files** (one
branch per NanoAOD file) — the workflow you most often submit to a batch system. With `--test N`
only the first file of each dataset is used, and at most `N` events of it. Branches are grouped
into jobs by estimated cost rather than in fixed-size chunks; see
[job composition](../workflow/htcondor.md#how-branches-become-jobs).

The producer runs in the FLAF environment, or inside CMSSW when the analysis sets
`use_cmssw_env_AnaTupleProduction: true` in its `global.yaml` (only HH→bb̄ττ does). An input file
with no events, or one that turns out to be corrupted, gives an empty anaTuple with a report
marked invalid, which the merge plan skips.

Each job fuses the outputs of the central selection and of every systematic shift into one file
([layout](../concepts/data-flow.md#what-an-anatuple-holds)), in which the columns sharing the text
before their first underscore form one collection. The step stops if a prefix mixes scalar and array
columns, or if any column would come out with another type than it went in; the fix is to rename the
columns in the analysis anaTuple definition.

The analysis `global.yaml` can list the columns that no shift changes:

```yaml
anaTuple_shift_invariant_columns:  # regular expressions, matched with re.search
  - "^weight_(gen|xs)$"
  - "^LHE_"
```

Those columns are stored in the central tree only; the central placeholder rows (events selected
only by a shift) get them from a shift that selected the event. The fuse step checks that every
variation selecting an event carries bit-identical values of every listed column and stops otherwise,
naming the column, the variation and an event. An array collection has to be listed as a whole. The
list must be the same for every file merged together, and `AnaTupleMergeTask` stops otherwise:
changing it means producing all anaTuples again under a new `--version`.

### `AnaTupleFileListBuilderTask` / `AnaTupleFileListTask`
Build the **merge plan**. `AnaTupleFileListBuilderTask` has one branch per MC dataset and a single
branch `data` for all data datasets; it reads the reports of the dataset's per-file anaTuples,
drops files marked invalid and, for MC, the fewest events' worth of files needed so that no
luminosity block appears in two files, and groups the rest into outputs of about
`nEventsPerFile` events (per process group in `global.yaml`; default 1 000 000 for data,
100 000 otherwise). It can run on a batch system and writes the plan and the combined reports to
`<version>/AnaTupleFileList/<era>/` and `<version>/AnaTupleFileList_reports/<era>/` on
`fs_anaTuple`. `AnaTupleFileListTask` always runs locally and only copies each plan into the
local `data/` area, where `AnaTupleMergeTask` reads it to build its branch map (on which the
analysis tasks' branch maps are built in turn). Both are pulled
in automatically; you rarely call them directly.

### `AnaTupleMergeTask`
Merges per-file anaTuples according to the merge plan. **One branch per merge-plan item**; each
writes one or more files of about `nEventsPerFile` events, `anaTuple_<N>.root` under
`<version>/AnaTuples/<era>/<dataset>/` on `fs_anaTuple`, so a dataset usually has several branches
and several files. All data datasets are merged into the single sample `data` (files named
`anaTuple_<eraLetter><eraVersion>_<N>.root`), filtered to the plan item's runs and with duplicate events
(same run, luminosity block and event) removed.

For MC the merge also computes the normalisation weight `weight_base` (sign of the generator
weight × luminosity × cross-section × shape weights / denominator), because the denominators of
the whole dataset are only known here — or, for a process whose processor (such as a
[stitcher](../concepts/stitching.md)) declares `dependency_level: {AnaTupleMerge: process}`, of
all the process's datasets;
in the 2024+ shared-MC eras it also writes `weight_base_cmb`. Columns matching the regular
expressions in `anaTupleMerge_drop_columns` are left out of the output.

- **Parameter:** `--delete-inputs-after-merge` (bool, default `false`) — meant to remove the
  per-file inputs once the merge has succeeded, to save space.

!!! warning "`--delete-inputs-after-merge` does not work in the current code"
    The deletion loop iterates over the keys of each input's output dictionary instead of its
    targets, so the task fails with an `AttributeError` right after the merged files have been
    written, and no input is removed. The merged outputs are complete, so a re-run marks the
    branch done.

## Analysis tasks (`Analysis`)

### `HistTupleProducerTask`
Runs `Analysis/HistTupleProducer.py` with the analysis's `histTupleDef` module over one merged
anaTuple file and writes one **histTuple** to `<version>/HistTuples/<era>/<dataset>/` on
`fs_HistTuple`: the analysis variables, the final event weights and the binned columns the
histogram step reads. **One branch per merged anaTuple file** (data as the single sample `data`;
plan items without events are skipped); several branches are grouped per batch job
(`--tasks-per-job`, default `10`).

Payload-producer columns (DNN scores and the like) are not computed here: they are read from the
[`AnalysisCacheTask`](#analysiscachetask) outputs, which this task requires for every histTuple
variable named `<producer>_<column>`, and from the aggregated caches of
[`AnalysisCacheAggregationTask`](#analysiscacheaggregationtask) where one applies.

### `HistFromNtupleProducerTask`
Fills **histograms** of the requested variables from the histTuples, with the Up/Down
variations when `compute_unc_histograms` is true (from the histTuple flavour, else `global.yaml` /
`user_custom.yaml`; default `false`). **Branches over (dataset, file-chunk):** each job reads its
chunk of input files and fills every active variable, one output file per variable under
`<version>/Hists_split/<era>/<variable>/<dataset>/`. Large datasets are parallelized by splitting
their files into chunks.

If the number of histograms booked in one RDataFrame pass — variables × selections ×
(Central + every Up/Down) — exceeds `hist_from_ntuple_max_hists` (default `4000`), the
producer repeats the event loop in batches instead of holding every histogram at once.
That keeps CI (8 GiB) from running out of memory when uncertainties are on. Set the
threshold in `global.yaml` / `user_custom.yaml`, or pass `--max-hists` to the producer
(`0` disables batching). LAW branches stay file-chunks; batching is inside the job.

- **Parameters:** `--variables` (string; restrict which variables), `--n-files-per-job` (int,
  default `20`; input files processed per branch).

### `HistMergerTask`
Merges the per-chunk histograms into per-process histograms ready for plotting and fitting,
one file per variable at `<version>/Hists_merged/<era>/<variable>/<variable>.root`. Each branch
(one per variable) merges **all uncertainty sources in a single pass**: every input file is read
once and all histograms are written directly to the final output file. The uncertainty sources
are those of `weights.yaml` when `compute_unc_histograms` is true (minus the era's
`uncs_to_exclude`), only `Central` otherwise.

- **Parameter:** `--variables` (string; restrict which variables).

### `AnalysisCacheTask`
Runs one **payload producer** (`Analysis/AnalysisCacheProducer.py`) and stores its per-event
output — typically a DNN score — for later stages. Payload producers are listed under
`payload_producers` in the analysis's `global.yaml`, and all three analyses define some
(HH→bb̄ττ: `ggF_DNN`; H→μμ: `DNN`, `VBFNet`, `PostVBFNetDNN`; HH→bb̄WW: `SingleLepHME`, `DoubleLepHME`,
`DNN`, `TwoStageDNN`, `DeepHME` and `BtagShape`). Same branches as `HistTupleProducerTask` (one per merged anaTuple file); output at
`<version>/AnalysisCache/<producer>/<era>/<dataset>/` on `fs_anaCacheTuple`.

Pulled in automatically by `HistTupleProducerTask` for every histTuple variable named
`<producer>_<column>` (a column listed under the producer's `columns`), so whether it runs depends
on the variables of the active histTuple flavour; also for an `is_global` producer that a
correction names as its `normCacheProducer` (through `AnalysisCacheAggregationTask` when the
producer also sets `needs_aggregation`). A producer's `dependencies` run first, and its
`n_cpus`, `max_runtime` (default 2 h), `save_as` (file type, default `root`) and `cmssw_env`
(run inside CMSSW) come from its `payload_producers` entry.

- **Parameter:** `--producer-to-run` (which payload producer to run; required).
- **Caveat:** on a cold cache this can be **time-consuming** (≈ 1 h per branch). Reuse it across
  runs via a [per-task version override](../workflow/arguments.md#per-task-version-overrides).

### `AnalysisCacheAggregationTask`
Aggregates all `AnalysisCacheTask` outputs of one producer for one dataset (`data` for all data)
into a single `aggregatedCache` file, which every `HistTupleProducerTask` branch of that dataset
receives. **One branch per dataset** of the producer's `target_groups`. Used for producers marked
`is_global` and `needs_aggregation` that a correction names as its `normCacheProducer` — today
HH→bb̄WW's `BtagShape`, the b-tag shape normalisation.

- **Parameter:** `--producer-to-aggregate` (required).

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
per variable, skipping variables configured with `plot_task: false`).

- **Parameter:** `--variables` (string; restrict which variables).

Plot styling comes from the analysis `config/plot/*.yaml` files (`cms_stacked.yaml`,
`histograms.yaml`, `<era>.yaml`) — unchanged from the legacy renderer. Signal overlays are scaled by
`signal_plot_scale` in `global.yaml`: a fixed factor (e.g. `100`) **or** `bkg` to normalise each
signal's integral to the summed background (shape comparison; the legend then reads
`… (norm. to bkg)`). PlotKit can also render outside FLAF; see its README for the standalone
`python -m PlotKit.cli` entry point.

## Statistical-inference tasks

Datacards and limits are not FLAF tasks: they come from the `StatInference` and
`inference`/`dhi` submodules, which only the two HH analyses include, and need the CMSSW/Combine
environment.

- **`StatInference` law chain** (`StatInference.law.tasks`, used by HH→bb̄WW and its CI):
  `PreprocessShapesTask` (only when the datacard configuration has a `preprocess` block) →
  `CreateDatacardsTask` → `ResonantLimitsTask` → `PlotResonantLimitsTask` /
  `PlotPullsAndImpactsTask`, plus `ResonantLimitsAndHistPlotTask`
  (limits together with the `HistPlotTask` plots of their input histograms). The chain reads the
  `Hists_merged` files of `--hists-version` (default: `--version`) as external inputs and never
  schedules their production — only `ResonantLimitsAndHistPlotTask`, through `HistPlotTask`,
  does. Eras, masses and variables come from the datacard configuration named by
  `StatInference.config` in `global.yaml`.
- **`dhi` tasks** from `inference.dhi.tasks` (e.g. `PlotResonantLimits`, `PlotPullsAndImpacts`,
  without the `Task` suffix), run directly on existing datacards. HH→bb̄ττ documents this manual
  route, after creating its datacards with `StatInference/dc_make/create_datacards.py`.

See each HH analysis's **Statistical inference** page (via [Analyses](../analyses.md)) and the
[walkthrough](../workflow/walkthrough.md#stage-5-statistical-inference).

!!! tip "Discover parameters from the command line"
    `law run <Task> --help` lists every parameter a task accepts, including the ones inherited from
    the base classes.
