# Full workflow walkthrough

This is the end-to-end tour of the pipeline: every stage, in order, with the command that runs it.
Read it once to understand the chain; in day-to-day work you usually run only the *last* stage you
need and let LAW produce the rest (see [the shortcut](#the-shortcut-just-ask-for-the-end)).

The commands use `FLAF.AnaProd.tasks.*` for production stages and `FLAF.Analysis.tasks.*` for
analysis stages — the fully-qualified task paths the framework registers.

## Setup recap

```sh
cd HH_bbtautau          # your analysis repository
source env.sh           # once per shell
voms-proxy-info         # confirm a valid proxy

# Pick a data-taking era and a label for this production:
ERA=Run3_2022
VER=dev
```

Throughout, `--period $ERA` selects the [era](../concepts/eras.md) and `--version $VER` namespaces
the [outputs](../concepts/data-flow.md#versions-keep-productions-apart). Add `--workflow local`
to run on this machine; switch to `--workflow htcondor` to scale up
([HTCondor guide](htcondor.md)).

---

## Stage 0 — Resolve the input files

`InputFileTask` turns "the datasets for this era" into a concrete list of NanoAOD files (from DAS).
Everything else depends on it, so it runs first — automatically when you launch a later stage, or
explicitly:

```sh
law run FLAF.AnaProd.tasks.InputFileTask --period $ERA --version $VER --workflow local
```

It is fast and cheap. If a from-scratch run unexpectedly *stays* in `InputFileTask` or fails here,
suspect a wrong `--period`/`--version` or an expired proxy.

## Stage 1 — Produce and merge analysis ntuples (anaTuples)

`AnaTupleFileTask` runs the analysis producer (`AnaProd/anaTupleProducer.py`, inside CMSSW) over
each NanoAOD file — **one branch per file** — applying the object selections and
[corrections](../concepts/architecture.md#common-vs-analysis-specific) and writing a slimmed
**anaTuple**. `AnaTupleMergeTask` then merges the per-file pieces into one anaTuple per dataset.

```sh
# Produce per-file anaTuples (heavy; normally on HTCondor):
law run FLAF.AnaProd.tasks.AnaTupleFileTask --period $ERA --version $VER --workflow local

# Merge them per dataset:
law run FLAF.AnaProd.tasks.AnaTupleMergeTask --period $ERA --version $VER --workflow local
```

!!! tip "Test on a few files first"
    `--branches 0,1,2` runs only the first three input files, and `--test 1000` processes only
    1000 events per file. Combine both to smoke-test ntuple production quickly.

## Stage 2 — Compute analysis observables (histTuples)

`HistTupleProducerTask` reads the merged anaTuples and computes the heavier analysis
**observables** (the "payload producers" configured in `global.yaml`), writing **histTuples**:

```sh
law run FLAF.Analysis.tasks.HistTupleProducerTask --period $ERA --version $VER --workflow local
```

!!! note "HH→bb̄WW: a caching step runs here first"
    In HH→bb̄WW, `AnalysisCacheTask` (and `AnalysisCacheAggregationTask`) pre-compute and aggregate
    per-event payloads — notably the **b-tag shape** weights — before histogramming. They are
    pulled in automatically and can be **time-consuming** (budget roughly an hour per branch on a
    cold cache). See the [HH_bbWW docs](../analyses.md) and the [Task reference](../reference/tasks.md).

## Stage 3 — Fill and merge histograms

`HistFromNtupleProducerTask` fills **histograms** of the requested variables from the histTuples —
**one branch per (dataset, file-chunk)**, filling all variables in a single pass — including
systematic variations. `HistMergerTask` merges the pieces into per-process histograms ready for
plotting and fitting.

```sh
# Fill histograms (restrict variables with --variables, set files/job with --n-files-per-job):
law run FLAF.Analysis.tasks.HistFromNtupleProducerTask --period $ERA --version $VER --workflow local

# Merge them:
law run FLAF.Analysis.tasks.HistMergerTask --period $ERA --version $VER --workflow local
```

Which variables are produced is controlled by the analysis config and can be narrowed with the
`--variables` parameter or the `variables:` list in `user_custom.yaml`.

## Stage 4 — Make the plots

`HistPlotTask` produces the final plots — **one branch per variable**:

```sh
law run FLAF.Analysis.tasks.HistPlotTask --period $ERA --version $VER --workflow local
# one variable only:
law run FLAF.Analysis.tasks.HistPlotTask --period $ERA --version $VER --workflow local --branches 0
```

This is the task you most often launch directly: asking for the plots makes LAW produce every
upstream product that is missing.

## Stage 5 — Statistical inference

The two HH analyses turn the merged histograms into datacards and then run limits and diagnostics
with [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/), via the
`StatInference` and `inference` submodules. H→μμ does not include this stage.

Because these commands run inside CMSSW/Combine, prefix them with `cmsEnv` (or open a `cmsEnv`
subshell once):

```sh
# 1) Create datacards from the produced shapes:
cmsEnv python3 StatInference/dc_make/create_datacards.py \
  --input  <PATH_TO_SHAPES> \
  --output <PATH_TO_CARDS> \
  --config <PATH_TO_CONFIG>      # e.g. StatInference/config/x_hh_bbww_run3.yaml

# 2) Run resonant limits:
law run PlotResonantLimits --version $VER --datacards '<PATH_TO_CARDS>/*.txt' --xsec fb --y-log

# 3) Pulls & impacts (per mass point — point at a single card):
PlotPullsAndImpacts --version $VER --datacards "<PATH_TO_CARDS>/<one_card>.txt" ...
```

The exact configs and options are analysis-specific — see each analysis's **Statistical
inference** page (linked from [Analyses](../analyses.md)) and the
[cms-hh inference docs](https://cms-hh.web.cern.ch/tools/inference/).

---

## The shortcut: just ask for the end

You rarely run the stages one by one. Because every task knows its dependencies, launching a late
stage runs all missing upstream stages automatically:

```sh
law run FLAF.Analysis.tasks.HistPlotTask --period $ERA --version $VER --workflow local
```

Run the individual stages explicitly only when you want to **stop at** an intermediate product
(e.g. produce anaTuples for someone else to use), or to inspect/debug one stage.

## See progress and redo selectively

```sh
# Status of the whole tree (task depth 3, file depth 1) — also prints output paths:
law run FLAF.Analysis.tasks.HistPlotTask --period $ERA --version $VER --print-status 3,1

# Force one stage to be recomputed:
law run FLAF.Analysis.tasks.HistMergerTask --period $ERA --version $VER --remove-output 0,a,y
```

See [Command arguments](arguments.md) for the full option list, and [Running on HTCondor](htcondor.md)
to take any of these commands to the batch system by swapping `--workflow local` for
`--workflow htcondor`.
