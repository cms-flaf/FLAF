# Eras

Every run targets one **era**, or several, passed as `--era` (see [Running several eras](#running-several-eras)).
An era is a CMS data-taking period; choosing one selects the matching datasets, corrections and
NanoAOD input.

!!! note "`--period` was renamed to `--era`"
    The former `--period` option, the `--<Task>-period` forms and a `[<Task>] period` entry in a
    law config now stop at once with `'period' was renamed to 'era' (--era)`, and a `period=`
    keyword with `'period' was renamed to 'era' (use era=...)`.
    Only the user-facing name changed: output paths (`<version>/<Kind>/<era>/…`), bundle URIs and
    job-file names are as before, so existing outputs are found unchanged. The AnaTuple column
    named `period` (the per-event C++ `Period` value), the `Period` enum and the `validity_period`
    cache settings keep their names.

## Run 3 eras (current)

| `--era` | Description | √s | NanoAOD (FLAF default version) | HH_bbWW, H_mumu |
|---|---|---|---|---|
| `Run3_2022` | 2022, pre-ECAL repair | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2022EE` | 2022, post-ECAL repair ("EE") | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2023` | 2023, pre-BPix | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2023BPix` | 2023, post-BPix install | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2024` | 2024 | 13.6 TeV | v15 | v15 (DAS) |
| `Run3_2025` | 2025 | 13.6 TeV | v15 | v15 (DAS) |
| `Run3_2026` | 2026 | 13.6 TeV | v15 | v15 (DAS) |

For 2022–2023BPix the framework default reads the HLepRare NanoAOD v14 skims
(`skim_2025_v1` under `/store/group/phys_higgs/HLepRare/`, the era's `fs_nanoAOD`); only
HH_bbtautau uses them. HH_bbWW and H_mumu override those four eras to read the central NanoAOD
from DAS instead. For 2024–2026 all three analyses read NanoAOD v15 from DAS (apart from the
few datasets that set their own `fs_nanoAOD`). FLAF itself sets only `nano_version: v15` for these
eras and no NanoAOD source: the analyses select DAS through `nanoAODVersions` in their era configs.

## Run 2 eras (legacy, not runnable)

`FLAF/config/` still has `Run2_2016_HIPM`, `Run2_2016`, `Run2_2017` and `Run2_2018` directories
(13 TeV), but they hold only legacy `samples.yaml`/`triggers.yaml` files — no `global.yaml` and
no `datasets.yaml` — and the analyses carry at most a Run 2 `weights.yaml`. Configuration loading
cannot build a run for these eras, and CI covers only the seven Run 3 eras. Treat Run 2 as
unsupported.

## Why the split into sub-eras?

The detector and its calibration change *within* a year, so CMS treats those segments as separate
eras for analysis:

- **2022** splits at the ECAL endcap repair → `Run3_2022` (before) and `Run3_2022EE` (after).
- **2023** splits at the pixel-detector "BPix" installation → `Run3_2023` and `Run3_2023BPix`.

Each sub-era has its own corrections and luminosity, which is exactly why the
[configuration system](configuration.md) has a **per-era layer**: `FLAF/config/<era>/` and
`<analysis>/config/<era>/` carry the era-specific datasets and overrides.

## What an era controls

- **Datasets** — `config/<era>/datasets.yaml` lists the samples available for that era; each
  entry's `nanoAOD:` map gives its DAS name per NanoAOD version (`22Sep23`, `v12`, `v15`, …).
- **NanoAOD input** — the table above, set by two keys of the era's `global.yaml`.
  `nanoAODVersions: { data: …, mc: … }` picks which entry of each dataset's `nanoAOD:` map is
  read, through Rucio; when it is not set, the files are listed from the era's `fs_nanoAOD` (the
  HLepRare skims). A dataset with its own `fs_nanoAOD` is always read from there.
  `nano_version` names the NanoAOD format the producers are written for (`v12` has different
  branches), so it has to match what is read.
- **Corrections** — pileup, b-tagging, trigger and other scale factors are era-specific.
- **Signals** — resonant/non-resonant signals exist for some eras and not others. For
  `Run3_2024`, VBF and non-resonant ggF HH are on DAS (new `Par-` naming); resonant
  Radion/BulkGraviton and X→YH→2B2W are not.
- **2024/2025/2026 shared MC** — there is no dedicated 2025 or 2026 MC campaign.
  All three years use the Summer24 NanoAOD, but **jet, PU and tau corrections
  differ**, so AnaTuple production runs once per era. `Run3_2025` and `Run3_2026`
  set `reuse_mc_from_era: Run3_2024` so the 2024 MC dataset list (and the
  `shared_mc` split) is reused. Each production stores `weight_base` (all events,
  this year's luminosity; use for a single-year run) and `weight_base_cmb` (the
  same events split by residue between 2024, 2025 and 2026; use for a combined
  run). The split applies only to MC; data keeps the full year in
  `weight_base_cmb`. Select which branch histograms use with
  `weight_base_branch`; all three eras default to `weight_base`, so by default every year
  uses all of its events with its own luminosity, and the residue split below feeds only
  `weight_base_cmb`.
  `shared_mc` lives only on the source era (`Run3_2024`): a 17:17:4 residue
  *target* over modulus 38 (`Run3_2024: [ 0, 16 ]`, `Run3_2025: [ 17, 33 ]`,
  `Run3_2026: [ 34, 37 ]`), matching the recorded luminosities
  109948.18 : 110730.86 : 25843.26. The actual event split need not match
  that target. AnaTupleFileTask therefore stores two denominators: the
  full-sample sum (for `weight_base`) and the in-era sum (for
  `weight_base_cmb`). Each weight is
  `gen × lumi × xs × PU / its_denominator`, so both yields stay `L·σ`.
  HistTuple multiplies the AnaTuple column named by `weight_base_branch`
  (`weight_base` for a single-year run, `weight_base_cmb` for the combined
  24+25+26 run).
  Until official UParTAK4 shape files exist, 2024+ era overlays omit
  `btag.normCacheProducer`, so HistTuple does not depend on the global
  `BtagShape` cache. `modes.<stage>: none` still loads the correction (needed
  for WP-id branches) but does not apply scale factors. Missing
  `uncs_to_exclude` era keys default to an empty list.
  The 2026 EGM folder has scale/smear files but no `electron.json`, so
  Electron-ID-SF is loaded from the 2025 file and evaluated with year
  `2025Prompt` (not `2026Prompt`).

## Running several eras

`--era` takes one era or several, separated by commas; the eras are set once per invocation
(`--<Task>-era` is refused with `the eras are set once per invocation, with --era`):

```sh
law run FLAF.Analysis.tasks.HistMergerTask --version v1 --era Run3_2022,Run3_2022EE --workflow htcondor
```

The list is sorted and duplicates are dropped, so `--era Run3_2022EE,Run3_2022` is the same run.
`--era Run3_2022` is a run over one era as before, apart from the `--branches` change described
under [Branch numbers](#branch-numbers).

!!! warning "`--era` must be one of the known eras"
    The known eras are the directories `FLAF/config/<era>/` whose `global.yaml` says
    `era: <era>`; any other `--era` stops at once with the list of the known ones
    ([Troubleshooting](../troubleshooting.md#unknown-eras-known-eras-the-configuration-says-era)).
    If a run unexpectedly drops into `InputFileTask` and queries Rucio for nothing, a wrong
    `--era`/`--version` combination is the usual cause. See
    [Troubleshooting](../troubleshooting.md).

### What a run over several eras does

- **Every stage up to the launched one runs for all the eras.** Each stage is **one workflow**
  whose branches are those of the eras that still need work there: one HTCondor or CRAB
  submission and one poll loop per stage, or, with `--workflow local`, the branches of all the
  eras in one driver.
- **An era whose part of a stage is complete is pruned.** It is not asked for anything, so the
  intermediate products its outputs were made from are not requested either, even when they have
  been removed since (the split AnaTuples after the merge, the histogram chunks replaced by
  `.merged` markers). Adding an era to a finished one submits nothing for the finished era.
- **A stage whose branch map is read from an earlier stage's output** (the file lists of
  `InputFileTask`, the merge plan of `AnaTupleFileListTask`) is built only once that output exists
  for every era of the run, so the slowest era sets the pace at that point.
- **Nothing is summed across eras.** Every output belongs to one era, as in a run over one era.
  Combining eras, such as 2024+2025+2026 with `weight_base_branch: weight_base_cmb`, stays a
  separate step: add the histograms of the eras.
- **Shared-MC eras** (`Run3_2025` and `Run3_2026` reuse the 2024 MC datasets, see above) each
  produce their own AnaTuples: in a run over `Run3_2024,Run3_2025,Run3_2026` the shared MC
  datasets are read three times and written as three sets of AnaTuples, so the work is not shared
  between the eras.
- **Every producer checks the era of its events.** `AnaTupleMergeTask`, `HistTupleProducerTask`
  and `AnalysisCacheTask` count, in the event loop they run anyway, the events whose `period`
  column (the C++ `Period` value of the AnaTuple) is not the era the branch runs for, and stop
  with an error when there is any (placeholder rows, `valid == false`, carry no era and are
  skipped). An AnaTuple of another era among the inputs is caught this way.
- **One driver per era.** The branches of an era are tracked in that era's job file (see
  [HTCondor](../workflow/htcondor.md#several-eras-in-one-submission)). Two instances of one stage
  that share an era never run at once inside one `law run` (a luigi resource per job-file
  directory, `flaf_jobs:<directory>`, of capacity 1 unless luigi's `[resources]` configuration
  section sets another), but two separate `law run` processes on the same era are not guarded
  against each other, exactly as before.

### Which tasks take several eras

The stages of the production and analysis chain: `InputFileTask`, `AnaTupleCostProbeTask`,
`AnaTupleFileTask`, `AnaTupleFileListBuilderTask`, `AnaTupleFileListTask`, `AnaTupleMergeTask`,
`AnalysisCacheTask`, `AnalysisCacheAggregationTask`, `HistTupleProducerTask`,
`PreHistTupleProductionTask`, `HistFromNtupleProducerTask`, `HistMergerTask` and `HistPlotTask`
(the classes that set `allow_multi_era` and `era_branches`).

Every other task takes one era and refuses several with
`<Task> takes one era; got --era Run3_2022,Run3_2023`. That covers the analyses' Studies tasks,
the `StatInference` tasks and `BundleTask`: bundles stay per era, the stages that ship one
require the bundles of each era that still has work, and an existing bundle of one era is never
rebuilt for another. Launch the tasks that refuse several eras once per era.

### Branch numbers

Over **one era** a branch has the number it has in that era's own branch map, as always. Over
**several eras** it is numbered

```text
era id × 1 000 000 + its number in that era's own map
```

where the era id is the C++ `Period` value:

| Era | `Run3_2022` | `Run3_2022EE` | `Run3_2023` | `Run3_2023BPix` | `Run3_2024` | `Run3_2025` | `Run3_2026` |
|---|---|---|---|---|---|---|---|
| Era id | 5 | 6 | 7 | 8 | 9 | 10 | 11 |

An era's branch numbers therefore never depend on the other eras of the run: branch 3 of
`Run3_2022EE` is 6000003 in every run. With `env.sh` sourced,
`python -m FLAF.Common.Setup --era-branch Run3_2022EE 3` prints it.

- **`--branches` takes these numbers over several eras**, for example
  `--era Run3_2022,Run3_2022EE --branches 5000000,5000001,6000000:6000010`. Every era of `--era`
  must have at least one selected branch (otherwise drop it from `--era`), a range has a start
  and an end and stays inside one era, and every number must belong to an era of the run. Once
  an era's branch map exists, at least one of its selected numbers must be a branch of it; a run
  over one era would take a selection naming none of its branches as the whole map, which a run
  over several eras refuses.
- **`--branch` is refused** over several eras: use `--era <era> --branch <its number in that era>`.
- **A run over one era takes that era's own numbers.** A number of a run over several eras
  (1 000 000 or more) stops with the decoded hint, for example
  `branch 6000001 is branch 1 of Run3_2022EE in a run over several eras; a run over one era takes that era's own numbers: --era Run3_2022EE --branches 1`.
  A number of no configured era stops with `branch <n> is beyond the 1000000 numbers of a run
  over one era`.
- **A `--branches` selection given before the stage's inputs exist** is honoured. For a stage
  whose branch map is read from an earlier stage's output (see above), a selection given before
  that output existed used to be widened to all of its branches; it is now kept, in runs over one
  era as well.

### Settings that must agree between the eras

A run over several eras reads some settings once, for all its eras. They must be the same in every
era of the run, or constructing the task stops with `EraDependentValueError` naming the key and
each era's value:

- `fs_default`;
- `crab` (the whole block), `runTokenServer`, and `WLCGFileSystem.remotePathCacheHost` and
  `remotePathCachePort`;
- `anaTuple_scheduling` and `use_cmssw_env_AnaTupleProduction`;
- of each entry of `payload_producers`: `n_cpus`, `max_runtime`, `crab_memory`, `save_as` and
  `cmssw_env`.

Everything else may differ per era: the datasets and processes, the corrections, the bundles and
the output filesystems `fs_anaTuple`, `fs_HistTuple`, `fs_anaCacheTuple` and `fs_plots` (for
instance in `config/<era>/user_custom.yaml`, see
[`user_custom.yaml`](../configuration/user-custom.md#several-eras-in-one-run)).

### What stays the same as in a run over one era

Branch tasks, jobs, payloads (one era per subprocess, `--era <the branch's era>`), output paths,
job files, logs, heartbeats, worker files, bundles and CRAB request names are per era and
identical to what `--era <that era>` alone writes. A run over several eras and runs over some of
them therefore find each other's outputs and unfinished jobs, in any order. The `--dataset`
selection applies to every era: a dataset that exists in only one of the eras fails the run
(`No datasets selected with a custom selection: …`), so run such a dataset on its own era.
`--parallel-jobs` is one cap for the whole run, not one per era
([HTCondor](../workflow/htcondor.md#several-eras-in-one-submission),
[CRAB](../workflow/crab.md#several-eras-in-one-run)). The error messages of this section are
explained in [Troubleshooting](../troubleshooting.md#task-takes-one-era-got-era).

In CI, the eras are listed in the `*_eras` variable (e.g.
`Run3_2022 Run3_2022EE Run3_2023 Run3_2023BPix Run3_2024 Run3_2025 Run3_2026`). See the
[integration pipeline](../ci/integration-pipeline.md). For a 2024+2025+2026 combination, run each
era with `weight_base_branch: weight_base_cmb` and add the histograms.

## Adding a new era

Adding an era means creating its per-era config directories in both the framework and the
analysis, wiring it into the CI era lists, and listing it in `test-setup-loading`. The full
procedure is in [Datasets](../configuration/datasets.md#adding-a-new-era).
