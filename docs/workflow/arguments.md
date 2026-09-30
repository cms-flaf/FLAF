# Command arguments

A reference for the options you pass to `law run`. The **common** ones are defined on FLAF's base
task classes (`FLAF/run_tools/law_customizations.py`), so they work on **every** FLAF task. LAW
also provides built-in options for status and cleanup.

!!! note "Underscores become dashes on the command line"
    A parameter named `transfer_logs` in the code is `--transfer-logs` on the CLI;
    `anaTuple_version` is `--anaTuple-version`, and so on.

!!! note "Boolean options take an explicit value"
    LAW makes boolean options explicit: `--bundle` on its own means `True`, and an option that is
    on by default is switched off with `False`, e.g. `--transfer-logs False`.

## Common task options

| Option | Default | Meaning |
|---|---|---|
| `--version` | *(required)* | Label that namespaces this run's outputs. Different versions never collide. |
| `--period` | *(required)* | The [era](../concepts/eras.md), e.g. `Run3_2022`. A task runs one era at a time; launch it once per era to [cover several](../concepts/eras.md#running-several-eras). |
| `--workflow` | `htcondor` | `local` (this machine), `htcondor` (CERN batch), or `crab` (WLCG). Without the option LAW picks the first workflow a task declares, which for FLAF tasks is HTCondor — so always pass it. `InputFileTask` and `AnaTupleFileListTask` always run locally. See [HTCondor](htcondor.md) and [CRAB](crab.md). |
| `--branches` | *(all)* | Which branches of the launched task to run, e.g. `0`, `0,2`, `5:8` (a `start:end` range excludes its end, so this is 5, 6, 7). Upstream tasks are then asked only for the branches these need — once their branch maps exist; on a from-scratch run the upstream stages that define the branch map are required in full. |
| `--test` | `-1` | AnaTuple production only: keep just the **first input file of every dataset** and process at most N events of it (`-1` = everything). See the warning below. |
| `--process` | `""` | Restrict to some processes: a comma-separated list of names or `^regex` patterns (e.g. `custom_CI_Signal`). A meta-process name expands to its members. |
| `--dataset` | `""` | Restrict to some datasets: a comma-separated list of names or `^regex` patterns. |
| `--model` | `""` | Override the physics model for this run. |
| `--customisations` | `""` | Ad-hoc `key=value;key=value` overrides of existing config entries (see below). |
| `--user-custom` | `""` | Path to an extra `user_custom`-style YAML, loaded last (see below). |

!!! warning "`--test` writes to the same paths as a full run"
    Nothing in the output paths records `--test`, so a truncated test output looks complete to a
    later full run of the same `--version`, which then reuses it. Give test runs their own
    `--version`. (Test jobs are also not used to calibrate the AnaTuple job-cost model.)

## HTCondor options (on every workflow task)

| Option | Default | Meaning |
|---|---|---|
| `--transfer-logs` | on | Keep each job's log (HTCondor and CRAB). With a remote `fs_default` the log is uploaded to `<version>/logs/<Task>/<period>/` there (plus the producer name for the analysis-cache tasks); with a local `fs_default` the log is part of the job's output sandbox, which HTCondor copies back to the task's local directory under `data/` only with `--htcondor-spool False` (with the default `-spool` it stays on the schedd until `condor_transfer_data` is run, which law does not do). Switch off with `--transfer-logs False`. |
| `--parallel-jobs` | unbounded (HTCondor) / **2000** (`AnaTupleFileTask` on HTCondor, `anaTuple_scheduling.parallel_jobs`) / **5000** (CRAB, `crab.parallel_jobs`) | Cap concurrent jobs. On CRAB this is also the max size of each CRAB task. |
| `--tasks-per-job` | `1` (`10` for `HistTupleProducerTask`) | Branches per job. Applies to the launched task only; set it for an upstream task with `--<Task>-tasks-per-job`. On `AnaTupleFileTask`, jobs are normally composed by [estimated cost](htcondor.md#how-branches-become-jobs) instead; passing this option explicitly restores fixed-size chunking. |
| `--max-runtime` | *(task default)* | Per-job wall-clock limit in hours: 12 unless the task sets its own (e.g. 40 for `AnaTupleFileTask`, 48 for `AnaTupleMergeTask`). On HTCondor, a resubmitted `AnaTupleFileTask` job gets a longer limit, unless `--tasks-per-job` is given. |
| `--n-cpus` | `1` (4 for `AnaTupleFileTask`, `AnaTupleCostProbeTask` and `HistTupleProducerTask`; 2 for `AnaTupleMergeTask`, `HistFromNtupleProducerTask` and `HistMergerTask`) | CPUs requested per job. `AnalysisCacheTask` always takes `n_cpus` and `max_runtime` from its producer's entry in `payload_producers`. |
| `--priority` | `0` | Job priority among your HTCondor jobs, from `-20` to `20`. |
| `--bundle` | off | Ship a code/environment tarball to the worker. See [HTCondor → bundles](htcondor.md#bundles-shipping-the-code-to-workers). Always on for `--workflow crab`. |
| `--htcondor-spool` | on | Pass `-spool` to `condor_submit`, so the input files (including the proxy) are sent to the schedd instead of being read from a shared filesystem. Switch off with `--htcondor-spool False`. |

The per-task defaults of `--n-cpus` and `--max-runtime` apply to the task you launch. A task
that LAW creates as a dependency is usually handed the requiring task's values instead, unless
FLAF resets them to the dependency's own defaults for that requirement.

## CRAB options (on every workflow task)

| Option | Default | Meaning |
|---|---|---|
| `--workflow crab` | — | Submit branches via CMS CRAB (WLCG). See [CRAB](crab.md). |

Optional site white/black lists go in `global.yaml` under `crab:` (not CLI flags).
Unset whitelist ⇒ all T1/T2/T3 sites. Default `--parallel-jobs` on CRAB is 5000
(`crab.parallel_jobs`); a new CRAB task is submitted only when at least
`crab.refill_fraction` (default 0.2) of those slots are free. `Site.storageSite`
/ `Data.outLFNDirBase` are derived from `fs_default`. Memory is
`2000 MB * n_cpus` (`crab.memory_mb_per_cpu`; CRAB / site-guaranteed default),
capped at the CRAB client limit (5000 MB for 1 core, `2500 MB * n_cpus` otherwise).
`--transfer-logs` uses FLAF's own log upload to `fs_default`; CRAB's log transfer stays off.

## Status & cleanup (LAW built-ins)

| Option | Meaning |
|---|---|
| `--print-status N,K` | Show the dependency tree status to task depth `N`, target-collection depth `K`. Also prints output paths. `--print-status 3,1` is a good default. |
| `--print-deps N` | Print the dependency tree to depth `N` without checking outputs. |
| `--remove-output N,a,y` | Remove outputs to task depth `N` (`0` = only the launched task). The mode `a` removes everything without asking (`i` asks per task, `d` is a dry run); the final `y` runs the task afterwards, so the outputs are recomputed (leave it out to only remove). **Deletes real files** — check the version first. |

## `--customisations`

Pass quick overrides of entries of the merged global configuration as a **semicolon**-separated
list, quoted so that the shell does not split it:

```sh
--customisations "key1=value1;key2=value2"
```

- Each item is exactly one `key=value`. Commas stay inside the value (`channels=eTau,muTau`),
  so `key1=a,key2=b` is read as one item and fails with `len of substring is not 2!`.
- A dotted key (`section.key=value`) addresses a nested entry.
- The key must already exist in the configuration, and the value is converted to the type of the
  value it replaces. Booleans are handled inconsistently: the configuration converts them with
  `bool()`, so `"False"` becomes `True`, while a few task-side checks compare the raw text with
  `"True"`. A list would be split into single characters. For booleans, lists and new keys use
  [`--user-custom`](#-user-custom-per-run-config-overlay).
- A customised run writes to the same output paths as a run without it: use a new `--version`.

!!! warning "AnaTuple production sees only three keys"
    `AnaTupleFileTask` does not pass the string on to `anaTupleProducer.py`: of its keys, only
    `channels`, `store_noncentral` and `compute_unc_variations` reach AnaTuple production (as
    options of the producer). Any other key changes only the configuration seen by the law tasks
    themselves and by the histTuple and histogram-filling scripts, which receive the string. For a setting the AnaTuple producer reads — such as
    HH→bb̄ττ's `deepTauVersion` — use `--user-custom`, which is forwarded to it.

!!! info "HH→bb̄ττ: restrict the channels"
    HH_bbtautau declares a top-level `channels` key, so `--customisations "channels=eTau,muTau"`
    limits every stage that takes a channel list (AnaTuple, analysis-cache, histTuple, histogram
    and plot tasks) to those channels.
    The other analyses do not declare it, and there the same option fails with
    `Key "channels" not found in global configuration.`

## `--user-custom`: per-run config overlay

`--user-custom <path>` appends an extra YAML **after every other configuration file** — including
`config/user_custom.yaml` and `config/<era>/global.yaml` — so its values win (see
[How values combine](../concepts/configuration.md#how-values-combine)). A missing file is an
error. Use an absolute path or one relative to `$ANALYSIS_PATH`. The file is also passed to the
producer scripts and shipped with remote jobs. It is the cleanest way to change settings for a
single run without touching your (git-ignored) `config/user_custom.yaml`:

```sh
law run FLAF.Analysis.tasks.HistPlotTask \
  --version test --period Run3_2022 --workflow local --branches 0 --test 1000 \
  --user-custom config/user_custom_test.yaml
```

See [`user_custom.yaml`](../configuration/user-custom.md).

## Per-task version overrides

Every task carries its own `--version`, so you can make one run **read** an existing upstream
production while **writing** its downstream outputs under a new version. Override an upstream task's
version with `--<TaskClassName>-version`:

```sh
law run FLAF.Analysis.tasks.HistTupleProducerTask \
  --version my_dev \
  --AnaTupleMergeTask-version v2605 \
  --AnaTupleFileListTask-version v2605 \
  --period Run3_2022EE --workflow local
```

Here the anaTuples are reused from the central `v2605` production (provided your `fs_anaTuple`
points at the central storage, see the warning below), while the histTuples are written under
`my_dev`. This is the key to fast, parallel development: many people can share one upstream
production without recomputing it.

### Shortcuts for the whole upstream

Listing every `--<Task>-version` is tedious. The base task exposes three shortcuts that set the
version of **all** upstream tasks at once:

| Flag | Sets the version of |
|---|---|
| `--anaTuple-version <v>` | every AnaTuple/AnaProd task (`InputFileTask`, `AnaTupleFileList*`, `AnaTupleMergeTask`, …) |
| `--anaCache-version <v>` | `AnalysisCacheTask` and `AnalysisCacheAggregationTask` |
| `--ana-version <v>` | **both of the above** — a single flag for the entire upstream production |

So the multi-flag example above (and any deeper fork) collapses to one flag:

```sh
law run FLAF.Analysis.tasks.HistTupleProducerTask \
  --version my_dev --ana-version v2605 \
  --period Run3_2022EE --workflow local
```

Use `--anaTuple-version` / `--anaCache-version` when you want to fork only one of the two upstream
stages.

!!! warning "A version override does not change *where* the outputs are looked up"
    These flags replace only the version label. The upstream outputs are still looked up on
    **your** `fs_anaTuple` / `fs_anaCacheTuple` (each falling back to `fs_default`). To reuse a
    central production, point those at the central storage as well (for example in a
    `--user-custom` overlay); otherwise LAW finds nothing there and re-runs the upstream chain
    under the central label in your own area.

!!! tip "`--<AnyTaskInTree>-<param>` works for parameters the requiring task does not pass on"
    LAW lets you set a parameter of any task in the dependency tree by prefixing it with the
    task's class name, e.g. `--HistFromNtupleProducerTask-n-files-per-job 10`. A parameter that
    the requiring task also declares (`--test`, `--workflow`, `--n-cpus`, …), or sets explicitly,
    is handed down by it and wins over the prefixed value — except for the ones FLAF prefers from
    the command line: `version`, the three version shortcuts and `tasks_per_job`.
