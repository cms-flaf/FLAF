# Command arguments

A reference for the options you pass to `law run`. The **common** ones are defined on FLAF's base
task classes (`FLAF/run_tools/law_customizations.py`), so they work on **every** FLAF task. LAW
also provides built-in options for status and cleanup.

!!! note "Underscores become dashes on the command line"
    A parameter named `transfer_logs` in the code is `--transfer-logs` on the CLI;
    `anaTuple_version` is `--anaTuple-version`, and so on.

## Common task options

| Option | Default | Meaning |
|---|---|---|
| `--version` | *(required)* | Label that namespaces this run's outputs. Different versions never collide. |
| `--period` | *(required)* | The [era](../concepts/eras.md), e.g. `Run3_2022`. |
| `--workflow` | `local` | `local` (this machine) or `htcondor` (batch). See [HTCondor](htcondor.md). |
| `--branches` | *(all)* | Which branches to run, e.g. `0`, `0,2`, `5-7`. Restricts only the launched task, not its dependencies. |
| `--test` | `-1` | Process only N events per input file (`-1` = all). Great for smoke tests. |
| `--process` | `""` | Restrict to one process (e.g. `custom_CI_Signal`). |
| `--dataset` | `""` | Restrict to one dataset. |
| `--model` | `""` | Override the physics model for this run. |
| `--customisations` | `""` | Ad-hoc `key=value,key=value` overrides (see below). |
| `--user-custom` | `""` | Path to an extra `user_custom`-style YAML, loaded last (see below). |

## HTCondor options (on every workflow task)

| Option | Default | Meaning |
|---|---|---|
| `--transfer-logs` | off | Bring job logs back to `data/`. Recommended. |
| `--parallel-jobs` | *(unbounded)* | Cap concurrent branches, e.g. `--parallel-jobs 100`. |
| `--max-runtime` | *(task default)* | Per-job wall-clock limit. |
| `--n-cpus` | `1` | CPUs requested per job. |
| `--priority` | `0` | Job priority. |
| `--bundle` | off | Ship a code/environment tarball to the worker. See [HTCondor → bundles](htcondor.md#bundles-shipping-the-code-to-workers). |
| `--htcondor-spool` | off | Spool job files to the schedd. |

## Status & cleanup (LAW built-ins)

| Option | Meaning |
|---|---|
| `--print-status N,K` | Show the dependency tree status to task depth `N`, file-collection depth `K`. Also prints output paths. `--print-status 3,1` is a good default. |
| `--print-deps N` | Print the dependency tree to depth `N` without checking outputs. |
| `--remove-output N,a,y` | Remove outputs to depth `N` (`a` = all branches, `y` = no prompt). Forces a recompute. **Deletes real files** — check the version first. |

## `--customisations`

Pass analysis-specific overrides as a comma-separated list:

```sh
--customisations key1=value1,key2=value2
```

!!! info "HH→bb̄ττ: select the DeepTau version"
    To run with DeepTau 2.5, add `--customisations deepTauVersion=2p5`. (See the HH_bbtautau docs.)

## `--user-custom`: per-run config overlay

`--user-custom <path>` loads an extra YAML **on top of** your `config/user_custom.yaml` (loaded
last, so its values win). Use an absolute path or one relative to `$ANALYSIS_PATH`. It is the
cleanest way to change settings for a single run without editing your committed file:

```sh
law run FLAF.Analysis.tasks.HistPlotTask \
  --version test --period Run3_2022 --workflow local --branches 0 --test 1000 \
  --user-custom /afs/.../config/user_custom/test_local/HH_bbtautau.yaml
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

Here the anaTuples are reused from the central `v2605` production, while the histTuples are written
under `my_dev`. This is the key to fast, parallel development: many people can share one upstream
production without recomputing it. The base task also exposes related shortcuts
(`--anaTuple-version`, `--anaCache-version`, `--ana-version`) used by some stages.

!!! tip "`--<AnyTaskInTree>-<param>` works generally"
    LAW lets you set *any* parameter of *any* task in the dependency tree by prefixing it with the
    task's class name. Version overrides are the most common case, but the same mechanism applies
    to other parameters.
