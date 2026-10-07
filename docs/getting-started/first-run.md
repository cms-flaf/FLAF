# Your first run

This page walks you through a **minimal end-to-end run**: a single command that exercises the
*whole* FLAF pipeline — from CMS NanoAOD all the way to a histogram plot — on just a handful of
events. It runs the same chain of tasks as the
[integration pipeline](../ci/integration-pipeline.md), only smaller, so if it works for you, your
setup is healthy.

## Before you start

You need a working environment from [Installation](installation.md):

```sh
cd HH_bbtautau          # or your analysis repository
source env.sh           # once per shell
voms-proxy-info         # confirm you have a valid proxy
```

You also need a `config/user_custom.yaml`. The quickest correct start is the settings file the
integration pipeline uses, which every analysis ships as `config/ci_custom.yaml`:

```sh
cp config/ci_custom.yaml config/user_custom.yaml
```

Then replace its `fs_default` (a path on the CI runner) with your own storage location. The file
already sets everything else the run needs:

- `phys_model: TestModel` — a small, fast subset of processes meant for testing;
- `analysis_config_area: config` — `HistPlotTask` fails with a `KeyError` without it;
- `store_noncentral` — read without a default by `AnalysisCacheTask`, which runs whenever a
  variable comes from a [payload producer](../glossary.md);
- `compute_unc_variations` and `compute_unc_histograms` — the systematic variations, switched on
  as in CI;
- in HH→bb̄ττ and H→μμ, `histTuple_flavor: CI`, a short list of variables;
- in HH→bb̄WW, a single region (`QCDRegions: [OS_Iso]`) and a CI datacard configuration.

!!! warning "H→μμ: the default histTuple flavor has no variables"
    H→μμ's `default` flavor lists no variables, so without `histTuple_flavor: CI` (or a
    `variables:` list) in your `user_custom.yaml` there is nothing to plot.

The [Configuration guide](../configuration/user-custom.md) explains every field and has a minimal
file written from scratch (for H→μμ, add `histTuple_flavor: CI` to it).

## Run it

```sh
law run FLAF.Analysis.tasks.HistPlotTask \
  --version my_first_run \
  --period Run3_2022 \
  --workflow local \
  --branches 0 \
  --test 1000
```

That one command asks LAW for the final plots. LAW notices that none of the inputs exist yet and
**automatically runs every upstream stage first** — resolving the input file list, producing and
merging analysis ntuples, computing observables (`HistTupleProducerTask`, preceded by
`AnalysisCacheTask` for variables that come from a payload producer), filling and merging
histograms — before making the plot. You do not run the intermediate tasks yourself.

### What each argument means

| Argument | Meaning |
|---|---|
| `FLAF.Analysis.tasks.HistPlotTask` | The task you want — here, the plotting task (the end of the chain). |
| `--version my_first_run` | A label for this run. Outputs are grouped under it, so you can keep runs apart. Use any name. |
| `--period Run3_2022` | Which data-taking [era](../concepts/eras.md) to process. |
| `--workflow local` | Run on this machine (not the batch system). Good for testing. |
| `--branches 0` | Only the first work unit. `HistPlotTask` has one branch per variable, so this plots a single variable (one PDF per channel × category × region). |
| `--test 1000` | Use only the **first input file** of each dataset and only its first 1000 events — fast, just to check the machinery. |

These and many more options are catalogued in [Command arguments](../workflow/arguments.md).

??? info "How this differs from what the integration pipeline runs"
    The [integration pipeline](../ci/integration-pipeline.md) is not run on every change: it
    starts when a member of the `cms-flaf` organisation posts a `@cms-flaf-bot please test`
    comment on a pull request. It then runs, per analysis, with `--version CI --workflow local
    --test 1000` (no `--branches 0`, so every variable), `config/ci_custom.yaml` as the
    `user_custom.yaml`, and **all seven Run 3 eras**:

    - one `HistFromNtupleProducerTask` job per era and per CI process (`--process <name>`);
    - then one `HistPlotTask` job per era;
    - for HH→bb̄WW, whose configured target is
      `StatInference.law.tasks.ResonantLimitsAndHistPlotTask`, a final job running that task.

## What to expect

- LAW prints a dependency tree and then runs the stages bottom-up. Because the early stages
  produce analysis ntuples from CMS NanoAOD (reading from the grid), **the first
  run is not instant** even with `--test 1000` — budget a little time.
- The ROOT files and plots are written under the storage you configured in `user_custom.yaml`
  (`fs_default`, or a more specific `fs_*`), organised by version and era. The plots of this run
  land in `my_first_run/Plots/Run3_2022/<variable>/<region>/<category>/<channel>_<variable>.pdf`.
- Small bookkeeping files — the input file lists and a local copy of the anaTuple merge plans —
  are written into the checkout under `data/my_first_run/`.
- When the top task finishes, LAW reports success for `HistPlotTask`.

!!! tip "Check progress without running anything"
    In another shell (after `source env.sh`), ask LAW for the status of the dependency tree. Pass
    the same task parameters as the run — `--test` in particular changes which tasks are meant:
    ```sh
    law run FLAF.Analysis.tasks.HistPlotTask \
      --version my_first_run --period Run3_2022 --workflow local --branches 0 --test 1000 \
      --print-status 3,1
    ```
    The numbers are *task depth* and *target-collection depth*. This is the quickest way to see
    which stage is done and where its output lives.

## If it fails

- **`InputFileTask` keeps appearing / Rucio errors** — usually a wrong era/version or an expired
  proxy. Re-run `voms-proxy-init`. See [Troubleshooting](../troubleshooting.md).
- **`law: command not found`** — you did not `source env.sh` in this shell.
- **Import errors / empty submodule dirs** — you cloned without `--recursive`; run
  `git submodule update --init --recursive`.

More symptoms and fixes are collected in [Troubleshooting](../troubleshooting.md).

## Next steps

You have run the whole pipeline once. Now learn what actually happened:

- [Tasks & LAW](../concepts/tasks-and-law.md) — what a task is and how LAW chains them.
- [Full workflow walkthrough](../workflow/walkthrough.md) — every stage, in order, with commands.
- [Running on HTCondor](../workflow/htcondor.md) — scale up from `local` to the batch system.
