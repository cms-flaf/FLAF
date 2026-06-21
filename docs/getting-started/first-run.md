# Your first run

This page walks you through a **minimal end-to-end run**: a single command that exercises the
*whole* FLAF pipeline — from CMS NanoAOD all the way to a histogram plot — on just a handful of
events. It is the same smoke test that the [continuous integration](../ci/integration-pipeline.md)
runs on every change, so if it works for you, your setup is healthy.

## Before you start

You need a working environment from [Installation](installation.md):

```sh
cd HH_bbtautau          # or your analysis repository
source env.sh           # once per shell
voms-proxy-info         # confirm you have a valid proxy
```

You also need two settings in `config/user_custom.yaml`:

- a storage location for outputs (the `fs_default` field), and
- `phys_model: TestModel` — a small, fast subset of processes meant for testing.

See the [Configuration guide](../configuration/user-custom.md) for a ready-to-copy minimal file.

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
merging analysis ntuples, computing observables, filling and merging histograms — before making
the plot. You do not run the intermediate tasks yourself.

### What each argument means

| Argument | Meaning |
|---|---|
| `FLAF.Analysis.tasks.HistPlotTask` | The task you want — here, the plotting task (the end of the chain). |
| `--version my_first_run` | A label for this run. Outputs are grouped under it, so you can keep runs apart. Use any name. |
| `--period Run3_2022` | Which data-taking [era](../concepts/eras.md) to process. |
| `--workflow local` | Run on this machine (not the batch system). Good for testing. |
| `--branches 0` | Only the first work unit. `HistPlotTask` has one branch per variable, so this plots a single variable. |
| `--test 1000` | Process only 1000 events per input file — fast, just to check the machinery. |

These and many more options are catalogued in [Command arguments](../workflow/arguments.md).

## What to expect

- LAW prints a dependency tree and then runs the stages bottom-up. Because the early stages
  produce analysis ntuples from CMS NanoAOD (using CMSSW and reading from the grid), **the first
  run is not instant** even with `--test 1000` — budget a little time.
- Output is written under the storage you configured in `user_custom.yaml`, organised by version
  and era. A local copy of small artifacts also appears under `data/`.
- When the top task finishes, LAW reports success for `HistPlotTask`.

!!! tip "Check progress without running anything"
    In another shell (after `source env.sh`), ask LAW for the status of the dependency tree:
    ```sh
    law run FLAF.Analysis.tasks.HistPlotTask \
      --version my_first_run --period Run3_2022 --print-status 3,1
    ```
    The numbers are *task depth* and *file-collection depth*. This is the quickest way to see
    which stage is done and where its output lives.

## If it fails

- **`InputFileTask` keeps appearing / DAS errors** — usually a wrong era/version or an expired
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
