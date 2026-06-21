# Troubleshooting & FAQ

The most common ways a FLAF run goes wrong, and how to fix them. If your symptom is not here, check
the job logs (run with `--transfer-logs` on HTCondor) and the task status
(`--print-status 3,1`).

## `law: command not found`
You did not `source env.sh` in this shell. Every new terminal needs it once
([Installation](getting-started/installation.md)).

## Import errors / empty submodule directories
You cloned without `--recursive`, so submodules (FLAF, PlotKit, physics tools) are empty. Fix:

```sh
git submodule update --init --recursive
```

## A run unexpectedly drops into `InputFileTask` / DAS errors
For a from-scratch production, `InputFileTask` running first is normal. But if a run that should
reuse existing outputs keeps re-resolving inputs, or fails here, the cause is almost always a
**wrong `--period` or `--version`** (so the expected upstream outputs aren't found and LAW falls
back to regenerating them), or an **expired proxy**. Double-check the era/version, and:

```sh
voms-proxy-info        # is it still valid?
voms-proxy-init -voms cms -rfc -valid 192:00
```

## "Permission denied" / "file not found" on storage
Usually an **expired VOMS proxy** — grid/EOS access needs a valid one. Re-run `voms-proxy-init`. If
it persists, confirm your `fs_*` paths in `user_custom.yaml` are correct and writable
([Storage](concepts/storage.md)).

## "Task not found" after adding a task
LAW's index is stale. Re-run:

```sh
law index --verbose
```

Needed after **adding/renaming/moving** a task class (not after editing an existing one's body).

## EOS read-after-write lag
EOS is eventually consistent: a file you just wrote can be briefly invisible to an existence check
(seconds, occasionally longer). In normal pipeline use FLAF tolerates this. If **your own** script
checks for freshly written outputs and intermittently "can't find" them, don't trust a single
`exists()` — list the parent directory and retry a few times with a short delay.

## Cross-analysis environment contamination
The environment caches paths in variables (`FLAF_PATH`, `ANALYSIS_PATH`, `ANALYSIS_SOFT_PATH`, …).
Reusing a shell that already set up a *different* analysis can pick up the wrong `flaf_env` and
produce baffling failures.

- **Interactive:** use a **fresh shell per analysis** and `source env.sh` there.
- **Scripted/background runs:** unset the FLAF/analysis variables before sourcing, but **keep**
  `LD_LIBRARY_PATH`, `HOME` and `PATH`:

```sh
unset FLAF_ENVIRONMENT_PATH ANALYSIS_SOFT_PATH LAW_HOME LAW_CONFIG_FILE \
      ANALYSIS_PATH ANALYSIS_DATA_PATH FLAF_PATH FLAF_CMSSW_BASE \
      FLAF_CMSSW_ARCH FLAF_CMSSW_VERSION FLAF_COMBINE_PATH \
      X509_USER_PROXY VIRTUAL_ENV PYTHONPATH
cd /path/to/<analysis>
source env.sh
```

## ROOT/cling library or JIT errors in a background run
You launched the environment under `env -i`, which strips `LD_LIBRARY_PATH` (ROOT/cling needs it).
Preserve it (and `HOME`, `PATH`) when starting a clean shell. See
[The environment](concepts/environment.md#sharp-edges).

## `source env.sh` sets the wrong path / fails to locate itself
You sourced it via `bash -c "source env.sh"`. That breaks `BASH_SOURCE` self-detection and sets the
wrong `ANALYSIS_PATH`. Source it directly in your interactive shell, or put your commands in a
**script file** and run that file.

## HH→bb̄WW: the run sits in `AnalysisCacheTask` for a long time
Expected on a cold cache: `AnalysisCacheTask` computes the b-tag shape weights and can take roughly
an hour per branch. Reuse an existing cache across runs with a
[per-task version override](workflow/arguments.md#per-task-version-overrides) instead of
recomputing it every time.

## A backgrounded `law run` won't stop when I kill it
Killing the parent leaves child `law`/job processes alive. Kill by pattern, and remove batch jobs:

```sh
pkill -f "version=<your_version>"
condor_rm <cluster>      # if you submitted to HTCondor
```

## My edits to FLAF/Corrections are ignored
You edited the submodule copy but the run used a different one — or vice-versa. The run uses
`FLAF_PATH`/`CORRECTIONS_PATH`; set them to your edited copy **before** `source env.sh`. See
[Developing shared submodules](concepts/environment.md#developing-shared-submodules).

## The first `source env.sh` takes forever
Expected: the first time it builds CMSSW and Combine (tens of minutes, a few GB under `soft/`).
Subsequent sources are quick. Don't interrupt the first build.
