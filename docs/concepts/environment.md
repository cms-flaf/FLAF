# The environment

`source env.sh` (from an analysis checkout) builds and activates everything FLAF needs. This page
explains what that environment contains, the variables it sets, and the few sharp edges to avoid.

## What `env.sh` sets up

The analysis `env.sh` sets `ANALYSIS_PATH` and `FLAF_PATH`, then hands off to `FLAF/env.sh`, which:

1. **Activates `flaf_env`** — a Python virtual environment built from the CVMFS `LCG_110a` stack
   (`x86_64-el9-gcc15-opt`), under `soft/flaf_env`. This provides Python, ROOT and the FLAF
   dependencies, and registers the `law` command with tab-completion.
2. **Provides CMSSW** — installs/uses `CMSSW_16_0_6` (compiler `gcc13`) under `soft/`. The ntuple
   production stages run inside it.
3. **Provides Combine** — builds standalone
   [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/) `v11.1.0` for statistical
   inference in `soft/HiggsAnalysis-CombinedLimit`, against the ROOT of `flaf_env`, and (for HH
   analyses) wires up the `inference`/`dhi` tooling. The build carries one FLAF patch
   (`run_tools/combine_root638_clipping.patch`): since ROOT 6.38 `RooRealVar::setVal` throws
   outside the variable's range instead of clipping, which makes Combine's `AsymptoticLimits`
   abort on the observed limit while still exiting 0, and the patch restores the clipping
   Combine relies on. The CMSSW area has its own `HiggsAnalysis/CombinedLimit`
   (`FLAF_CMSSW_COMBINE_VERSION`, `v10.4.2`), used by CombineHarvester, by the tasks that run in
   CMSSW and by jobs that run from bundles.
4. **Sets up grid access** — points `X509_USER_PROXY` at `data/voms.proxy` and initialises Rucio.
5. **Defines the `cmsEnv` helper** (see below).

!!! note "First source is slow, the rest are fast"
    The CMSSW and Combine builds happen only on the first `source env.sh`. After that it is a
    quick activation. You must source it **once in every new shell**.

!!! warning "When FLAF moves to another LCG release"
    The LCG release is pinned in `FLAF/env.sh`. The first `source env.sh` after an update of
    FLAF that changes it deletes and rebuilds `soft/flaf_env` and rebuilds the standalone Combine,
    which links against the ROOT of `flaf_env`. Run it in a fresh shell, with network access, and
    not in a checkout whose jobs are still queued or running: they use the same `soft/`. A
    production that reuses an existing `--version` with bundles also needs its unhashed `soft` and
    `cmssw` bundles deleted, see
    [Bundles are named after what they contain](../workflow/htcondor.md#bundles-are-named-after-what-they-contain);
    with an old `soft` bundle the jobs stop with
    `ERROR: FLAF environment not found at … and FLAF_NO_INSTALL=1`. `LCG_110a` brings Python 3.13
    and ROOT 6.40, which matters for personal scripts run in `flaf_env`. The standalone Combine
    built by earlier versions inside the CMSSW area (`soft/CMSSW_16_0_6/src/HiggsAnalysis/CombinedLimit/build`)
    is no longer used and can be deleted.

## Key environment variables

| Variable | Meaning |
|---|---|
| `ANALYSIS_PATH` | The analysis checkout (set by the analysis `env.sh`). |
| `FLAF_PATH` | The FLAF code in use. Defaults to `$ANALYSIS_PATH/FLAF`; override to develop FLAF (below). |
| `CORRECTIONS_PATH` | The Corrections code in use. Defaults to `$ANALYSIS_PATH/Corrections`. |
| `ANALYSIS_SOFT_PATH` | Where the built software lives (`$ANALYSIS_PATH/soft`). |
| `FLAF_ENVIRONMENT_PATH` | The `flaf_env` virtual environment (`$ANALYSIS_SOFT_PATH/flaf_env`). |
| `FLAF_CMSSW_BASE` | The CMSSW area used by the pipeline. |
| `FLAF_COMBINE_PATH` | The standalone Combine checkout and build (`soft/HiggsAnalysis-CombinedLimit`). |
| `FLAF_COMBINE_VERSION` / `FLAF_CMSSW_COMBINE_VERSION` | Combine versions of the standalone build and of the CMSSW area; `none` switches Combine off. |
| `ANALYSIS_DATA_PATH` | The local `data/` working area. |
| `X509_USER_PROXY` | Your VOMS proxy (`data/voms.proxy`). |
| `LAW_HOME` / `LAW_CONFIG_FILE` | LAW's home (`.law`) and config (`config/law.cfg`). |
| `FLAF_NO_INSTALL` | When `1`, `env.sh` refuses to build anything (used on batch workers). |

## `cmsEnv`: running inside CMSSW

Some commands must run inside the CMSSW runtime. The `cmsEnv` alias runs a command in a clean
shell with just the CMSSW variables set:

```sh
cmsEnv python3 my_cmssw_script.py
cmsEnv /bin/zsh         # an interactive CMSSW subshell
```

You will see it most often around statistical-inference commands that call `combine`.

## Developing shared submodules

`FLAF` and `Corrections` are pinned **submodules** inside the analysis. If you edit the framework
in place and run the pipeline, your edits may be ignored — because the run uses the submodule copy.
The environment solves this cleanly: `FLAF_PATH` and `CORRECTIONS_PATH` are **inputs** to
`env.sh`. If they are already set when you source it, they are respected; otherwise they default to
the submodule copies.

So to run against an edited copy of FLAF, set `FLAF_PATH` to that copy **before** sourcing:

```sh
export FLAF_PATH=/path/to/your/edited/FLAF
source env.sh           # everything downstream now uses the edited FLAF
```

Everything derived from it — `PYTHONPATH`, the code shipped in batch bundles, the worker bootstrap
— follows automatically. When `FLAF_PATH`/`CORRECTIONS_PATH` differ from the submodule copy,
`env.sh` also enables `PYTHONSAFEPATH` and prepends the right parent directory so the edited copy
wins for `import FLAF` / `import Corrections` (which are namespace packages). On HTCondor, non-bundle
jobs receive these paths (the AFS area is mounted on workers); bundle jobs ship the edited code
inside the tarball instead. See [Running on HTCondor](../workflow/htcondor.md) and
[Contributing](../contributing.md).

## Sharp edges

!!! danger "Do not strip `LD_LIBRARY_PATH` (`env -i`)"
    Running the environment under `env -i` (a fully empty environment) removes `LD_LIBRARY_PATH`,
    which ROOT/cling needs — you get cryptic library/JIT failures. If you must launch a clean
    background shell, preserve `LD_LIBRARY_PATH` (and `HOME`, `PATH`).

!!! danger "Do not source via `bash -c \"source env.sh\"`"
    `env.sh` locates itself through `BASH_SOURCE`/`$0`. Sourcing it inside `bash -c "..."` breaks
    that detection and sets the wrong `ANALYSIS_PATH`. Source it directly in your shell, or put the
    commands in a script file and run that script.

!!! warning "One environment per shell — beware cross-analysis contamination"
    The environment caches paths in variables (`FLAF_PATH`, `ANALYSIS_SOFT_PATH`, …). Sourcing a
    *second* analysis's `env.sh` in the same shell, or reusing a shell that already has another
    analysis's variables, can pick up the wrong `flaf_env`. Use a fresh shell per analysis. When
    scripting background runs, unset the `FLAF_*`/`ANALYSIS_*` variables first (see
    [Troubleshooting](../troubleshooting.md#cross-analysis-environment-contamination)).
