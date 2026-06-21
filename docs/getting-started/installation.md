# Installation

## You install an *analysis*, not FLAF itself

This is the single most important thing to understand about setting up FLAF:

!!! warning "FLAF is a submodule — never clone it on its own to run an analysis"
    FLAF is shared machinery that lives **inside** each analysis repository as a git submodule.
    Its `env.sh` deliberately refuses to run unless an analysis has set it up first. To work with
    FLAF you clone one of the **analysis repositories**, which brings the right version of FLAF
    (and the other shared submodules) with it.

The available analysis repositories are:

| Repository | Channel | Clone URL |
|---|---|---|
| [`HH_bbtautau`](https://github.com/cms-flaf/HH_bbtautau) | HH → bb̄ττ | `git@github.com:cms-flaf/HH_bbtautau.git` |
| [`HH_bbWW`](https://github.com/cms-flaf/HH_bbWW) | HH → bb̄WW | `git@github.com:cms-flaf/HH_bbWW.git` |
| [`H_mumu`](https://github.com/cms-flaf/H_mumu) | H → μμ | `git@github.com:cms-flaf/H_mumu.git` |

The examples below use **HH_bbtautau** (the reference analysis); the steps are identical for the
others.

## 1. Clone the analysis repository (with submodules)

```sh
git clone --recursive git@github.com:cms-flaf/HH_bbtautau.git
cd HH_bbtautau
```

!!! danger "Do not forget `--recursive`"
    FLAF, Corrections, the inference tooling and analysis-specific submodules (e.g. SVfit,
    HHKinFit2, HHbtag) are all git submodules. Without `--recursive` you get empty directories and
    confusing import errors. If you already cloned without it, run:
    ```sh
    git submodule update --init --recursive
    ```

## 2. Set up the environment

```sh
source env.sh
```

Sourcing the analysis's `env.sh` is how you enter the FLAF environment. It:

1. sets `ANALYSIS_PATH` to the repository and points `FLAF_PATH` at the bundled `FLAF/` submodule;
2. **the first time**, builds everything it needs (this is the slow part):
    - a Python virtual environment `flaf_env` (from the CVMFS `LCG_108a` stack) under `soft/`;
    - a CMSSW area (`CMSSW_16_0_6`) used by the parts of the pipeline that need CMS software;
    - a standalone [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/)
      (`v10.4.2`) for statistical inference;
3. activates that environment and registers the `law` command and tab-completion;
4. defines a `cmsEnv` helper for running commands inside CMSSW;
5. points your VOMS proxy location at `data/voms.proxy`.

!!! note "The first `source env.sh` is slow; later ones are fast"
    The initial build compiles CMSSW and Combine and can take **tens of minutes** and a few GB of
    disk under `soft/`. It only happens once. Afterwards, `source env.sh` takes a few seconds and
    just activates the already-built environment. You must `source env.sh` **once per shell**
    (every new terminal).

??? info "Advanced: skipping the build on worker nodes (`FLAF_NO_INSTALL`)"
    Setting `FLAF_NO_INSTALL=1` makes `env.sh` fail instead of building anything if the
    environment is missing. This is used on batch workers, where the environment is shipped in a
    bundle rather than built on the node. You will not normally set it by hand. See
    [The environment](../concepts/environment.md) and [Running on HTCondor](../workflow/htcondor.md).

## 3. Index the LAW tasks

LAW needs to discover the available tasks. Run this once after installation (and again whenever
you add or rename a task):

```sh
law index --verbose
```

You should see the FLAF tasks listed (`InputFileTask`, `AnaTupleFileTask`, `HistPlotTask`, …).

## 4. Get a VOMS proxy

The pipeline reads and writes grid storage, which needs a short-lived **VOMS proxy**. Create one
(valid here for 8 days):

```sh
voms-proxy-init -voms cms -rfc -valid 192:00
```

Because `env.sh` sets `X509_USER_PROXY` to `data/voms.proxy`, the proxy is written where FLAF
expects it. Check it any time with:

```sh
voms-proxy-info
```

!!! warning "Expired proxy = mysterious failures"
    A common cause of "permission denied" / "file not found" errors on grid storage is simply an
    expired proxy. If something that worked yesterday fails today, re-run `voms-proxy-init`.

## 5. Create your `user_custom.yaml`

Finally, tell FLAF where *your* outputs should go and which physics model to use, by creating
`config/user_custom.yaml`. This file holds your personal, uncommitted settings (storage paths,
test vs. production model). The [Configuration guide](../configuration/user-custom.md) explains
every field; a minimal file is enough to start.

## What now?

- Smoke-test the whole chain on a handful of events: [Your first run](first-run.md).
- Not sure what "task", "era" or "process" mean here? [Key terms](key-terms.md).
- Want the full story? The [Concepts](../concepts/architecture.md) section explains the
  architecture and the environment in depth.
