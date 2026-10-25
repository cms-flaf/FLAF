# Contributing

How to make a change to FLAF (or an analysis) and get it merged. The same workflow applies to the
shared submodules (Corrections, StatInference, PlotKit).

## Branch, don't commit to `main`

Always work on a **topic branch** and open a pull request — never commit directly to `main`.
Nearly all PRs come from personal **forks**: pushing a branch to `cms-flaf/<repo>` itself needs
write access to that repository. Fork the repository on GitHub once and add the fork as a remote:

```sh
git remote add fork git@github.com:<your-user>/<repo>.git      # once per checkout
git fetch origin
git checkout -b my-short-topic-name origin/main
# ... make changes ...
git commit -m "short, clear one-line description"
git push fork my-short-topic-name        # then open a PR against cms-flaf/<repo> on GitHub
```

If your change spans repositories (e.g. FLAF **and** an analysis), use the **same branch name** in
each affected repo so reviewers can find the matching pieces, and test them together by listing
the other PRs in the bot comment (see
[Integration pipeline](ci/integration-pipeline.md#triggering-it-cms-flaf-bot-please-test)).

!!! note "Changing FLAF or Corrections from an analysis checkout"
    To run an analysis against your edited framework, point `FLAF_PATH`/`CORRECTIONS_PATH` at the
    edited copy before sourcing `env.sh` — see
    [The environment](concepts/environment.md#developing-shared-submodules). Mind its two
    warnings: the edited copy's directory must itself be named `FLAF` (or `Corrections`),
    because `env.sh` puts its *parent* directory on `PYTHONPATH`; and `Setup` reads the framework
    configuration from `<analysis>/FLAF/config` rather than from `FLAF_PATH`, so make
    configuration edits in the submodule copy.

## Format your commits

Formatting is CI-enforced ([GitHub Actions](ci/github-actions.md)). `apply_format.sh` applies all
formatters at once, but only to the files changed by the **commits** on your branch
(`git log origin/main..HEAD`), so the order is commit, format, amend. Run it from the root of the
repository you changed, with the environment active:

```sh
source env.sh                            # in the analysis checkout
git commit -m "..."
bash FLAF/run_tools/apply_format.sh      # from an analysis; in FLAF itself: bash run_tools/apply_format.sh
git commit --amend --no-edit             # if it changed anything
```

black (Python) and clang-format (C++) fix the files in place; yamllint (YAML) only reports
problems, which you fix by hand. `bash FLAF/run_tools/apply_format.sh --dry-run` checks without
changing anything. You can also run the tools on individual files — see
[GitHub Actions](ci/github-actions.md#passing-the-checks-before-you-push).

## Re-index after adding a task

If you added, renamed or moved a LAW task class, refresh the index so it can be found:

```sh
law index --verbose
```

## Writing code for several eras

A stage over several eras (`--era A,B`) is one workflow whose branches each belong to one era
([Eras](concepts/eras.md#running-several-eras)); the second model, a workflow without era
branches, is described below. The base class `Task` in
`FLAF/run_tools/law_customizations.py` keeps the two scopes apart, and a task class follows these
rules:

- **Class flags.** `allow_multi_era = True` lets a class be constructed over several eras (without
  it `Task.__init__` raises `<Task> takes one era`). `era_branches = True` marks a workflow whose
  branches carry the era: its branch map over several eras is the union of the eras' own maps,
  numbered `era id × 1 000 000 + the number in the era's own map`, with per-era requirements and
  per-era submission state. It needs `allow_multi_era`. Every stage class of `AnaProd/tasks.py` and
  `Analysis/tasks.py` sets both.
- **Read per-era values through `only_era`, never `self.era[0]`.** `self.only_era` is the era of
  a task over one era and raises `EraScopeError` on a task over several; `self.setup` and the paths
  and names built from it behave the same. Over several eras use `setup_for(era)`, `single(era)`
  (this stage over one era) or `common(getter, what)` for a setting that holds for all the eras,
  which raises `EraDependentValueError` when they disagree (the settings read this way are listed
  in [Eras](concepts/eras.md#settings-that-must-agree-between-the-eras)). `fs_default` and
  `cmssw_env` are readable on any task: the first is checked to agree when the task is
  constructed, the second does not depend on the era.
- **Branch-level code runs on one era.** A branch of a run over several eras is the branch task of
  its era's own workflow, so `output()`, `run()`, a branch's `requires()` and the payload calls see
  one era and that era's own branch numbers.
- **The requirements of a stage go in `stage_workflow_requires()`**, not in `workflow_requires()`:
  the latter is a dispatcher that, over several eras, asks each era that still needs work for its
  `stage_workflow_requires()` and merges the answers into one requirement per stage. Call
  `super().stage_workflow_requires()` for the bundles.
- **Build the branch map through `cached_branch_map`**: `create_branch_map` returns
  `self.cached_branch_map(self._build_branch_map)`, and the union over the eras is made from the
  single-era maps.
- **A workflow over several eras without era branches** (`allow_multi_era` only; StatInference's
  card stages are the example) keeps law's own branches, and every branch task keeps all the eras
  of the run, so branch-level code reads an era's values through `setup_for(era)` and run-wide ones
  through `common()`. FLAF's `store_parts()` and `remote_log_dir_target()` are per era and refuse
  several: override both so that the job file and the logs do not depend on `--era` (a later run
  over more eras then resumes the same jobs). Build its map without `cached_branch_map`, which
  refuses several eras without era branches, and keep anything that must refuse an era set out of
  `__init__`: FLAF constructs the workflow over each single era for its luigi resource tokens.
  Bundles (and with them CRAB) read per-era configuration and are not available to such a
  workflow. `test/test_era_jobs.py` (`EraCardWorkflow`) guards this shape.
- **Name eras in requirements only through `with_eras(eras, branches)`**, which derives this stage
  over other eras and hands on the per-task values (`--tasks-per-job`, `--max-runtime`,
  `--n-cpus`, `--crab-memory`). `run_tools/check_era_rename.py` reports an `era` keyword in
  `X.req(...)`, `req_params`, `req_branch` or `req_workflow` outside `with_eras`, and `self.era[...]`
  in a task class outside `only_era`; a deliberate exception carries `# noqa: era-scope` on the
  line (the rename findings use `# noqa: era-rename`).
- **A producer that reads AnaTuples checks the era of their events:** `Utilities.book_era_check(df, era)`
  before the event loop and `Utilities.check_era(count, era, what)` after it, as `AnaTupleMergeTask`,
  `HistTupleProducerTask` and `AnalysisCacheTask` do.

## Validate config changes

- Edited `datasets.yaml`? Run the
  [consistency check](configuration/datasets.md#validate-the-dataset-config).
- Added an era or changed config loading? Make sure `Setup` still loads for every era — see
  [Run the tests](#run-the-tests). CI's `test-setup-loading` does this only on analysis PRs,
  so a FLAF change is not checked there.

## Run the tests

Most of the framework cannot be tested without CERN infrastructure, but a few checks run locally,
from the analysis checkout with `env.sh` sourced:

- **Config loading** — `python3 FLAF/test/test_setup_loading.py Run3_2022 Run3_2022EE …` loads
  `Setup` for each listed era, as `test-setup-loading` does in CI. It finds the analysis from its
  own location, so run the copy inside the analysis checkout.
- **Smoke test** — `python3 FLAF/test/test_hello_world.py --version <v> --workflow local`
  (or `htcondor` or `crab`; add `--bundle` to go through a bundle, `--force-fail` to check the log
  transfer of a failing job, `--user-custom <file>` for an extra configuration file) runs
  `FLAF.test.hello_world_task.HelloWorldTask` for one branch and prints `PASS` or `FAIL`. `--era`
  defaults to `Run3_2022EE`; with several (`--era Run3_2022,Run3_2023`) the run is one workflow
  over all of them, with branch 0 of each era selected by its number in that run, and every era's
  log and output are checked where a run over that era alone puts them. It first deletes
  `data/<v>/` and the task's remote log, so use a throwaway version name.
- **Every stage over several eras** — `python3 FLAF/test/test_era_dimension.py Run3_2022 Run3_2022EE …`
  (from the analysis checkout, at least two eras) constructs every FLAF stage over those eras on
  the analysis's own configuration, with each workflow type it supports, and takes it through what
  the scheduler does before anything runs (proxy, resources, requirements, outputs,
  completeness). It fails on a setting that differs between the eras where it has to agree, or on
  a per-era value read on a task over several eras. No grid is needed: the `fs_*` are redirected
  to a local directory, and ROOT and Corrections are stand-ins. The reusable GitHub workflow
  `test-era-dimension.yaml` runs it ([GitHub Actions](ci/github-actions.md)).
- **Unit tests** — the other `FLAF/test/test_*.py` files (path cache, bundle hashing, stitching
  variables, cost model, …), and the `test/test_*.py` of an analysis, are pytest modules, mostly
  `unittest` classes; see [Unit tests](#unit-tests) for where they go, how to mark the ones that
  need ROOT, CVMFS or a grid proxy, and how to run them. Run the ones covering the code you
  changed, and extend them when you change that code.
- **Undefined names** — `run_tools/apply_format.sh`, when run in FLAF, also runs
  `flake8 --select=F821,F811` over the whole package, as `formatting-check` does in CI.

### Unit tests

A unit test goes into `test/test_<topic>.py` of the repository whose code it covers, FLAF or an
analysis. The `unit-tests` workflow collects every `test/**/test_*.py` there, so a new test needs
no change under `.github/` (a PR that changes `.github/` can be
[tested by the bot](ci/integration-pipeline.md) on a comment by `kandrosov` only). The GitHub runner has only `pyyaml`, `luigi`, `law` and `pytest` (9.0.3, as in the LCG view): no ROOT,
no `/cvmfs`, no grid proxy. A test that needs one of them declares it with a marker, and CI
deselects it ([GitHub Actions](ci/github-actions.md#unit-tests)):

| Marker | The test needs |
|---|---|
| `needs_root` | ROOT or another package of the analysis environment (numpy, uproot, awkward, Corrections) |
| `needs_cvmfs` | `/cvmfs`, e.g. the CRAB client or `SITECONF` |
| `needs_proxy` | a valid grid proxy |

```python
import pytest

pytestmark = pytest.mark.needs_root  # the whole module


@pytest.mark.needs_cvmfs  # or one class, or one test method
class TheCrabClientItself(unittest.TestCase):
    ...
```

- A test that uses no ROOT itself but reaches it through an import (`FLAF.Common.Setup` and
  `FLAF.Common.Utilities` import ROOT at module level, and so does `run_tools/law_customizations.py`
  through them) takes no marker. It imports through a stand-in where ROOT is missing, as
  `test_crab_credentials.py` and `test_bundle_hash.py` do, and so it runs in CI. Keep `needs_root`
  for tests that use ROOT themselves.
- A module that cannot be imported without what it needs (`import ROOT` at the top, for the
  tests' own use) declares the marker module-wide with `pytestmark`. The marker is read from the
  source, and the failed import is reported as one test that carries it, so the marker deselects
  the module. A module without a marker that fails to import stops the run.
- The test modules share one process, so a stand-in serves the import only and is removed from
  `sys.modules` right after: a module collected later must still get the real ROOT, or fail to
  import it. A module that leaves a stand-in for ROOT behind stops the run.
- `FLAF/test/conftest.py` defines the markers and these rules. Its `collect_ignore` lists the
  scripts named `test_*.py` that are not unit tests (`test_setup_loading.py`,
  `test_era_dimension.py`, `test_hello_world.py`, `GenProcess/`).

Run them from the root of the repository:

```sh
python -m pytest                     # FLAF: every test, which needs an analysis environment
python -m pytest -m "not needs_root and not needs_cvmfs and not needs_proxy"   # what CI runs
python -m pytest -p FLAF.test.conftest test   # an analysis: its test/, with the pinned FLAF's markers
```

The marked tests run in an analysis environment: in the analysis checkout, after
`source env.sh`, `cd FLAF && python -m pytest -m "needs_root or needs_cvmfs or needs_proxy"`
(`test_shape_weight_registry.py` takes Corrections from `ANALYSIS_PATH`). The end of the output
lists the deselected test files with their markers and every test file that ran no test.

## Open the PR and run the checks

On the pull request:

1. The GitHub Actions checks run automatically: repository sanity on every PR, formatting on PRs
   to `main` in every repo, the unit tests on PRs to `main` in FLAF and the analyses, `test-setup-loading` on analysis PRs to `main`, the cross-section and
   dataset checks on FLAF PRs to `main` that change those files, and the docs build when the docs
   change — see
   [GitHub Actions](ci/github-actions.md).
2. For a real physics check, a member of the `cms-flaf` GitHub organisation triggers the full
   pipeline with a `@cms-flaf-bot please test` comment (if the PR changes `.github/`, only
   `kandrosov` can) — see [Integration pipeline](ci/integration-pipeline.md).

### Pre-PR checklist

- [ ] On a topic branch from a fresh `origin/main` (not `main`)
- [ ] `apply_format.sh` run on the committed changes, and the result amended into the commit
- [ ] `law index --verbose` run if you added/renamed a task
- [ ] dataset consistency check run if you touched `datasets.yaml`
- [ ] `test_setup_loading.py` run if you changed configuration or its loading
- [ ] the unit tests for the code you changed pass, including the marked ones CI deselects
- [ ] no binary files staged (use Git LFS for those)
- [ ] docs updated if behaviour or interfaces changed (see below)

## Editing the documentation

These docs are [MkDocs](https://www.mkdocs.org/) with the
[Material](https://squidfunk.github.io/mkdocs-material/) theme; the sources are the Markdown files
under `docs/` and the navigation is in `mkdocs.yml`. To preview locally:

```sh
pip install mkdocs-material          # once, e.g. in a throwaway venv
mkdocs serve                         # live preview at http://127.0.0.1:8000
mkdocs build --strict                # what to run before committing: fails on broken links
```

`mkdocs build --strict` catches broken internal links, missing nav entries and missing assets —
run it before you commit doc changes. The same strict build runs on every PR that touches the
docs, and merging to `main` publishes the site. There is a :material-pencil: **edit** action on
every page that takes you straight to the source file on GitHub.

Guidelines for docs changes:

- Keep framework-wide material here in FLAF; put analysis-specific material in that analysis's
  `docs/` (see [Analyses](analyses.md)). Link rather than duplicate.
- Prefer concrete, copy-pasteable commands, and flag caveats with admonitions
  (`!!! warning`, `!!! tip`).
- Remember the audience includes physicists new to the tooling — define terms or link the
  [Glossary](glossary.md).
