# GitHub Actions

FLAF uses **two** continuous-integration systems:

| System | Where | Purpose |
|---|---|---|
| **GitHub Actions** | GitHub | Fast code-quality and sanity checks on pull requests, and the documentation build. |
| **FLAF integration** | GitLab CI (CERN) | The full pipeline run that checks physics correctness. Triggered by a bot comment — see [Integration pipeline](integration-pipeline.md). |

This page covers the GitHub Actions checks. (The integration test can also run on GitHub Actions
as a backup; that backend is described on the [Integration pipeline](integration-pipeline.md#the-github-actions-backend)
page.)

## Shared, reusable workflows

The other repositories (the three analyses, Corrections, StatInference, PlotKit) don't duplicate
CI logic. Each workflow is a thin wrapper that calls the shared implementation in FLAF:

```yaml
jobs:
  my-job:
    uses: cms-flaf/FLAF/.github/workflows/<workflow>.yaml@main
    secrets: inherit
```

So fixing a check in FLAF fixes it everywhere. The exceptions run their own steps:
`deploy-docs.yaml`, which each analysis carries as a standalone copy; the `era-rename-check` job
that the three analyses, Corrections and StatInference add to their `repo-sanity-checks.yaml`,
which runs FLAF's `run_tools/check_era_rename.py` from a pinned FLAF commit; and StatInference's
`unit-tests.yaml`.

A checkout helper inside the shared workflows (`.github/actions/checkout-flaf`) makes the FLAF
tooling — `.yamllint`, `.clang-format`, `test/` — available in the other repos. It checks out the
`FLAF` submodule at **FLAF's `main`** (or clones it), not at the commit the repository pins, so an
analysis PR is checked with the current FLAF tooling and configuration. `unit-tests.yaml` is the
exception: an analysis's unit tests import FLAF, so they run against the FLAF commit the analysis
pins (see [Unit tests](#unit-tests)).

## The standard checks

| Workflow | Runs on | What it checks | Repos |
|---|---|---|---|
| `formatting-check.yaml` | PRs to `main` | The files the PR changes: `black --check` (Python), `clang-format --dry-run --Werror` (C++: `.cpp`, `.h`, `.hpp`, `.cc`), `yamllint -s` (YAML); `black` and `yamllint` are installed at the versions `run_tools/mk_flaf_env.sh` pins, so that the check agrees with `run_tools/apply_format.sh` run in `flaf_env`. The repository's own `.clang-format`/`.yamllint` is used if it has one, otherwise FLAF's. In FLAF, when a Python file changed, it also runs `flake8 --select=F821,F811` (undefined or shadowed names) over the whole package: a dropped import surfaces only where the name is used, which can be a branch that runs on a CRAB worker alone. | all seven (the `flake8` step: FLAF only) |
| `unit-tests.yaml` | PRs to `main` and pushes to `main`; reusable (`workflow_call`, no inputs) | Every `test/**/test_*.py` with `pytest`, except the tests marked as needing ROOT, CVMFS or a grid proxy — see [Unit tests](#unit-tests) — against the law and luigi that `run_tools/mk_flaf_env.sh` pins (`test_flaf_env.py` checks that every workflow installs a package that script pins at the same version). `test_flaf_env.py` runs `env.sh` in bash and in zsh; the workflow installs zsh, and under CI a missing zsh fails those cases. Many of them cover code that otherwise runs on a batch node only. Another step, in FLAF only, runs `run_tools/check_era_rename.py` over FLAF's tracked `.py`, `.yaml`, `.yml`, `.sh`, `.cfg`, `.ini` and `.json` files (not the docs, not C++): no leftover of the `period` → `era` rename (a `period` identifier outside its allow-list, `getattr`/`.get` with `"period"`, `--period`, `{period}`, and, inside task classes, the era tuple used as a string); a line carrying `# noqa: era-rename` and the files the step excludes are skipped. The same script reports era-scope leaks, which keep a task over several eras from reading another era's value: an `era` keyword in `X.req(...)`, `req_params`, `req_branch` or `req_workflow` outside `with_eras`, and `self.era[...]` read inside a task class outside `only_era` (a line carrying `# noqa: era-scope` is skipped). | FLAF, and the three analyses through their own `unit-tests.yaml` wrapper; StatInference has its own, which runs `test/test_law_chain.py` (its law tasks on the real FLAF and law classes) against a pinned FLAF commit |
| `repo-sanity-checks.yaml` | every PR | Two jobs: the growth of the repository after a simulated squash merge (fails above 1024 KiB unless the PR has the `big changes` label), and no binary files among the changed files (use Git LFS for those). | all seven |
| `test-setup-loading.yaml` | PRs to `main` in the analyses | Actually loads `Setup` for the seven Run 3 eras listed in the analysis's wrapper (a real load with ROOT mocked, not a dry run) and checks that every shape weight in each era's `weights.yaml` comes from a correction that era enables — catches config typos and broken references early. | HH_bbtautau, HH_bbWW, H_mumu |
| `test-era-dimension.yaml` | reusable only (`workflow_call`, input `eras`: space-separated, at least two) | Constructs every FLAF stage over all the given eras on the analysis's own configuration, with each workflow type it supports, and takes it through what the scheduler does before anything runs (`test/test_era_dimension.py`; no grid, ROOT and Corrections are stand-ins). It fails when a setting that a run over several eras reads once (`fs_default`, `crab`, a producer's resources, …) differs between the eras, or when a per-era value is read on a task over several. See [Eras](../concepts/eras.md#settings-that-must-agree-between-the-eras). | called from an analysis's wrapper, which passes its eras |
| `trigger-flaf-integration.yaml` | new or edited PR comments | Parses a `@cms-flaf-bot` comment and starts the integration test. See [Integration pipeline](integration-pipeline.md). | all seven |
| `deploy-docs.yaml` | PRs (to any branch) and pushes to `main` that touch `docs/**`, `mkdocs.yml` or the workflow itself; manual dispatch | `mkdocs build --strict`; on a push or a manual dispatch it also publishes the site with `mkdocs gh-deploy`. | FLAF and the three analyses |

!!! warning "What does *not* run automatically"
    - `test-setup-loading` does **not** run on FLAF, Corrections, StatInference or PlotKit PRs:
      FLAF's copy is a reusable workflow only, called by the analyses. A FLAF change that breaks
      config loading shows up in the [integration pipeline](integration-pipeline.md), or on the
      first analysis PR after it is merged into FLAF's `main`.
    - `test-era-dimension` has no trigger of its own in FLAF either: it is a reusable workflow that
      an analysis calls with its eras.
    - `formatting-check`, `test-setup-loading` and FLAF's two config checks below run only on PRs
      whose base is `main`; a PR to another branch (e.g. `next_prod`) gets only
      `repo-sanity-checks` (and `deploy-docs` if it touches the docs).
    - The unit tests marked `needs_root`, `needs_cvmfs` or `needs_proxy` (`test_stitching_variables.py`,
      the `test_fuse_*.py` suites, the checks against the CRAB client on CVMFS, …) are deselected
      by `unit-tests` and run by no workflow: run them in an analysis environment — see
      [Contributing](../contributing.md#run-the-tests).

FLAF itself additionally runs, on PRs to `main`:

| Workflow | What it checks |
|---|---|
| `cross-section-check.yaml` | When a `config/crossSections*.yaml` changes: `test/checkCrossSections.py` for the seven Run 3 eras — the cross-section files parse, their entries are valid, and every cross-section a dataset references exists. |
| `ds-consistency-check.yaml` | When a `config/…/datasets.yaml` changes: `datasets.yaml` entries are well-formed (generator, resolvable cross-section, consistency across eras) via `test/checkDatasetConfigConsistency.py`, and follow the naming rules via `test/checkDatasetNaming.py`. |

FLAF also holds `integration-test.yaml`, the [GitHub Actions backend](integration-pipeline.md#the-github-actions-backend)
of the integration test; it runs only when the bot trigger dispatches it (`ci_backend: github`).

### Unit tests

`unit-tests.yaml` runs, on Python 3.13 with only `pyyaml`, `luigi`, `law` and `pytest` installed
(pytest 9.0.3, the version of the LCG view, since `test/conftest.py` builds on pytest's collection
classes),

```sh
python -m pytest -q -ra --strict-markers -m "not needs_root and not needs_cvmfs and not needs_proxy" test
```

pytest collects every `test/**/test_*.py`, so adding a test changes nothing under `.github/`. The
markers and the collection rules are in FLAF's `test/conftest.py`; how to mark a test is in
[Contributing](../contributing.md#unit-tests).

- A module that fails to import and declares none of the markers stops the run: a test never
  disappears because the runner lacks something. `--strict-markers` rejects a misspelt marker,
  and a module that leaves a stand-in for ROOT in `sys.modules` stops the run as well: the next
  module would import the stand-in instead of failing to import ROOT.
- The end of the log lists every deselected test file with its marker (a module that the runner
  cannot even import shows its import error), and every test file that ran no test: in FLAF,
  the scripts named `test_*.py` that other workflows and the docs run by path. `-ra` lists the
  skipped tests with their reason.
- An analysis without a `test/` directory, or whose tests are all deselected (pytest exit code
  5), passes with a notice in the run summary: it has no unit test until it adds one. In FLAF,
  which has tests, either fails the job, as it can only mean that the collection broke (a moved
  `test/`, an edited `-m` expression, a broken `test/conftest.py`).

Each analysis calls it from its own `.github/workflows/unit-tests.yaml`
(`uses: cms-flaf/FLAF/.github/workflows/unit-tests.yaml@main`, no inputs). The workflow then checks
out the analysis with its `FLAF` submodule only, at the commit the analysis pins (Corrections would
need ROOT, and `inference` is on CERN GitLab), and loads that FLAF's `test/conftest.py` with
`-p FLAF.test.conftest`. An analysis whose FLAF pin has no `test/conftest.py` yet fails there
with `Error importing plugin "FLAF.test.conftest"` until it advances the pin. The
`period` → `era` leftover step runs in FLAF only; the analyses run theirs in `repo-sanity-checks`.

## Passing the checks before you push

Formatting is enforced. The convenience script applies the formatters to every file changed by
the commits on your branch (`git log origin/main..HEAD`) — so it only sees **committed** changes.
Commit first, then format and amend. Run it from the root of the repository you changed, with
`env.sh` sourced (it takes `.clang-format`/`.yamllint` from `$ANALYSIS_PATH`, else `$FLAF_PATH`):

```sh
git commit -m "..."
bash FLAF/run_tools/apply_format.sh     # from an analysis; in FLAF itself: bash run_tools/apply_format.sh
git commit --amend --no-edit            # if it reformatted anything
```

black and clang-format rewrite files in place; yamllint only **reports** problems, which you fix
by hand. `--dry-run` checks without changing anything, like CI does.

Or run them individually. CI uses the `.clang-format`/`.yamllint` at the root of the repository
being checked if it has one (H_mumu and Corrections carry their own `.clang-format`), otherwise
FLAF's — use the same file. For a Corrections change, that is `Corrections/.clang-format`, which
`apply_format.sh` does not pick up.

```sh
black <file.py>                                               # Python
clang-format -i --style "file:$FLAF_PATH/.clang-format" <f>   # C++
yamllint -s -c $FLAF_PATH/.yamllint <file.yaml>               # YAML
```

If you edited `datasets.yaml`, also run the consistency check from
[Datasets](../configuration/datasets.md#validate-the-dataset-config). See
[Contributing](../contributing.md) for the full pre-PR checklist.

!!! note "Required secrets"
    The bot-trigger workflow needs the secrets `FLAF_INTEGRATION_TOKEN` (GitLab trigger) and
    `FLAF_GITHUB_TOKEN` (organisation-membership check and the reply comment), passed on with
    `secrets: inherit`, so they must be available to every repo with the trigger. The GitHub
    Actions integration backend also uses `GRID_USERCERT`, `GRID_USERKEY`, `GRID_PASSWORD` and
    `SSH_PRIVATE_KEY`. The quality checks need no secrets.
