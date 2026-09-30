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
  (or `htcondor`; add `--bundle` to go through a bundle, `--force-fail` to check the log transfer
  of a failing job) runs `FLAF.test.hello_world_task.HelloWorldTask` for one branch and prints
  `PASS` or `FAIL`. `--period` defaults to `Run3_2022EE`. It first deletes `data/<v>/` and the
  task's remote log, so use a throwaway version name.
- **Unit suites** — the other `FLAF/test/test_*.py` files (path cache, bundle hashing, stitching
  variables, cost model, …) are standalone scripts, mostly `unittest` modules; run one directly,
  e.g. `python3 FLAF/test/test_path_cache.py`. **No CI workflow runs them**, so run the ones covering
  the code you changed, and extend them when you change that code.

## Open the PR and run the checks

On the pull request:

1. The GitHub Actions checks run automatically: repository sanity on every PR, formatting on PRs
   to `main` in every repo, `test-setup-loading` on analysis PRs to `main`, the cross-section and
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
- [ ] the unit suites for the code you changed pass
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
