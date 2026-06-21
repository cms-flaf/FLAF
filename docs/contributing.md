# Contributing

How to make a change to FLAF (or an analysis) and get it merged. The same workflow applies to the
shared submodules (Corrections, StatInference).

## Branch, don't commit to `main`

Always work on a **topic branch** and open a pull request — never commit directly to `main`.

```sh
git checkout -b my-short-topic-name
# ... make changes ...
git commit -m "short, clear one-line description"
git push origin my-short-topic-name        # then open a PR on GitHub
```

If your change spans repositories (e.g. FLAF **and** an analysis), use the **same branch name** in
each affected repo so reviewers can find the matching pieces.

## Format before every commit

Formatting is CI-enforced ([GitHub Actions](ci/github-actions.md)). Apply all formatters at once
(with `flaf_env` active):

```sh
source env.sh
bash run_tools/apply_format.sh
```

This runs black (Python), clang-format (C++) and yamllint (YAML). You can also run them on
individual files — see [GitHub Actions](ci/github-actions.md#passing-the-checks-before-you-push).

## Re-index after adding a task

If you added, renamed or moved a LAW task class, refresh the index so it can be found:

```sh
law index --verbose
```

## Validate config changes

- Edited `datasets.yaml`? Run the
  [consistency check](configuration/datasets.md#validate-the-dataset-config).
- Added an era or changed config loading? Make sure `Setup.py` still loads for every era (this is
  what `test-setup-loading` does in CI).

## Open the PR and run the checks

On the pull request:

1. The GitHub Actions checks (formatting, sanity, setup-loading) run automatically.
2. For a real physics check, ask an authorised user to trigger the full pipeline with a
   `@cms-flaf-bot please test` comment — see [Integration pipeline](ci/integration-pipeline.md).

### Pre-PR checklist

- [ ] On a topic branch (not `main`)
- [ ] `bash run_tools/apply_format.sh` is clean
- [ ] `law index --verbose` run if you added/renamed a task
- [ ] dataset consistency check run if you touched `datasets.yaml`
- [ ] no binary files staged
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
run it before you commit doc changes. There is a :material-pencil: **edit** action on every page
that takes you straight to the source file on GitHub.

Guidelines for docs changes:

- Keep framework-wide material here in FLAF; put analysis-specific material in that analysis's
  `docs/` (see [Analyses](analyses.md)). Link rather than duplicate.
- Prefer concrete, copy-pasteable commands, and flag caveats with admonitions
  (`!!! warning`, `!!! tip`).
- Remember the audience includes physicists new to the tooling — define terms or link the
  [Glossary](glossary.md).
