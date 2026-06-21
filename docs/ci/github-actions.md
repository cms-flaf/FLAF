# GitHub Actions

FLAF uses **two** continuous-integration systems:

| System | Where | Purpose |
|---|---|---|
| **GitHub Actions** | GitHub | Fast code-quality and sanity checks on every pull request. |
| **FLAF integration** | GitLab CI (CERN) | The full pipeline run that checks physics correctness. Triggered by a bot comment — see [Integration pipeline](integration-pipeline.md). |

This page covers the GitHub Actions checks.

## Shared, reusable workflows

The analysis repositories don't duplicate CI logic. Each workflow is a thin wrapper that calls the
shared implementation in FLAF:

```yaml
jobs:
  my-job:
    uses: cms-flaf/FLAF/.github/workflows/<workflow>.yaml@main
    secrets: inherit
```

So fixing a check in FLAF fixes it everywhere. (A checkout helper inside the shared workflows makes
the FLAF tooling — `.yamllint`, `.clang-format` — available even though FLAF is a submodule.)

## The standard checks

| Workflow | Runs on | What it checks |
|---|---|---|
| `formatting-check.yaml` | PRs | Code style: **flake8**/black (Python), **clang-format** (C++), **yamllint** (YAML). |
| `repo-sanity-checks.yaml` | PRs | Submodule-pointer consistency, repository health, no stray binary files. |
| `test-setup-loading.yaml` | PRs | Actually loads `Setup.py` for **every configured era** — catches config typos and broken references early (a real run, not a dry run). |
| `trigger-flaf-integration.yaml` | PR comments | Parses a `@cms-flaf-bot` comment and triggers the GitLab pipeline. See [Integration pipeline](integration-pipeline.md). |

FLAF itself additionally runs:

| Workflow | What it checks |
|---|---|
| `cross-section-check.yaml` | Cross-section values are consistent/valid. |
| `ds-consistency-check.yaml` | `datasets.yaml` entries are well-formed (generator, resolvable cross-section, naming) via `test/checkDatasetConfigConsistency.py`. |

## Passing the checks before you push

Formatting is enforced, so format **before** committing. The convenience script applies all
formatters at once (with `flaf_env` active):

```sh
bash run_tools/apply_format.sh
```

Or run them individually:

```sh
black <file.py>                                   # Python
clang-format -i --style "file:.clang-format" <f>  # C++
yamllint -s -c .yamllint <file.yaml>              # YAML
```

If you edited `datasets.yaml`, also run the consistency check from
[Datasets](../configuration/datasets.md#validate-the-dataset-config). See
[Contributing](../contributing.md) for the full pre-PR checklist.

!!! note "Required secrets"
    The bot-trigger workflow needs the org-level secrets `FLAF_INTEGRATION_TOKEN` (GitLab trigger)
    and `FLAF_GITHUB_TOKEN` (to post the reply comment), inherited via `secrets: inherit`. The
    quality checks need no secrets.
