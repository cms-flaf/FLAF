# FLAF - Copilot Instructions

## Overview

**FLAF** (Flexible LAW-based Analysis Framework) is a CMS (CERN) high-energy physics analysis framework for HH → bb𝜏𝜏 analysis. The project uses [LAW](https://github.com/riga/law) (Luigi Analysis Workflow) for task management and is designed to run on CERN computing infrastructure (lxplus, HTCondor, CVMFS).

- **Repository size**: ~2 MB
- **Languages**: Python (~13,000 lines), C++ headers, Bash scripts, YAML configs
- **Target environment**: CERN lxplus (AlmaLinux 9), requires CVMFS and CMSSW
- **Submodules**: `RunKit`, `PlotKit` (must be cloned with `--recursive`)

## Project Structure

```
FLAF/
├── .github/                    # CI workflows and GitHub Actions
│   ├── workflows/              # formatting-check, repo-sanity-checks, ds-consistency-check, trigger-flaf-integration
│   └── integration_cfg.yaml    # Integration test configuration
├── Analysis/                   # Analysis task implementations (histograms, merging, plotting)
├── AnaProd/                    # AnaTuple production tasks
├── Common/                     # Shared utilities (Setup.py, Utilities.py, etc.)
├── config/                     # Era-specific configs (Run2_*, Run3_*), cross-sections
├── docs/                       # MkDocs documentation
├── include/                    # C++ headers for ROOT analysis
├── run_tools/                  # Helper scripts (mk_flaf_env.sh, law_customizations.py)
├── test/                       # Test scripts (checkDatasetConfigConsistency.py)
├── RunKit/                     # Git submodule for workflow utilities
├── PlotKit/                    # Git submodule for plotting
├── env.sh                      # Main environment setup script
├── bootstrap.sh                # HTCondor bootstrap script
└── cmsEnv.sh                   # CMSSW environment wrapper
```

## Code Formatting Requirements (CI Enforced)

**All pull requests must pass formatting checks.** Always format code before committing:

### Python Files (`.py`)
```bash
pip install black
black --check --diff <file.py>   # Check formatting
black <file.py>                   # Apply formatting
```

### YAML Files (`.yaml`, `.yml`, `.yamllint`)
```bash
pip install yamllint
yamllint -s -c .yamllint <file.yaml>
```
Key rules: 2-space indentation, spaces inside braces `{ }` and brackets `[ ]`, no document-start marker required.

### C++ Files (`.cpp`, `.h`, `.hpp`, `.cc`)
```bash
clang-format --dry-run --Werror --style "file:.clang-format" <file.cpp>
clang-format -i --style "file:.clang-format" <file.cpp>  # Apply formatting
```

## GitHub Workflows (CI Checks)

These run automatically on PRs to `main`:

1. **formatting-check.yaml**: Validates Python (black), YAML (yamllint), C++ (clang-format) formatting
2. **repo-sanity-checks.yaml**: Checks for binary files (must use git LFS), calculates repo size delta
3. **ds-consistency-check.yaml**: Validates `config/*/samples.yaml` files with:
   ```bash
   python3 test/checkDatasetConfigConsistency.py --exception config/dataset_exceptions.yaml Run3_2022 Run3_2022EE Run3_2023 Run3_2023BPix
   ```
4. **trigger-flaf-integration.yaml**: Triggers GitLab integration pipeline via `@cms-flaf-bot test` comments

## Key Configuration Files

| File | Purpose |
|------|---------|
| `.yamllint` | YAML linting rules |
| `.clang-format` | C++ formatting (Google-based, 120 column limit) |
| `.editorconfig` | Editor settings (4-space indent, UTF-8) |
| `config/law.cfg` | LAW task module registration |
| `config/<era>/samples.yaml` | Dataset definitions per era |
| `config/crossSections*.yaml` | Cross-section values |
| `mkdocs.yml` | Documentation site configuration |

## Important Notes for Code Changes

1. **Dataset configs** (`config/*/samples.yaml`): When modifying, ensure:
   - MC samples have `crossSection` and `generator` fields
   - `crossSection` values exist in `crossSections*.yaml`
   - Run consistency check: `python3 test/checkDatasetConfigConsistency.py --exception config/dataset_exceptions.yaml <eras>`

2. **LAW tasks** (`AnaProd/tasks.py`, `Analysis/tasks.py`): Inherit from `Task` or `HTCondorWorkflow` in `run_tools/law_customizations.py`. After adding new tasks, run `law index --verbose`.

3. **Environment**: The framework requires CERN infrastructure (CVMFS, grid certificates). Local development needs `env.sh` sourced, which sets up a Python venv at `$ANALYSIS_SOFT_PATH/flaf_env/` using LCG_107_cuda. The `ANALYSIS_SOFT_PATH` environment variable (defaulting to `$ANALYSIS_PATH/soft`) is set automatically by `env.sh`.

4. **Binary files**: Never commit binary files directly. Use git LFS if needed.

5. **Git submodules**: `RunKit` and `PlotKit` are submodules. Python imports use `from FLAF.RunKit.* import ...` and `from FLAF.PlotKit.* import ...` patterns.

## Validation Checklist

Before submitting changes:
- [ ] Run `black --check` on modified Python files
- [ ] Run `yamllint -s -c .yamllint` on modified YAML files
- [ ] Run `clang-format --dry-run --Werror` on modified C++ files
- [ ] If modifying `config/*/samples.yaml`, run dataset consistency check
- [ ] Ensure no binary files are staged (`git status` should show only text files)

## Trust These Instructions

These instructions are validated. Only search the codebase if:
- Information appears incomplete or outdated
- A command fails unexpectedly
- The task requires understanding code not covered here
