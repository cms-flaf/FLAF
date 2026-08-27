# FLAF — instructions for Copilot code review

FLAF is the shared framework behind the CMS analyses HH_bbtautau, HH_bbWW and H_mumu. It builds
[LAW](https://github.com/riga/law)/luigi task graphs that run on HTCondor and CRAB, reads and
writes multi-TB datasets over GFAL, and JIT-compiles C++ into RDataFrame.

**A change here reaches every analysis and, through them, productions that take days of grid
time.** The failures that matter are silent ones: a job that exits 0 having written nothing, a
histogram normalised by the wrong denominator, a task that reports "complete" because a stale
path exists. Those cost days. A misplaced import costs seconds.

## What a useful review comment looks like here

Prioritise, in order:

1. **Silent wrongness** — a code path that produces a plausible but incorrect number, or reports
   success without doing the work. Say what input triggers it and what the wrong output is.
2. **Violations of the framework invariants below.** They are not deducible from the diff; they
   are why this file exists.
3. **Concurrency and remote-storage assumptions** — shared state under `law --workers`, ordering
   between tasks, anything assuming a remote write is immediately visible.
4. **Genuine logic errors** — off-by-one, wrong branch, mishandled empty input.

Anchor a comment to a concrete failure: *"with `--workflow local` this also forces
`AnaTupleFileTask` local, so 10k branches run on the submit node"* is actionable. *"consider
adding error handling"* is not.

If the diff is fine, say so briefly. Volume is not value: three real findings beat thirty
observations.

## Framework invariants

Each of these has caused a production incident. They are ordered by how much damage they do.

### law semantics

- **`workflow` is a *significant* luigi parameter and `req()` copies it.** A task pinned to
  `workflow="local"` (e.g. because its output is a `local_target`) therefore drags every task it
  requires onto the local scheduler. Upstream workflow choice must travel on a separate
  **insignificant** carrier parameter (`upstream_workflow` in `AnaProd/tasks.py`) that is
  forwarded explicitly. It has to be a real `luigi.Parameter`, not an attribute, or branch and
  workflow fall out of sync. Flag any new `req()` that forwards `workflow` implicitly.
- **`@law.dynamic_workflow_condition` objects are shared with subclasses.** A subclass that
  decorates with the *parent's* `workflow_condition.output` mutates the shared object and
  corrupts the parent task. It must `.copy()` first. Review the **parent** task too — the damage
  shows up there, not in the subclass being changed.
- **law fixes the "already exists" branch set when luigi schedules a remote workflow**, before
  its requirements run. A workflow whose own requirement creates its outputs will resubmit
  everything. Clearing `_existing_branches`/`_skip_jobs` at the top of `run()` is the fix.
- **`poll()` snapshots the job count once.** Changing `job_data` mid-poll hangs or ends the loop
  early, and a resumed run never calls `submit()` — hooks belong at the top of `poll()` too.

### Bundles (`run_tools/law_customizations.py`)

- **A bundle's output is a plain path, so law treats an existing one as complete forever.** Any
  flavour packing code or configuration must be `hashed: true`, or jobs keep unpacking whatever
  was built first and rebuild their branch map from *that* config — branch indices then mean
  different datasets than the submitter intended.
- **A flavour must list every task whose output it packs** in `task_requires`. Miss one and the
  tarball is built while that task is still writing. FLAF warns about a packed
  `data/<version>/<Task>/<period>` directory that nothing requires; do not silence that warning.
- Bundles preserve symlinks inside a packed directory verbatim. An absolute symlink into AFS
  therefore still sends every job back to AFS — which is what the bundle exists to avoid.

### Remote storage (`RunKit/law_gfal.py`)

- **`exists()` is answered from a cached directory listing, not a per-file stat.** Absence may
  only be inferred from a *valid listing marker*. Never add a path that concludes "the directory
  is known to exist, the file is not in what we have, therefore it is absent" — that reported
  2260 of 11066 existing outputs as missing in production.
- The cache is two-level (in-process + a shared server). A change that makes every job do its own
  `gfal-ls` will work in a test and DDoS the storage in production. Reviews should ask what a
  change costs at 10k concurrent jobs.
- **Freshly written remote files can be invisible for seconds.** Code that writes and then
  immediately checks must retry, not conclude absence.

### Processors and stitching

- **`stages` accepts only `AnaTuple` and `AnaTupleMerge`** — any other value is silently ignored.
  A stitcher must appear at **both**: the first writes its denominator into the anaCache, the
  second combines those caches. Present at `AnaTuple` alone, every merge of that process dies
  with `combineAnaCaches: processor <name> not provided`. `dependency_level` is read only for
  `AnaTupleMerge`.
- **Stitching variables must be readable at the merge stage.** Bins select on gen-level
  quantities; an anaTuple that drops `GenPart`/`LHEPart` cannot evaluate them later. New bin
  variables need the analysis to store them (`genInfo`) with a nanoAOD fallback.
- An empty stitching bin is not a bug: each bin's denominator is summed over the very events that
  later read it, so a bin no event falls into is never divided by.

### Concurrency

- **Producers must not write bare-relative temp files.** CWD is shared between branches under
  `law --workers`, so two branches race on the same name. Write under the job's working
  directory.

## Configuration invariants

- `config_path_order` merges four directories: **scalars override, lists concatenate**. A list
  added in an analysis config *extends* the framework one rather than replacing it.
- Dataset split: SM backgrounds and data live in `FLAF/config/<era>/datasets.yaml`; signals and
  CI samples live in the analysis. A signal added to the framework config is misplaced.
- `Run3_2025` and `Run3_2026` carry no MC of their own — they set `reuse_mc_from_era: Run3_2024`.
  A dataset list edited for 2024 changes all three.
- Cross-section keys referenced by a dataset must exist in `crossSections*.yaml`; CI checks this,
  so flag it only when the diff adds a reference CI cannot see.

## Testing expectations

Physics correctness cannot be unit-tested without CERN infrastructure, but the framework's
mechanics can, and `test/` has suites for the ones that bit us (path cache, bundle hashing,
stitching variables, cost model). Changes to those areas should extend them.

When a test uses a fake, the fake must call the **real** `__init__` and patch only what is
genuinely unavailable. Hand-mirroring a class's attributes creates a copy that silently stops
matching — that is how the path-cache suite went red for a whole merge cycle.

Note that CI runs only `test/test_setup_loading.py` (via the `test-setup-loading` workflow);
the pytest suites are not run anywhere, so a broken one is not caught automatically.

## Already enforced by CI — do not comment on these

`formatting-check` (black, yamllint, clang-format), `repo-sanity-checks` (binary files, repo size),
`ds-consistency-check`, `cross-section-check`, `test-setup-loading` (loads `Setup` for all seven
Run 3 eras). Formatting, indentation, quote style and trailing whitespace are settled by tooling;
comments about them are pure noise.

## Do not flag

- **Comment density or missing docstrings.** House policy is comments only where the *why* is
  non-obvious; do not ask for narration of what the code already says.
- **PyROOT idioms** — C++ passed as strings to `ROOT.gInterpreter.Declare()` / RDataFrame
  `Define()`, `from FLAF.Common.HistHelper import *`. These are deliberate.
- **Per-era config duplication.** Eras are kept explicit on purpose; "factor this out" is wrong.
- **Requests for unit tests of code that needs CVMFS, a grid proxy, or real NanoAOD.**
- **Broad refactors** of code the diff merely touches.
- **Speculative hardening** with no failure mode behind it.

## Repository facts

Verified 2026-08-27; re-check before relying on any of it.

| | |
|---|---|
| Layout | `AnaProd/` (anaTuple production tasks), `Analysis/` (histogram/plot tasks), `Common/` (`Setup.py`, utilities), `Processors/` (stitching), `RunKit/` (vendored grid/job tools), `run_tools/` (`law_customizations.py`), `config/`, `include/` (C++ headers), `test/`, `docs/` |
| Submodule | `PlotKit` only. **`RunKit` is vendored**, not a submodule; imports are `from FLAF.RunKit.<module> import …` |
| Datasets | `config/<era>/datasets.yaml` for Run 3. Run 2 eras still use the older `samples.yaml` |
| Eras | `Run3_2022`, `Run3_2022EE`, `Run3_2023`, `Run3_2023BPix`, `Run3_2024`, `Run3_2025`, `Run3_2026`; Run 2 legacy |
| Workflows | `formatting-check`, `repo-sanity-checks`, `ds-consistency-check`, `cross-section-check`, `test-setup-loading`, `deploy-docs`, `integration-test`, `trigger-flaf-integration` |
| Integration test | Triggered by `@cms-flaf-bot please test`. Its configuration (process lists, eras, versions) lives in **`cms-flaf/FLAF_ci`**, not in this repo |
| Docs | `docs/`, built with `mkdocs build --strict`. A user-visible change should update them in the same PR |
