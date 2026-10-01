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
5. **Documentation that did not ship with the change** — see the section below; a user-visible
   change with no documentation update is an incomplete PR, not a nitpick.

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

### anaTuple columns (`AnaProd/FuseAnaTuples.py`, `Common/TupleHelpers.py`)

- Columns sharing the text before their first underscore are stored as one collection; array
  collections share one counter. `defineColumnGrouping` refuses a collection that mixes scalars and
  arrays (the scalars would silently become arrays) or holds arrays of different lengths, and
  `fuseAnaTuples` checks that every column keeps its type. The fix for either is a rename in the analysis anaTuple
  definition, never a reader that accepts both layouts.

### anaTuple layout (`AnaProd/FuseAnaTuples.py`)

- Every tree of an anaTuple has one row per event selected in any variation, aligned by row.
  `valid == false` rows of the central tree are placeholders for events selected only by a shift.
  Shifted trees store `<name>__delta`; readers attach the central tree as the friend `Central`.
- Columns in `anaTuple_shift_invariant_columns` are stored in the central tree only, and the fuse
  step fills them into the placeholder rows after checking bit-identity across every variation that
  selects the event. Skipping a column without filling the placeholders would give events selected
  only by a shift the placeholder's zeros, so both halves must stay together. `MergeAnaTuples`
  refuses inputs produced with different lists, since a chain takes its columns from the first
  file. A column that a shifted tree takes from `Central` answers `HasColumn` but is not listed by
  `GetColumnNames()`, and `Define` of that name fails: check existence with `HasColumn`.
- Listing a whole collection as shift-invariant also drops its counter from the shifted trees.

### Concurrency

- **Producers must not write bare-relative temp files.** CWD is shared between branches under
  `law --workers`, so two branches race on the same name. Write under the job's working
  directory.

### Shifted-tree array counters (`AnaProd/FuseAnaTuples.py`)

- ROOT reads an array of a friend tree with the main tree's counter of the same name
  (`TTreeReaderArray` looks the counter up by name). Shifted trees are read with the central tree as
  the friend `Central`, so their array collections are counted by `n<collection>__shifted`, and the
  deltas are computed against the first `central.n<collection>` elements of `central.<array>`.
  Writing a shifted tree with the central counter names, or dropping the clipping, silently
  corrupts every shifted collection that is longer than the central one.
- `__shifted` is a column suffix like `__delta`: `Common/Utilities.CreateDataFrame` skips it, and
  any other code that splits column names on `__` has to accept it.

### Writing trees with uproot

- **Write trees with `Common/TupleHelpers.writeTree` (or an explicit `mktree`), never
  `file[name] = arrays`.** Since uproot 5.7 (LCG_110a) that assignment writes an RNTuple, which
  TChain, tree friends and the anaTuple readers do not accept.

## Configuration invariants

- `config_path_order` layers four directories (`FLAF/config`, `FLAF/config/<era>`, `config`,
  `config/<era>`; for `global.yaml` each directory's `user_custom.yaml` joins in, and
  `--user-custom` comes last). The files are **concatenated as text and parsed once**, so a
  repeated top-level key **replaces the earlier value wholesale** — lists and nested dicts
  included; nothing is merged or concatenated. An analysis or era that redefines a block such as
  `corrections:` must repeat every entry it still needs. Datasets from several layers combine only
  because each dataset is its own top-level key.
- Dataset split: SM backgrounds and data live in `FLAF/config/<era>/datasets.yaml`; signals and
  CI samples live in the analysis. A signal added to the framework config is misplaced.
- A dataset whose NanoAOD lacks some weights is handled in its entry: `exclude_files` drops the
  individual files (InputFileTask refuses a name the dataset does not have), and
  `disabled_corrections` sets a shape weight to 1 for every member (its branches still exist) or
  drops any other correction. Neither belongs in a correction's `processes:` list or in a code
  guard on the branch size.
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

Note what CI actually runs from `test/`: `test_setup_loading.py` (via `test-setup-loading`, on
analysis PRs only) and the config checkers `checkCrossSections.py`,
`checkDatasetConfigConsistency.py` and `checkDatasetNaming.py` (on FLAF PRs that change those
config files). The other suites (`test_*.py`) are not run anywhere, so a broken one is not
caught automatically.

## Documentation must ship with the change

A PR must update the documentation **in the same PR** whenever it changes anything a user of the
framework can observe. Treat this as a review item of the same weight as correctness — docs
drifting from the code is the failure that motivated the current documentation, and a PR that
lands without them is not complete.

Ask, for every diff: does it add, rename or remove any of these?

- a task or DAG node, or the arguments/parameters of one;
- a command, a CLI flag, or the meaning of an existing one;
- a configuration key — `global.yaml`, `user_custom.yaml`, `processes.yaml`, `phys_models.yaml`,
  cross-sections, `fs_*` storage keys, bundle flavours, processor entries;
- a dataset, era, process or physics-model name;
- the environment, installation or setup steps;
- storage locations, output paths or log locations;
- a CI workflow, or how the integration test is triggered or configured;
- any behaviour a user relies on, including a default that changes.

If the answer is yes and the diff touches **no** documentation file, say so and name the page that
should have changed. If the author states the change is internal-only, that is a legitimate
answer — a pure refactor or bugfix with no user-visible effect is exempt — but it should be
stated in the PR, not left implicit.

Also flag the inverse: documentation edited to describe behaviour the diff does not implement, and
new pages added without being wired into `mkdocs.yml`'s `nav` (the build fails on that, but the
review should catch it first).

Where it goes in this repository:

- `docs/` is the source of truth for framework-wide material. Keep generic content here and link
  to it from the analyses rather than duplicating it.
- `docs/reference/tasks.md`, `docs/workflow/arguments.md` and `docs/configuration/*` are the pages
  that go stale first, because they enumerate tasks, arguments and config keys.
- New pages must be added to `nav:` in `mkdocs.yml`.
- Verified with `mkdocs build --strict`, which fails on a broken link, a missing nav entry or a
  missing asset.

An ecosystem-level change — a new analysis, a new site, a renamed entry point — also needs the
landing page in `cms-flaf/cms-flaf.github.io`, which is a separate repository and therefore a
separate PR; say so in the review rather than assuming it will be noticed.

## Already enforced by CI — do not comment on these

`formatting-check` (black, yamllint, clang-format), `repo-sanity-checks` (binary files, repo size),
`ds-consistency-check`, `cross-section-check`. Formatting, indentation, quote style and trailing
whitespace are settled by tooling; comments about them are pure noise.

`test-setup-loading` (loads `Setup` for all seven Run 3 eras) runs on **analysis** PRs only —
FLAF's copy is a reusable workflow with no PR trigger. On a FLAF PR, a change that can break
config loading (`Common/Setup.py`, `config/`) is therefore not checked, and is worth a comment.

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

Verified 2026-09-30; re-check before relying on any of it.

| | |
|---|---|
| Layout | `AnaProd/` (anaTuple production tasks), `Analysis/` (histogram/plot tasks), `Common/` (`Setup.py`, utilities), `Processors/` (stitching), `RunKit/` (vendored grid/job tools), `run_tools/` (`law_customizations.py`), `config/`, `include/` (C++ headers), `test/`, `docs/` |
| Submodule | `PlotKit` only. **`RunKit` is vendored**, not a submodule; imports are `from FLAF.RunKit.<module> import …` |
| Datasets | `config/<era>/datasets.yaml` for Run 3. Run 2 eras still use the older `samples.yaml` |
| Eras | `Run3_2022`, `Run3_2022EE`, `Run3_2023`, `Run3_2023BPix`, `Run3_2024`, `Run3_2025`, `Run3_2026`; Run 2 legacy |
| Workflows | `formatting-check`, `repo-sanity-checks`, `ds-consistency-check`, `cross-section-check`, `test-setup-loading`, `deploy-docs`, `integration-test`, `trigger-flaf-integration` |
| Integration test | Triggered by `@cms-flaf-bot please test`. Its configuration (process lists, eras, versions) lives in **`cms-flaf/FLAF_ci`**, not in this repo |
| Docs | `docs/`, built with `mkdocs build --strict`; see the documentation section above |
