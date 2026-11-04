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

- **Package versions are pinned in `run_tools/mk_flaf_env.sh`.** Every package it installs is
  pinned (`==`, or a commit archive URL), law included; `env.sh` names no package of `flaf_env`
  and probes no version, and nothing checks a version at import. (It does check that the system
  `/usr/bin/python3` imports `gfal2`, to set `GFAL_PYTHONBIN` for the system `gfal-*` commands,
  outside law jobs only.) `env.sh` identifies `flaf_env` by the LCG
  release and a hash of that script (`FLAF_ENVIRONMENT_ID`, also its marker
  `.<FLAF_ENVIRONMENT_ID>`), so a pin change rebuilds it on the next `source env.sh` on the
  submitting machine; inside a law job (`LAW_JOB_HOME` set, non-bundle HTCondor jobs included) or
  with `FLAF_NO_INSTALL=1` it refuses instead. Whether `flaf_env` exists and lacks its marker is
  decided from listings, never from a stat alone, and a check that fails (hash, listing, a listed
  path no stat shows as a directory, an unlisted path a stat shows, a listed marker no stat shows)
  refuses without removing anything. The marker is written only after the build's own status is
  checked, never through `run_cmd`, whose `kill -INT` a shell that ignores SIGINT carries past.
  A dependency bump is a pin change in `run_tools/mk_flaf_env.sh` plus the same version in each
  CI workflow that installs that package (`test/test_flaf_env.py` enforces it); no version is
  repeated in `env.sh` or the Python code (adapting the code to a new release is a separate
  change). A `getattr`/`try` or version branch that keeps another release working is a silent
  fallback, not compatibility.
- **The job script is law's `law_job.sh` with its per-branch loop replaced** by the grouped run
  (`grouped_law_job_script`: all branches of a job in one `--branches=… --workflow=local`
  process, `--print-deps=0`). Its generator raises unless law's loop is found exactly once and
  names the file after its content; it must never fall back to law's unmodified script.
- **`workflow` is a *significant* luigi parameter and `req()` copies it.** A task pinned to
  `workflow="local"` (e.g. because its output is a `local_target`) therefore drags every task it
  requires onto the local scheduler. Upstream workflow choice must travel on a separate
  **insignificant** carrier parameter (`upstream_workflow` in `AnaProd/tasks.py`) that is
  forwarded explicitly. It has to be a real `luigi.Parameter`, not an attribute, or branch and
  workflow fall out of sync. Flag any new `req()` that forwards `workflow` implicitly.
- **A `req()` within one task family passes `_skip_task_excludes`.** law drops the per-task
  resources of `exclude_params_req` (`tasks_per_job`, `max_runtime`, `n_cpus`, `crab_memory`)
  whatever the two families are, so a plain `X.req(x)` with `x` an `X` would take the class
  defaults and, `n_cpus` being significant, could become a second task for the same outputs.
  `Task.req` raises `SameFamilyReqError` unless the call passes `_skip_task_excludes=True` (the
  same task) or `=False` (a separate task with its own defaults, e.g. the `AnalysisCacheTask` of a
  dependency producer). Flag a helper that derives a task from whatever task it is handed
  (`sub_era_task`, `_anaTuple_outputs`, `InputFileTask.WF_complete`) where it can be handed one of
  its target's own family.
- **`@law.dynamic_workflow_condition` objects are shared with subclasses.** A subclass that
  decorates with the *parent's* `workflow_condition.output` mutates the shared object and
  corrupts the parent task. It must `.copy()` first. Review the **parent** task too — the damage
  shows up there, not in the subclass being changed.
- **Whether the AnaTuple merge plans are available is decided by `anaTuple_plans_ready`
  (`AnaProd/tasks.py`) alone.** It copies an existing production's plans into `data/` when they
  are on `fs_anaTuple`. Every dynamic workflow condition from `AnaTupleMergeTask` down and every
  choice between narrow and whole-era requirements asks it; a direct
  `AnaTupleFileListTask.req(...).complete()` there disagrees with it in a new checkout
  (placeholder maps, whole-era requirements, `InputFileTask` under bundles; issue #352).
  `test/test_anatuple_plan_reuse.py` scans for one.
- **law fixes the "already exists" branch set when luigi schedules a remote workflow**, before
  its requirements run. A workflow whose own requirement creates its outputs will resubmit
  everything. Clearing `_existing_branches`/`_skip_jobs` at the top of `run()` is the fix; both
  proxies do it in `SubmissionGuards.run()`, after `require_fresh_negatives()`.
- **In its first poll of a resumed run, law caches each job's skip verdict before it queries the
  batch system**, so a job that finishes in between comes back as "initially missing task
  outputs". `SubmissionGuards._stop_on_mass_lost_outputs` looks at every such job again; one
  whose branches are complete is marked skippable, not resubmitted.
- **`poll()` snapshots the job count once.** Changing `job_data` mid-poll hangs or ends the loop
  early, and a resumed run never calls `submit()` — hooks belong at the top of `poll()` too.
  law's `submit()` drops backlog jobs whose branches exist; inside the loop that leaves the count
  one too high when the job is also in `jobs` (`process_resources` recorded it finished on a
  fresh run). When FLAF skips the round before `poll()`, `SubmissionGuards.poll()` drops them
  first (`_drop_complete_backlog`).

### Bundles (`run_tools/law_customizations.py`)

- **A bundle's output is a plain path, so law treats an existing one as complete forever.** Any
  flavour packing code or configuration must be `hashed: true`, or jobs keep unpacking whatever
  was built first and rebuild their branch map from *that* config — branch indices then mean
  different datasets than the submitter intended.
- **The unhashed flavour that packs `flaf_env` is named `<flavour>_<FLAF_ENVIRONMENT_ID>`**, so an
  environment rebuilt from another `run_tools/mk_flaf_env.sh` is packed anew instead of shipping
  the old packages with a job script rendered from the new ones. `BundleTask.output` raises when
  the identity is not set; it must never fall back to the plain name.
- **A flavour must list every task whose output it packs** in `task_requires`. Miss one and the
  tarball is built while that task is still writing. FLAF warns about a packed
  `data/<version>/<Task>/<era>` directory that nothing requires; do not silence that warning.
- Bundles preserve symlinks inside a packed directory verbatim. An absolute symlink into AFS
  therefore still sends every job back to AFS — which is what the bundle exists to avoid.
- **A task that ships bundles must ship `flaf_env`.** `_bundle_tasks` raises at submission when
  none of its flavours packs `$FLAF_ENVIRONMENT_PATH` (`BundleTask.packs_environment`): the job
  sources `env.sh` from the bundles, and `env.sh` refuses without the environment.
- **A job's bootstrap fails unless `env.sh` exported `FLAF_ENV_LOADED=1`.** `env.sh` stops with
  `kill -INT $$`. On HTCondor law runs the job in that very shell, which the signal would kill
  before the log stageout, so `bootstrap.sh` traps SIGINT while it sources `env.sh`, and only then
  (a trap left set would carry the job past an interrupt of the task). On CRAB law runs the job
  in a pipeline into `tee`, `$$` names the shell around it, and `env.sh` goes on. Neither the
  signal nor the status of `source` (the analysis `env.sh` returns 0 whatever `FLAF/env.sh` did)
  tells a refusal there: `load_flaf_env` clears the sentinel first and exports it only when
  nothing refused, so every refusal path, `install()` included, must mark the load as refused.
- **Worker files are keyed on their content.** Without spooling, `_stage_and_update_jdl` uploads
  each job input to `worker_files/<era>/<sha256[:12]>/<name>` and trusts an existing copy. law
  names a copied input after a hash of its *source path* (`job/base.py`,
  `postfix_file(add_hash=True)`), not its content, so a key on the name alone hands jobs another
  task's or older code's files.

### Remote storage (`RunKit/law_gfal.py`)

- **`exists()` is answered from a cached directory listing, not a per-file stat.** Absence may
  only be inferred from a *valid listing marker*. Never add a path that concludes "the directory
  is known to exist, the file is not in what we have, therefore it is absent" — that reported
  2260 of 11066 existing outputs as missing in production.
- The cache is two-level (in-process + one per-user, per-machine server on a Unix socket,
  `/tmp/flaf-path-cache-<uid>/server_v1.sock`, started on demand by `ensure_local_server`; the
  central `cms-flaf.cern.ch:5000` server is gone). Jobs (HTCondor and CRAB) never contact a
  server: they read the snapshot their submission shipped. An "absent" and a listing marker live
  10 min on the server (`negative_validity`), an "exists" 24 h. A server address must come from
  `checked_server_address` (the directory private to the user), and only a same-uid peer is
  believed (`_check_peer`, SO_PEERCRED): another user on the node could otherwise answer every
  question with "exists". A server-learned listing marker is never kept locally (`RemotePathCache._keep`).
  A change that makes every job do its own
  `gfal-ls` will work in a test and DDoS the storage in production. Reviews should ask what a
  change costs at 10k concurrent jobs.
- **Freshly written remote files can be invisible for seconds.** Code that writes and then
  immediately checks must retry, not conclude absence.
- **A failed listing must never be cached (or answered) as "absent".** Only gfal's own
  not-found (errno `ENOENT`, `gfal_ls_checked`) is an absence; a timeout, SSL error or missing
  credential is a failure, and a negative cached from it is published to the cache server and
  hides existing files from every client. `exists()` may answer `False` for a failed listing but
  must cache nothing — not even a "this directory failed" memo: the next lookup must list again,
  or a short blip condemns every file of the directory. Do not reintroduce `gfal_ls_safe` where a
  path's existence is decided.
- **Uploads are published by rename.** `gfal_copy_safe(..., copy_mode="copy_rename")` uploads to
  `<target>.flaf-tmp-<pid>-<uuid>`, verifies the checksum and renames onto the target; the target
  must never exist while partial, and an existing good file must not be removed before the
  replacement is complete, and no "absent" may be published for the target during the upload.
  Orphaned `.flaf-tmp-*` files match no output pattern but take quota.
- Jobs publish nothing to the path cache, so a driver that believes an "absent" from a
  cached listing reads files the jobs wrote as missing. On both backends the driver never trusts
  the batch system's "finished": every poll lists the output directories of the jobs reported
  finished (every "absent" resting on that poll's listing; a missing output is retried only after
  one more poll, an unlistable directory is tolerated for at most 30 minutes after the job was
  reported finished, then the run stops; `check_finished_jobs`, and `_stop_on_mass_lost_outputs`
  for the first poll of a resumed run, where law demotes before the hook),
  and the start of every batch workflow takes fresh listings
  first (`require_fresh_negatives`, and the existing-branch memos law filled while scheduling are
  dropped).
- The submit-time snapshot (`path_cache_<taskhash>_<time_ns>.json.gz`, every round, HTCondor and
  CRAB) is a job's only path cache, and luigi checks every requirement
  of a job's branches before running them: `_stage_path_cache_input` must ship each existing
  output of those requirements whatever the age of the driver's own entries (only "exists",
  never an absent for them). A job that has to list them itself pays one `gfal-ls` per directory
  (a HistMergerTask branch: every HistFromNtupleProducerTask branch, one directory per variable).
  The snapshot differs per submission, so its job input is not `share=True`: law builds every CRAB
  task of a run in one directory and reuses an existing shared input there (`skip_existing`).

### Locking finished productions (`run_tools/eos_lock.py`)

- A lock is `z:i` on **every** directory plus mode `0555`, with everything owned by the locker.
  Never put `!d` in a lock ACL (it lets a file's owner move the file out, even with `i`), never
  edit a locked ACL with `eos acl` (it drops `z:i` silently), and set the mode before `z:i`
  (`chmod` is refused below an immutable parent).
- An original is deleted only once a copy with the same size and adler32 is in place; an entry
  whose content changed while the lock ran is never deleted, and a copy is renamed into place only
  after it was verified against an original that still exists. A directory of another member is
  never removed from a directory carrying `!d`: EOS refuses that even to the parent's owner, so the
  original is moved into the rebuilt directory (which gets the caller's ACL first) and removed there.
- Every original file stays at its path while the lock runs, except inside the swap of a rebuilt
  directory: that step has no safe point (signals only set a flag, checked between directories
  and files), is retried as a whole from a fresh look, and is the first thing a rerun finishes. Every intermediate state must be
  resumable by a rerun: `test/test_eos_lock.py` kills the lock after each operation, and a new
  operation must keep that test green.

### CRAB backend (`run_tools/law_customizations.py`, `run_tools/crab_sites.py`, `run_tools/crab_watchdog.py`)

- **CRAB gives the whitelist precedence over the blacklist.** Excluded sites (configured blacklist
  and quarantine alike) must be cut out of the whitelist itself (`resolve_whitelist`), with globs
  expanded from CRIC's Processing Site Name list (`preset=site-names`, type `psn`), never from the
  compute-unit list: a non-PSN name gets the whole task refused. A blacklist entry overlapping a
  whitelist glob always expands the glob, even if the excluded site is missing from the list.
- **A run is stopped only from `crab_poll_callback`** (via `stop_reason` on the job manager). law
  runs `query()` in a thread pool and swallows a raise there as one more failed query, so the run
  would end many polls later, or never.
- **CRAB server states are classified on law's result too, not only on its errors.** law 0.1.21
  reports `WAITING on command SUBMIT` pending (without a bound) and `SUBMITFAILED` failed (and
  resubmits); `parse_query_output` raises `CrabTaskNotScheduledYet` / `CrabTaskSubmitFailed` for
  them when the response has no per-job JSON, so the 60-poll wait bound and the
  second-failed-submission stop still apply. A task that publishes per-job JSON keeps its states.
  Only this run's own submissions (`_submitted_projects`) count towards the second-refusal and
  second-failed-submission stops; the 60-poll wait bound and the unreadable limit count every
  task, those left by earlier runs included.
- **The site record is harvested keyed by job id from the parsed status**, never from the
  positional `extra` that law syncs onto `job_data`, and a failure without a job-level exit code
  (kill, never started, refused task, watchdog verdict) is never charged to a site: counted, a mass
  kill drives every baseline to ~100 % and the quarantine can no longer fire.
- **The wave gate must keep `len(job_data)`**: parked jobs go into `unsubmitted_jobs` (counted by
  `JobData.__len__`, dumped to disk), never out of `job_data` altogether, because the poll loop
  snapshots the job count once. `--no-poll` bypasses the gate; law applies it to the workflow it
  is given to only, not to the remote workflows that one requires. Only the backlog is measured against
  the wave size, and the oldest parked retry is released after `retry_release_minutes`.
- **The lost-outputs brake (`SubmissionGuards`) runs before anything can park a retry
  generation** (a skipped round, the wave gate): a parked mass retry would be released later
  where the brake no longer sees it. It re-checks candidates with `as_branch(b).complete()` on
  fresh listings (outputs that appeared, marker-aware completeness), and before raising it puts
  the jobs back in the job file, because law has already rewritten and dumped them.
- **CRAB `maxMemoryMB` is a kill line** (exit 50660, never retried by CRAB): an explicit request is
  honoured or refused, never clamped down; cores are snapped up to 1/2/4/8.
- **CLI-presence helpers match exactly** (`_cli_has_param`, `_cli_has_tasks_per_job`): an option
  addressed to one task must not disable a default for every other task; the bare
  `--tasks-per-job` belongs to the root task only. `parallel_jobs` and `poll_interval` are in
  `prefer_params_cli`, so another task's prefixed form reaches that task, while the bare and the
  root task's prefixed form are copied to every task by `req` and count for all of them.
- **CRAB's `Site.storageSite` / `Data.outLFNDirBase` are a fixed placeholder
  (`CRAB_PLACEHOLDER_STAGEOUT`), never derived from `fs_default` or any other user setting.**
  CRAB transfers nothing for FLAF, but the pair must pass the REST server (a `/store/user/<name>/`
  LFN whose `<name>` is not the submitter is refused, so a shared `fs_default` such as
  `/store/user/lpcflaf/` failed every submission) and the task worker's Rucio write-PFN
  resolution at the site (T3_CH_CERNBOX maps `/store/user/` only). The source of each check is cited at
  the constant.
- **Workers never require bundles, and neither renew nor delete the delegated proxy.**
- **Code behind `LAW_CRAB_JOB_NUMBER` / `LAW_JOB_HOME` / `on_batch_node()` runs only on a worker,
  so it needs a test that sets the variable** (a `NameError` there shipped past a green
  suite); the whole package is checked with `flake8 --select=F821,F811` in CI.
- **On the submit node `CreateVomsProxy` replaces a proxy only once its renewal has succeeded**
  (`RunKit/grid_helper_tasks.py`): `voms-proxy-init` needs the certificate passphrase, so a run
  without a controlling terminal fails it, and removing the old proxy first left such a run with
  none, although it was still valid for hours.

### Eras (`run_tools/law_customizations.py`, `Common/Setup.py`)

- **The FLAF era is `era`; `period` no longer exists as a parameter, option, keyword or pattern.**
  `--period`, `--<Task>-period` and `[<Task>] period` must keep failing with `'period' was renamed to
  'era' (--era)`, `period=` with `'period' was renamed to 'era' (use era=...)`, and `--<Task>-era`
  with `the eras are set once per invocation, with --era`;
  do not add a silent alias. The task parameter is `era = EraParameter()` (a sorted tuple of unique
  known eras; `max_len` is gone, and which classes may be constructed over several is
  `Task.allow_multi_era`: any other raises `<Task> takes one era`). Payload scripts take `--era` (one
  value) and `Setup(ana_path, era, version)` exposes `Setup.era`. Bundle patterns use `{era}`;
  `{period}` is refused.
- **The era is a branch dimension: `--era A,B` is one workflow per stage whose branches each belong
  to one era.** `era_branches = True` (it needs `allow_multi_era`) marks the stage classes
  of `AnaProd/tasks.py` and `Analysis/tasks.py`; a branch is numbered `era id × 1 000 000 + its number
  in its era's own map` (`ERA_IDS` = the C++ `Period` values, `Common.Setup.era_branch`), and over one era
  nothing is renumbered. What holds, and what to flag:
  - **Branch tasks, jobs, payloads and every path (outputs, job files `data/<v>/<Task>/<era>/…`,
    logs, heartbeats, worker files, bundles, CRAB request names) are one era's and identical to a
    run over that era alone.** A path, job-file name, resource key or cache keyed by the era *set*,
    or a branch number of a run over several eras written into an era's own file or command line,
    breaks the rerun sequences (`--era A` then `--era A,B`, and the reverse) that rely on the two
    runs sharing state. A job's command line carries `--era <its era>` and the era's own numbers.
  - **A per-era value is read through `only_era`, `setup_for(era)` or `single(era)`, never
    `self.era[0]`.** `only_era` raises `EraScopeError` over several eras; `EraScopeError`,
    `EraDependentValueError` and `BranchSelectionError` must not subclass `AttributeError`, which
    `getattr(task, name, default)` and `hasattr` would swallow. A setting a run reads once for all its
    eras goes through `common(getter, what)`, which raises when the eras disagree (`fs_default`,
    `crab`, `runTokenServer`, `anaTuple_scheduling`,
    `use_cmssw_env_AnaTupleProduction`, a producer's `n_cpus`/`max_runtime`/`crab_memory`/`save_as`/
    `cmssw_env`); a new run-wide reader without it silently takes one era's value.
  - **A stage's requirements live in `stage_workflow_requires()`; `workflow_requires()` is the
    dispatcher** that, over several eras, prunes each era that is complete (its consumed
    intermediates must not be requested) and lifts the rest into one requirement per stage. Eras are
    named in a requirement only through `with_eras`, which keeps the per-task values
    (`--tasks-per-job`, `--max-runtime`, `--n-cpus`, `--crab-memory`); `run_tools/check_era_rename.py`
    flags `era=` in `req`/`req_params`/`req_branch`/`req_workflow` outside it and `self.era[...]` in
    a task class outside `only_era` (`# noqa: era-scope`).
  - **Branch maps go through `cached_branch_map`**, which restores a `--branches` selection that law
    reset while the map was its placeholder; a selection is in the numbering of its run
    (`BranchSelectionError` otherwise), and `--branch` with several eras is refused.
  - **Submission state is per era.** `EraMergedProxy` (HTCondor and CRAB, active over several eras
    only) reads and writes each era's job file in the era's own numbers and delegates building and
    submission to a proxy of that era; the lost-outputs brake is judged per era, the CRAB watchdog
    per era, and `parallel_jobs` and the refused/failed-submission stops are run-wide. Two
    instances of a stage sharing an era never run at once (the luigi resource
    `flaf_jobs:<job-file directory>`).
  - **A second model: `allow_multi_era` without `era_branches`** (StatInference's card stages,
    guarded by `EraCardWorkflow` in `test/test_era_jobs.py`): law's own branches, every branch task
    over all eras of the run, `store_parts()`/`remote_log_dir_target()` overridden without the era,
    the map built without `cached_branch_map`, the plain (not the merged) proxy, one job file and one
    luigi token. FLAF code must not build the map of, or ask completeness of, the `single(era)`
    instances such a workflow is constructed into for its resource tokens: their maps may refuse an
    era set (a card needs all of its eras).
  - **The merge, HistTuple and AnalysisCache producers check the `period` column of every valid
    event against their era** (`Utilities.book_era_check`/`check_era`) and must keep doing it.
- **Names that stay `period` on purpose:** the AnaTuple column `period` (per-event C++ `Period`), the
  C++ `Period` enum, `validity_period` cache settings. Output paths (`<version>/<Kind>/<era>/…`), bundle
  URIs and job-file names are unchanged, so existing outputs must keep being found. Do not rename them.
- **No global `Corrections`.** Payloads build one per dataset/era/stage and pass it to the analysis
  hooks as the required keyword `corrections=`: `anaTupleProducer.py` constructs `Corrections(...)`
  directly for `anaTupleDef.addAllVariables`; `HistTupleProducer.py` and `AnalysisCacheProducer.py`
  use `Utilities.CreateCorrections(setup, dataset_name, stage, aggregated_caches=None)` for
  `histTupleDef.GetDfw` and `DefineWeightForHistograms`; `MergeAnaTuples.py` constructs one directly
  per merge (no hook). It must stay referenced for the dataframe's lifetime (`dfw.corrections`, or a
  local of the function that runs the event loop): its C++ providers are referenced by address. Flag a
  hook reading a module-level instance, and a `Corrections` created inside a function whose dataframe
  outlives it.

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
- A stitched process's bin denominators are summed over every dataset whose events fall into the bin, so a dataset may cover several bins only if it covers each of them completely or not at all; a dataset reaching part of a bin (e.g. the 0-parton events of a pT(ll)-nested DY sample) must be cut back to whole bins with an `event_filter`.

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

### Process-level state

- **One law process can handle several eras, versions and `user_custom` files.** A class-level
  cache or latch must be keyed by everything that determines its value (the whole era tuple
  included) — for branch maps and
  what is derived from them, `Task._branch_map_cache_key()` through `Task.cached_branch_map` — and
  must never keep law's placeholder map (`[None]`) or a negative completeness verdict. An
  unkeyed one hands one era's answer to another (cms-flaf/FLAF#338: the Builder selected the
  second era's branches from the first era's `AnaTupleFileTask` map).
- `Setup.get_processors(..., create_instances=True)` returns new processor instances on every
  call; the caller owns them (`mergeAnaTuples` adds its `ds` processor to the dict).
- `Setup` accepts only `known_eras()` (the `FLAF/config/<era>/` directories whose `global.yaml`
  says `era: <era>`) and raises when any later layer, `user_custom` file or customisation changes
  `era`. `ERA_IDS` mirrors the C++ `Period` enum; `test/test_era_process_state.py` compares them.

### Concurrency

- **Producers must not write bare-relative temp files.** CWD is shared between branches under
  `law --workers`, so two branches race on the same name. Write under the job's working
  directory.
- **Every `hadd -j` passes `-d` with a fresh directory of that merge alone**
  (`RunKit.run_tools.hadd_parallel` does). Without `-d`, hadd writes its partial files
  (`partial<i>_<TUUID>.root`) to the shared temporary directory, where two concurrent merges have
  drawn the same TUUID and swapped content (FLAF#361).

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

### ROOT object ownership in Python

- **A ROOT object detached with `SetDirectory(0)` that no other C++ object owns must be given to
  Python: `ROOT.SetOwnership(obj, True)`** (or `obj.__python_owns__ = True` in a module that does
  not import ROOT). An object still held by a C++ owner (an RDataFrame result's `shared_ptr`, an
  owning collection) must not be. PyROOT 6.40 never owns what `Clone()`, `TKey::ReadObj()`, `TDirectory::Get()`,
  `Rebin(n, newname, …)` or `Projection*()` return, a TH1 constructed while a file is the current
  directory belongs to that file, and `SetDirectory(0)` does not give ownership back: without the
  call every such histogram stays allocated until the job ends. Never give ownership while the
  object is still in a file's directory: `TFile::Close` deletes it, and Python deletes it again.

### NN inference (`env.sh`, `Common/Setup.py`, `Common/OnnxSession.py`)

- **`TF_ENABLE_ONEDNN_OPTS=0` everywhere TensorFlow runs.** TensorFlow turns its oneDNN kernels on
  by itself on CPUs with AVX512_VNNI/AVX_VNNI/AVX512_BF16/AMX only, which makes HHbtag and DNN
  outputs depend on the batch node (cms-flaf/FLAF#357). `env.sh` exports it, and every environment
  built from an empty one must set it too: `Setup.cmssw_env` and the `cmsEnv` alias do. A new
  `env -i`/whitelist environment that can run TensorFlow without it is a regression.
- **ONNX Runtime sessions are created through `FLAF.Common.OnnxSession`** (`make_session`, or
  `session_options()` handed to code that creates its own sessions, as DeepHME's
  `session_options`): one intra-op and one inter-op thread. A bare `ort.InferenceSession` with the
  default options sizes its pool by the host's physical cores and gives outputs that depend on it.
- What no setting of these builds removes is accepted and documented
  (`docs/concepts/environment.md`), not a defect of a change: ONNX Runtime's AVX2 vs AVX-512 kernels
  (`x86_64-v3` vs `x86_64-v4` nodes), TensorFlow's Intel vs AMD results from Eigen's
  `rsqrtps`/`rcpps`-based `Rsqrt` and reciprocal, and, in `flaf_env`, TensorFlow's AVX-512-dependent
  matrix products (oneDNN `sgemm` inside Eigen contractions).
- **A TensorFlow thread pool sized by the host is a reproducibility defect, as for ONNX Runtime:**
  the number of intra-op threads changes the outputs, and a fixed number removes that
  (`TF_NUM_INTRAOP_THREADS`/`TF_NUM_INTEROP_THREADS` or `tf.config.threading`). `env.sh` does not set
  it; the HH→bb̄ττ DNN in `flaf_env` still runs with the default pool (a follow-up of
  cms-flaf/FLAF#357), so new TensorFlow inference code should fix its thread count.

## Configuration invariants

- `config_path_order` layers four directories (`FLAF/config`, `FLAF/config/<era>`, `config`,
  `config/<era>`; for `global.yaml` each directory's `user_custom.yaml` joins in, and
  `--user-custom` comes last). The files are **concatenated as text and parsed once**, so a
  repeated top-level key **replaces the earlier value wholesale** — lists and nested dicts
  included; nothing is merged or concatenated. An analysis or era that redefines a block such as
  `corrections:` must repeat every entry it still needs, or inherit them with
  `<<: *corrections_default` from an anchor on the top-level block and override only what
  differs (one level deep). Every reader of an era's `global.yaml` — the `Config` for
  `reuse_mc_from_era` included, and any script that parses the file with `yaml.safe_load` on its
  own — must layer the top-level directories too, or such an alias is undefined. Datasets from
  several layers combine only because each dataset is its own top-level key.
- Dataset split: SM backgrounds and data live in `FLAF/config/<era>/datasets.yaml`; signals and
  CI samples live in the analysis. A signal added to the framework config is misplaced.
- A dataset whose NanoAOD lacks some weights is handled in its entry, per NanoAOD source (`v12`,
  `HLepRare`, …, as `fileNamePattern`: one analysis reads DAS, another a skim of it):
  `exclude_files` drops individual files (InputFileTask refuses a name the listing does not have),
  `event_filter` removes events before anything is booked (as if they did not exist), and
  `disabled_corrections` sets a shape weight to 1 for every member (its branches still exist)
  or drops any other correction. An option not keyed by source is a finding. None belongs in a
  correction's `processes:` list or in a code guard on the branch size.
- `Run3_2025` and `Run3_2026` carry no MC of their own — they set `reuse_mc_from_era: Run3_2024`.
  A dataset list edited for 2024 changes all three.
- Cross-section keys referenced by a dataset must exist in `crossSections*.yaml`; CI checks this,
  so flag it only when the diff adds a reference CI cannot see.
- Process hierarchy: `Setup.process_parent` / `process_ancestors` / `original_process` lead from a
  dataset's process (an expanded meta-process member, a sub-process) to the entry the physics model
  lists; analysis code should decide by that entry, not by substrings of process names. Within the
  groups the model reaches, a process may belong to one group only. The `parent_process` key on a
  base process is different: the model process after expansion, which histograms are merged under.
- Plot blinding (`Common/PlotBlinding.py`, applied by `Analysis/HistPlotter.py`): a variable's
  `histograms.yaml` entry takes `blind` and `blind_ranges` only; any other key starting with
  `blind`, and a blinding key in the page-level `data_hist`, raises. A `blind_ranges` window hides
  (NaN content and error) every data bin it overlaps and every bin the rebinning or the overflow
  merge fills from a stored bin it overlaps, in every region, control regions included. A change
  that lifts a window in some region, or draws a hidden bin as a marker at zero, shows blinded
  data.

## Testing expectations

Physics correctness cannot be unit-tested without CERN infrastructure, but the framework's
mechanics can, and `test/` has suites for the ones that bit us (path cache, bundle hashing,
stitching variables, cost model). Changes to those areas should extend them.

When a test uses a fake, the fake must call the **real** `__init__` and patch only what is
genuinely unavailable. Hand-mirroring a class's attributes creates a copy that silently stops
matching — that is how the path-cache suite went red for a whole merge cycle.

Note what CI actually runs from `test/`: `test_setup_loading.py` (via `test-setup-loading`, on
analysis PRs only), the config checkers `checkCrossSections.py`,
`checkDatasetConfigConsistency.py` and `checkDatasetNaming.py` (on FLAF PRs that change those
config files), and, through `unit-tests.yaml` on every FLAF PR and push to `main`, every
`test/**/test_*.py` that pytest collects except the tests marked `needs_root`, `needs_cvmfs` or
`needs_proxy`, plus `run_tools/check_era_rename.py` over FLAF's own files (no leftover of the
`period` → `era` rename). `test/conftest.py` defines the markers and the collection rules (its
`collect_ignore` lists the scripts named `test_*.py`); the analyses load it from their pinned FLAF
when they run their own `test/` through the same workflow. The marked tests are not run anywhere,
so a broken one is not caught automatically. A new test needs no workflow change; one that needs
ROOT, CVMFS or a grid proxy must carry the marker (module-wide `pytestmark` when the module cannot
be imported without it), or it fails the job; a test that reaches ROOT only through an import
(`FLAF.Common.Utilities`) takes a withdrawn stand-in instead and runs in CI. All test modules share one process: a stand-in for a
missing module (an empty `ROOT`) must be removed from `sys.modules` after the import it serves;
`test/conftest.py` stops the run on a module that leaves one for ROOT.

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

`formatting-check` (black, yamllint, clang-format, and in FLAF `flake8 --select=F821,F811` over
the whole package), `unit-tests` (every test in `test/` not marked `needs_root`/`needs_cvmfs`/`needs_proxy`, the `period` → `era` leftover check and the era-scope rules of `check_era_rename.py`), `repo-sanity-checks` (binary
files, repo size), `ds-consistency-check`, `cross-section-check`. Formatting, indentation, quote
style and trailing whitespace are settled by tooling; comments about them are pure noise.

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

Verified 2026-10-25; re-check before relying on any of it.

| | |
|---|---|
| Layout | `AnaProd/` (anaTuple production tasks), `Analysis/` (histogram/plot tasks), `Common/` (`Setup.py`, utilities), `Processors/` (stitching), `RunKit/` (vendored grid/job tools), `run_tools/` (`law_customizations.py`, CRAB helpers, `eos_lock.py`), `config/`, `include/` (C++ headers), `test/`, `docs/` |
| Submodule | `PlotKit` only. **`RunKit` is vendored**, not a submodule; imports are `from FLAF.RunKit.<module> import …` |
| Datasets | `config/<era>/datasets.yaml` for Run 3. Run 2 eras still use the older `samples.yaml` |
| Eras | `Run3_2022`, `Run3_2022EE`, `Run3_2023`, `Run3_2023BPix`, `Run3_2024`, `Run3_2025`, `Run3_2026`; Run 2 legacy. `--era` takes several for the stage classes; ids 5 (`Run3_2022`) … 11 (`Run3_2026`), branch number over several eras `era id × 1 000 000 + local` |
| Workflows | `formatting-check`, `unit-tests` (also reusable: each analysis runs its own `test/` through it, against its pinned FLAF), `repo-sanity-checks`, `ds-consistency-check`, `cross-section-check`, `test-setup-loading`, `test-era-dimension` (reusable: an analysis calls it with its eras), `deploy-docs`, `integration-test`, `trigger-flaf-integration` |
| Integration test | Triggered by `@cms-flaf-bot please test`. Its configuration (process lists, eras, versions) lives in **`cms-flaf/FLAF_ci`**, not in this repo |
| Docs | `docs/`, built with `mkdocs build --strict`; see the documentation section above |
| Package pins | `run_tools/mk_flaf_env.sh` (law `0.1.21`, luigi `3.8.1`, pip `26.2.1`, …); the CI workflows install the same versions, and `unit-tests` also the `onnx`/`onnxruntime` of the LCG release |
