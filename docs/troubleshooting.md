# Troubleshooting & FAQ

The most common ways a FLAF run goes wrong, and how to fix them. If your symptom is not here, check
the job logs and the task status (`--print-status 3,1`). HTCondor and CRAB jobs keep their logs by
default (`--transfer-logs` defaults to on, so it need not be passed): when `fs_default` is remote
storage they are staged to `<version>/logs/<task>/<era>/` there (the two analysis-cache tasks
add a `<producer>/` level; locally only `AnalysisCacheTask` does). With a local `fs_default` the log stays in the job's output sandbox: HTCondor
copies it to `data/<version>/<task>/<era>/` in the checkout only with `--htcondor-spool False`;
with the default `-spool` it remains on the schedd until retrieved with `condor_transfer_data`. LAW names a failed job's log file in its output.

## `law: command not found`
You did not `source env.sh` in this shell. Every new terminal needs it once
([Installation](getting-started/installation.md)).

## Import errors / empty submodule directories
You cloned without `--recursive`, so submodules (FLAF, PlotKit, physics tools) are empty. Fix:

```sh
git submodule update --init --recursive
```

## A batch job stops in `env.sh`: `… was not built by the current …/mk_flaf_env.sh`
The environment the job runs from is not the one the installation script of its FLAF defines
(another pin, another LCG release), and a job never builds one, because other jobs run from that
environment too. A job without `--bundle` uses the `flaf_env` of the checkout on AFS: source
`env.sh` on the submitting machine, which rebuilds it, and resubmit. A bundle job unpacked a `soft`
bundle of another environment. That bundle is named after the environment
(`soft_<FLAF_ENVIRONMENT_ID>`), so this happens when a bundle was replaced by hand, or when
`run_tools/mk_flaf_env.sh` changed (a `git pull`, say) after `env.sh` was sourced in the shell that
submitted: source it again there and resubmit. `ERROR: cannot identify the FLAF environment` or
`ERROR: cannot tell whether …` means that the check itself could not be made (storage that did not
answer, for example), and nothing was removed or built; source `env.sh` again once the storage
answers. When it says `… is listed in … but not seen as a directory` and persists, something that
is no reachable directory (a file, a broken link) sits at `soft/flaf_env`: move it away by hand. See
[The environment](concepts/environment.md#what-envsh-sets-up).

## `unknown era(s) …; known eras: …` / `the configuration says era: …`
`--era` names one era, and the known eras are the directories `FLAF/config/<era>/` whose
`global.yaml` says `era: <era>`: the seven Run 3 eras ([Eras](concepts/eras.md)). A typo or a
meta-era name (a datacard era such as a sum of sub-eras) is not one; a payload script started by
hand with such an `--era` stops with `unknown era '…'; known eras: …` instead. A comma-separated
list takes several known eras, for the tasks that support them (see
[below](#task-takes-one-era-got-era)). The `era` key itself is set only by that file: an
analysis layer, a `user_custom` file or a `--customisations` entry that sets it to another value
stops the run.

## `'period' was renamed to 'era' (--era)`
The `period` parameter is gone. Use `--era Run3_2022` instead of `--period Run3_2022`, `era=` instead of
`period=` in code (a `period=` keyword stops with `<Task>: 'period' was renamed to 'era' (use era=...)`),
and `{era}` instead of `{period}` in a bundle pattern (the message then reads
`… use {period}, which was renamed to {era}`). The same message is raised for `--<Task>-period` and for a
`period` entry in the `[<Task>]` section of a law/luigi config file: replace it by `--era` on the command
line. Outputs written before the rename are found as before, since the paths never changed.

## `the eras are set once per invocation, with --era`
`--<Task>-era` (or `era` in a `[<Task>]` config section) is refused: the eras are given to
`law run` once, with `--era` (one era, or several separated by commas), and passed to every task of
the run. An unknown era lists the known ones.

## `<Task> takes one era; got --era …`
The task does not support several eras: `--era Run3_2022,Run3_2023` was given to `BundleTask` or
an analysis Studies task. The stages of the production and analysis chain and the `StatInference`
tasks take several eras ([Eras](concepts/eras.md#which-tasks-take-several-eras)); launch the
others once per era.

## `the cards of <configuration> need <card>: <eras>, which --era does not include`
A `StatInference` task was run over fewer eras than its datacard configuration needs: every real
era behind a card (the card's own era, or the members of a group era in `era_groups:`) must be in
`--era`. Add the eras the message names, e.g. `--era Run3_2022,Run3_2022EE,Run3_2023,Run3_2023BPix`
for a configuration whose cards cover those four. Nothing has run.

- `ResonantLimitsTask`, `ResonantLimitsAndHistPlotTask` and the plot tasks check when they are
  created: `law run` ends at once with a traceback whose last line is this message (exit code 60).
- `CreateDatacardsTask` or `PreprocessShapesTask` started directly check while luigi schedules
  them. Over `--workflow htcondor` the message ends a traceback (exit code 60): luigi asks the
  workflow for its resources outside its error handling. Over `--workflow local` luigi reports
  `1 failed scheduling` and `law run` exits with code 50.

## `--branch <n> with --era <eras>: a branch belongs to one era`
`--branch` selects one branch, and over several eras a branch belongs to one era. Use
`--era <era> --branch <its number in that era>`, or `--branches` with the numbers of the run over
several eras (`era id × 1 000 000 + the number in the era's own map`, see
[Eras](concepts/eras.md#branch-numbers)).

## Branch selection errors: `branch <n> is branch <m> of <era> …`
A `--branches` selection (`BranchSelectionError`) does not fit the numbering of the run:

- `branch 6000001 is branch 1 of Run3_2022EE in a run over several eras; a run over one era takes
  that era's own numbers: --era Run3_2022EE --branches 1`: a number of a run over several eras was
  given to a run over one era. Use the number the hint names.
- `branch <n> is branch <m> of <era>, not of an era of this run`: over several eras, a number that
  belongs to an era that `--era` does not list (or to no era).
- `an open range in --branches` / `the range <a>:<b> spans <era> and <era>; a range stays inside
  one era`: over several eras every range has a start and an end and stays inside one era.
- `--branches selects no branch of <era>; select a branch of each era or drop it from --era`: every
  era of the run needs a number; an era without one is still part of the run.
- `--branches selects no branch of the <N> of <era>`: over several eras, every number given for
  `<era>` lies outside its branch map (`<N>` branches, numbered `era id × 1 000 000` upwards). A run
  over that era alone would take such a selection as the whole map; select an existing branch.
- `branch <n> is beyond the 1000000 numbers of a run over one era`: a run over one era was given a
  number of 1 000 000 or more that belongs to no configured era.
- `branch <n> is in the numbering of the stages with era branches; this workflow has none and
  numbers its branches as its own map does over any eras`: a workflow without era branches (the
  `StatInference` card workflows) was given such a number over several eras. Its branches keep
  their own numbers: `--branches 1` is its second card.

## `… differs between the eras (…), but one run over several eras uses one value for all of them`
`EraDependentValueError`: a setting that a run over several eras reads once differs between its
eras. The message names the key and each era's value. Make it the same in every era, or run the
eras separately. The settings are `fs_default`, the `crab` block, `runTokenServer`,
`anaTuple_scheduling`,
`use_cmssw_env_AnaTupleProduction` and, per `payload_producers` entry, `n_cpus`, `max_runtime`,
`crab_memory`, `save_as` and `cmssw_env`; everything else, including `fs_anaTuple`, `fs_HistTuple`,
`fs_anaCacheTuple` and `fs_plots`, may differ ([Eras](concepts/eras.md#settings-that-must-agree-between-the-eras)).

## `a value of one era was read on a task over several eras`
`EraScopeError`: code read a per-era value (the setup, an output path, the era name) on a task
over several eras, where it has no single answer. In a task class the era is read through
`only_era` and the setup through `setup`, which are both for a task over one era; over several eras
`setup_for(era)`, `single(era)` or `common()` are used instead, as the message says
([Contributing](contributing.md#writing-code-for-several-eras)). On the command line the error
points to a bug in the task class.

## `No datasets selected with a custom selection: …` over several eras
`--dataset` has to match at least one dataset in every era of a run over several eras. A dataset
that exists in only one of the eras fails the run, because the configuration of each of the other
eras then selects nothing: run that dataset with `--era <its era>`.

## `… event(s) were produced for another era than …`
A producer (`AnaTupleMergeTask`, `HistTupleProducerTask`, `AnalysisCacheTask`) found events of
another era in its inputs: the AnaTuple column `period` (the C++ `Period` value) of a counted
event is not the era of the branch. The job fails: find the input file that carries the events
of the other era, regenerate or remove it, and run again.

## A run unexpectedly drops into `InputFileTask` / Rucio errors
For a from-scratch production, `InputFileTask` running first is normal. But if a run that should
reuse existing outputs keeps re-resolving inputs, or fails here, the cause is almost always a
**wrong `--era` or `--version`** (so the expected upstream outputs aren't found and LAW falls
back to regenerating them), or an **expired proxy**. Double-check the era/version, and:

```sh
voms-proxy-info        # is it still valid?
voms-proxy-init -voms cms -rfc -valid 192:00
```

If instead you see a transient Rucio error (e.g. `server returned 503`), it usually clears on a
retry. FLAF caches resolved storage locations (LFN→PFN) on disk, so once a path has been resolved
a Rucio outage no longer blocks commands that reuse it; the cache lives at
`$ANALYSIS_DATA_PATH/lfn_pfn_cache.json` (override with `FLAF_LFN_PFN_CACHE`, delete to reset).
`env.sh` also pins the Rucio version and warns if a newer one is available on cvmfs — pin a
different one with `FLAF_RUCIO_VERSION` if the default ever misbehaves.

## A job is slow to start reading a Rucio input file
The job log shows how the input copy went (see
[Where the inputs come from](concepts/storage.md#where-the-inputs-come-from)): `Trying <replica>`
with the limits of the attempt, `Joining with <replica>` when the first one is slower than
expected, `Copy attempt from <replica> failed: …` with the reason (`exit code …` and the end of
the tool's output, `no data for N s`, `no result after N s`, a size or adler32 mismatch), and
`Copied from <replica> in N s`. A site that keeps appearing in the failures is broken or
overloaded; the copy works around it. A job that fails with `Unable to copy …: every source
failed` lists every attempt and why it failed; `no copy within 21600 s` instead means that the
copy was still trying after 6 h.

## "Permission denied" / "file not found" on storage
Usually an **expired VOMS proxy** — grid/EOS access needs a valid one. Re-run `voms-proxy-init`. If
it persists, confirm your `fs_*` paths in `user_custom.yaml` are correct and writable
([Storage](concepts/storage.md)).

## "Task not found" after adding a task
LAW's index is stale. Re-run:

```sh
law index --verbose
```

Needed after **adding/renaming/moving** a task class (not after editing an existing one's body).

A batch job that fails with `task family 'FLAF.….<Task>' not found in index` is a different
problem: law reports a task module it cannot import this way, so the FLAF package was not
importable on the worker (a bootstrap or bundle that did not set it up).

## EOS read-after-write lag
EOS is eventually consistent: a file you just wrote can be briefly invisible to an existence check
(seconds, occasionally longer). In normal pipeline use FLAF tolerates this. If **your own** script
checks for freshly written outputs and intermittently "can't find" them, don't trust a single
`exists()` — list the parent directory and retry a few times with a short delay.

## `--print-status` reports outputs as missing although the files are on storage
Existence checks go through a path cache, so a status check does not stat every file
individually. Each user runs one path-cache server per machine, started on demand by the first
law process that needs it (`RunKit/pathCacheClient.py`, `ensure_local_server`). It listens on a Unix
socket, `/tmp/flaf-path-cache-<uid>/server_v1.sock` (the directory is mode 0700 and the socket 0600;
`FLAF_PATH_CACHE_DIR` chooses another directory, the socket path being at most 107 bytes). A process
uses it only when that directory belongs to its user and nobody else can enter it, and only a server
running as the same user is believed (checked on every connection on Linux); otherwise the process
works from its own cache. The server shares what one process learns with every other law process
of that user on that machine: the forked luigi workers, later runs and `--print-status`. A directory is listed once and the listing is
published to the server, so every later check is answered from the cache without touching storage.
The server keeps its state in `cache.json` next to the socket (written every 60 s while in use and
on SIGTERM/SIGHUP, reloaded on start), logs to `server.log` there, and exits after 24 h without a
request. On lxplus (`KillUserProcesses=yes`) it ends with the session that started it, and the next
law process restarts it from its state. If the server cannot be used, a process works from its own
in-process cache and says so once on stderr.

An "exists" is kept for 24 h. An "absent" and a directory listing marker (which implies an absence
for every file it does not list) are kept for 10 minutes only (`negative_validity` in
`RunKit/pathCacheServerCfg.yaml`), because no job publishes the files it writes: a file created
after a listing becomes visible when that listing expires. Jobs never contact a server (see
[CRAB → Caveats](workflow/crab.md#caveats) and
[HTCondor → Submission safeguards](workflow/htcondor.md#submission-safeguards)).

If a status check still disagrees with what you see on the storage element, list the directory
yourself (`gfal-ls`) to establish the truth, and drop the cached entries for that subtree:

```sh
python3 $FLAF_PATH/RunKit/pathCacheClient.py \
  --command invalidate_regex --path '.*/<version>/.*'
```

The client talks to this user's server on this machine (`--socket` chooses another one);
`--command` is `get`, `set`, `invalidate` or `invalidate_regex`. The pattern is matched against
stored paths, which are normalised — repeated slashes are collapsed, so a pattern containing the
URL scheme (`davs://…`) never matches. Match on the part of the path you care about, as above.

The caches are per machine. After an out-of-band deletion (a version removed by hand), invalidate
on each machine that ran law for that version, or stop the server (its pid is in `server.log`) and
remove `/tmp/flaf-path-cache-<uid>/cache.json`. Files that jobs remove (e.g. inputs consumed by
`AnaTupleMergeTask` with `delete_inputs_after_merge`, or by `HistMergerTask` with
`remove_merged_inputs`) are not published either: the server reads them as existing for up to 24 h.

### Outputs written by a job read as missing

Jobs publish nothing to the path cache, so a directory listing taken before a job wrote its file
keeps answering "absent" for that file until it expires (10 minutes). On both batch backends the
driver therefore never takes the batch system's "finished" at its word: at every poll it lists the
output directories of the jobs reported finished and accepts an absence only on a listing taken in
that poll (see [CRAB → Status handling](workflow/crab.md#status-handling) and
[HTCondor → Submission safeguards](workflow/htcondor.md#submission-safeguards)); a (re)started driver
takes fresh listings before it judges which outputs exist. A *plain* status check
(`--print-status`) started separately has no such guarantee: drop the cached entries as above if
it disagrees with the storage.

A listing that **fails** (timeout, SSL error, expired proxy) is never cached as "absent": only a
confirmed not-found is. Such a listing prints `GFALFileInterface: could not list …` (once per
directory per minute); every file looked up in it reads as missing while it lasts, and each
lookup lists again, so the first successful listing settles the rest of the directory.
Leftover `<name>.flaf-tmp-<pid>-<uuid>` files next to outputs are orphans of killed uploads
(uploads are published by rename,
[Storage](concepts/storage.md#how-uploads-are-published-and-absence-is-decided)); they take
quota until deleted, and can be deleted.

## A resumed run stops with "… jobs of this resumed workflow came back for missing outputs"
More than 10 % of the jobs of a resumed batch workflow (at least 2) came back for missing
outputs and were still missing on a fresh check, so the run stopped before resubmitting them;
their entries in the job file were left as they were. If the storage was unreachable, run again
once it is back. To redo the work on purpose use `--ignore-submission`. See
[HTCondor → Submission safeguards](workflow/htcondor.md#submission-safeguards).

## `… is not readable (…), so no job file can be built` / skipped submission rounds
A file the job is built from (law's job scripts, FLAF's `bootstrap.sh` or `stageout_logs.sh`)
cannot be read, usually because the Kerberos ticket or AFS token expired. Submission rounds are
skipped, with no loss, until it is readable again, and the run stops after 30 minutes. `klist -f`
shows the Kerberos expiry and the renewable window, `tokens` the AFS token.

## CRAB: the server refused a task (`SUBMITREFUSED`)
The message carries the server's reason and the project directory. The usual cause is a
`Site.whitelist` entry that is not a CMS Processing Site Name; the cached site list
(`<analysis>/data/cms_psn_sites.json`) is dropped automatically so the next submission re-reads
CRIC. The jobs are resubmitted as a new task; a second refusal in the same run stops it. Check
`crab.whitelist` and `crab.blacklist`. See [CRAB → Status handling](workflow/crab.md#status-handling).

## CRAB: the server failed to submit a task (`SUBMITFAILED`)
law reports the task's jobs failed and submits them again as a new task, and FLAF prints the
server's `Failure message from server` once with the project directory. A second failed
submission in the same run stops it, quoting the message: the cause did not go away. A common
one is a MyProxy credential the TaskWorker cannot retrieve (see the next entry); otherwise
`crab status -d <project directory>` and the task URL it prints show what the server says. See
[CRAB → Status handling](workflow/crab.md#status-handling).

## CRAB: "MyProxy credential valid for at least 5 days" is refused
The credential must be stored under the SHA1 of the DN with CRAB's retrieval policy. A bare
`myproxy-init` does not do that. Create it with `cmsEnv crab createmyproxy --days 30`
([CRAB → Prerequisites](workflow/crab.md#prerequisites)).

## Cross-analysis environment contamination
The environment caches paths in variables (`FLAF_PATH`, `ANALYSIS_PATH`, `ANALYSIS_SOFT_PATH`, …).
Reusing a shell that already set up a *different* analysis can pick up the wrong `flaf_env` and
produce baffling failures.

- **Interactive:** use a **fresh shell per analysis** and `source env.sh` there.
- **Scripted/background runs:** unset the FLAF/analysis variables before sourcing, but **keep**
  `LD_LIBRARY_PATH`, `HOME` and `PATH`:

```sh
unset FLAF_ENVIRONMENT_PATH ANALYSIS_SOFT_PATH LAW_HOME LAW_CONFIG_FILE \
      ANALYSIS_PATH ANALYSIS_DATA_PATH FLAF_PATH FLAF_CMSSW_BASE \
      FLAF_CMSSW_ARCH FLAF_CMSSW_VERSION FLAF_COMBINE_PATH \
      X509_USER_PROXY VIRTUAL_ENV PYTHONPATH
cd /path/to/<analysis>
source env.sh
```

## `Collection '…' mixes scalar columns … with array columns …` in `AnaTupleFileTask`
The anaTuple stores all columns that share the text before their first underscore as one collection
with one counter, so a scalar next to arrays of the same prefix would become an array. Rename the
columns in the analysis anaTuple definition so that scalars and arrays do not share a prefix (for
example a scalar `TTInfo_nLeptonicW` next to per-top arrays `genTop_pt`, `genTop_eta`, … rather
than `TTInfo_top_pt`).
The same step stops on `Columns changed type while fusing`, which lists every column whose type the
fused file does not preserve.

## `Column '…' is declared shift-invariant but differs in …` in `AnaTupleFileTask`
A column listed in `anaTuple_shift_invariant_columns` does change under the named shift, so it cannot
be taken from the central tree. It is usually a generator quantity attached to a reconstructed object
(a matched gen jet or lepton, a flavour label of a jet), which follows the selected object. Remove the
pattern that matches it and produce the anaTuples again under a new `--version`: the files already
produced with the old list cannot be merged with new ones (`AnaTupleMergeTask` stops with
`anaTuples of dataset … were produced with different anaTuple_shift_invariant_columns`). Related
messages: an array collection listed only in part, a listed column missing from some shifted trees,
or a pattern matching `valid`/`FullEventId`.

## ROOT/cling library or JIT errors in a background run
You launched the environment under `env -i`, which strips `LD_LIBRARY_PATH` (ROOT/cling needs it).
Preserve it (and `HOME`, `PATH`) when starting a clean shell. See
[The environment](concepts/environment.md#sharp-edges).

## `source env.sh` sets the wrong path / fails to locate itself
`env.sh` sets `ANALYSIS_PATH` to the directory of the file it was sourced from (`BASH_SOURCE` in
bash, `%x` in zsh). That breaks when its *text* is run instead of the file — `eval "$(cat env.sh)"`,
piping it into a shell, `source /dev/stdin` — which makes `ANALYSIS_PATH` the current directory or
`/dev`. Source the file itself (`source /path/to/<analysis>/env.sh`) in the shell you run `law` from.
Sourcing it in a child shell (`bash -c "source env.sh"`) locates it correctly, but the settings
only last for that command — put such commands in a **script file** that sources `env.sh` first.

## HH→bb̄WW: the run sits in `AnalysisCacheTask` for a long time
Expected on a cold cache. `AnalysisCacheTask` runs every payload producer whose columns the
selected variables use — with HH→bb̄WW's default variables, `DeepHME` and the two DNNs (`DNN`,
`TwoStageDNN`) — plus the global `BtagShape` producer that the `btag` correction needs, each as its
own workflow with one branch per merged anaTuple file; several of them are configured for jobs of
many hours. Reuse an existing cache across runs with a
[per-task version override](workflow/arguments.md#per-task-version-overrides) instead of
recomputing it every time.

## A backgrounded `law run` won't stop when I kill it
Killing the parent leaves child `law`/job processes alive. Kill by a pattern that matches the
command line as you typed it, and remove batch jobs:

```sh
pkill -f "law run .*--version[= ]<your_version>"
condor_rm <cluster>      # if you submitted to HTCondor
```

## My edits to FLAF/Corrections are ignored
You edited the submodule copy but the run used a different one — or vice-versa. The run uses
`FLAF_PATH`/`CORRECTIONS_PATH`; set them to your edited copy **before** `source env.sh`. See
[Developing shared submodules](concepts/environment.md#developing-shared-submodules).

Two limits of that mechanism:

- The edited copy's directory must be named `FLAF` (or `Corrections`): `env.sh` puts its *parent*
  directory on `PYTHONPATH`, and Python imports it by that name.
- The framework configuration (`FLAF/config/*.yaml`) is always read from the analysis's own
  `FLAF/` submodule, whatever `FLAF_PATH` says. Edit those files there.

## The first `source env.sh` takes forever
Expected: the first time it builds CMSSW and Combine (tens of minutes, a few GB under `soft/`).
Subsequent sources are quick. Don't interrupt the first build.

## Arrays of a shifted tree differ from the nanoAOD values beyond the central length
anaTuples produced before shifted trees got their own array counters (`n<collection>__shifted`, see
[Array counters in the shifted trees](concepts/data-flow.md#array-counters-in-the-shifted-trees))
store the elements of a shifted collection beyond the length of the central one as differences
from values that are not part of the central event (rounded, for floating-point columns, like every
delta), and readers may add different such values back. `Central.<array>` read from such a shifted tree also comes with the size of the
shifted collection. The shifted values of an event whose collection is not longer after the shift
are correct. The anaTuples have to be produced again; there is no way to repair them afterwards.
