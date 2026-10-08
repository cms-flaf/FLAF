# Troubleshooting & FAQ

The most common ways a FLAF run goes wrong, and how to fix them. If your symptom is not here, check
the job logs and the task status (`--print-status 3,1`). HTCondor and CRAB jobs keep their logs by
default (`--transfer-logs` defaults to on, so it need not be passed): when `fs_default` is remote
storage they are staged to `<version>/logs/<task>/<period>/` there (the two analysis-cache tasks
add a `<producer>/` level; locally only `AnalysisCacheTask` does). With a local `fs_default` the log stays in the job's output sandbox: HTCondor
copies it to `data/<version>/<task>/<period>/` in the checkout only with `--htcondor-spool False`;
with the default `-spool` it remains on the schedd until retrieved with `condor_transfer_data`. LAW names a failed job's log file in its output.

## `law: command not found`
You did not `source env.sh` in this shell. Every new terminal needs it once
([Installation](getting-started/installation.md)).

## Import errors / empty submodule directories
You cloned without `--recursive`, so submodules (FLAF, PlotKit, physics tools) are empty. Fix:

```sh
git submodule update --init --recursive
```

## A run unexpectedly drops into `InputFileTask` / Rucio errors
For a from-scratch production, `InputFileTask` running first is normal. But if a run that should
reuse existing outputs keeps re-resolving inputs, or fails here, the cause is almost always a
**wrong `--period` or `--version`** (so the expected upstream outputs aren't found and LAW falls
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

## EOS read-after-write lag
EOS is eventually consistent: a file you just wrote can be briefly invisible to an existence check
(seconds, occasionally longer). In normal pipeline use FLAF tolerates this. If **your own** script
checks for freshly written outputs and intermittently "can't find" them, don't trust a single
`exists()` — list the parent directory and retry a few times with a short delay.

## `--print-status` reports outputs as missing although the files are on storage
Existence checks go through a path cache (a per-process cache backed by the shared cache server
configured under `WLCGFileSystem` in `global.yaml`), so a status check does not stat every file
individually. A directory is listed once; that listing is published to the cache server, so every
later check — in any process, including jobs — is answered from the cache without touching
storage, and a file created afterwards is published by the job that writes it. Only a path that
the cache knows nothing about, in a directory whose listing has expired, costs a real listing.
That is also what makes the cache self-healing: if a job's own cache update is lost, its file
becomes visible again as soon as the directory's cached listing expires.

If a status check still disagrees with what you see on the storage element, list the directory
yourself (`gfal-ls`) to establish the truth, and drop the cached entries for that subtree:

```sh
python3 $FLAF_PATH/RunKit/pathCacheClient.py --host cms-flaf.cern.ch --port 5000 \
  --command invalidate_regex --path '.*/<version>/.*'
```

The pattern is matched against stored paths, which are normalised — repeated slashes are
collapsed, so a pattern containing the URL scheme (`davs://…`) never matches. Match on the part
of the path you care about, as above.

### CRAB: outputs written by a job read as missing

CRAB workers cannot reach the path-cache server, so a directory listing published before a CRAB
job wrote its file keeps answering "absent" for that file until it expires (24 h by default). On
`--workflow crab` the driver therefore takes one fresh listing per output directory per poll
before it accepts an absence, and believes a CRAB `FINISHED` only when the outputs are on
storage (see [CRAB → Status handling](workflow/crab.md#status-handling)); a (re)started driver
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
