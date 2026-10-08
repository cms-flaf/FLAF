# Running on CRAB (WLCG)

HTCondor covers the CERN local batch farm. For jobs that should run anywhere on the
**WLCG** (CMS CRAB), FLAF tasks can be submitted with `--workflow crab`. The implementation
uses [law's CMS CRAB workflow](https://github.com/riga/law) (`law.contrib.cms.CrabWorkflow`).

Analysis **outputs and job logs** use FLAF remote I/O only (`fs_default` via gfal/`davs://`,
plus `stageout_logs.sh`). CRAB `transferOutputs` / `transferLogs` are forced **off** so
nothing is duplicated onto CRAB's stageout area. The CRAB client still needs
`Site.storageSite` / `Data.outLFNDirBase` for a submit-time write check — those
fields are derived from `fs_default`, not configured separately.

## Prerequisites

1. A valid **VOMS proxy** for the CMS VO (`voms-proxy-init --voms cms -valid 192:00`).
2. A **MyProxy** credential valid for **at least 5 days**, in the form **CRAB** reads it:
   stored under the SHA1 of your DN (the username CRAB looks for), with the retrieval policy
   of the CRAB task workers. FLAF fails early if the VOMS proxy is missing or expired, or if
   no credential under the SHA1 username with at least 5 days left is found (instead of
   waiting for a server-side `SUBMITFAILED`). It cannot see the retrieval policy, so a
   credential made by hand under that name without one passes the check and is refused by
   the server; create it as below. There is no password-file fallback.
   Create it with the CRAB client, in a shell with the analysis `env.sh` sourced:

    ```sh
    cmsEnv crab createmyproxy --days 30   # asks for the grid certificate passphrase
    ```

    CRAB fetches the task-worker retrieval policy itself, so nothing needs to be spelled
    out by hand. A bare `myproxy-init` is **not** enough: it stores the credential under
    the plain DN and without the policy, so CRAB never finds it.

    Nothing in a run renews the credential (law submits with `crab submit --proxy`, which
    skips CRAB's own delegation), so create it with enough days for the whole campaign.

    ??? note "Without the certificate passphrase (stop-gap)"
        With only a VOMS proxy at hand, the client can delegate from the proxy itself:

        ```sh
        cmsEnv env X509_USER_CERT=$X509_USER_PROXY X509_USER_KEY=$X509_USER_PROXY \
          crab createmyproxy --days 7
        ```

        The credential then cannot outlive the proxy: its lifetime is clamped to the whole
        days left on the proxy, so a fresh 8-day proxy gives a 7-day credential, which clears
        the 5-day minimum for about two days. The client's warning that "your user
        certificate is going to expire in 7 days" refers to the proxy standing in for the
        certificate. The two variables are set for this one command only and must
        **never be exported**: they precede the proxy in the GSI search order and break
        other commands.

3. CRAB client available (via CMSSW / the law CMSSW sandbox; the sandbox is set by
   `crab_sandbox_name` in the `[job]` section of the analysis `config/law.cfg`, currently
   `CMSSW_14_0_0::arch=el9_amd64_gcc12`).
4. **Bundles**: CRAB workers do not mount AFS. FLAF always ships code via `BundleTask`
   when `--workflow crab` is used (same tarballs as HTCondor `--bundle`). Tasks already
   declare `bundle_flavours`.
5. Remote `fs_default` (e.g. `davs://eoshome-...`) so bundles and analysis outputs are on a
   grid-accessible filesystem.

## Config

CRAB's write-check site is taken from `fs_default`:

| `fs_default` | CRAB `storageSite` + `outLFNDirBase` |
|---|---|
| `T3_CH_CERNBOX:/store/user/<you>/...` | as written |
| `davs://eoshome-<initial>.cern.ch:.../eos/user/<initial>/<you>/...` | `T3_CH_CERNBOX` + `/store/user/<you>/...` |

Verify write access before the first campaign:

```sh
crab checkwrite --site=T3_CH_CERNBOX --lfn=/store/user/$USER
```

Every CRAB setting is optional and lives in `global.yaml` / `user_custom.yaml` under `crab:`:

```yaml
crab:
  # whitelist: [T2_CH_CERN]   # omit to use all T1/T2/T3 sites
  # blacklist: [T2_US_MIT]
  # parallel_jobs: 5000       # default --parallel-jobs; CLI wins if set
  # refill_fraction: 0.2      # minimum wave size as a fraction of parallel_jobs
  # retry_release_minutes: 45 # a parked retry goes out after this long, whatever the wave size
  # poll_interval: 5          # minutes between crab status polls; CLI wins if set
  # min_runtime_min: 60       # floor for CRAB maxJobRuntimeMin
  # auto_blacklist:           # automatic site quarantine (on by default)
  #   enabled: true
  # watchdog:                 # stall watchdog (on by default)
  #   enabled: true
  # ignore_global_blacklist: false   # waive CMS's own site blacklist (not recommended)
```

!!! note "A `crab:` block in `user_custom.yaml` replaces the `global.yaml` one wholesale"
    The config layers are concatenated and parsed as one YAML document, so a later
    `crab:` mapping wins as a whole — repeat the keys you want to keep.

| Key | Meaning |
|---|---|
| `whitelist` | Restricts `Site.whitelist`. Default: `T1_*`, `T2_*`, `T3_*`. |
| `blacklist` | Sites (or glob patterns) to exclude. Removed from the whitelist itself — see [Sites and quarantine](#sites-and-quarantine). |
| `parallel_jobs` | Default for `--parallel-jobs` on CRAB (CLI wins). Default: `5000`. Caps how many CRAB jobs are in flight and thus the size of each CRAB task. CRAB itself refuses more than 10 000 jobs in one task. |
| `refill_fraction` | Minimum wave size, as a fraction of `parallel_jobs`. Default: `0.2`. See [The wave gate](#the-wave-gate). A value that is not a number is an error. |
| `retry_release_minutes` | How long the wave gate may hold a retry back before releasing it whatever the wave size. Default: `45`. A value that is not a number is an error. |
| `poll_interval` | Minutes between `crab status` polls (CLI `--poll-interval` wins). Default: `5`. Each poll is one multi-MB `crab status --json` per live CRAB task. |
| `min_runtime_min` | Lower bound, in minutes, for CRAB `JobType.maxJobRuntimeMin` (`--max-runtime` converted to minutes), since every job first downloads and unpacks its bundles. Default: `60`. It must parse as a whole number: an unparseable value raises an error instead of silently leaving CRAB's own 1250 min default (which would kill every longer job). |
| `auto_blacklist` | Mapping (or `false`). Automatic site quarantine, on by default — see [below](#automatic-site-quarantine). |
| `watchdog` | Mapping (or `false`). Stall watchdog, on by default — see [Stall watchdog](#stall-watchdog). |
| `ignore_global_blacklist` | Set `true` to waive CMS's own blacklist of known-broken sites (`Site.ignoreGlobalBlacklist`). Not recommended: with an open site pool it is the main protection against burning jobs at bad sites. |

!!! warning "`crab.memory_mb_per_cpu` is retired"
    CRAB memory is no longer derived from a per-core figure. A `memory_mb_per_cpu` key in the
    `crab:` block raises an error that names the replacements below
    ([`--<Task>-crab-memory`](#resources)); remove the key.

## Resources

A CRAB job is described by the cores and the memory CRAB is asked for, both derived from the
task's `--n-cpus` and `--crab-memory`.

**Cores.** CRAB accepts only 1, 2, 4 or 8. `n_cpus` is rounded up to the next of these (3 cores
is requested as 4) and `n_cpus > 8` is refused; the payload still runs `n_cpus` threads. The
generated PSet carries the same thread count, as CRAB requires.

**Memory.** `JobType.maxMemoryMB` is a **kill threshold**, not a reservation: a job above it is
removed (exit code 50660) and CRAB never retries that, while law's retries would repeat the same
peak. With no explicit request a job therefore asks for the most CRAB grants for its cores,
`max(3000, 2500 × cores)` MB:

| Cores | Default `maxMemoryMB` |
|---|---|
| 1 | 3000 |
| 2 | 5000 |
| 4 | 10000 |
| 8 | 20000 |

A task that needs a different amount asks for it with `--crab-memory` (MB), per task:

| Where | Form |
|---|---|
| command line | `--<Task>-crab-memory 6000` (e.g. `--AnaTupleFileTask-crab-memory 6000`) |
| `law.cfg`, in a `[luigi_<Task>]` section (law hands only `luigi_*` sections to luigi) | `crab_memory: 6000` |
| `AnalysisCacheTask` | `payload_producers.<producer>.crab_memory` in `global.yaml`; overrides the CLI value, like `n_cpus` and `max_runtime` |

An explicit request is honoured exactly. When it exceeds what the payload's cores may hold, cores
are bought (a request of 8000 MB with `n_cpus: 1` is submitted with 4 cores, whose ceiling is
10000 MB). A request above 20000 MB (what 8 cores allow) or below 1000 (taken to be a value not
given in MB) is refused at submission, never shrunk. Like `--n-cpus` and `--max-runtime`,
`--crab-memory` belongs to the task it is given for and is not handed to the tasks it requires.

**Runtime.** `--max-runtime` (hours) becomes `maxJobRuntimeMin`, at least `crab.min_runtime_min`.

## Sites and quarantine

The CRAB client requires `Site.whitelist` because law uses dummy `userInputFiles` (no input
dataset). FLAF defaults that list to `T1_*`, `T2_*`, `T3_*` so jobs can run at every CMS
processing site.

CRAB gives the **whitelist precedence** over the blacklist: a site matched by both lists is
*kept* (the client only prints a warning). FLAF therefore removes excluded sites — the
configured `blacklist` and the automatic quarantine alike — from the whitelist itself,
expanding a glob that covers an excluded site into the concrete sites it matches, minus the
excluded ones. Blacklist entries may be globs too.

The expansion uses the CMS **Processing Site Names** (PSNs) from CRIC (`preset=site-names`,
rows of type `psn`): the list the CRAB server validates a whitelist against. A name outside it
gets the whole task refused. The older compute-unit based list admitted names that are not PSNs
(`T3_CH_CERN_HelixNebula_REHA`) and missed 39 that are. The list is cached for 24 h in
`<analysis>/data/cms_psn_sites.json`; a list with fewer than 50 names is taken for a changed
payload and refused on every path (cache included); when CRIC cannot be read, a stale cache is
used with a message that gives its age. A glob covering nothing excluded is passed through
unchanged, so without a blacklist no CRIC lookup happens at all. A blacklist entry that equals
or matches a whitelist glob always expands that glob, even when the excluded site is missing
from the list, so a stale list cannot leave the glob in place.

!!! note "`T2_CH_CERN` does not exclude all of CERN"
    The CERN processing sites include `T2_CH_CERN_HLT`, `T2_CH_CERN_P2`, `T2_CH_CERN_P5` and
    `T3_CH_CERN_DOMA`. A blacklist entry `T2_CH_CERN` matches the site of that exact name only;
    `T2_CH_CERN*` excludes the others as well.

### Automatic site quarantine

One broken worker node fails jobs in seconds, frees its slot and takes the next job, so
a single black hole can eat a large share of a production. FLAF keeps a rolling per-site
record of job outcomes (`<analysis>/data/crab_site_stats.json`, harvested from `crab
status`) and keeps a site out of the *next* CRAB task — retries included — when its
recent jobs mostly fail. The failure rate is measured over jobs *sent* to the site
(ended + still in flight), judged against the other sites' record, so a bug of your own
(which fails everywhere) never quarantines anything. The record is per analysis and advisory —
deleting the JSON file resets it. One record is shared by every CRAB workflow of a law
process; with `--workers` above 1, luigi runs workflows in separate processes, each of which
saves its own view, so the last one to save wins.

- **Rate test:** a site is quarantined when it has at least `min_failures` failures, at least
  `min_failure_rate` of the jobs sent to it failed, that is `relative_factor` times the other
  sites' rate, and the other sites have at least `min_baseline_jobs` jobs.
- **Burst test:** a black hole cycles through slots faster than any healthy site and its earlier
  successes hold the 24 h rate down, so a site that fails `burst_failures` jobs within
  `burst_minutes` is quarantined at once. The same relative guards apply, and the rate is read
  over the jobs that ended inside the window.
- **Escalation:** the first quarantine lasts `quarantine_hours`; each further one doubles the
  last, up to `max_quarantine_hours` (24, 48, 96, 192, 384, then 768 h = 32 days). The number
  of quarantines is kept when one expires, so a site that is still broken is held out for longer
  each time. Outcomes recorded before a quarantine was lifted (`cleared_at`) no longer judge the
  site.
- **What is charged to a site:** only failures that carry a job-level exit code. Kills,
  never-started jobs and tasks refused by the server say nothing about the site they were last
  seen at (a mass kill would otherwise drive every site's baseline
  to 100 % failure and stop the quarantine from ever firing).

Tune or disable it with `crab.auto_blacklist` (`false` or `{enabled: false}` disables it); the
settings and their defaults (`DEFAULTS` in `FLAF/run_tools/crab_sites.py`):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` keeps only the statically configured `crab.blacklist`. |
| `min_failures` | `5` | Failures a site needs before it can be quarantined by the rate test. |
| `min_failure_rate` | `0.5` | Share of the jobs sent to the site that must have failed. |
| `relative_factor` | `2.0` | The site must fail this many times more often than the other sites. |
| `min_baseline_jobs` | `20` | Jobs elsewhere the site is judged against. |
| `quarantine_hours` | `24.0` | Length of a site's first quarantine; doubled for each further one. |
| `max_quarantine_hours` | `768.0` | Ceiling of the doubling (32 days). |
| `burst_failures` | `20` | Failures within `burst_minutes` that quarantine a site at once. A very large value leaves only the rate test. |
| `burst_minutes` | `15.0` | Length of the burst window. |
| `window_hours` | `24.0` | Outcomes older than this stop counting. |
| `max_sites` | `10` | Never quarantine more than this many sites at once (the worst offenders are kept). |

## Submit

```sh
law run FLAF.Analysis.tasks.HistTupleProducerTask \
  --period Run3_2022EE --version my_crab \
  --workflow crab \
  --branches 0 \
  --test 1000 \
  --user-custom /path/to/user_custom_with_crab.yaml
```

| Option | Why |
|---|---|
| `--workflow crab` | Submit via CRAB instead of local/HTCondor. |
| `--parallel-jobs` | Jobs in flight (default **5000** on CRAB; on HTCondor unlimited, except 2000 for `AnaTupleFileTask`). Each refill is one CRAB task. Also `crab.parallel_jobs` in `global.yaml`. The bare form, and the prefixed form of the task that is launched, are copied to every task it requires; `--<Task>-parallel-jobs` for another task applies to that task. |
| `--max-runtime` / `--n-cpus` | Same as HTCondor; mapped to CRAB `maxJobRuntimeMin` (at least `crab.min_runtime_min`) / `numCores` (rounded up to 1, 2, 4 or 8). See [Resources](#resources). |
| `--crab-memory` | CRAB `maxMemoryMB` in MB (CRAB only); default the most CRAB grants for the cores. See [Resources](#resources). |
| `--poll-interval` | Minutes between `crab status` polls (default `5`). Like `--parallel-jobs`: the bare and the launched task's prefixed form reach every required task, `--<Task>-poll-interval` for another task applies to that task. |
| `--transfer-logs` | On by default; each job uploads its log to `<version>/logs/<Task>/<period>/stdall_<first branch>_crab<tag>.<N>.txt` on `fs_default` (with the producer name after `<period>` for per-producer tasks). `<N>` is the CRAB job number and `<tag>` the unique suffix of the CRAB task: CRAB numbers the jobs of every task from 1 and a production is many tasks, so the number alone would let one job's log overwrite another's. CRAB's own log transfer stays off. |

You do **not** need `--bundle` for CRAB — bundles are forced whenever the workflow is `crab`.

## How it fits with HTCondor + bundles

| Mode | Code on worker | Typical use |
|---|---|---|
| `--workflow local` | Submit machine | Development, small tests |
| `--workflow htcondor` | AFS | CERN farm production |
| `--workflow htcondor --bundle` | Tarball from `fs_default` | HTCondor without AFS dependency |
| `--workflow crab` | Tarball from `fs_default` (always) | Full WLCG via CRAB |

## The wave gate

Creating a CRAB task is expensive and a task holds a few thousand jobs, so a production is
submitted in waves of at least `refill_fraction × parallel_jobs` jobs.

- Only the **backlog** — branches never submitted plus retries an earlier poll parked — is
  measured against the wave size. The generation of retries offered by the current poll has
  waited for nothing yet and is parked once, in front of the backlog.
- Jobs are held back only while a full wave is still achievable. Once running plus waiting jobs
  cannot fill one (the tail of a production, or any production smaller than a wave), whatever is
  waiting is submitted at once.
- A retry parked for `crab.retry_release_minutes` (default 45) is released however small the
  wave it makes. The window starts with the first retry parked and is not moved by later ones,
  so the oldest parked retry waits at most one window; retries a release could not take get a
  fresh window. Parked retries are recognised from the attempt counters in the job file, so a
  restarted driver delays a release by at most one window and loses no job.
- `--no-poll` bypasses the gate: a no-poll run resubmits failures exactly once and returns, so a
  parked job would not be offered again.

## Status handling

The driver asks `crab status` for every live CRAB task each poll and reads what comes back as
follows.

- **Refused task.** A task the CRAB server refuses (`SUBMITREFUSED`, e.g. a site name outside
  the PSN list) never runs and cannot be resubmitted. Its jobs are reported failed, and law
  submits them as a new CRAB task. The server's reason is printed once with the project
  directory, together with a hint to check the whitelist; the cached CRIC site list is dropped,
  so the next submission re-reads it. A **second** refused submission made by this run stops
  the run — a refusal is a verdict on what was sent, and retrying would spend every branch's
  attempts on the same verdict. Refusals left by earlier runs are reported but never count.
- **Waiting task.** `WAITING` (accepted by the server, not yet on a scheduler) is read as
  pending, not as an error. It is reported on the first poll and every 12 polls after that, and
  the run stops once one task has been in that state for more than 60 consecutive polls (about
  five hours at the default cadence): the server accepted the task, so the TaskWorker is the place to look.
- **Unreadable response.** `crab status` occasionally returns output that cannot be parsed. The
  query is retried (3 times, 15 s apart), then the task's jobs are reported *pending*, with one
  message per task naming the first lines of what crab returned. After more than 10 consecutive
  unreadable polls of one task the run stops. Any query failure is ridden out this way (an
  expired proxy or a deleted project directory included), so a genuinely dead task surfaces only
  when the tolerance runs out — about an hour at the default cadence. While a task is degraded
  law sees no failures and resubmits nothing for it; other CRAB tasks are unaffected.
- **Finished means on storage.** A CRAB `FINISHED` is believed only when the branch's outputs
  are on storage. CRAB parks a job in `transferring` between the payload exiting and its
  classification — a payload that exited non-zero as well — and with transfers skipped law maps
  that to finished; `transferring` and `transferred` count as finished here, so without the
  output check a failed job would be written off as done. Every "absent" answer of a poll rests
  on a listing taken after the status: one listing per output directory per poll, which also
  republishes the directory to the path-cache server. CRAB workers cannot reach that server, so
  without this the files they write would read as absent for up to 24 h. For the same reason a
  driver that is (re)started takes fresh listings before it judges which outputs exist: jobs
  that finished while no driver was polling are accepted, not sent back to the grid.
- **Why a job failed.** Each newly failed job that has a job-level exit code gets one line with
  the exit code, the site and the last exception line of its stdout, fetched from the schedd
  with your proxy: at most 5 jobs per CRAB task and poll, the last 4 MB of the stdout, 30 s per socket
  operation and 60 s per transfer; the rest are counted. A stdout that cannot be read, or
  carries no exception, is said to be so, once per attempt. Jobs failed without an exit code
  (kills, refused tasks, watchdog verdicts) get no such line.

All conditions that end the run (the second refusal, the waiting limit, the unreadable limit) are
raised from the poll callback, not from the query: law runs queries in a thread pool and turns
an exception raised there into one more failed query, so the run would only end several polls
later.

## Stall watchdog

CRAB reports a job as `running` for as long as the batch system holds the slot, which is not the
same as the payload doing anything: a production of 600 branches once sat at 598 done for hours
behind two such jobs. The watchdog finds them. It is on by default for CRAB.

- Each CRAB job's branch refreshes a **flag file** every `interval_minutes`, at
  `<fs_default>/<version>/heartbeat/<Task>/<period>[/<producer>]/<branch>`. The flag is removed
  when the payload ends, however it ends.
- The driver lists that one directory once per interval (however many jobs are running) and
  reads only the modification times. A job that CRAB still reports as running, and whose flag
  has not moved for `missed_checks` intervals, is failed, so law's normal retry resubmits it. A
  running job that never wrote a flag is failed after `missed_checks + 1` intervals.
- CRAB has no per-job kill, so the slot is **abandoned**, not freed: it is reclaimed when
  `maxJobRuntimeMin` expires.
- A flag that disappears is read as the job exiting, and is not a verdict. A flag older than
  the current attempt is ignored.
- Over `davs://` the flag is rewritten as a delete followed by a fresh upload. On storage that
  keeps deleted files (CERNBox, the usual `fs_default`), every beat therefore leaves an entry
  in the recycle bin, about `2 × running jobs` per hour at the default interval; and a beat that
  fails between the two leaves no flag until the next one, which reads as a job on its way out.
- Brakes: no more than `max_per_interval` verdicts in one interval, `max_per_branch` rescues
  per branch (a branch that stalls wherever it runs is the branch's problem), and no verdicts at
  all when more than `max_stale_fraction` of the running jobs look stale while at least 10 are
  running: that is read as a storage fault. With fewer running jobs — the tail of a production,
  where the last stalled jobs hold its completion — the proportion says nothing, and they get
  their verdicts. A job failed by the watchdog is charged to the site it was last seen at,
  once.

`crab.watchdog: false` disables it. Settings (`DEFAULTS` in `FLAF/run_tools/crab_watchdog.py`);
unknown keys are refused:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Switch. |
| `interval_minutes` | `30` | How often a job refreshes its flag and how often the driver lists the directory. Must be at least 1. |
| `missed_checks` | `2` | Intervals a flag may stay unchanged before the job is failed. Must be at least 1. |
| `max_per_interval` | `5` | Most verdicts in one interval. |
| `max_per_branch` | `1` | Most rescues of one branch. |
| `max_stale_fraction` | `0.5` | A listing in which more than this fraction of the running jobs look stale issues no verdicts. |
| `dry_run` | `false` | Log the verdicts that would have been issued and issue none. |

## Submission safeguards

The safeguards that guard the submission itself apply to both batch backends and are described
under [HTCondor](htcondor.md#submission-safeguards): an unreadable software tree skips a
submission round instead of ending the run, a resumed workflow that lost most of its outputs
stops instead of resubmitting them, and the expensive producers refuse to run inline inside a job
submitted for another task.

## Monitor

```sh
law run FLAF.Analysis.tasks.HistTupleProducerTask \
  --period Run3_2022EE --version my_crab \
  --workflow crab --branches 0 --test 1000 \
  --user-custom /path/to/user_custom_with_crab.yaml \
  --print-status 1,1
```

Repeat the parameters of the submission (`--workflow`, `--branches`, `--test`, …): with different
ones LAW looks at a different set of tasks.

CRAB project directories live under `data/jobs/` (see `job.job_file_dir` in `law.cfg`). You can
also use `crab status -d <project_dir>` from a CMSSW environment.

## Caveats

!!! warning "MyProxy must stay valid"
    The TaskWorker retrieves the MyProxy credential for the whole life of a task, and nothing in
    a run renews it (status polls use `--proxy`, not MyProxy). Create it with enough days
    before large campaigns (`cmsEnv crab createmyproxy`, see Prerequisites).

!!! note "Path-existence cache is shipped with the job"
    `WLCGFileSystem.remotePathCacheHost` (`cms-flaf.cern.ch`) is behind the CERN
    firewall, so CRAB workers do not use it. At submit time FLAF dumps the
    in-process path cache and ships it with the job; the worker loads that
    snapshot and uses a longer local TTL (`24 × localPathCacheValidity`, at
    least 24 h, or `WLCGFileSystem.crabLocalPathCacheValidity` when set) so concurrent
    jobs do not re-stat the same remote paths. The driver compensates for the missing
    server link on its side: see [Finished means on storage](#status-handling).

!!! note "Workers read their proxy with `voms-proxy-info -dont-verify-ac`"
    A stale CRL for the VOMS server makes plain `voms-proxy-info` exit non-zero while the proxy
    is usable, which used to kill a job before it did any work. The attribute-certificate check
    is skipped; a missing or unreadable proxy still fails.

!!! warning "First-time CRAB / grid mapfile"
    New users may need a CRAB username mapping and write access to the chosen storage site
    LFN. At CERN, prefer `T3_CH_CERNBOX` for `/store/user/...` (maps to personal EOS and
    usually passes `crab checkwrite`); `T2_CH_CERN /store/user` often does not exist.

!!! note "Distant sites still read `fs_default`"
    The default whitelist lets jobs run anywhere, but the bundle and outputs stay
    on `fs_default`. Personal EOS (`davs://eoshome-*.cern.ch`) can fail or stall
    from far-away sites (gfal 112, HTTP 404, hung DNN). Law retries usually
    recover; set `crab.whitelist` closer to CERN if that I/O is a problem.

!!! warning "Do not replace a live bundle mid-campaign"
    The analyses mark `core` as `hashed: true`, so it is published as
    `core_<hash>.tar.bz2` and a code change produces a new file instead of replacing
    the live one (see [Bundles are named after what they contain](htcondor.md#bundles-are-named-after-what-they-contain)).
    Unhashed flavours (`soft.tar.bz2`, `cmssw.tar.bz2`, …) keep their name. `BundleTask`
    can stay DONE after such a bundle is deleted because of the path-existence cache,
    and workers then get HTTP 404. Rebuild into a sibling file and `mv` it over the
    live path; do not `cp` onto a file jobs may be downloading (a mid-copy can stage
    out 0 bytes).

!!! note "Test small first"
    Validate with `--workflow local --branches 0 --test 1000`, then a single CRAB branch,
    before large submissions.

!!! note "The CRAB client runs with its own HOME"
    CRAB rewrites its task cache `~/.crab3` on **every** command, status polls included —
    with `$HOME` on AFS a multi-day production dies with `PermissionError` the moment the
    AFS token lapses. FLAF therefore runs every `crab` invocation with
    `HOME=$TMPDIR/flaf_crab_home_<uid>` and, except for `submit`, from that directory
    (so `crab.log` does not land in the working area). `--proxy` is always passed
    explicitly, so nothing from the real home is needed.

!!! warning "Every job reports `unknown job id`"
    This usually means the submission itself failed and law swallowed the cause — most
    often the CMSSW sandbox it runs `crab` in could not be built. FLAF builds that
    sandbox eagerly before the first submission and raises an actionable error; if you
    still see it, check that `python` on PATH resolves to a python3 (the sandbox dumps
    its environment with bare `python`, which modern CMSSW does not ship — the flaf_env
    provides one) and inspect `$LAW_HOME/cms/cmssw_cache`.
