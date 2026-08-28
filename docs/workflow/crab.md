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
2. A **MyProxy** credential valid for **at least 5 days**, registered so **CRAB task
   workers can retrieve it**. A plain `myproxy-init -d -n` is not enough: the credential
   must use the SHA1 username law/CRAB expect and include CRAB retriever DNs. From an
   existing VOMS proxy (no grid-cert passphrase):

   ```sh
   export X509_USER_PROXY=/tmp/x509up_u$(id -u)   # or your proxy path
   # identity DN → SHA1 username (same as law)
   SHA1=$(python3 - <<'PY'
   import hashlib, subprocess
   out = subprocess.check_output(["voms-proxy-info", "-identity"], text=True).strip()
   print(hashlib.sha1(out.encode()).hexdigest())
   PY
   )
   RETR='/DC=ch/DC=cern/OU=computers/CN=crab-(preprod|prod|dev)-tw(01|02|03).cern.ch|/DC=ch/DC=cern/OU=computers/CN=stefanov(m|m2).cern.ch|/DC=ch/DC=cern/OU=computers/CN=dciangot-tw.cern.ch|/DC=ch/DC=cern/OU=computers/CN=crab-(preprod|prod)-tw(01|02).cern.ch|/DC=ch/DC=cern/OU=computers/CN=crab-dev-tw(01|02|03|04).cern.ch|/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=cmscrab/CN=(817881|373708)/CN=Robot: cms crab|/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=crabint1/CN=373708/CN=Robot: CMS CRAB Integration 1'
   GT_PROXY_MODE=rfc myproxy-init -n -s myproxy.cern.ch \
     -C "$X509_USER_PROXY" -y "$X509_USER_PROXY" \
     -l "$SHA1" -t 168 -c 168 \
     -x -R "$RETR" -x -Z "$RETR" -m cms
   myproxy-info -s myproxy.cern.ch -l "$SHA1"   # expect timeleft >= 5 days + retrieval policy
   ```

   FLAF fails early if the VOMS proxy or a suitable MyProxy credential is missing
   (instead of waiting for server-side `SUBMITFAILED`). There is no password-file
   fallback.
3. CRAB client available (via CMSSW / the law CMSSW sandbox; default sandbox name is set in
   law's config as `job.crab_sandbox_name`).
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

The CRAB client requires `Site.whitelist` because law uses dummy `userInputFiles`
(no input dataset). FLAF defaults that list to `T1_*`, `T2_*`, `T3_*` so jobs
can run at every CMS processing site. Restrict or exclude sites only if you need
to, in `global.yaml` / `user_custom.yaml`:

```yaml
crab:
  # whitelist: [T2_CH_CERN]   # omit to use all T1/T2/T3 sites
  # blacklist: [T2_US_MIT]
  # parallel_jobs: 5000       # default --parallel-jobs; CLI wins if set
  # refill_fraction: 0.2      # minimum wave size as a fraction of parallel_jobs
  # poll_interval: 5          # minutes between crab status polls; CLI wins if set
  # memory_mb_per_cpu: 2000   # CRAB maxMemoryMB / n_cpus
  # auto_blacklist:           # automatic site quarantine (on by default)
  #   enabled: true
  # ignore_global_blacklist: false   # waive CMS's own site blacklist (not recommended)
```

!!! note "A `crab:` block in `user_custom.yaml` replaces the `global.yaml` one wholesale"
    The config layers are concatenated and parsed as one YAML document, so a later
    `crab:` mapping wins as a whole — repeat the keys you want to keep.

CRAB gives the **whitelist precedence** over the blacklist: a site matched by both
lists is *kept* (the client only prints a warning). FLAF therefore removes excluded
sites — the configured `blacklist` and the automatic quarantine alike — from the
whitelist itself, expanding a tier glob that covers an excluded site into the concrete
sites it matches. The expansion uses the CRIC processing-site list (cached 24 h in
`<analysis>/data/cms_sites.json`; a stale cache is reused if CRIC is unreachable).
Globs covering nothing excluded are passed through unchanged, so without a blacklist
no CRIC lookup happens at all.

Memory is `2000 MB * n_cpus` (override with `crab.memory_mb_per_cpu`), matching
the CRAB default that all sites guarantee per core. Then capped at the CRAB
client limit: 5000 MB for 1 core, `2500 MB * n_cpus` otherwise. There is no
separate CRAB memory CLI flag. AnaTuple production defaults to 4 cores (8 GB)
so tautau CMSSW jobs fit; 2 cores only allow 5 GB.

Verify write access before the first campaign:

```sh
crab checkwrite --site=T3_CH_CERNBOX --lfn=/store/user/$USER
```

| Key | Meaning |
|---|---|
| `whitelist` | Optional. Restricts `Site.whitelist`. Default: `T1_*`, `T2_*`, `T3_*`. |
| `blacklist` | Optional. Sites to exclude. Removed from the whitelist itself (CRAB gives the whitelist precedence, so passing them only as `Site.blacklist` would do nothing) — tier globs covering an excluded site are expanded from the CRIC processing-site list. |
| `parallel_jobs` | Optional. Default for `--parallel-jobs` on CRAB (CLI wins). Default: `5000`. Caps how many CRAB jobs are in flight and thus the size of each CRAB task. CRAB itself refuses more than 10 000 jobs in one task. |
| `refill_fraction` | Optional. Minimum wave size, as a fraction of `parallel_jobs`. Default: `0.2`. Jobs — unsubmitted and retries alike — are held back and aggregated into one CRAB task while a full wave is still achievable, and released immediately once running + waiting can no longer fill one (the tail of a production, and any production smaller than a wave). |
| `memory_mb_per_cpu` | Optional. CRAB `JobType.maxMemoryMB` is this times `--n-cpus`, capped at 5000 MB (1 core) or `2500 MB * n_cpus`. Default: `2000` (CRAB / site-guaranteed per-core default). |
| `poll_interval` | Optional. Minutes between `crab status` polls (CLI `--poll-interval` wins). Default: `5`. Each poll is one multi-MB `crab status --json` per live CRAB task. |
| `min_runtime_min` | Optional. Floor for CRAB `maxJobRuntimeMin` (bundles must be downloaded and unpacked before the payload starts). Default: `60`. |
| `auto_blacklist` | Optional mapping (or `false`). Automatic site quarantine, on by default — see below. |
| `ignore_global_blacklist` | Optional. Set `true` to waive CMS's own blacklist of known-broken sites (`Site.ignoreGlobalBlacklist`). Not recommended: with an open site pool it is the main protection against burning jobs at bad sites. |

### Automatic site quarantine

One broken worker node fails jobs in seconds, frees its slot and takes the next job, so
a single black hole can eat a large share of a production. FLAF keeps a rolling per-site
record of job outcomes (`<analysis>/data/crab_site_stats.json`, harvested from `crab
status`) and keeps a site out of the *next* CRAB task — retries included — when its
recent jobs mostly fail. The failure rate is measured over jobs *sent* to the site
(ended + still in flight), judged against the other sites' record, so a bug of your own
(which fails everywhere) never quarantines anything. Tune or disable it with
`crab.auto_blacklist`; the knobs and their defaults are documented in
`FLAF/run_tools/crab_sites.py` (`DEFAULTS`): `min_failures: 5`, `min_failure_rate: 0.5`,
`relative_factor: 2.0`, `min_baseline_jobs: 20`, `quarantine_hours: 6`,
`window_hours: 24`, `max_sites: 10`. The record is per analysis and advisory — deleting
the JSON file resets it.

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
| `--parallel-jobs` | Jobs in flight (default **5000** on CRAB, unlimited on HTCondor). Each refill is one CRAB task. Also `crab.parallel_jobs` in `global.yaml`. |
| `--max-runtime` / `--n-cpus` | Same as HTCondor; mapped to CRAB `maxJobRuntimeMin` / `numCores` / memory (`2000 MB * n_cpus`, CRAB-capped). |
| `--transfer-logs` | On by default; enables remote log stageout when `fs_default` is WLCG. |

You do **not** need `--bundle` for CRAB — bundles are forced whenever the workflow is `crab`.

## How it fits with HTCondor + bundles

| Mode | Code on worker | Typical use |
|---|---|---|
| `--workflow local` | Submit machine | Development, small tests |
| `--workflow htcondor` | AFS (or `--bundle` tarball) | CERN farm production |
| `--workflow htcondor --bundle` | Tarball from `fs_default` | HTCondor without AFS dependency |
| `--workflow crab` | Tarball from `fs_default` (always) | Full WLCG via CRAB |

## Monitor

```sh
law run FLAF.Analysis.tasks.HistTupleProducerTask \
  --period Run3_2022EE --version my_crab --print-status 1,1
```

CRAB project directories live under `data/jobs/` (see `job.job_file_dir` in `law.cfg`). You can
also use `crab status -d <project_dir>` from a CMSSW environment.

## Caveats

!!! warning "MyProxy must stay valid"
    CRAB polls through MyProxy. Delegate a long-lived proxy before large campaigns
    (`myproxy-init` as in Prerequisites).

!!! note "Path-existence cache is shipped with the job"
    `WLCGFileSystem.remotePathCacheHost` (`cms-flaf.cern.ch`) is behind the CERN
    firewall, so CRAB workers do not use it. At submit time FLAF dumps the
    in-process path cache and ships it with the job; the worker loads that
    snapshot and uses a longer local TTL (`24 × localPathCacheValidity`, at
    least 24 h) so concurrent jobs do not re-stat the same remote paths.

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
    `BundleTask` can stay DONE after `core.tar.bz2` is deleted because of the
    path-existence cache. Workers then get HTTP 404. Rebuild into a sibling file
    and `mv` it over the live path; do not `cp` onto a file jobs may be
    downloading (a mid-copy can stage out 0 bytes).

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

!!! note "An unreadable `crab status` response is ridden out"
    `crab status` occasionally returns output law cannot parse. FLAF retries the query
    (3x, 15 s apart), then reports that task's jobs as *pending* — with one message per
    task naming the first lines of what crab returned — and only raises after 10
    consecutive unreadable polls. While a task is degraded this way law sees no failures
    and resubmits nothing for it; jobs at other sites and other CRAB tasks are unaffected.

!!! warning "Every job reports `unknown job id`"
    This usually means the submission itself failed and law swallowed the cause — most
    often the CMSSW sandbox it runs `crab` in could not be built. FLAF builds that
    sandbox eagerly before the first submission and raises an actionable error; if you
    still see it, check that `python` on PATH resolves to a python3 (the sandbox dumps
    its environment with bare `python`, which modern CMSSW does not ship — the flaf_env
    provides one) and inspect `$LAW_HOME/cms/cmssw_cache`.
