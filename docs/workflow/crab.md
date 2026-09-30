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
   workers can retrieve it**. CRAB looks the credential up under the SHA1 of your DN and needs
   the CRAB retriever DNs in its retrieval policy; a plain `myproxy-init -d -n` stores it under
   the plain DN with no such policy. From an existing VOMS proxy (no grid-cert passphrase):

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

    FLAF fails early if the VOMS proxy is missing or expired, or if no MyProxy credential
    with at least 5 days left is found (instead of waiting for server-side `SUBMITFAILED`).
    There is no password-file fallback.

3. CRAB client available (via CMSSW / the law CMSSW sandbox; the sandbox is set by
   `crab_sandbox_name` in the `[job]` section of the analysis `config/law.cfg`, currently
   `CMSSW_14_0_0::arch=el9_amd64_gcc12`).
4. **Bundles**: CRAB workers do not mount AFS. FLAF always ships code via `BundleTask`
   when `--workflow crab` is used (same tarballs as HTCondor `--bundle`). Tasks already
   declare `bundle_flavours`.
5. Remote `fs_default` (e.g. `davs://eoshome-...`) so bundles and analysis outputs are on a
   grid-accessible filesystem.

!!! note "FLAF's MyProxy check is looser than what CRAB needs"
    The pre-submission check accepts a credential under either username — the plain DN or
    its SHA1 — and its error message suggests the plain form
    (`myproxy-init -d -n -s myproxy.cern.ch`, verified with
    `myproxy-info -d -s myproxy.cern.ch`). A credential created that way passes the check
    but lacks what CRAB's task workers look for, so create it as in step 2 and verify the
    SHA1 username with the `myproxy-info -l "$SHA1"` line.

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
  # refill_fraction: 0.2      # new CRAB task only when this fraction of slots is free
  # memory_mb_per_cpu: 2000   # CRAB maxMemoryMB / n_cpus
  # min_runtime_min: 60       # floor for CRAB maxJobRuntimeMin
```

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
| `blacklist` | Optional. CRAB `Site.blacklist` (applied on top of the whitelist). |
| `parallel_jobs` | Optional. Default for `--parallel-jobs` on CRAB (CLI wins). Default: `5000`. Caps how many CRAB jobs are in flight and thus the size of each CRAB task. CRAB itself refuses more than 10 000 jobs in one task. |
| `refill_fraction` | Optional. Submit a new CRAB task only when `parallel_jobs - n_active >= refill_fraction * parallel_jobs`. Default: `0.2`. Prevents a 1-job task every time a single job finishes. |
| `memory_mb_per_cpu` | Optional. CRAB `JobType.maxMemoryMB` is this times `--n-cpus`, capped at 5000 MB (1 core) or `2500 MB * n_cpus`. Default: `2000` (CRAB / site-guaranteed per-core default). |
| `min_runtime_min` | Optional. Lower bound, in minutes, for CRAB `JobType.maxJobRuntimeMin` (`--max-runtime` converted to minutes), since every job first downloads and unpacks its bundles. Default: `60`. |

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
| `--parallel-jobs` | Jobs in flight (default **5000** on CRAB; on HTCondor unlimited, except 2000 for `AnaTupleFileTask`). Each refill is one CRAB task. Also `crab.parallel_jobs` in `global.yaml`. |
| `--max-runtime` / `--n-cpus` | Same as HTCondor; mapped to CRAB `maxJobRuntimeMin` (at least `crab.min_runtime_min`) / `numCores` / memory (`2000 MB * n_cpus`, CRAB-capped). |
| `--transfer-logs` | On by default; each job uploads its log to `<version>/logs/<Task>/<period>/stdall_crab<N>.txt` on `fs_default` (with the producer name after `<period>` for per-producer tasks). CRAB's own log transfer stays off. |

You do **not** need `--bundle` for CRAB — bundles are forced whenever the workflow is `crab`.

## How it fits with HTCondor + bundles

| Mode | Code on worker | Typical use |
|---|---|---|
| `--workflow local` | Submit machine | Development, small tests |
| `--workflow htcondor` | AFS | CERN farm production |
| `--workflow htcondor --bundle` | Tarball from `fs_default` | HTCondor without AFS dependency |
| `--workflow crab` | Tarball from `fs_default` (always) | Full WLCG via CRAB |

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
    CRAB polls through MyProxy. Delegate a long-lived proxy before large campaigns
    (`myproxy-init` as in Prerequisites).

!!! note "Path-existence cache is shipped with the job"
    `WLCGFileSystem.remotePathCacheHost` (`cms-flaf.cern.ch`) is behind the CERN
    firewall, so CRAB workers do not use it. At submit time FLAF dumps the
    in-process path cache and ships it with the job; the worker loads that
    snapshot and uses a longer local TTL (`24 × localPathCacheValidity`, at
    least 24 h, or `WLCGFileSystem.crabLocalPathCacheValidity` when set) so concurrent
    jobs do not re-stat the same remote paths.

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
