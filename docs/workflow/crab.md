# Running on CRAB (WLCG)

HTCondor covers the CERN local batch farm. For jobs that should run anywhere on the
**WLCG** (CMS CRAB), FLAF tasks can be submitted with `--workflow crab`. The implementation
uses [law's CMS CRAB workflow](https://github.com/riga/law) (`law.contrib.cms.CrabWorkflow`).

Analysis **outputs and job logs** use FLAF remote I/O only (`fs_default` via gfal/`davs://`,
plus `stageout_logs.sh`). CRAB `transferOutputs` / `transferLogs` are forced **off** so
nothing is duplicated onto CRAB's stageout area. `crab.storage_site` /
`crab.out_lfn_base` are still required by the CRAB client for a valid config and a
submit-time write check — they are not where FLAF stores analysis products.

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

   Alternatively set `job.crab_password_file` in `law.cfg` to a file with the grid
   certificate passphrase; law will call `delegate_myproxy` with the same CRAB retrievers.

   FLAF fails early if no suitable MyProxy credential is found (instead of waiting for
   server-side `SUBMITFAILED`).
3. CRAB client available (via CMSSW / the law CMSSW sandbox; default sandbox name is set in
   law's config as `job.crab_sandbox_name`).
4. **Bundles**: CRAB workers do not mount AFS. FLAF always ships code via `BundleTask`
   when `--workflow crab` is used (same tarballs as HTCondor `--bundle`). Tasks already
   declare `bundle_flavours`.
5. Remote `fs_default` (e.g. `davs://eoshome-...`) so bundles and analysis outputs are on a
   grid-accessible filesystem.

## Config

Add a `crab:` block to `user_custom.yaml` (or pass via `--user-custom`):

```yaml
crab:
  # Stageout site for CRAB bookkeeping (analysis outputs still use fs_default).
  # At CERN, T3_CH_CERNBOX maps /store/user/<you> to personal EOS and usually
  # passes `crab checkwrite`; T2_CH_CERN /store/user often does not exist.
  storage_site: T3_CH_CERNBOX
  out_lfn_base: /store/user/<your_username>/FLAF
  # optional:
  # whitelist: [T2_CH_CERN]   # where jobs run (can differ from storage_site)
  # blacklist: [T2_US_MIT]
  # max_memory_mb: 4000
```

Verify write access before the first campaign:

```sh
crab checkwrite --site=T3_CH_CERNBOX --lfn=/store/user/$USER
```

Alternatively set environment variables:

```sh
export FLAF_CRAB_STORAGE_SITE=T3_CH_CERNBOX
export FLAF_CRAB_OUT_LFN_BASE=/store/user/$USER/FLAF
```

| Key | Meaning |
|---|---|
| `storage_site` | CRAB `Site.storageSite` (required for submission). |
| `out_lfn_base` | CRAB `Data.outLFNDirBase` (required; not where analysis outputs go). |
| `whitelist` / `blacklist` | Optional site lists. Whitelist implies `ignoreLocality`. |
| `max_memory_mb` | Default memory when `--crab-memory` is not set (`n_cpus * 2500` otherwise). |

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
| `--crab-memory 4000` | Override max memory (MB) per job. |
| `--crab-whitelist T2_CH_CERN` | Restrict to listed sites. |
| `--max-runtime` / `--n-cpus` | Same as HTCondor; mapped to CRAB `maxJobRuntimeMin` / `numCores` / memory. |
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
    (`myproxy-init -d -n` or law's password-file path).

!!! warning "First-time CRAB / grid mapfile"
    New users may need a CRAB username mapping and write access to the chosen storage site
    LFN. At CERN, prefer `T3_CH_CERNBOX` for `/store/user/...` (maps to personal EOS and
    usually passes `crab checkwrite`); `T2_CH_CERN /store/user` often does not exist.

!!! note "Test small first"
    Validate with `--workflow local --branches 0 --test 1000`, then a single CRAB branch,
    before large submissions.
