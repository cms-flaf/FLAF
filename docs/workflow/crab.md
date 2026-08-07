# Running on CRAB (WLCG)

HTCondor covers the CERN local batch farm. For jobs that should run anywhere on the
**WLCG** (CMS CRAB), FLAF tasks can be submitted with `--workflow crab`. The implementation
uses [law's CMS CRAB workflow](https://github.com/riga/law) (`law.contrib.cms.CrabWorkflow`).

Analysis **outputs** still go through the normal FLAF remote targets (`fs_default` via
gfal/`davs://` etc.). CRAB's own stageout path is only used for CRAB bookkeeping (automatic
output collection is disabled).

## Prerequisites

1. A valid **VOMS proxy** for the CMS VO (`voms-proxy-init --voms cms -valid 192:00`).
2. A **MyProxy** credential valid for **at least 5 days**. The CRAB *server* pulls the proxy
   from `myproxy.cern.ch`; without it the client can accept the task and the server then
   returns `SUBMITFAILED`. Set up once:

   ```sh
   myproxy-init -d -n -s myproxy.cern.ch
   # or, for non-interactive delegation, put the grid cert passphrase in a file and set
   # job.crab_password_file in law.cfg
   ```

   FLAF fails early with a clear error if MyProxy is missing (instead of waiting for
   `SUBMITFAILED`).
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
  storage_site: T2_CH_CERN
  out_lfn_base: /store/user/<your_username>/FLAF
  # optional:
  # whitelist: [T2_CH_CERN]
  # blacklist: [T2_US_MIT]
  # max_memory_mb: 4000
```

Alternatively set environment variables:

```sh
export FLAF_CRAB_STORAGE_SITE=T2_CH_CERN
export FLAF_CRAB_OUT_LFN_BASE=/store/user/$USER/FLAF
```

| Key | Meaning |
|---|---|
| `storage_site` | CRAB `Site.storageSite` (required for submission). |
| `out_lfn_base` | CRAB `Data.outLFNDirBase` (required; not where analysis outputs go). |
| `whitelist` / `blacklist` | Optional site lists. Whitelist implies `ignoreLocality`. |
| `max_memory_mb` | Default memory when `--crab-memory` is not set (`n_cpus * 2000` otherwise). |

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
    LFN. Prefer a site you already use for CMS jobs (`T2_CH_CERN` is the usual CERN EOS
    choice for `/store/user/...`).

!!! note "Test small first"
    Validate with `--workflow local --branches 0 --test 1000`, then a single CRAB branch,
    before large submissions.
