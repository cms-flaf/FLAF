# Running on HTCondor

Producing ntuples and histograms for a full era means processing thousands of files — far too much
for one machine. FLAF tasks are **workflows** ([Tasks & LAW](../concepts/tasks-and-law.md)), so
their branches can be submitted to CERN's **HTCondor** batch system. The recommended pattern is to
**develop and test with `--workflow local`, then switch to `--workflow htcondor` for production** —
the command is otherwise the same.

## Submit a task to the batch system

```sh
law run FLAF.AnaProd.tasks.AnaTupleFileTask \
  --period Run3_2022 --version prod \
  --workflow htcondor \
  --transfer-logs \
  --parallel-jobs 100
```

| Option | Why you want it |
|---|---|
| `--workflow htcondor` | Submit branches as batch jobs instead of running locally. |
| `--transfer-logs` | Bring each job's stdout/stderr back to your `data/` area. **Highly recommended** — without it, debugging a failed job is painful. |
| `--parallel-jobs 100` | Cap how many jobs are in flight at once. Be a good citizen on the shared pool; very large uncapped submissions are discouraged. |
| `--branches 0-99` | Submit only a subset (e.g. to retry a range). |

Other HTCondor parameters available on every workflow task: `--max-runtime`, `--n-cpus`,
`--priority`, `--htcondor-spool`. See [Command arguments](arguments.md).

## Monitor and resume

LAW tracks which branches have finished (by checking their outputs), so a re-run only resubmits the
missing ones — batch jobs fail and time out, and resuming is normal. Check progress with:

```sh
law run FLAF.AnaProd.tasks.AnaTupleFileTask \
  --period Run3_2022 --version prod --print-status 1,1
```

Standard `condor_q` / `condor_status` work for the underlying jobs.

## Bundles: shipping the code to workers

A batch worker needs your code and environment. FLAF supports two modes:

- **Non-bundle jobs** rely on the shared AFS area being mounted on the worker: the job receives
  `FLAF_PATH`/`CORRECTIONS_PATH` and runs the code straight from AFS (including any edits you made
  via the [dev overlay](../concepts/environment.md#developing-shared-submodules)).
- **Bundle jobs** ship a tarball of the code/environment to the worker (the `--bundle` flag and the
  `BundleTask` machinery). The worker runs from the tarball and never reaches back to AFS, so it is
  deliberately *not* given `FLAF_PATH`/`CORRECTIONS_PATH`. Bundles also set `FLAF_NO_INSTALL=1` so
  the worker never tries to build the environment.

For most work the defaults are correct; you only think about bundles when a stage explicitly needs
one (e.g. it declares a CMSSW bundle flavour) or when AFS is not available on the target pool.

### A bundle waits for everything it packs

A flavour that packs the output of a task declares it in `task_requires`, and a flavour that
packs the output of *several* tasks lists all of them:

```yaml
  AnaTupleFileList:
    patterns:
      - data/{version}/AnaTupleFileListBuilderTask/{period}
      - data/{version}/AnaTupleFileListTask/{period}
    task_requires:
      - module: FLAF.AnaProd.tasks
        class: AnaTupleFileListBuilderTask
      - module: FLAF.AnaProd.tasks
        class: AnaTupleFileListTask
```

With only the first one listed the bundle is packed as soon as the builder is done, while the
per-dataset lists of the second are still being written, and the jobs receive a tarball that
is missing them. FLAF warns when a packed `data/<version>/<Task>/<period>` directory has no
matching entry. The producers are requested with the workflow the submission was started
with, so they are the same task instances the rest of the graph waits on.

### Bundles are named after what they contain

A bundle's output is a path, and law treats an existing one as complete forever — so an edit
made after it was first built would never reach the workers, which rebuild their branch map
from the config *inside the tarball*. Adding one dataset shifts every branch index after it,
and jobs then work on a different file than the one they were submitted for.

Flavours therefore opt into a content hash in `global.yaml`:

```yaml
bundles:
  core:            # code and configuration: small, edited often
    hashed: true
    patterns: [ FLAF, AnaProd, Analysis, config, env.sh, Corrections, include ]
  soft:            # the installed environment: large, changes only on a reinstall
    patterns: [ soft/flaf_env ]
```

A hashed flavour is published as `core_<hash>.tar.bz2`, so changing any packed file yields a
new name, `BundleTask` sees a missing output and rebuilds, and the jobs are handed the URL of
the bundle matching the code they were submitted with. Files up to 1 MB are hashed by their
content and larger ones by size and modification time, which keeps the cost at a fraction of
a second per submission even for an analysis shipping ~150 MB of models.

Splitting a big immutable payload into its own unhashed flavour is what keeps it that cheap.
The trade-off is that such a flavour is **not** rebuilt when its content changes: after
reinstalling the environment, or changing anything else packed without a hash, delete the
bundle so that the next submission recreates it.

!!! warning "A symlink can send a bundle job back to AFS anyway"
    Symlinks *inside* a packed directory are kept as symlinks — deliberately, so that the CVMFS
    links in `soft/flaf_env` are not dereferenced into the tarball. An absolute symlink pointing
    into the analysis area therefore still resolves to the submit host on the worker, and every
    job reads that payload over AFS. A few thousand jobs pulling a model or a correction file this
    way is enough for CERN to answer with *"Batch submission limited due to high AFS load"*, while
    the jobs themselves fail on timeouts (`DEADLINE_EXCEEDED … Connection timed out`). Reference
    the real content the bundle packs — for HH_bbtautau the HHbtag models are taken from
    `$ANALYSIS_PATH/HHbtag/models`, not through `$CMSSW_BASE/src/HHTools/HHbtag`, which is such a
    symlink.

For jobs that should run on the full CMS WLCG (not only CERN HTCondor), use
[`--workflow crab`](crab.md) — that path always uses bundles.


!!! tip "Your edits to FLAF *do* reach the workers"
    Thanks to the dev overlay, non-bundle jobs run your edited `FLAF`/`Corrections`, and bundle
    jobs include them in the tarball — so testing framework changes on HTCondor works without
    committing first. See [Contributing](../contributing.md).

## Caveats

!!! warning "Keep your proxy valid for the whole run"
    Jobs that outlive your VOMS proxy lose grid access mid-flight. Create a long-lived proxy
    (`-valid 192:00`) before a big submission, and refresh it for long campaigns.

!!! warning "Killing a background `law` leaves its jobs/children"
    Pressing `Ctrl-C` or `kill`-ing a backgrounded `law` process does not necessarily stop the
    branches it spawned. To stop everything for a run, match the processes by pattern, e.g.
    `pkill -f "version=prod"`, and `condor_rm` the submitted jobs if needed.

!!! note "Test small, then scale"
    Validate a task with `--workflow local --branches 0 --test 1000` before submitting the full
    workflow to HTCondor. A bug found on one local branch is far cheaper than one found across a
    thousand batch jobs.
