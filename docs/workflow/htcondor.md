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

## How branches become jobs

By default LAW puts a fixed number of consecutive branches into each job (`--tasks-per-job`). That
works when branches cost about the same. AnaTuple production is not like that: a dilepton-skim file
where half the events are selected costs twenty times what a hadronic one does, and because
branches are ordered by dataset the expensive ones are neighbours — so fixed-size chunks collect
them into the same job, which then runs into `--max-runtime`, is removed by HTCondor, and is
retried with exactly the same grouping.

`AnaTupleFileTask` therefore composes jobs by **estimated cost**:

```
seconds(file) = overhead + sec_per_event(dataset) x n_events(file)
```

- `sec_per_event` is measured, not configured — first by [`AnaTupleCostProbeTask`](../reference/tasks.md#anatuplecostprobetask),
  then refined from the durations of the production jobs themselves, which are recorded as they
  finish and used by the next run.
- `n_events` comes from `InputFileTask`'s catalogue, or is inferred from the file size when the
  storage does not report event counts.

Branches are then packed into jobs up to `target_job_hours`, largest first. A file that costs more
than that on its own gets a job to itself; cheap ones are combined until the target is reached.
Where the estimate is a guess rather than a measurement (a dataset the probe could not reach, a new
sample), the packing is deliberately more conservative, so a wrong guess cannot rebuild an
over-long job.

Three consequences worth knowing:

- **Each resubmission of a failed job gets more runtime and, if configured, more memory**, up to
  `retry_max_factor`.
- **Jobs are grouped once per run, from everything measured so far**, and never regrouped while
  that run is polling: LAW fixes the total job count when polling starts, so changing it mid-run
  would corrupt its accounting. Durations observed during a run are recorded and applied by the
  *next* one — including a plain resume of the same version, which regroups whatever is still
  unsubmitted before it resumes polling. So restarting is how an over-long group gets broken up,
  and unlike the manual `--tasks-per-job 1` restart it replaces, the regrouping is automatic and
  applies only where the measurements say it is needed. A resume only regroups what is still
  *unsubmitted*, though: a group already recorded in the jobs file comes back with its original
  composition, so use `--ignore-submission` to regroup those as well.
- **`--parallel-jobs` defaults to 2000** for this task, which is good queue hygiene.

Passing `--tasks-per-job` (or `--AnaTupleFileTask-tasks-per-job`) turns the cost-aware grouping
and the per-attempt escalation off for that task and restores plain fixed-size chunking — the
escape hatch if an estimate ever misbehaves.
Setting the option for a *different* task does not affect it.

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
