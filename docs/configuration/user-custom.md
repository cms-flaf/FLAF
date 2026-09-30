# `user_custom.yaml`

`config/user_custom.yaml` holds **your personal, uncommitted settings** — where your outputs go,
which physics model to use, and a handful of options. It is git-ignored: it never gets committed,
so it changes the settings of *your* runs without changing anything for anyone else.

!!! note "Where it sits in the load order"
    FLAF reads `global.yaml` and then `user_custom.yaml` in each
    [configuration directory](../concepts/configuration.md#four-layers-merged-in-order), so
    `config/user_custom.yaml` overrides the framework defaults and the analysis-wide
    `config/global.yaml` — but `config/<era>/global.yaml` is read **after** it and wins for every
    top-level key it sets (in the analyses e.g. `corrections`, `nanoAODVersions`, `met_type`,
    `triggerFile`). A repeated top-level key replaces the earlier value wholesale, so a map such as
    `corrections:` has to be given in full. Only a file passed with
    [`--user-custom`](#per-run-overrides-user-custom) is read after everything else.

## A minimal file to get started

The quickest correct start is the file the CI uses, which every analysis ships:
`cp config/ci_custom.yaml config/user_custom.yaml`, then replace its `fs_default` (a path on the CI
runner) with your own storage. In HH→bb̄WW that file also carries CI-only `QCDRegions` and
`StatInference` overrides (see [below](#per-run-overrides-user-custom)); drop them for real
results. Written from scratch, the minimum is:

```yaml
# Where outputs go (your EOS / CERNBox user area):
fs_default: davs://eoshome-<initial>.cern.ch:8444/eos/user/<initial>/<user>/FLAF/HH_bbtautau/

# Use the small, fast set of processes while testing:
phys_model: TestModel

# Standard options:
analysis_config_area: config
compute_unc_variations: true
compute_unc_histograms: true
store_noncentral: true
```

Replace `<initial>`/`<user>` with yours (e.g. `k` / `kandroso`). `analysis_config_area` and
`store_noncentral` are not optional: `HistPlotTask` and `AnalysisCacheTask` read them without a
default and fail with a `KeyError` when they are missing. For H→μμ also add
`histTuple_flavor: CI` — its `default` flavor lists no variables. With just this, you can run the
[first-run smoke test](../getting-started/first-run.md).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `fs_default` | string or list | **Required.** Default storage for all outputs. The fallback for every other `fs_*`. See [Storage](../concepts/storage.md). |
| `fs_anaTuple`, `fs_HistTuple`, `fs_anaCacheTuple`, `fs_plots`, … | string/list | Optional per-output-type storage. Unset ⇒ uses `fs_default`. |
| `phys_model` | string | Which [physics model](processes-and-models.md) to run: `TestModel` (small, for testing/CI) or the analysis's production model (e.g. `BaseModel`). |
| `analysis_config_area` | string | **Required by `HistPlotTask`.** The analysis config directory, relative to the checkout — normally `config`; the plotter reads its `global.yaml` from there. |
| `compute_unc_variations` | bool | Whether to compute systematic (up/down) variations during production. |
| `compute_unc_histograms` | bool | Whether to also fill histograms for those variations. Prefer setting this **per histTuple flavor** in `global.yaml` (`histTuple_flavors.<flavor>.compute_unc_histograms`) — uncertainties are usually only needed for the limit-setting shape flavor, so the flavor should dictate it. The value here (or in `global.yaml`) is used as the fallback when the active flavor does not set it. |
| `store_noncentral` | bool | **Required by `AnalysisCacheTask`.** Whether to keep the non-central (systematic-shift) outputs, not just the central one. `AnalysisCacheTask` also uses it to decide whether the cache producers compute the variations. |
| `remove_merged_inputs` | bool | If `true`, `HistMergerTask` deletes each variable's per-chunk split histograms (`HistFromNtupleProducerTask` outputs) after merging, to save space, leaving a tiny per-chunk `.merged` marker in place of each. Safe: the producer stays "complete" for exactly the chunks that were merged (it finds the split *or* its marker), so the task graph stays consistent (no re-run); a chunk that was never produced has no marker and is still produced. Default `false` — intermediates are kept. |
| `variables` | list | Restrict which variables are produced/plotted (applied to the active `histTuple_flavor` list). If that flavor's variable list is empty (e.g. H_mumu `default`), this list is used as the active set. Omit for the full flavor set. |
| `histTuple_flavor` | string | Optional. Selects which `histTuple_flavors` entry drives the variable lists. The analyses set `default` in `config/global.yaml`; HH→bb̄ττ and H→μμ also define a short `CI` flavor. |
| `hist_from_ntuple_max_hists` | int | Max histograms `HistFromNtupleProducerTask` books in one RDataFrame pass. The count is variables × selections × (Central + every Up/Down). Default `4000`; `0` disables batching. Lower this (do not raise CI memory) if a job OOMs. |
| `anaTuple_shift_invariant_columns` | list | Set in the analysis `config/global.yaml`, not here. Regular expressions (`re.search`) of the anaTuple columns no systematic shift changes; they are stored in the central tree only. See [`AnaTupleFileTask`](../reference/tasks.md#anatuplefiletask). A list in a later config file replaces an earlier one. Default: empty. |
| `anaTuple_scheduling` | map | Tunes how `AnaTupleFileTask` branches are composed into HTCondor jobs. Every key has a default; see below. |

### `anaTuple_scheduling`

Optional. Controls the [cost-based job composition](../workflow/htcondor.md#how-branches-become-jobs)
of AnaTuple production. Sensible defaults apply when the block is absent.

| Key | Default | Meaning |
|---|---|---|
| `target_job_hours` | `6.0` | Wall time a job is packed up to. A branch that costs more than this is submitted on its own. |
| `max_units_per_job` | `50` | Upper bound on branches per job, whatever their cost. |
| `runtime_safety` | `2.5` | The packing capacity never exceeds `max_runtime / runtime_safety`, so a packed job cannot be built into the wall clock. |
| `parallel_jobs` | `2000` | Default queue footprint (also what creates the submission waves that let estimates improve mid-run). `--parallel-jobs` overrides it. |
| `probe_enabled` | `true` | Whether to run [`AnaTupleCostProbeTask`](../reference/tasks.md#anatuplecostprobetask) before production. |
| `probe_events` | `5000` | Events a probe scans per dataset. |
| `overhead_sec` | `300` | Fixed per-job cost (worker setup, JIT, corrections); re-measured from the probes. |
| `default_sec_per_event` | `0.02` | Per-event prior, used only until a dataset has been measured. |
| `default_events_per_byte` | `4.5e-4` | Events-per-byte prior, used when neither an event count nor a measurement is available. |
| `tier_safety` | see below | Divides the packing capacity according to how well the estimate is known: `job` 1.0, `probe`/`catalogue` 1.3, `process` 2.0, `group` 3.0, `default` 4.0. |
| `retry_runtime_factor` / `retry_memory_factor` / `retry_max_factor` | `1.5` / `1.25` / `3.0` | Per-attempt escalation of a resubmitted job's runtime and memory, and the cap on both. |
| `request_memory_mb` | *(unset)* | When set, requests this much memory explicitly instead of letting the site derive it from `--n-cpus`. |

The calibration is stored in `data/<version>/AnaTupleCost/cost_model.json` and is keyed by
**version, not era** — a version fixes the physics selection by convention, so a cost measured
while producing one era is reused by the others. Delete the file to force a full recalibration.

!!! tip "`TestModel` is the fast path"
    `TestModel` selects a reduced set of processes so the pipeline runs quickly end-to-end. Use it
    for development and local testing; switch to the production model only when you need full
    results. This is exactly what CI does.

## A production-style example

```yaml
fs_default: davs://eoshome-k.cern.ch:8444/eos/user/k/kandroso/FLAF/HH_bbtautau/
# A separate, roomier site for the big ntuples:
fs_anaTuple: T3_US_FNALLPC:/store/user/lpcflaf/HH_bbtautau/

phys_model: BaseModel
analysis_config_area: config
compute_unc_variations: true
compute_unc_histograms: true
store_noncentral: true
```

## Per-run overrides (`--user-custom`)

To change settings for a **single run** without editing your `user_custom.yaml`, pass an extra
YAML with `--user-custom`. It is loaded *after* every configuration file, `user_custom.yaml` and
`config/<era>/global.yaml` included, so its values win:

```sh
law run FLAF.Analysis.tasks.HistPlotTask \
  --version my_test --period Run3_2022 --workflow local --branches 0 --test 1000 \
  --user-custom /path/to/extra.yaml
```

The path may be absolute or relative to `$ANALYSIS_PATH`; a path that does not exist is an error.
Remote jobs receive the file as a job input. This is the preferred way to run one-off
variants (a different model, a different storage area, a short `variables:` list) — it keeps your
`user_custom.yaml` clean and is reproducible.

!!! note "The CI uses a dedicated file"
    Each analysis commits a `config/ci_custom.yaml`, which the integration pipeline copies to
    `config/user_custom.yaml` instead of using a personal file, so tests never touch real storage.
    All three set a local `fs_default`, `TestModel` and the standard options above. HH→bb̄ττ and
    H→μμ select `histTuple_flavor: CI`; HH→bb̄WW runs its `default` flavor (its `variables:` list
    is commented out) and overrides `QCDRegions` and the `StatInference` datacard configuration
    (`config/Datacards/CI_card.yaml`). See [Integration pipeline](../ci/integration-pipeline.md).
