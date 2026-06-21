# `user_custom.yaml`

`config/user_custom.yaml` holds **your personal, uncommitted settings** — where your outputs go,
which physics model to use, and a handful of options. It is loaded on top of the merged
[configuration](../concepts/configuration.md), so it overrides the defaults for *your* runs without
changing anything for anyone else. It is git-ignored: it never gets committed.

## A minimal file to get started

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

Replace `<initial>`/`<user>` with yours (e.g. `k` / `kandroso`). With just this, you can run the
[first-run smoke test](../getting-started/first-run.md).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `fs_default` | string or list | **Required.** Default storage for all outputs. The fallback for every other `fs_*`. See [Storage](../concepts/storage.md). |
| `fs_anaTuple`, `fs_HistTuple`, `fs_anaCacheTuple`, `fs_plots`, … | string/list | Optional per-output-type storage. Unset ⇒ uses `fs_default`. |
| `phys_model` | string | Which [physics model](processes-and-models.md) to run: `TestModel` (small, for testing/CI) or the analysis's production model (e.g. `BaseModel`). |
| `analysis_config_area` | string | The analysis config directory, relative to the checkout — normally `config`. |
| `compute_unc_variations` | bool | Whether to compute systematic (up/down) variations during production. |
| `compute_unc_histograms` | bool | Whether to also fill histograms for those variations. |
| `store_noncentral` | bool | Whether to keep the non-central (systematic-shift) outputs, not just the central one. |
| `variables` | list | Restrict which variables are produced/plotted. Omit for the full set. |

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

To change settings for a **single run** without editing your committed file, pass an extra YAML
with `--user-custom`. It is loaded *after* `user_custom.yaml`, so its values win:

```sh
law run FLAF.Analysis.tasks.HistPlotTask \
  --version my_test --period Run3_2022 --workflow local --branches 0 --test 1000 \
  --user-custom /path/to/extra.yaml
```

The path may be absolute or relative to `$ANALYSIS_PATH`. This is the preferred way to run one-off
variants (a different model, a different storage area, a short `variables:` list) — it keeps your
`user_custom.yaml` clean and is reproducible.

!!! note "The CI uses a dedicated file"
    The integration pipeline supplies its own `ci_custom.yaml` (local storage, `TestModel`, a short
    `variables:` list) instead of a personal file, so tests never touch real storage. See
    [Integration pipeline](../ci/integration-pipeline.md).
