# Configuration system

FLAF's behaviour — which datasets exist, which corrections apply, where outputs go, which
processes make up the analysis — is driven by **YAML configuration**. This page explains *how* the
configuration is assembled. For *how to change* specific things, see the
[Configuration guide](../configuration/user-custom.md).

The implementation lives in `FLAF/Common/Setup.py`.

## Four layers, merged in order

When you run a task with `--period <era>`, FLAF loads configuration from **four directories**, in
this order:

```python
config_path_order = [
    "<analysis>/FLAF/config",          # 1. framework defaults (all analyses)
    "<analysis>/FLAF/config/<era>",    # 2. framework defaults for this era
    "<analysis>/config",               # 3. analysis-wide settings
    "<analysis>/config/<era>",         # 4. analysis settings for this era  ← wins
]
```

Think of it as **general → specific**: the framework provides sensible cross-analysis defaults
(layer 1–2), and each analysis overrides or extends them (layer 3–4). The per-era directories let
2022 and 2023 differ without duplicating everything.

### How values combine

The merge rule depends on the value type, and this distinction matters:

- **Scalars** (a string, a number, a bool): a later layer **overrides** an earlier one. So
  `analysis/config/<era>` has the final say.
- **Lists** (most importantly `datasets.yaml`): later layers **extend** (concatenate) earlier
  ones. Nothing is lost.

The list behaviour is why datasets are *split* across files instead of duplicated: SM backgrounds
and data live in the framework's `FLAF/config/<era>/datasets.yaml`, while signals and custom
samples live in the analysis's `config/<era>/datasets.yaml`. After merging, **all** of them are
available together. See [Datasets](../configuration/datasets.md).

## The objects you may meet in code

| Class (in `Common/Setup.py`) | Role |
|---|---|
| `Config` | Loads and merges the YAML files from the four directories for one logical config (e.g. "datasets"). Accessed like a dict: `cfg["key"]`, `cfg.get("key", default)`. |
| `Setup` | The master configuration object. Built once per `(analysis, period)`; holds the merged datasets, processes, the physics model and more. |
| `PhysicsModel` | Classifies each process as **background**, **signal** or **data**, and expands *meta-processes* (parameterised families, e.g. all resonant masses). |

`Setup` is a **singleton**: anywhere in the code, `Setup.getGlobal()` returns the one instance for
the current run, so all tasks see a consistent configuration.

## The key configuration files

| File | Lives in | Holds |
|---|---|---|
| `user_custom.yaml` | analysis `config/` | **Your** personal, uncommitted settings: storage, model, options. [Guide](../configuration/user-custom.md). |
| `global.yaml` | both layers | Global settings: anaTuple/histTuple definitions, corrections, payload producers, signal types. |
| `datasets.yaml` | per-era, both layers | Dataset (sample) definitions. [Guide](../configuration/datasets.md). |
| `processes.yaml` | analysis `config/` | Logical processes built from datasets. [Guide](../configuration/processes-and-models.md). |
| `phys_models.yaml` | analysis `config/` | Which processes are background/signal/data for a model. [Guide](../configuration/processes-and-models.md). |
| `crossSections13p6TeV.yaml` | `FLAF/config/` | Cross-section values referenced by datasets. |

!!! tip "Validate your config without running the pipeline"
    Loading `Setup.py` for an era is exactly what the `test-setup-loading` CI check does — it
    catches typos and missing references early. You can do the same locally by constructing the
    setup for an era; if it loads, the config is internally consistent.

## `user_custom.yaml` is part of the merge too

Your `config/user_custom.yaml` overlays the merged configuration with personal values (storage
locations, `phys_model`, options like `compute_unc_variations`). For a single run you can layer an
*extra* file on top with `--user-custom <path>`, which is loaded last and therefore wins — handy
for one-off tests without editing your committed file. See
[`user_custom.yaml`](../configuration/user-custom.md#per-run-overrides-user-custom).
