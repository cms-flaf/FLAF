# Eras & periods

Every run targets one **era** (also called a **period**), passed as `--period`. An era is a CMS
data-taking period; choosing one selects the matching datasets, corrections and NanoAOD version.

## Run 3 eras (current)

| `--period` | Description | √s | NanoAOD |
|---|---|---|---|
| `Run3_2022` | 2022, pre-ECAL repair | 13.6 TeV | v12 |
| `Run3_2022EE` | 2022, post-ECAL repair ("EE") | 13.6 TeV | v12 |
| `Run3_2023` | 2023, pre-BPix | 13.6 TeV | v13 |
| `Run3_2023BPix` | 2023, post-BPix install | 13.6 TeV | v13 |
| `Run3_2024` | 2024 | 13.6 TeV | v15 |
| `Run3_2025` | 2025 (future) | 13.6 TeV | — |

## Run 2 eras (legacy)

`Run2_2016_HIPM`, `Run2_2016`, `Run2_2017`, `Run2_2018` (13 TeV). Still defined, but new
development targets Run 3.

## Why the split into sub-eras?

The detector and its calibration change *within* a year, so CMS treats those segments as separate
eras for analysis:

- **2022** splits at the ECAL endcap repair → `Run3_2022` (before) and `Run3_2022EE` (after).
- **2023** splits at the pixel-detector "BPix" installation → `Run3_2023` and `Run3_2023BPix`.

Each sub-era has its own corrections and luminosity, which is exactly why the
[configuration system](configuration.md) has a **per-era layer**: `FLAF/config/<era>/` and
`<analysis>/config/<era>/` carry the era-specific datasets and overrides.

## What an era controls

- **Datasets** — `config/<era>/datasets.yaml` lists the samples available for that era, including
  the correct NanoAOD version path on DAS.
- **NanoAOD version** — the table above; the dataset entries point at the right `vNN` campaign.
- **Corrections** — pileup, b-tagging, trigger and other scale factors are era-specific.
- **Signals** — resonant/non-resonant signals exist for some eras and not others (for instance,
  several signal families are not produced for `Run3_2024`).

## Running several eras

A task runs **one era at a time**. To cover multiple eras, launch the task once per era (often
scripted), or, in CI, list them in the `*_eras` variable (e.g.
`Run3_2022 Run3_2022EE Run3_2023 Run3_2023BPix`, or `ALL`). See the
[integration pipeline](../ci/integration-pipeline.md).

!!! warning "`--period` must match an existing era directory"
    If you pass an era that has no `config/<era>/` (or whose datasets are not defined), config
    loading fails — and if a run unexpectedly drops into `InputFileTask` and queries DAS for
    nothing, a wrong `--period`/`--version` combination is the usual cause. See
    [Troubleshooting](../troubleshooting.md).

## Adding a new era

Adding an era means creating its per-era config directories in both the framework and the
analysis, wiring it into the CI era lists, and listing it in `test-setup-loading`. The full
procedure is in [Datasets](../configuration/datasets.md#adding-a-new-era).
