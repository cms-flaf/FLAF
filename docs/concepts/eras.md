# Eras & periods

Every run targets one **era** (also called a **period**), passed as `--period`. An era is a CMS
data-taking period; choosing one selects the matching datasets, corrections and NanoAOD input.

## Run 3 eras (current)

| `--period` | Description | √s | NanoAOD (FLAF default version) | HH_bbWW, H_mumu |
|---|---|---|---|---|
| `Run3_2022` | 2022, pre-ECAL repair | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2022EE` | 2022, post-ECAL repair ("EE") | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2023` | 2023, pre-BPix | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2023BPix` | 2023, post-BPix install | 13.6 TeV | v14, HLepRare skim | v12 MC / `22Sep23` data (DAS) |
| `Run3_2024` | 2024 | 13.6 TeV | v15 | v15 (DAS) |
| `Run3_2025` | 2025 | 13.6 TeV | v15 | v15 (DAS) |
| `Run3_2026` | 2026 | 13.6 TeV | v15 | v15 (DAS) |

For 2022–2023BPix the framework default reads the HLepRare NanoAOD v14 skims
(`skim_2025_v1` under `/store/group/phys_higgs/HLepRare/`, the era's `fs_nanoAOD`); only
HH_bbtautau uses them. HH_bbWW and H_mumu override those four eras to read the central NanoAOD
from DAS instead. For 2024–2026 all three analyses read NanoAOD v15 from DAS (apart from the
few datasets that set their own `fs_nanoAOD`). FLAF itself sets only `nano_version: v15` for these
eras and no NanoAOD source: the analyses select DAS through `nanoAODVersions` in their era configs.

## Run 2 eras (legacy, not runnable)

`FLAF/config/` still has `Run2_2016_HIPM`, `Run2_2016`, `Run2_2017` and `Run2_2018` directories
(13 TeV), but they hold only legacy `samples.yaml`/`triggers.yaml` files — no `global.yaml` and
no `datasets.yaml` — and the analyses carry at most a Run 2 `weights.yaml`. Configuration loading
cannot build a run for these eras, and CI covers only the seven Run 3 eras. Treat Run 2 as
unsupported.

## Why the split into sub-eras?

The detector and its calibration change *within* a year, so CMS treats those segments as separate
eras for analysis:

- **2022** splits at the ECAL endcap repair → `Run3_2022` (before) and `Run3_2022EE` (after).
- **2023** splits at the pixel-detector "BPix" installation → `Run3_2023` and `Run3_2023BPix`.

Each sub-era has its own corrections and luminosity, which is exactly why the
[configuration system](configuration.md) has a **per-era layer**: `FLAF/config/<era>/` and
`<analysis>/config/<era>/` carry the era-specific datasets and overrides.

## What an era controls

- **Datasets** — `config/<era>/datasets.yaml` lists the samples available for that era; each
  entry's `nanoAOD:` map gives its DAS name per NanoAOD version (`22Sep23`, `v12`, `v15`, …).
- **NanoAOD input** — the table above, set by two keys of the era's `global.yaml`.
  `nanoAODVersions: { data: …, mc: … }` picks which entry of each dataset's `nanoAOD:` map is
  read, through Rucio; when it is not set, the files are listed from the era's `fs_nanoAOD` (the
  HLepRare skims). A dataset with its own `fs_nanoAOD` is always read from there.
  `nano_version` names the NanoAOD format the producers are written for (`v12` has different
  branches), so it has to match what is read.
- **Corrections** — pileup, b-tagging, trigger and other scale factors are era-specific.
- **Signals** — resonant/non-resonant signals exist for some eras and not others. For
  `Run3_2024`, VBF and non-resonant ggF HH are on DAS (new `Par-` naming); resonant
  Radion/BulkGraviton and X→YH→2B2W are not.
- **2024/2025/2026 shared MC** — there is no dedicated 2025 or 2026 MC campaign.
  All three years use the Summer24 NanoAOD, but **jet, PU and tau corrections
  differ**, so AnaTuple production runs once per era. `Run3_2025` and `Run3_2026`
  set `reuse_mc_from_era: Run3_2024` so the 2024 MC dataset list (and the
  `shared_mc` split) is reused. Each production stores `weight_base` (all events,
  this year's luminosity; use for a single-year run) and `weight_base_cmb` (the
  same events split by residue between 2024, 2025 and 2026; use for a combined
  run). The split applies only to MC; data keeps the full year in
  `weight_base_cmb`. Select which branch histograms use with
  `weight_base_branch`; all three eras default to `weight_base`, so by default every year
  uses all of its events with its own luminosity, and the residue split below feeds only
  `weight_base_cmb`.
  `shared_mc` lives only on the source era (`Run3_2024`): a 17:17:4 residue
  *target* over modulus 38 (`Run3_2024: [ 0, 16 ]`, `Run3_2025: [ 17, 33 ]`,
  `Run3_2026: [ 34, 37 ]`), matching the recorded luminosities
  109948.18 : 110730.86 : 25843.26. The actual event split need not match
  that target. AnaTupleFileTask therefore stores two denominators: the
  full-sample sum (for `weight_base`) and the in-era sum (for
  `weight_base_cmb`). Each weight is
  `gen × lumi × xs × PU / its_denominator`, so both yields stay `L·σ`.
  HistTuple multiplies the AnaTuple column named by `weight_base_branch`
  (`weight_base` for a single-year run, `weight_base_cmb` for the combined
  24+25+26 run).
  Until official UParTAK4 shape files exist, 2024+ era overlays omit
  `btag.normCacheProducer`, so HistTuple does not depend on the global
  `BtagShape` cache. `modes.<stage>: none` still loads the correction (needed
  for WP-id branches) but does not apply scale factors. Missing
  `uncs_to_exclude` era keys default to an empty list.
  The 2026 EGM folder has scale/smear files but no `electron.json`, so
  Electron-ID-SF is loaded from the 2025 file and evaluated with year
  `2025Prompt` (not `2026Prompt`).

## Running several eras

A task runs **one era at a time**. To cover multiple eras, launch the task once per era (often
scripted), or, in CI, list them in the `*_eras` variable (e.g.
`Run3_2022 Run3_2022EE Run3_2023 Run3_2023BPix Run3_2024 Run3_2025 Run3_2026`). See the
[integration pipeline](../ci/integration-pipeline.md). For a 2024+2025+2026
combination, run each era with `weight_base_branch: weight_base_cmb` and add
the histograms.

!!! warning "`--period` must match an existing era directory"
    If you pass an era that has no `config/<era>/` (or whose datasets are not defined), config
    loading fails — and if a run unexpectedly drops into `InputFileTask` and queries Rucio for
    nothing, a wrong `--period`/`--version` combination is the usual cause. See
    [Troubleshooting](../troubleshooting.md).

## Adding a new era

Adding an era means creating its per-era config directories in both the framework and the
analysis, wiring it into the CI era lists, and listing it in `test-setup-loading`. The full
procedure is in [Datasets](../configuration/datasets.md#adding-a-new-era).
