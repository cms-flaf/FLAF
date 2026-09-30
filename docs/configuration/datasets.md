# Datasets

A **dataset** is one CMS sample — a simulated signal/background, or a chunk of real data —
identified by its DAS name. Datasets are declared per era in `datasets.yaml` files. Each dataset
is its own top-level key, so after the
[configuration merge](../concepts/configuration.md#how-values-combine) the datasets of the
framework and of the analysis are all available together. A name defined in both files takes the
analysis's entry as a whole — the two entries are not merged field by field.

## The split rule (where a dataset belongs)

| Kind of sample | Goes in |
|---|---|
| SM background, real data | `FLAF/config/<era>/datasets.yaml` (framework — common to all analyses) |
| Signal, custom/CI sample | `<analysis>/config/<era>/datasets.yaml` (analysis-specific) |

This keeps the common SM samples in one shared place while each analysis owns its signals.

`Run3_2025` and `Run3_2026` have no dedicated MC campaign. Set
`reuse_mc_from_era: Run3_2024` in those eras' `FLAF/config/<era>/global.yaml` and
Setup copies every 2024 MC (and analysis-signal) dataset that is not already
defined and is not data (`eraLetter`), and inherits `shared_mc` from that era.
Those `datasets.yaml` files therefore only need that year's data plus any
era-local custom/CI samples. 2026 PromptReco NanoAOD (eras A–D) is listed in
`FLAF/config/Run3_2026/datasets.yaml`.

## Dataset entry format

```yaml
DatasetName:
  generator: powheg          # madgraph, powheg, pythia, ...
  mass: 125                  # optional (signals)
  spin: 0                    # optional (resonant signals)
  crossSection: 1pb          # the name of an entry in the era's cross-section file
  nanoAOD:
    v12: /DAS/path/to/dataset/NANOAODSIM   # NanoAOD campaign tag -> one DAS name
    v15: /DAS/path/to/dataset/NANOAODSIM
```

Each tag under `nanoAOD` maps to **one** DAS dataset name (a string, not a list). Which tag is
used is set per era by `nanoAODVersions` (`data:` and `mc:` tags) in `global.yaml`. An era that
does not set it does not use Rucio at all: it lists the directory `dirName` on the global
`fs_nanoAOD` (for `Run3_2022` … `Run3_2023BPix` the HLepRare skims) — see
[Where the inputs come from](../concepts/storage.md#where-the-inputs-come-from).

Real-data entries carry `eraLetter` (the run era, e.g. `C`) instead of `generator` and
`crossSection`. Two optional fields control how the input files are listed: `dirName` (the
directory to list on `fs_nanoAOD`, default: the dataset name) and `fileNamePattern` (a map from
NanoAOD tag to a regular expression the file names must match, default `.*\.root$`).

For **custom/local** samples (e.g. CI test inputs) that are not official DAS datasets, point at
your own storage instead:

```yaml
custom_CI:
  generator: powheg
  mass: 125
  spin: 0
  crossSection: 1pb
  fs_nanoAOD: T3_CH_CERNBOX:/store/user/<user>/
  dirName: "directory_name"
```

## Cross-sections

MC datasets reference a **cross-section** by name: `crossSection` is always a key into the
cross-section file that the era's `global.yaml` selects with `crossSectionsFile` —
`FLAF/config/crossSections13p6TeV.yaml` for every Run 3 era (the 13 TeV file
`crossSections13TeV.yaml` is kept for Run 2, whose FLAF era directories have no `global.yaml`).
An unknown name fails AnaTuple production. For signals whose normalisation is set elsewhere, the
entry `1pb` (a cross-section of 1 pb, defined in that file) is conventional.

## Adding a dataset

1. **Choose the file** by the split rule above.
2. **Add the entry** in the right era's `datasets.yaml`, following the format.
3. Make sure the `crossSection` resolves — add it to `crossSections13p6TeV.yaml` if needed.
4. For **Run3_2024 signals**, check the actual DAS name: the naming changed (e.g. VBF drops the
   `_fixedTauDecays` suffix and uses a `_Par-` form). The conventions are summarised in the
   analysis docs and the project notes.
5. **Validate** (below).

## Validate the dataset config

The same check the CI runs (`ds-consistency-check`) verifies that MC entries have a generator and a
resolvable cross-section, that names are well-formed, etc.:

```sh
python3 test/checkDatasetConfigConsistency.py \
  --exception config/dataset_exceptions.yaml \
  Run3_2022 Run3_2022EE Run3_2023 Run3_2023BPix Run3_2024 Run3_2025 Run3_2026
```

Run it from the FLAF checkout after editing a framework `datasets.yaml`: it reads
`config/<era>/datasets.yaml` of FLAF only, not the analyses' files. It also checks that each MC dataset
exists in every listed era with the same `crossSection`; known, intentional exceptions live in
`config/dataset_exceptions.yaml`.
The same CI job checks the dataset names against `config/dataset_naming_rules.yaml`:

```sh
python3 test/checkDatasetNaming.py --rules config/dataset_naming_rules.yaml \
  Run3_2022 Run3_2022EE Run3_2023 Run3_2023BPix Run3_2024 Run3_2025 Run3_2026
```

See [CI / GitHub Actions](../ci/github-actions.md).

## Adding a new era

1. Create `FLAF/config/<new_era>/` with at least `datasets.yaml` and `global.yaml`.
2. Add the era to the C++ `Period` enum in `FLAF/include/AnalysisTools.h` and to the
   `PeriodToHHbTagInput` maps in `FLAF/include/HHbTagScores.h`. AnaTuple production
   JIT-compiles `Period::<era>` (`anaTupleProducer.py`); a missing enumerator fails
   immediately with `no member named '<era>' in 'Period'`.
3. Create `<analysis>/config/<new_era>/` with the analysis-specific overrides and signals.
   In `triggers.yaml`, every `jsonTRGcorrection_key` map must include the
   Corrections period name (`2026_Summer24` for `Run3_2026`). Missing keys
   fail MC jobs with `KeyError` in `TrigCorrProducer`. Until a 2026
   Electron-ID-SF JSON is published, `EleCorrProducer` evaluates that
   correction with year `2025Prompt` (the only year key in the 2025 file
   that `2026_Summer24` loads). A raw `2026Prompt` key fails HistTuple
   with `Index not available in Category`.
4. Add the era to `.github/workflows/test-setup-loading.yaml` in each affected analysis (so CI
   loads `Setup.py` for it and catches config errors early), and to the era lists of FLAF's
   `.github/workflows/ds-consistency-check.yaml` and `.github/workflows/cross-section-check.yaml`.
5. Add the era to the `<analysis>_eras` variables in `integration_cfg.yaml` of
   [`cms-flaf/FLAF_ci`](https://github.com/cms-flaf/FLAF_ci) if it should be part of CI runs.
   See [Integration pipeline](../ci/integration-pipeline.md#integration_cfgyaml).

See also [Eras & periods](../concepts/eras.md).
