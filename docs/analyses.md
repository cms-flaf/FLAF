# Analyses

FLAF is shared by three analyses. The **common** pipeline is documented here; each analysis adds
its own physics — extra submodules, observables, signals and (for the HH analyses) statistical
inference — documented in that analysis's own `docs/`.

| Analysis | Channel | Adds on top of FLAF | Docs |
|---|---|---|---|
| **HH→bb̄ττ** | HH → bb̄ττ | SVfit (`ClassicSVfit`, `SVfitTF`), `HHKinFit2`, `HHbtag`, DeepTau; resonant + non-resonant signals; `StatInference`. | [github.com/cms-flaf/HH_bbtautau](https://github.com/cms-flaf/HH_bbtautau) → `docs/` |
| **HH→bb̄WW** | HH → bb̄WW | `DeepHME` mass reconstruction; the `BtagShape` normalisation (`AnalysisCacheAggregationTask`); `StatInference`. | [github.com/cms-flaf/HH_bbWW](https://github.com/cms-flaf/HH_bbWW) → `docs/` |
| **H→μμ** | H → μμ | Single-Higgs; the simplest setup (just `FLAF` + `Corrections`); **no** statistical-inference submodule. | [github.com/cms-flaf/H_mumu](https://github.com/cms-flaf/H_mumu) → `docs/` |

## What is common vs analysis-specific

- **Common (here, in FLAF):** the [task graph](concepts/data-flow.md), the
  [configuration system](concepts/configuration.md), the [environment](concepts/environment.md),
  [storage](concepts/storage.md), [eras](concepts/eras.md) and [CI](ci/integration-pipeline.md).
  The [full-workflow walkthrough](workflow/walkthrough.md) applies to every analysis.
- **Analysis-specific (in each repo's `docs/`):** the extra physics submodules and how to set them
  up, the analysis's signals and processes, its observables and any analysis-only steps, and — for
  HH→bb̄ττ and HH→bb̄WW — the statistical-inference configuration.

## HH→bb̄ττ — the reference analysis

The most feature-complete analysis: SVfit and HHKinFit2 mass reconstruction, the HHbtag b-jet
identifier, DeepTau-based τ identification (the version is the `deepTauVersion` key of
`config/global.yaml`, `2p5` by default), and resonant + non-resonant signal models. Used throughout
these docs as the worked example.

## HH→bb̄WW

Uses `DeepHME` for mass reconstruction instead of SVfit. Like every analysis, it computes its
payload-producer observables (HME, DeepHME and DNN scores; the `default` flavour uses the DeepHME
and DNN ones) in `AnalysisCacheTask` before histogramming. On top of that, its `BtagShape` producer is a global one: its per-file results are
aggregated per sample by `AnalysisCacheAggregationTask` into the b-tag shape normalisation that the
`btag` correction reads — see the
[walkthrough](workflow/walkthrough.md#stage-2-compute-analysis-observables-histtuples) and
[Task reference](reference/tasks.md#analysiscacheaggregationtask).

## H→μμ

A single-Higgs analysis with the leanest submodule set (no `StatInference`/`inference`). Like the
HH analyses, its CI runs over all seven Run 3 eras; its CI process names differ in case
(`custom_CI_signal`, `custom_CI_background_TT`, … instead of `custom_CI_Signal`, …) — see
[Processes & models](configuration/processes-and-models.md).
