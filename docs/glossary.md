# Glossary

Framework and CMS-computing vocabulary, in plain terms. For the quick on-ramp version see
[Key terms](getting-started/key-terms.md).

**anaTuple**
: The analysis-level ntuple FLAF produces from NanoAOD — a slimmed, skimmed ROOT tree with the
  objects, weights and flags an analysis needs. Produced by `AnaTupleFileTask`, merged by
  `AnaTupleMergeTask`.

**AnaProd**
: The part of FLAF (`AnaProd/`) that produces anaTuples from NanoAOD, including
  `anaTupleProducer.py` (run in the FLAF environment, or inside CMSSW when the analysis sets
  `use_cmssw_env_AnaTupleProduction`, as HH→bb̄ττ does).

**Branch**
: One independent work unit of a workflow task. What it represents depends on the task — an input
  file, a dataset, or a variable. Select with `--branches`.

**Bundle**
: A tarball of code/environment shipped to a batch worker so a job can run without the shared AFS
  area. See [HTCondor](workflow/htcondor.md#bundles-shipping-the-code-to-workers).

**Combine**
: The CMS statistical tool (`HiggsAnalysis/CombinedLimit`) used for limits and fits. FLAF builds a
  standalone `v10.4.2`.

**Corrections**
: The shared submodule providing object corrections and systematic variations (pileup, b-tag,
  triggers, …). The analysis configuration (`corrections:` in `global.yaml`, optionally per
  dataset or process) gives each correction the stage(s) where it is applied: `AnaTuple`
  (anaTuple production), `AnaTupleMerge`, `AnalysisCache` or `HistTuple` (e.g. HH→bb̄ττ applies
  its τ-ID, muon, electron and trigger scale factors at `HistTuple`).

**CMSSW**
: The CMS software framework, installed by `env.sh` under `soft/`. Pipeline steps that need it
  (e.g. HH→bb̄ττ's anaTuple production, datacard creation and Combine, some payload producers)
  are run inside it by the tasks themselves; for interactive commands `env.sh` defines the
  `cmsEnv` alias.

**CRAB**
: The CMS service that submits jobs to WLCG sites. FLAF submits workflow branches to it with
  `--workflow crab`. See [Running on CRAB](workflow/crab.md).

**CVMFS**
: The CERN read-only software-distribution filesystem (`/cvmfs/…`) from which FLAF gets compilers,
  Python, ROOT (LCG stacks), CMSSW and the POG correction files (`/cvmfs/cms-griddata.cern.ch`).

**DAS**
: The CMS Data Aggregation System — the catalogue of official datasets and the source of the
  dataset **name** convention (`/A/B/TIER`). File lists and disk locations are resolved via Rucio
  (see below); DAS is queried only for per-file event counts, which feed the job-cost estimate
  and are optional.

**Dataset**
: One CMS sample (a simulated process or a chunk of data), declared under a short name (e.g.
  `TTto2L2Nu`) in `datasets.yaml`, with its DAS name per NanoAOD version in `nanoAOD:`. See [Datasets](configuration/datasets.md).

**Era** / **period**
: A CMS data-taking period (`Run3_2022`, `Run3_2023BPix`, …), passed as `--period`. Selects
  datasets, corrections and the NanoAOD version. See [Eras](concepts/eras.md).

**Filesystem (`fs_*`)**
: A named storage location (local, EOS or a WLCG site) where a given output type is read/written.
  Configured in `user_custom.yaml`. See [Storage](concepts/storage.md).

**FLAF**
: The Flexible LAW-based Analysis Framework — the shared machinery (tasks, config, environment, CI)
  included as a submodule in each analysis.

**histTuple**
: An ntuple, derived from anaTuples, that carries the computed analysis observables, ready to be
  histogrammed. Produced by `HistTupleProducerTask`.

**HTCondor**
: CERN's batch system. FLAF submits workflow branches to it with `--workflow htcondor`. See
  [HTCondor](workflow/htcondor.md).

**LAW**
: [Luigi Analysis Workflow](https://github.com/riga/law) — the layer over Luigi that gives FLAF its
  command-line interface, remote-storage handling and batch submission.

**Luigi**
: The Python workflow engine that tracks task dependencies and outputs underneath LAW.

**Meta-process**
: A process template that expands into a family of concrete processes (e.g. all resonant mass
  points). Marked `is_meta_process: true`. See [Processes & models](configuration/processes-and-models.md).

**NanoAOD**
: The compact CMS data format that is the input to the whole pipeline.

**Payload producer**
: A configured component (under `payload_producers` in `global.yaml`) that computes analysis
  observables such as DNN scores or a mass reconstruction. Each producer runs in its own
  `AnalysisCacheTask` (`--producer-to-run <name>`), one branch per merged anaTuple file;
  `HistTupleProducerTask` then reads the cached columns, named `<producer>_<column>`. All three
  analyses define payload producers.

**Physics model**
: The named set of processes (background/signal/data) an analysis uses. `TestModel` is the small
  testing set; production uses the full model. Defined in `phys_models.yaml`.

**PlotKit**
: FLAF's plotting toolkit (a submodule of FLAF, [cms-flaf/PlotKit](https://github.com/cms-flaf/PlotKit)).
  Renders the stacked CMS plots with **matplotlib + mplhep** by default (no ROOT required) and can
  optionally render through **ROOT + cmsstyle**. It reads the analysis `config/plot/*.yaml` files and
  can also run standalone (`python -m PlotKit.cli`).

**Process**
: A physics object built from one or more datasets (e.g. "TT", "DY", "signal") — what you plot and
  fit. Defined in `processes.yaml`.

**Proxy (VOMS)**
: A short-lived credential derived from your grid certificate that authorises grid/EOS access.
  Created with `voms-proxy-init`. FLAF uses the file `X509_USER_PROXY` points to; `env.sh` sets
  that variable to `data/voms.proxy` only when it is not already set.

**Rucio**
: The CMS data-management service. For a dataset read by its DAS name (a NanoAOD version other
  than `HLepRare`), `InputFileTask` queries it for the list of NanoAOD files and their disk
  locations; skims listed from a storage directory (`fs_nanoAOD`) do not use it. See
  [Storage](concepts/storage.md#where-the-inputs-come-from).

**RunKit**
: Workflow utilities vendored into FLAF as a regular directory (formerly a submodule). Imported as
  `FLAF.RunKit.<module>`.

**StatInference**
: The shared submodule for datacard creation and limit/fit tooling (used by the HH analyses).

**Task**
: One stage of the pipeline with defined inputs, outputs and a run step. The unit LAW schedules.
  See [Tasks & LAW](concepts/tasks-and-law.md) and the [Task reference](reference/tasks.md).

**Version**
: The `--version` label that namespaces a run's outputs so productions and tests don't collide.

**WLCG**
: The Worldwide LHC Computing Grid — the federation of sites (`T1_*`, `T2_*`, `T3_*`) where CMS
  data and FLAF outputs are stored.

**Workflow**
: A task that splits into many branches, runnable `local`, on `htcondor` or on `crab`.
