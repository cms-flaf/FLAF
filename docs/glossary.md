# Glossary

Framework and CMS-computing vocabulary, in plain terms. For the quick on-ramp version see
[Key terms](getting-started/key-terms.md).

**anaTuple**
: The analysis-level ntuple FLAF produces from NanoAOD — a slimmed, skimmed ROOT tree with the
  objects, weights and flags an analysis needs. Produced by `AnaTupleFileTask`, merged by
  `AnaTupleMergeTask`.

**AnaProd**
: The part of FLAF (`AnaProd/`) that produces anaTuples from NanoAOD, including the CMSSW-based
  `anaTupleProducer.py`.

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
  triggers, …) applied during ntuple production.

**CMSSW**
: The CMS software framework. Some stages run inside it; FLAF wraps it with the `cmsEnv` helper.

**CVMFS**
: The CERN read-only software-distribution filesystem (`/cvmfs/…`) from which FLAF gets compilers,
  Python, ROOT (LCG stacks) and CMSSW.

**DAS**
: The CMS Data Aggregation System — the catalogue of official datasets. `InputFileTask` queries it
  to turn dataset names into file lists.

**Dataset**
: One CMS sample (a simulated process or a chunk of data), identified by its DAS name. Declared in
  `datasets.yaml`. See [Datasets](configuration/datasets.md).

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
: A configured component that computes an analysis observable during `HistTupleProducerTask`.

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
  Created with `voms-proxy-init`; FLAF expects it at `data/voms.proxy`.

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
: A task that splits into many branches, runnable `local` or on `htcondor`.
