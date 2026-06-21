# Key terms

FLAF borrows vocabulary from software-workflow tools (LAW/Luigi) and from CMS computing. If your
background is physics, a few words may be unfamiliar or may mean something slightly different
from what you expect. This page is the quick crosswalk; the [Glossary](../glossary.md) has the
full list.

| Term | What it means in FLAF |
|---|---|
| **Task** | One stage of the pipeline (e.g. "produce analysis ntuples", "make plots"). A unit of work LAW knows how to run, with defined inputs and outputs. You request a task and LAW runs whatever it depends on. |
| **Workflow** | A task that splits into many independent **branches**. Run it `--workflow local` (on your machine) or `--workflow htcondor` (on the batch system). |
| **Branch** | One independent work unit inside a workflow — for example one input file, one dataset, or one variable. `--branches 0,2,5-7` selects which ones to run. |
| **Era** / **period** | A CMS data-taking period, e.g. `Run3_2022`, `Run3_2023BPix`. Passed as `--period`. Selects which datasets, corrections and NanoAOD version apply. See [Eras](../concepts/eras.md). |
| **Dataset** | A specific CMS sample, identified by its DAS name (a simulated signal/background, or a chunk of real data). |
| **Process** | A *physics* object built from one or more datasets (e.g. "TT", "DY", "signal"). What you actually plot and fit. Defined in `processes.yaml`. |
| **Sample** | Loosely used for "a dataset or a process". When precision matters, prefer *dataset* (CMS file set) vs *process* (physics grouping). |
| **anaTuple** | The analysis-level ntuple FLAF produces from NanoAOD: a slimmed, skimmed ROOT tree with the objects and flags the analysis needs. |
| **histTuple** | A further ntuple with the (heavier) analysis observables computed, ready to be turned into histograms. |
| **Histogram** | A binned distribution of one variable for one process, including its systematic variations. |
| **Version** | A label (`--version`) that namespaces a run's outputs, so different productions or tests don't collide. |
| **Customisation** | An ad-hoc `key=value` override passed via `--customisations` (e.g. `deepTauVersion=2p5`). |
| **Filesystem** (`fs_*`) | A named storage location (local or grid/EOS) where a given output type is read/written. Configured in `user_custom.yaml`. See [Storage](../concepts/storage.md). |
| **Bundle** | A tarball of the code/environment shipped to batch workers so a job can run without your AFS area. See [HTCondor](../workflow/htcondor.md). |
| **Physics model** | The set of processes (backgrounds, signals, data) used for an analysis. `TestModel` is the small set used in testing; production uses the full model. Defined in `phys_models.yaml`. |
| **Corrections** | Scale factors and systematic variations (pileup, b-tagging, triggers, …) applied during ntuple production, from the shared `Corrections` submodule. |
| **LAW / Luigi** | The workflow engine. [Luigi](https://luigi.readthedocs.io/) tracks task dependencies and outputs; [LAW](https://github.com/riga/law) adds the command-line interface, remote storage and batch submission FLAF uses. |
| **CMSSW** | The CMS software framework. Some stages run inside it; FLAF wraps this with the `cmsEnv` helper. |

!!! tip "The mental model in one sentence"
    You ask LAW for the **task** whose output you want, for a given **version** and **period**;
    LAW runs the chain of tasks needed, splitting big tasks into **branches** that can go to
    **HTCondor**, reading and writing each output to its configured **filesystem**.

Ready to see the chain in action? Continue to the
[full-workflow walkthrough](../workflow/walkthrough.md), or read
[Tasks & LAW](../concepts/tasks-and-law.md) to understand the engine first.
