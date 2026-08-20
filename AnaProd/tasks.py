import contextlib
import json
import law
import luigi
import os
import shutil
import re
import yaml
from pathlib import Path

from FLAF.RunKit.run_tools import (
    ps_call,
    PsCallError,
    natural_sort,
    check_root_file_integrity,
    get_tree_entries,
)
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    CrabWorkflow,
    copy_param,
)
from FLAF.Common.Utilities import getCustomisationSplit, ServiceThread
from .AnaTupleFileList import CreateMergePlan
from .CostModel import CostModel, entry_key, merged_params
from .MergeAnaTuples import mergeAnaTuples


class InputFileTask(Task, law.LocalWorkflow):
    def __init__(self, *args, **kwargs):
        kwargs["workflow"] = "local"
        super(InputFileTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        for dataset_id, dataset_name in self.iter_datasets():
            branches[dataset_id] = dataset_name
        return branches

    def output(self):
        dataset_name = self.branch_data
        return self.local_target(f"{dataset_name}.json")

    def run(self):
        dataset_name = self.branch_data
        print(f"{dataset_name}: creating input file list into {self.output().abspath}")
        dataset = self.datasets[dataset_name]
        process_group = dataset["process_group"]
        ignore_missing = self.global_params.get("ignore_missing_nanoAOD_files", {}).get(
            process_group, False
        )
        fs_nanoAOD, folder_name, include_folder_name = self.get_fs_nanoAOD(dataset_name)
        nano_version = self.get_nano_version(dataset_name)
        pattern_dict = self.datasets[dataset_name].get("fileNamePattern", {})
        pattern = pattern_dict.get(nano_version, r".*\.root$")
        entries = fs_nanoAOD.listdir(folder_name)
        # After the listing, so the metadata comes from it instead of a second query.
        listing_info = self.list_file_info(fs_nanoAOD, folder_name)
        input_files = []
        inactive_files = []
        file_info = {}
        for file in entries:
            if not re.match(pattern, file):
                continue
            file_path = os.path.join(folder_name, file) if include_folder_name else file
            if hasattr(fs_nanoAOD, "file_interface"):

                if hasattr(fs_nanoAOD.file_interface, "is_available"):
                    if not fs_nanoAOD.file_interface.is_available(
                        folder_name, file, verbose=1
                    ):
                        if ignore_missing:
                            print(
                                f"{file_path}: will be ignored because no sites are found."
                            )
                            inactive_files.append(file_path)
                            continue
                        else:
                            raise RuntimeError(f"No sites found for {file_path}")
            input_files.append(file_path)
            info = listing_info.get(file)
            if info:
                file_info[file_path] = info

        if len(input_files) == 0:
            raise RuntimeError(f"No input files found for {dataset_name}")

        input_files = natural_sort(input_files)
        output = {
            "input_files": input_files,
            "inactive_files": inactive_files,
            "file_info": file_info,
        }
        with self.output().localize("w") as out_local_file:
            with open(out_local_file.abspath, "w") as f:
                json.dump(output, f, indent=2)

        print(f"{dataset_name}: {len(input_files)} input files are found.")

    @staticmethod
    def list_file_info(fs, folder_name):
        """``{file_name: {"size": .., "n_events": ..}}`` for one input folder.

        The size comes from the directory listing that the file interface performs
        anyway; ``n_events`` is filled in only where the backend can supply it cheaply
        (DAS, for Rucio-discovered datasets).  Purely advisory: any failure yields an
        empty mapping and job-cost estimation falls back to coarser tiers.
        """
        interface = getattr(fs, "file_interface", None)
        if interface is None or not hasattr(interface, "listdir_info"):
            return {}
        try:
            return interface.listdir_info(folder_name)
        except Exception as e:
            print(f"{folder_name}: unable to collect input file metadata: {e}")
            return {}

    input_file_cache = {}

    @staticmethod
    def _load(input_file_list):
        if input_file_list not in InputFileTask.input_file_cache:
            with open(input_file_list, "r") as f:
                InputFileTask.input_file_cache[input_file_list] = json.load(f)
        return InputFileTask.input_file_cache[input_file_list]

    @staticmethod
    def load_input_files(input_file_list, test=False):
        input_files = InputFileTask._load(input_file_list)["input_files"]
        active_files = (
            [input_files[0]] if test and len(input_files) > 0 else input_files
        )
        return active_files

    @staticmethod
    def load_file_info(input_file_list):
        """Per-file metadata, empty for lists produced before it was recorded."""
        return InputFileTask._load(input_file_list).get("file_info", {})

    WF = None
    WF_complete_ = False

    @staticmethod
    def WF_complete(ref_task):
        if InputFileTask.WF_complete_:
            return True
        if InputFileTask.WF is None:
            InputFileTask.WF = InputFileTask.req(ref_task, branch=-1, branches=())
        InputFileTask.WF_complete_ = InputFileTask.WF.complete()
        return InputFileTask.WF_complete_


class AnaTupleProducerMixin:
    """Shared invocation of `anaTupleProducer.py` (production jobs and cost probes)."""

    @property
    def bundle_flavours(self):
        flavours = ["core", "inputFileList"]
        if self.global_params.get("use_cmssw_env_AnaTupleProduction", False):
            flavours.append("cmssw")
        return flavours

    def producer_settings(self):
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = (
            customisation_dict["channels"]
            if "channels" in customisation_dict.keys()
            else self.global_params["channelSelection"]
        )
        if type(channels) == list:
            channels = ",".join(channels)
        store_noncentral = (
            customisation_dict["store_noncentral"] == "True"
            if "store_noncentral" in customisation_dict.keys()
            else self.global_params.get("store_noncentral", False)
        )
        compute_unc_variations = (
            customisation_dict["compute_unc_variations"] == "True"
            if "compute_unc_variations" in customisation_dict.keys()
            else self.global_params.get("compute_unc_variations", False)
        )
        return channels, store_noncentral, compute_unc_variations

    def cost_params(self):
        """Scheduling parameters from `anaTuple_scheduling` in the global config."""
        return merged_params(self.global_params.get("anaTuple_scheduling"))

    def producer_env(self):
        if self.global_params.get("use_cmssw_env_AnaTupleProduction", False):
            return self.cmssw_env
        return None

    def anatuple_cmd(
        self,
        dataset_name,
        local_input,
        in_file_name,
        out_dir,
        output_name,
        report_path,
        extra_args=None,
    ):
        channels, store_noncentral, compute_unc_variations = self.producer_settings()
        cmd = [
            "python3",
            "-u",
            os.path.join(self._flaf_root(), "AnaProd", "anaTupleProducer.py"),
            "--period",
            self.period,
            "--inFile",
            local_input,
            "--outDir",
            out_dir,
            "--dataset",
            dataset_name,
            "--anaTupleDef",
            os.path.join(self.ana_path(), self.global_params["anaTupleDef"]),
            "--channels",
            channels,
            "--inFileName",
            in_file_name,
            "--reportOutput",
            report_path,
            "--LAWrunVersion",
            self.version,
            "--output-name",
            output_name,
        ]
        if compute_unc_variations:
            cmd.append("--compute-unc-variations")
        if store_noncentral:
            cmd.append("--store-noncentral")
        if self.test > 0:
            cmd.extend(["--nEvents", str(self.test)])
        if self.user_custom:
            cmd.extend(["--user-custom", self.user_custom])
        cmd.extend(extra_args or [])
        return cmd


class AnaTupleFileTask(
    AnaTupleProducerMixin, Task, HTCondorWorkflow, CrabWorkflow, law.LocalWorkflow
):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 40.0)
    # tautau CMSSW AnaTuple used ~7.5 GB RSS on CRAB; 2 cores cap at 5000 MB.
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["inputFile"] = InputFileTask.req(self, branches=())
        if self.cost_params()["probe_enabled"]:
            reqs["costProbe"] = AnaTupleCostProbeTask.req(
                self,
                branches=(),
                max_runtime=AnaTupleCostProbeTask.max_runtime._default,
                n_cpus=AnaTupleCostProbeTask.n_cpus._default,
            )
        return reqs

    def requires(self):
        return []

    # ------------------------------------------------------------ cost-aware packing
    #
    # The HTCondor workflow proxy calls these to build jobs of bounded duration instead
    # of fixed-size groups of branches. A task that does not define them keeps law's
    # default `tasks_per_job` chunking.

    def cost_model_path(self):
        """Location of the calibration store.

        Keyed by version only, not by era: a version fixes the physics selection, so a
        cost measured while producing one era applies to the others and a multi-era
        production is calibrated once.
        """
        return os.path.join(
            self.ana_data_path(), self.version, "AnaTupleCost", "cost_model.json"
        )

    _dataset_peers_cache = None

    def _dataset_peers(self):
        """(by_process, by_group): dataset names sharing a process / process group."""
        if self._dataset_peers_cache is None:
            by_process = {}
            by_group = {}
            for _, dataset_name in self.iter_datasets():
                dataset = self.datasets[dataset_name]
                by_process.setdefault(dataset["process_name"], []).append(dataset_name)
                by_group.setdefault(dataset["process_group"], []).append(dataset_name)
            self._dataset_peers_cache = (by_process, by_group)
        return self._dataset_peers_cache

    def _input_file_info(self, dataset_name):
        input_file_list = (
            InputFileTask.req(
                self, branch=self.get_dataset_id(dataset_name), branches=()
            )
            .output()
            .abspath
        )
        return InputFileTask.load_file_info(input_file_list)

    _cost_model = None

    def load_cost_model(self):
        """Calibration store, refreshed once per process with any probe results it has
        not seen yet.  Kept on the instance because the probe lookup goes to remote
        storage and because everything in this process must share one writer."""
        if self._cost_model is not None:
            return self._cost_model
        model = CostModel.load(self.cost_model_path(), self.cost_params())
        self._cost_model = model
        failed = []
        for _, dataset_name in self.iter_datasets():
            nano_version = self.get_nano_version(dataset_name)
            if entry_key(nano_version, dataset_name) in model.entries:
                continue
            target = AnaTupleCostProbeTask.probe_target(self, dataset_name)
            try:
                if not target.exists():
                    continue
                with target.localize("r") as local_probe:
                    with open(local_probe.abspath, "r") as f:
                        probe = json.load(f)
            except Exception as e:
                print(f"{dataset_name}: unable to read the cost probe: {e}")
                continue
            if not model.set_from_probe(nano_version, dataset_name, probe):
                failed.append(dataset_name)
        if failed:
            print(
                f"cost model: no usable probe for {len(failed)} dataset(s), falling back "
                f"to a coarser estimate: {', '.join(sorted(failed)[:10])}"
                + (" ..." if len(failed) > 10 else "")
            )
        if model.dirty:
            model.save(self.cost_model_path())
        return model

    def branch_cost_map(self):
        """{branch -> (estimated seconds, tier)} for every branch of this workflow."""
        model = self.load_cost_model()
        by_process, by_group = self._dataset_peers()
        file_info = {}
        costs = {}
        for branch, (dataset_name, input_file, _) in self.branch_map.items():
            if dataset_name not in file_info:
                info = self._input_file_info(dataset_name)
                file_info[dataset_name] = info
                model.set_events_per_byte_from_catalogue(
                    self.get_nano_version(dataset_name), dataset_name, info
                )
            dataset = self.datasets[dataset_name]
            costs[branch] = model.estimate(
                self.get_nano_version(dataset_name),
                dataset_name,
                file_info[dataset_name].get(input_file),
                peers=by_process.get(dataset["process_name"]),
                group_peers=by_group.get(dataset["process_group"]),
                max_events=self.test if self.test > 0 else None,
            )
        if model.dirty:
            model.save(self.cost_model_path())
        return costs

    def cost_capacity_seconds(self):
        """Packing capacity: the target job duration, never so large that a job built up
        to it could run into `max_runtime`."""
        params = self.cost_params()
        target = float(params["target_job_hours"]) * 3600.0
        safety = max(float(params["runtime_safety"]), 1.0)
        return max(min(target, float(self.max_runtime) * 3600.0 / safety), 60.0)

    def record_job_durations(self, samples):
        """Fold observed job durations into the calibration.

        *samples* is an iterable of ``(branches, seconds)``. Only groups whose branches
        all belong to one dataset are used, since a mixed group cannot be attributed.
        Returns True when something changed, so the caller can re-pack what is still
        unsubmitted.

        A ``--test`` run records nothing: it processes a prefix of every file, so its
        durations say nothing about the cost of the whole one, and the store is shared
        with the production runs of the same version.
        """
        if self.test > 0:
            return False
        model = self.load_cost_model()
        branch_map = self.branch_map
        file_info = {}
        changed = False
        for branches, seconds in samples:
            datasets = {branch_map[b][0] for b in branches if b in branch_map}
            if len(datasets) != 1:
                continue
            dataset_name = datasets.pop()
            nano_version = self.get_nano_version(dataset_name)
            if dataset_name not in file_info:
                file_info[dataset_name] = self._input_file_info(dataset_name)
            n_events = 0.0
            for b in branches:
                if b not in branch_map:
                    continue
                events, _ = model.n_events(
                    nano_version,
                    dataset_name,
                    file_info[dataset_name].get(branch_map[b][1]),
                )
                if events is None:
                    n_events = 0.0
                    break
                n_events += events
            if n_events <= 0:
                continue
            changed |= model.add_measurement(
                nano_version, dataset_name, seconds, n_events
            )
        if changed:
            model.save(self.cost_model_path())
        return changed

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return InputFileTask.WF_complete(self)

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_idx = 0
        branches = {}
        for dataset_id, dataset_name in self.iter_datasets():
            input_file_list = (
                InputFileTask.req(self, branch=dataset_id, branches=()).output().abspath
            )
            input_files = InputFileTask.load_input_files(
                input_file_list, test=self.test > 0
            )

            for input_file_idx, input_file in enumerate(input_files):
                output_name = f"anaTupleFile_{input_file_idx}"
                branches[branch_idx] = (
                    dataset_name,
                    input_file,
                    output_name,
                )
                branch_idx += 1
        return branches

    @workflow_condition.output
    def output(self):
        dataset_name, _, output_name = self.branch_data
        output_path = os.path.join(
            self.version, "AnaTuples_split", self.period, dataset_name
        )
        root_output = os.path.join(output_path, f"{output_name}.root")
        report_output = os.path.join(output_path, f"{output_name}.json")
        return {
            "root": self.remote_target(root_output, fs=self.fs_anaTuple),
            "report": self.remote_target(report_output, fs=self.fs_anaTuple),
        }

    def run(self):
        with ServiceThread() as service_thread:
            dataset_name, input_file_name, output_name = self.branch_data
            dataset = self.datasets[dataset_name]
            process_group = dataset["process_group"]

            fs_nanoAOD, _, _ = self.get_fs_nanoAOD(dataset_name)
            input_file = self.remote_target(input_file_name, fs=fs_nanoAOD)

            job_home, remove_job_home = self.law_job_home()
            print(f"dataset_name: {dataset_name}")
            print(f"process_group: {process_group}")
            print(f"input_file = {input_file.uri()}")

            print("step 1: nanoAOD -> raw anaTuples")
            outdir_anatuples = os.path.join(job_home, "rawAnaTuples")
            reportFileName = "report.json"
            rawReportPath = os.path.join(outdir_anatuples, reportFileName)
            input_ok = True
            with contextlib.ExitStack() as stack:
                local_input = stack.enter_context(input_file.localize("r")).abspath
                inFileName = os.path.basename(input_file.abspath)
                print(f"inFileName {inFileName}")
                # A NanoAOD file with no events cannot be processed by the producer;
                # treat it like a corrupted input and emit an empty anaTuple + invalid
                # report so it is skipped when building the merge plan.
                if get_tree_entries(local_input, verbose=1) == 0:
                    input_ok = False
                    print(
                        f"{inFileName}: input file has 0 entries. "
                        "Will create empty anaTuple and report."
                    )
                else:
                    anatuple_cmd = self.anatuple_cmd(
                        dataset_name,
                        local_input,
                        inFileName,
                        outdir_anatuples,
                        output_name,
                        rawReportPath,
                    )
                    try:
                        ps_call(anatuple_cmd, env=self.producer_env(), verbose=1)
                    except PsCallError as e:
                        print(f"anaTupleProducer failed: {e}")
                        print("Checking input file integrity...")
                        input_ok = check_root_file_integrity(local_input, verbose=1)
                        if input_ok:
                            raise RuntimeError("anaTupleProducer failed.")
                        print(
                            "Input file is corrupted. Will create empty anaTuple and report."
                        )

            producer_fuseTuples = os.path.join(
                self._flaf_root(), "AnaProd", "FuseAnaTuples.py"
            )
            outdir_fusedTuples = os.path.join(job_home, "fusedAnaTuples")
            outFileName = os.path.basename(input_file.abspath)
            outFilePath = os.path.join(outdir_fusedTuples, outFileName)
            finalReportPath = os.path.join(outdir_fusedTuples, reportFileName)
            if input_ok:
                print("step 2: raw anaTuples -> fused anaTuples")
                verbosity = "1"
                fuseTuple_cmd = [
                    "python",
                    "-u",
                    producer_fuseTuples,
                    "--input-config",
                    rawReportPath,
                    "--work-dir",
                    outdir_fusedTuples,
                    "--tuple-output",
                    outFileName,
                    "--report-output",
                    reportFileName,
                    "--verbose",
                    verbosity,
                ]
                ps_call(fuseTuple_cmd, verbose=1)
            else:
                os.makedirs(outdir_fusedTuples, exist_ok=True)
                Path(outFilePath).touch()
                report = {
                    "valid": False,
                    "nano_file_name": inFileName,
                    "anaTuple_file_name": output_name,
                    "dataset_name": dataset_name,
                }
                with open(finalReportPath, "w") as f:
                    json.dump(report, f, indent=2)

            with self.output()["root"].localize("w") as local_file:
                shutil.move(outFilePath, local_file.abspath)
            with self.output()["report"].localize("w") as local_file:
                shutil.move(finalReportPath, local_file.abspath)

            if remove_job_home:
                shutil.rmtree(job_home)


class AnaTupleCostProbeTask(
    AnaTupleProducerMixin, Task, HTCondorWorkflow, CrabWorkflow, law.LocalWorkflow
):
    """Times the AnaTuple producer on a short prefix of one file per dataset.

    The per-event cost of a dataset spans more than an order of magnitude (a dilepton skim
    selects half of its events, a hadronic one a few percent) and depends on the analysis
    selection, so `AnaTupleFileTask` measures it instead of guessing: a few thousand events
    take minutes and turn a blind packing into an informed one.

    Outputs are keyed by version and nano source but deliberately not by era, so a
    multi-era production probes each dataset once and the later eras skip this stage.
    A probe that fails still writes a result, marked not ok: calibration is an
    optimisation and must never block production.
    """

    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 3.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    @staticmethod
    def probe_target(task, dataset_name):
        return task.remote_target(
            task.version,
            "AnaTupleCost",
            task.get_nano_version(dataset_name),
            f"{dataset_name}.json",
            fs=task.fs_anaTuple,
        )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["inputFile"] = InputFileTask.req(self, branches=())
        return reqs

    def requires(self):
        return []

    def create_branch_map(self):
        return {
            idx: dataset_name
            for idx, (_, dataset_name) in enumerate(self.iter_datasets())
        }

    def output(self):
        return AnaTupleCostProbeTask.probe_target(self, self.branch_data)

    def probe_input(self, dataset_name):
        """(file name, metadata) of the file to time: the median-sized one when sizes are
        known, so a truncated first file does not distort events-per-byte."""
        input_file_list = (
            InputFileTask.req(
                self, branch=self.get_dataset_id(dataset_name), branches=()
            )
            .output()
            .abspath
        )
        input_files = InputFileTask.load_input_files(
            input_file_list, test=self.test > 0
        )
        if not input_files:
            raise RuntimeError(f"no input files for {dataset_name}")
        file_info = InputFileTask.load_file_info(input_file_list)
        sized = sorted(
            (info["size"], name)
            for name, info in file_info.items()
            if name in set(input_files) and info.get("size")
        )
        name = sized[len(sized) // 2][1] if sized else input_files[0]
        return name, file_info.get(name, {})

    def run(self):
        dataset_name = self.branch_data
        probe_events = int(self.cost_params()["probe_events"])
        if self.test > 0:
            probe_events = min(probe_events, self.test)
        result = {
            "ok": False,
            "dataset_name": dataset_name,
            "era": self.period,
            "probe_events": probe_events,
        }
        job_home, remove_job_home = self.law_job_home()
        try:
            input_file_name, info = self.probe_input(dataset_name)
            result["input_file"] = input_file_name
            result["input_size"] = info.get("size")
            fs_nanoAOD, _, _ = self.get_fs_nanoAOD(dataset_name)
            input_file = self.remote_target(input_file_name, fs=fs_nanoAOD)
            out_dir = os.path.join(job_home, "probe")
            report_path = os.path.join(out_dir, "report.json")
            with input_file.localize("r") as local_file:
                cmd = self.anatuple_cmd(
                    dataset_name,
                    local_file.abspath,
                    os.path.basename(input_file.abspath),
                    out_dir,
                    "probe",
                    report_path,
                    extra_args=["--max-scan-events", str(probe_events)],
                )
                # Retried once: a probe result is written even when it fails, and it is
                # reused by every era of this version, so a transient failure would
                # otherwise cost the dataset its calibration for the whole production.
                for attempt in range(2):
                    try:
                        ps_call(cmd, env=self.producer_env(), verbose=1)
                        break
                    except PsCallError:
                        if attempt == 1:
                            raise
                        print(
                            f"{dataset_name}: probe attempt {attempt + 1} failed, retrying"
                        )
            with open(report_path, "r") as f:
                report = json.load(f)
            for key in (
                "n_scanned_events",
                "n_original_events",
                "loop_seconds",
                "setup_seconds",
                "n_trees",
            ):
                result[key] = report.get(key)
            result["ok"] = bool(result["n_scanned_events"]) and bool(
                result["loop_seconds"]
            )
        except Exception as e:
            result["error"] = str(e)
            print(
                f"!!! cost probe failed for {dataset_name}: {e}\n"
                "!!! production will fall back to a coarser cost estimate for it."
            )

        with self.output().localize("w") as local_output:
            with open(local_output.abspath, "w") as f:
                json.dump(result, f, indent=2)
        if remove_job_home:
            shutil.rmtree(job_home, ignore_errors=True)


class AnaTupleFileListBuilderTask(
    Task, HTCondorWorkflow, CrabWorkflow, law.LocalWorkflow
):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 24.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    bundle_flavours = ["core", "inputFileList"]

    def __init__(self, *args, **kwargs):
        ana_v = kwargs.get("ana_version") or kwargs.get("anaTuple_version")
        if ana_v:
            kwargs["version"] = ana_v
        super(AnaTupleFileListBuilderTask, self).__init__(*args, **kwargs)

    _anaTuple_map_cache = None

    @classmethod
    def _get_anaTuple_map(cls, ref_task):
        if cls._anaTuple_map_cache is None:
            cls._anaTuple_map_cache = AnaTupleFileTask.req(
                ref_task, branch=-1, branches=()
            ).create_branch_map()
        return cls._anaTuple_map_cache

    def workflow_requires(self):
        # When the Builder's own outputs (plan + reports on fs_anaTuple at this version) already
        # exist (central production case), we do not need to run this task at all.
        # Return empty early (before super, which would pull bundles when --bundle is used,
        # and before any production logic). This prevents bundle flavours like "inputFileList"
        # (whose requires can pull InputFileTask) from being required for an "existent" Builder.
        if self.complete():
            return {}

        reqs = super().workflow_requires()

        input_file_task_complete = InputFileTask.WF_complete(self)
        if not input_file_task_complete:
            reqs["anaTuple"] = AnaTupleFileTask.req(self, branches=())
            reqs["inputFile"] = InputFileTask.req(self, branches=())
            return reqs

        if not isinstance(self._get_anaTuple_map(self), dict):
            return reqs
        branch_set = set()
        for idx, (dataset_name, process_group) in self.branch_map.items():
            branch_set |= self._get_branch_set_for_dataset(dataset_name, process_group)

        reqs["AnaTupleFileTask"] = AnaTupleFileTask.req(
            self,
            version=self.version,
            branches=tuple(branch_set),
            max_runtime=AnaTupleFileTask.max_runtime._default,
            n_cpus=AnaTupleFileTask.n_cpus._default,
        )
        return reqs

    def _get_branch_set_for_dataset(self, dataset_name, process_group):
        AnaTuple_map = self._get_anaTuple_map(self)
        branch_set = set()
        for br_idx, (anaTuple_dataset_name, _, _) in AnaTuple_map.items():
            match = dataset_name == anaTuple_dataset_name
            if not match and process_group == "data":
                anaTuple_dataset = self.datasets[anaTuple_dataset_name]
                anaTuple_process_group = anaTuple_dataset["process_group"]
                match = anaTuple_process_group == "data"
            if match:
                branch_set.add(br_idx)
        return branch_set

    def requires(self):
        dataset_name, process_group = self.branch_data
        # Prune production sources when our plan already exists on the (central) target.
        # Complements the workflow_requires early return; prevents the per-file AnaTupleFileTask
        # (and thus their InputFileTask deps via workflow_condition) from appearing in the graph.
        if self.complete():
            return []
        if not InputFileTask.WF_complete(self):
            return []
        branch_set = self._get_branch_set_for_dataset(dataset_name, process_group)
        # leaf gets explicit from resolved version (set via ana in our __init__ or per-task for Builder)
        return [
            AnaTupleFileTask.req(
                self,
                version=self.version,
                max_runtime=AnaTupleFileTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
            )
            for prod_br in tuple(branch_set)
        ]

    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        branches = {}
        k = 0
        data_done = False
        for dataset_id, dataset_name in self.iter_datasets():
            dataset = self.datasets[dataset_name]
            process_group = dataset["process_group"]
            if process_group == "data":
                if data_done:
                    continue  # Will have multiple data datasets, but only need one branch
                dataset_name = "data"
                data_done = True
            branches[k] = (dataset_name, process_group)
            k += 1
        return branches

    def get_output_path(self, dataset_name, output_name):
        output_file = f"{dataset_name}.json"
        base_name = "AnaTupleFileList"
        if output_name != "plan":
            base_name += f"_{output_name}"
        return os.path.join(self.version, base_name, self.period, output_file)

    def output(self):
        dataset_name, process_group = self.branch_data
        outputs = {}
        for output_name in ["plan", "reports"]:
            output_path = self.get_output_path(dataset_name, output_name)
            outputs[output_name] = self.remote_target(output_path, fs=self.fs_anaTuple)
        return outputs

    def run(self):
        dataset_name, process_group = self.branch_data
        with contextlib.ExitStack() as stack:

            print("Localizing inputs")
            local_inputs = [
                stack.enter_context(inp["report"].localize("r")).abspath
                for inp in self.input()
            ]
            print(f"Localized {len(local_inputs)} inputs")

            job_home, remove_job_home = self.law_job_home()

            nEventsPerFile = self.setup.global_params.get(
                "nEventsPerFile", {"data": 1_000_000}
            )
            if isinstance(nEventsPerFile, dict):
                nEventsPerFile = nEventsPerFile.get(process_group, 100_000)
            is_data = process_group == "data"

            raw_result = CreateMergePlan(
                setup=self.setup,
                local_inputs=local_inputs,
                n_events_per_file=nEventsPerFile,
                is_data=is_data,
            )

            result = {
                "plan": raw_result["plan"],
                "reports": {
                    "reports": raw_result["reports"],
                    "ignored_files": raw_result["ignored_files"],
                    "n_events_plan": raw_result["n_events_plan"],
                    "n_events_ignored": raw_result["n_events_ignored"],
                },
            }

            for output_name, output_remote in self.output().items():
                output_path_tmp = os.path.join(job_home, f"{output_name}_tmp.json")
                with open(output_path_tmp, "w") as f:
                    json.dump(result[output_name], f, indent=2)
                with output_remote.localize("w") as output_localized:
                    shutil.move(output_path_tmp, output_localized.abspath)

            if remove_job_home:
                shutil.rmtree(job_home)


class AnaTupleFileListTask(AnaTupleFileListBuilderTask):
    bundle_flavours = []
    # This task only copies the Builder's plan into a local_target, so it is forced to run locally
    # (its output must land on the shared FS, not an ephemeral HTCondor worker). Forcing
    # workflow="local" would otherwise leak through req() into the upstream Builder and the heavy
    # AnaTupleFileTask, making production run locally too. This carrier parameter remembers the
    # originally requested workflow and is forwarded to the Builder, so Builder and AnaTupleFileTask
    # run on the requested workflow while this task stays local. It is a parameter (not an attribute)
    # so it survives req_branch and is identical for the workflow and all its branches, keeping
    # workflow_requires() and requires() consistent; insignificant so it never affects the task id.
    upstream_workflow = luigi.Parameter(default=law.NO_STR, significant=False)

    def __init__(self, *args, **kwargs):
        ana_v = kwargs.get("ana_version") or kwargs.get("anaTuple_version")
        if ana_v:
            kwargs["version"] = ana_v
        if kwargs.get("upstream_workflow", law.NO_STR) in (law.NO_STR, None):
            kwargs["upstream_workflow"] = kwargs.get("workflow") or law.NO_STR
        kwargs["workflow"] = "local"
        super(AnaTupleFileListTask, self).__init__(*args, **kwargs)

    def workflow_requires(self):
        return {
            "AnaTupleFileListBuilderTask": AnaTupleFileListBuilderTask.req(
                self, workflow=self.upstream_workflow
            )
        }

    def requires(self):
        return AnaTupleFileListBuilderTask.req(self, workflow=self.upstream_workflow)

    def output(self):
        dataset_name, process_group = self.branch_data
        return self.local_target(f"{dataset_name}.json")

    def run(self):
        with self.input()["plan"].localize("r") as input_local:
            self.output().makedirs()
            shutil.copy(input_local.abspath, self.output().abspath)


class AnaTupleMergeTask(Task, HTCondorWorkflow, CrabWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 48.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)
    delete_inputs_after_merge = luigi.BoolParameter(default=False)
    bundle_flavours = ["core", "inputFileList", "AnaTupleFileList"]

    def __init__(self, *args, **kwargs):
        ana_v = kwargs.get("ana_version") or kwargs.get("anaTuple_version")
        if ana_v:
            kwargs["version"] = ana_v
        super(AnaTupleMergeTask, self).__init__(*args, **kwargs)

    def workflow_requires(self):
        # If this merge's output already exists on the target (central for dev-on-existing),
        # we don't need to run it or its bundles/production organization.
        if self.complete():
            return {}

        reqs = super().workflow_requires()
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            reqs["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
                self,
                version=self.version,
                branches=(),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
            return reqs

        branch_set = set()
        for _, (
            _,
            _,
            ds_branch,
            dataset_dependencies,
            _,
            _,
            _,
            _,
        ) in self.branch_map.items():
            branch_set.add(ds_branch)
            branch_set.update(dataset_dependencies.values())

        reqs["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
            self,
            version=self.version,
            branches=tuple(branch_set),
            max_runtime=AnaTupleFileListTask.max_runtime._default,
            n_cpus=AnaTupleFileListTask.n_cpus._default,
        )
        return reqs

    def requires(self):
        # Need both the AnaTupleFileTask for the input ROOT file, and the AnaTupleFileListTask for the json structure
        (
            dataset_name,
            process_group,
            ds_branch,
            dataset_dependencies,
            input_file_list,
            _,
            skip_future_tasks,
            runs,
        ) = self.branch_data
        # When the merged anaTuple already exists on the target (central for dev-on-existing-anaTuples,
        # or after a previous successful merge), do not pull the per-file production (FileTask + InputFileTask).
        # This (together with the Builder/List pruning) stops the unwanted InputFileTask deps from
        # appearing in HistTupleProducerTask / cache graphs when using --AnaTuple*Task-version on central.
        if self.complete():
            return {"root": {}, "json": {}}
        if not InputFileTask.WF_complete(self):
            return {"root": {}, "json": {}}
        anaTuple_branch_map = AnaTupleFileTask.req(
            self, version=self.version, branch=-1, branches=()
        ).create_branch_map()
        required_branches = {"root": {}}
        if not isinstance(anaTuple_branch_map, dict):
            return required_branches
        for prod_br, (
            anaTuple_dataset_name,
            anaTuple_input_file,
            anaTuple_output_name,
        ) in anaTuple_branch_map.items():
            match = dataset_name == anaTuple_dataset_name
            if not match and process_group == "data":
                anaTuple_dataset = self.datasets[anaTuple_dataset_name]
                anaTuple_process_group = anaTuple_dataset["process_group"]
                match = anaTuple_process_group == "data"
            dependency_type = None
            if match:
                key = f"{anaTuple_dataset_name}/{anaTuple_output_name}"
                if key in input_file_list:
                    dependency_type = "root"
            if dependency_type:
                if anaTuple_dataset_name not in required_branches[dependency_type]:
                    required_branches[dependency_type][anaTuple_dataset_name] = []
                required_branches[dependency_type][anaTuple_dataset_name].append(
                    AnaTupleFileTask.req(
                        self,
                        version=self.version,
                        max_runtime=AnaTupleFileTask.max_runtime._default,
                        branch=prod_br,
                        branches=(prod_br,),
                    )
                )

        required_branches["json"] = {}
        if process_group != "data":
            anaTupleFileListBuilder_branch_map = AnaTupleFileListBuilderTask.req(
                self, branch=-1, branches=()
            ).create_branch_map()

            for builder_branch, (
                builder_dataset_name,
                _,
            ) in anaTupleFileListBuilder_branch_map.items():
                if (
                    builder_dataset_name == dataset_name
                    or builder_dataset_name in dataset_dependencies
                ):
                    required_branches["json"][builder_dataset_name] = (
                        AnaTupleFileListBuilderTask.req(
                            self,
                            version=self.version,
                            max_runtime=AnaTupleFileListBuilderTask.max_runtime._default,
                            branch=builder_branch,
                            branches=(builder_branch,),
                        )
                    )

        return required_branches

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        branches = {}
        nBranch = 0
        # Use req(self) for controlled FileList ds map; version via copy of ana* or per-task on FileList.
        ds_branch_map = AnaTupleFileListTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()

        ds_branches = {}
        for ds_branch, (dataset_name, process_group) in ds_branch_map.items():
            if dataset_name in ds_branches:
                raise RuntimeError(
                    f"Dataset {dataset_name} appears multiple times in AnaTupleFileListTask branch map!"
                )
            ds_branches[dataset_name] = ds_branch

        for ds_branch, (dataset_name, process_group) in ds_branch_map.items():
            dataset_dependencies = self.collect_extra_dependencies(
                dataset_name, ds_branches, process_group
            )
            this_dataset_dict = self.setup.getAnaTupleFileList(
                dataset_name,
                AnaTupleFileListTask.req(self, branch=ds_branch, branches=()).output(),
            )
            for this_dict in this_dataset_dict:
                input_file_list = this_dict["inputs"]
                output_file_list = this_dict["outputs"]
                skip_future_tasks = this_dict["n_events"] == 0
                runs = this_dict.get("runs", [])
                branches[nBranch] = (
                    dataset_name,
                    process_group,
                    ds_branch,
                    dataset_dependencies,
                    input_file_list,
                    output_file_list,
                    skip_future_tasks,
                    runs,
                )
                nBranch += 1
        return branches

    def collect_extra_dependencies(self, dataset_name, ds_branches, process_group):
        other_datasets = {}
        if process_group != "data":
            dataset = self.datasets[dataset_name]
            processors = self.setup.get_processors(
                dataset["process_name"], stage="AnaTupleMerge"
            )
            require_whole_process = any(
                p.get("dependency_level", {}).get("AnaTupleMerge", "file") == "process"
                for p in processors
            )
            if require_whole_process:
                process = self.setup.base_processes[dataset["process_name"]]
                for p_dataset_name in process.get("datasets", []):
                    if p_dataset_name != dataset_name:
                        other_datasets[p_dataset_name] = ds_branches[p_dataset_name]
        return other_datasets

    def _branch_output_targets(self, branch_data):
        dataset_name = branch_data[0]
        output_file_list = branch_data[5]
        output_dir = os.path.join(self.version, "AnaTuples", self.period, dataset_name)
        return [
            self.remote_target(os.path.join(output_dir, out_file), fs=self.fs_anaTuple)
            for out_file in output_file_list
        ]

    def all_branch_outputs(self):
        """{branch -> [output targets]} for the whole workflow, built directly from the
        branch map without instantiating a task per branch. Downstream tasks
        (HistTupleProducer, AnalysisCache) use this to derive their output names cheaply
        instead of resolving the anaTuple requirement per branch (O(nBranches) each)."""
        return {
            br: self._branch_output_targets(branch_data)
            for br, branch_data in self.branch_map.items()
        }

    @workflow_condition.output
    def output(self):
        return self._branch_output_targets(self.branch_data)

    def run(self):
        (
            dataset_name,
            process_group,
            ds_branch,
            dataset_dependencies,
            input_file_list,
            output_file_list,
            skip_future_tasks,
            runs,
        ) = self.branch_data
        is_data = process_group == "data"
        job_home, remove_job_home = self.law_job_home()
        tmpFiles = [
            os.path.join(job_home, f"AnaTupleMergeTask_{dataset_name}_{i}.root")
            for i in range(len(self.output()))
        ]
        print(f"dataset: {dataset_name}")
        with contextlib.ExitStack() as stack:

            print("Localizing root inputs")
            local_root_inputs = []
            for ds_name, files in self.input()["root"].items():
                for file_list in files:
                    local_input = stack.enter_context(
                        file_list["root"].localize("r")
                    ).abspath
                    local_root_inputs.append(local_input)
            print(f"Localized {len(local_root_inputs)} root inputs")

            print("Localizing reports")
            reports = {}
            for ds_name, file_list in self.input()["json"].items():
                report_file = stack.enter_context(
                    file_list["reports"].localize("r")
                ).abspath
                with open(report_file, "r") as f:
                    ds_details = yaml.safe_load(f)

                if "reports" in ds_details:
                    ignored_files = set(ds_details["ignored_files"])
                    ds_reports = ds_details["reports"]
                else:
                    # workaround for a backward compatibility with the old report format
                    ignored_files = set()
                    ds_reports = ds_details
                selected_ds_reports = [
                    report
                    for key, report in ds_reports.items()
                    if key not in ignored_files
                ]
                print(
                    f"  {ds_name}: selected {len(selected_ds_reports)} out of {len(ds_reports)} reports"
                )
                reports[ds_name] = selected_ds_reports
            print(f"Localized reports from {len(reports)} datasets")

            mergeAnaTuples(
                setup=self.setup,
                dataset_name=dataset_name,
                is_data=is_data,
                work_dir=job_home,
                input_reports=reports,
                input_roots=local_root_inputs,
                root_outputs=tmpFiles,
                runs=runs,
            )

        for outFile, tmpFile in zip(self.output(), tmpFiles):
            with outFile.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.abspath
                shutil.move(tmpFile, out_local_path)

        if self.delete_inputs_after_merge:
            print(f"Finished merging, lets delete remote AnaTupleFile targets")
            for ds_name, files in self.input()["root"].items():
                for remote_targets in files:
                    for target in remote_targets:
                        target.remove()

        if remove_job_home:
            shutil.rmtree(job_home)
