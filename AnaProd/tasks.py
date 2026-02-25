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
)
from FLAF.run_tools.law_customizations import Task, HTCondorWorkflow, copy_param
from FLAF.Common.Utilities import getCustomisationSplit, ServiceThread
from .AnaTupleFileList import CreateMergePlan
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
        print(f"{dataset_name}: creating input file list into {self.output().path}")
        dataset = self.datasets[dataset_name]
        process_group = dataset["process_group"]
        ignore_missing = self.global_params.get("ignore_missing_nanoAOD_files", {}).get(
            process_group, False
        )
        fs_nanoAOD, folder_name, include_folder_name = self.get_fs_nanoAOD(dataset_name)
        nano_version = self.get_nano_version(dataset_name)
        pattern_dict = self.datasets[dataset_name].get("fileNamePattern", {})
        pattern = pattern_dict.get(nano_version, r".*\.root$")
        input_files = []
        inactive_files = []
        for file in fs_nanoAOD.listdir(folder_name):
            if not re.match(pattern, file):
                continue
            file_path = os.path.join(folder_name, file) if include_folder_name else file
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

        if len(input_files) == 0:
            raise RuntimeError(f"No input files found for {dataset_name}")

        input_files = natural_sort(input_files)
        output = {
            "input_files": input_files,
            "inactive_files": inactive_files,
        }
        with self.output().localize("w") as out_local_file:
            with open(out_local_file.path, "w") as f:
                json.dump(output, f, indent=2)

        print(f"{dataset_name}: {len(input_files)} input files are found.")

    input_file_cache = {}

    @staticmethod
    def load_input_files(input_file_list, test=False):
        if input_file_list not in InputFileTask.input_file_cache:
            with open(input_file_list, "r") as f:
                input_files = json.load(f)["input_files"]
            InputFileTask.input_file_cache[input_file_list] = input_files
        input_files = InputFileTask.input_file_cache[input_file_list]
        active_files = (
            [input_files[0]] if test and len(input_files) > 0 else input_files
        )
        return active_files

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


class AnaTupleFileTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 40.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)

    def workflow_requires(self):
        return {
            "inputFile": InputFileTask.req(self, branches=()),
        }

    def requires(self):
        return []

    _req_params = None

    @classmethod
    def req(cls, inst, **kwargs):
        if cls._req_params is None:
            cls._req_params = cls.req_params(inst, **kwargs)
        for param_name in ["branch", "branches"]:
            param_value = kwargs.get(param_name, getattr(inst, param_name))
            cls._req_params[param_name] = param_value
        return cls(**cls._req_params)

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return InputFileTask.WF_complete(self)

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_idx = 0
        branches = {}
        for dataset_id, dataset_name in self.iter_datasets():
            input_file_list = (
                InputFileTask.req(self, branch=dataset_id, branches=()).output().path
            )
            input_files = InputFileTask.load_input_files(
                input_file_list, test=self.test > 0
            )

            for input_file in input_files:
                output_name = f"anaTupleFile_{branch_idx}"
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
            producer_anatuples = os.path.join(
                self.ana_path(), "FLAF", "AnaProd", "anaTupleProducer.py"
            )

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

            fs_nanoAOD, _, _ = self.get_fs_nanoAOD(dataset_name)
            input_file = self.remote_target(input_file_name, fs=fs_nanoAOD)

            job_home, remove_job_home = self.law_job_home()
            print(f"dataset_name: {dataset_name}")
            print(f"process_group: {process_group}")
            print(f"input_file = {input_file.uri()}")

            print("step 1: nanoAOD -> raw anaTuples")
            outdir_anatuples = os.path.join(job_home, "rawAnaTuples")
            anaTupleDef = os.path.join(
                self.ana_path(), self.global_params["anaTupleDef"]
            )
            reportFileName = "report.json"
            rawReportPath = os.path.join(outdir_anatuples, reportFileName)
            input_ok = True
            with contextlib.ExitStack() as stack:
                local_input = stack.enter_context(input_file.localize("r")).path
                inFileName = os.path.basename(input_file.path)
                print(f"inFileName {inFileName}")
                anatuple_cmd = [
                    "python3",
                    "-u",
                    producer_anatuples,
                    "--period",
                    self.period,
                    "--inFile",
                    local_input,
                    "--outDir",
                    outdir_anatuples,
                    "--dataset",
                    dataset_name,
                    "--anaTupleDef",
                    anaTupleDef,
                    "--channels",
                    channels,
                    "--inFileName",
                    inFileName,
                    "--reportOutput",
                    rawReportPath,
                    "--LAWrunVersion",
                    self.version,
                    "--output-name",
                    output_name,
                ]
                if compute_unc_variations:
                    anatuple_cmd.append("--compute-unc-variations")
                if store_noncentral:
                    anatuple_cmd.append("--store-noncentral")

                if self.test > 0:
                    anatuple_cmd.extend(["--nEvents", str(self.test)])
                env = None
                if self.global_params.get("use_cmssw_env_AnaTupleProduction", False):
                    env = self.cmssw_env
                try:
                    ps_call(anatuple_cmd, env=env, verbose=1)
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
                self.ana_path(), "FLAF", "AnaProd", "FuseAnaTuples.py"
            )
            outdir_fusedTuples = os.path.join(job_home, "fusedAnaTuples")
            outFileName = os.path.basename(input_file.path)
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
                shutil.move(outFilePath, local_file.path)
            with self.output()["report"].localize("w") as local_file:
                shutil.move(finalReportPath, local_file.path)

            if remove_job_home:
                shutil.rmtree(job_home)


class AnaTupleFileListBuilderTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 24.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def workflow_requires(self):
        input_file_task_complete = InputFileTask.WF_complete(self)
        if not input_file_task_complete:
            return {
                "anaTuple": AnaTupleFileTask.req(self, branches=()),
                "inputFile": InputFileTask.req(self, branches=()),
            }

        AnaTuple_map = AnaTupleFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        branch_set = set()
        for idx, (dataset_name, process_group) in self.branch_map.items():
            for br_idx, (anaTuple_dataset_name, _, _) in AnaTuple_map.items():
                match = dataset_name == anaTuple_dataset_name
                if not match and process_group == "data":
                    anaTuple_dataset = self.datasets[anaTuple_dataset_name]
                    anaTuple_process_group = anaTuple_dataset["process_group"]
                    match = anaTuple_process_group == "data"
                if match:
                    branch_set.add(br_idx)

        deps = {
            "AnaTupleFileTask": AnaTupleFileTask.req(
                self,
                branches=tuple(branch_set),
                max_runtime=AnaTupleFileTask.max_runtime._default,
                n_cpus=AnaTupleFileTask.n_cpus._default,
            )
        }
        return deps

    def requires(self):
        dataset_name, process_group = self.branch_data
        AnaTuple_map = AnaTupleFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        branch_set = set()
        for br_idx, (anaTuple_dataset_name, _, _) in AnaTuple_map.items():
            match = dataset_name == anaTuple_dataset_name
            if not match and process_group == "data":
                anaTuple_dataset = self.datasets[anaTuple_dataset_name]
                anaTuple_process_group = anaTuple_dataset["process_group"]
                match = anaTuple_process_group == "data"
            if match:
                branch_set.add(br_idx)

        reqs = [
            AnaTupleFileTask.req(
                self,
                max_runtime=AnaTupleFileTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
            )
            for prod_br in tuple(branch_set)
        ]
        return reqs

    def create_branch_map(self):
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
                stack.enter_context(inp["report"].localize("r")).path
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

            result = CreateMergePlan(self.setup, local_inputs, nEventsPerFile, is_data)

            for output_name, output_remote in self.output().items():
                output_path_tmp = os.path.join(job_home, f"{output_name}_tmp.json")
                with open(output_path_tmp, "w") as f:
                    json.dump(result[output_name], f, indent=2)
                with output_remote.localize("w") as output_localized:
                    shutil.move(output_path_tmp, output_localized.path)

            if remove_job_home:
                shutil.rmtree(job_home)


class AnaTupleFileListTask(AnaTupleFileListBuilderTask):
    def workflow_requires(self):
        return {"AnaTupleFileListBuilderTask": AnaTupleFileListBuilderTask.req(self)}

    def requires(self):
        return AnaTupleFileListBuilderTask.req(self)

    def output(self):
        dataset_name, process_group = self.branch_data
        return self.local_target(self.get_output_path(dataset_name, "plan"))

    def run(self):
        with self.input()["plan"].localize("r") as input_local:
            self.output().makedirs()
            shutil.copy(input_local.path, self.output().path)


class AnaTupleMergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 48.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)
    delete_inputs_after_merge = luigi.BoolParameter(default=False)

    def workflow_requires(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            return {
                "AnaTupleFileListTask": AnaTupleFileListTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleFileListTask.max_runtime._default,
                    n_cpus=AnaTupleFileListTask.n_cpus._default,
                ),
            }

        branch_set = set()
        for _, (
            _,
            _,
            ds_branch,
            dataset_dependencies,
            _,
            _,
            _,
        ) in self.branch_map.items():
            branch_set.add(ds_branch)
            branch_set.update(dataset_dependencies.values())

        return {
            "AnaTupleFileListTask": AnaTupleFileListTask.req(
                self,
                branches=tuple(branch_set),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
        }

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
        ) = self.branch_data
        anaTuple_branch_map = AnaTupleFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        required_branches = {"root": {}}
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
        branches = {}
        nBranch = 0
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
                branches[nBranch] = (
                    dataset_name,
                    process_group,
                    ds_branch,
                    dataset_dependencies,
                    input_file_list,
                    output_file_list,
                    skip_future_tasks,
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

    @workflow_condition.output
    def output(self):
        (
            dataset_name,
            process_group,
            ds_branch,
            dataset_dependencies,
            input_file_list,
            output_file_list,
            skip_future_tasks,
        ) = self.branch_data
        output_dir = os.path.join(self.version, "AnaTuples", self.period, dataset_name)
        outputs = [os.path.join(output_dir, out_file) for out_file in output_file_list]
        return [
            self.remote_target(out_path, fs=self.fs_anaTuple) for out_path in outputs
        ]

    def run(self):
        (
            dataset_name,
            process_group,
            ds_branch,
            dataset_dependencies,
            input_file_list,
            output_file_list,
            skip_future_tasks,
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
                    ).path
                    local_root_inputs.append(local_input)
            print(f"Localized {len(local_root_inputs)} root inputs")

            print("Localizing reports")
            reports = {}
            for ds_name, file_list in self.input()["json"].items():
                report_file = stack.enter_context(
                    file_list["reports"].localize("r")
                ).path
                with open(report_file, "r") as f:
                    ds_reports = yaml.safe_load(f)
                reports[ds_name] = list(ds_reports.values())
            print(f"Localized {len(reports)} reports")

            mergeAnaTuples(
                setup=self.setup,
                dataset_name=dataset_name,
                is_data=is_data,
                work_dir=job_home,
                input_reports=reports,
                input_roots=local_root_inputs,
                root_outputs=tmpFiles,
            )

        for outFile, tmpFile in zip(self.output(), tmpFiles):
            with outFile.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.path
                shutil.move(tmpFile, out_local_path)

        if self.delete_inputs_after_merge:
            print(f"Finished merging, lets delete remote AnaTupleFile targets")
            for ds_name, files in self.input()["root"].items():
                for remote_targets in files:
                    for target in remote_targets:
                        target.remove()

        if remove_job_home:
            shutil.rmtree(job_home)
