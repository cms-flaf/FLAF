import contextlib
import law
import luigi
import os
import shutil
import re

from FLAF.RunKit.run_tools import ps_call, natural_sort
from FLAF.run_tools.law_customizations import Task, HTCondorWorkflow, copy_param
from FLAF.Common.Utilities import getCustomisationSplit, ServiceThread, SerializeObjectToString


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
        return self.local_target("input_files", f"{dataset_name}.txt")

    def run(self):
        dataset_name = self.branch_data
        folder_name = (
            self.datasets[dataset_name]["dirName"]
            if "dirName" in self.datasets[dataset_name]
            else dataset_name
        )
        print(
            f"Creating inputFile for dataset {dataset_name} into {self.output().path}"
        )

        fs_nanoAOD = self.fs_nanoAOD
        if self.datasets[dataset_name].get("fs_nanoAOD", None) is not None:
            fs_nanoAOD = self.setup.get_fs(
                f"fs_nanoAOD_{dataset_name}", self.datasets[dataset_name]["fs_nanoAOD"]
            )
        if fs_nanoAOD is None:
            raise RuntimeError(f"fs_nanoAOD is not defined for dataset {dataset_name}")

        with self.output().localize("w") as out_local_file:
            input_files = []
            pattern = self.datasets[dataset_name].get("fileNamePattern", r".*\.root$")
            for file in natural_sort(fs_nanoAOD.listdir(folder_name)):
                if re.match(pattern, file):
                    input_files.append(file)
            with open(out_local_file.path, "w") as inputFileTxt:
                for input_line in input_files:
                    inputFileTxt.write(input_line + "\n")
        print(
            f"inputFile for dataset {dataset_name} is created in {self.output().path}"
        )

    @staticmethod
    def load_input_files(
        input_file_list, folder_name, fs=None, return_uri=False, test=False
    ):
        input_files = []
        with open(input_file_list, "r") as txt_file:
            for file in txt_file.readlines():
                file_path = os.path.join(folder_name, file.strip())
                file_full_path = fs.uri(file_path) if return_uri else file_path
                input_files.append(file_full_path)
        if len(input_files) == 0:
            raise RuntimeError(f"No input files found for {folder_name}")
        active_files = [input_files[0]] if test else input_files
        return active_files


class AnaTupleFileTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 40.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def workflow_requires(self):
        input_file_task_complete = InputFileTask.req(self, branches=()).complete()
        if not input_file_task_complete:
            return { "inputFile": InputFileTask.req(self, branches=()), }

        branches_set = set()
        for branch_idx, (
            dataset_id,
            _,
            _,
        ) in self.branch_map.items():
            branches_set.add(dataset_id)
        branches = tuple(branches_set)
        return { "inputFile": InputFileTask.req(self, branches=branches), }

    def requires(self):
        return []

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return InputFileTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branch_idx = 0
        branches = {}
        for dataset_id, dataset_name in self.iter_datasets():
            fs_nanoAOD = self.fs_nanoAOD
            dataset = self.datasets[dataset_name]

            if dataset.get("fs_nanoAOD", None) is not None:
                fs_nanoAOD = self.setup.get_fs(
                    f"fs_nanoAOD_{dataset_name}", dataset["fs_nanoAOD"]
                )
            dir_to_list = dataset.get("dirName", dataset_name)
            input_file_list = (
                InputFileTask.req(self, branch=dataset_id, branches=(dataset_id,))
                .output()
                .path
            )
            input_files = InputFileTask.load_input_files(
                input_file_list, dir_to_list, test=self.test > 0
            )
            if fs_nanoAOD is None:
                raise RuntimeError(
                    f"fs_nanoAOD is not defined for dataset {dataset_name}"
                )
            for input_file in input_files:
                fileintot = self.remote_target(input_file, fs=fs_nanoAOD)
                branches[branch_idx] = (
                    dataset_id,
                    dataset_name,
                    fileintot,
                )
                branch_idx += 1
        return branches

    @workflow_condition.output
    def output(self):
        _, dataset_name, input_file = self.branch_data
        output_name = os.path.basename(input_file.path)
        json_name = f"{output_name.split('.')[0]}.json"
        output_path = os.path.join(
            "anaTuples", self.version, self.period, dataset_name, "split", output_name
        )
        json_path = os.path.join(
            "anaTuples", self.version, self.period, dataset_name, "split", json_name
        )
        return [
            self.remote_target(output_path, fs=self.fs_anaTuple),
            self.remote_target(json_path, fs=self.fs_anaTuple),
        ]

    def run(self):
        with ServiceThread() as service_thread:
            dataset_id, dataset_name, input_file = self.branch_data
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

            job_home, remove_job_home = self.law_job_home()
            print(f"dataset_id: {dataset_id}")
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
                ps_call(anatuple_cmd, env=env, verbose=1)

            print("step 2: raw anaTuples -> fused anaTuples")
            producer_fuseTuples = os.path.join(
                self.ana_path(), "FLAF", "AnaProd", "FuseAnaTuples.py"
            )
            outdir_fusedTuples = os.path.join(job_home, "fusedAnaTuples")
            outFileName = os.path.basename(input_file.path)
            outFilePath = os.path.join(outdir_fusedTuples, outFileName)
            finalReportPath = os.path.join(outdir_fusedTuples, reportFileName)
            # verbosity = "2" if self.test > 0 else "1"
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

            tuple_output = self.output()[0]
            report_output = self.output()[1]
            with tuple_output.localize("w") as local_file:
                shutil.move(outFilePath, local_file.path)
            with report_output.localize("w") as local_file:
                shutil.move(finalReportPath, local_file.path)

            if remove_job_home:
                shutil.rmtree(job_home)


class AnaTupleFileListBuilderTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def workflow_requires(self):
        input_file_task_complete = InputFileTask.req(self, branches=()).complete()
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
            for br_idx, (
                anaTuple_dataset_id,
                anaTuple_dataset_name,
                anaTuple_fileintot,
            ) in AnaTuple_map.items():
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
                customisations=self.customisations,
            )
        }
        return deps

    def requires(self):
        dataset_name, process_group = self.branch_data
        AnaTuple_map = AnaTupleFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        branch_set = set()
        for br_idx, (
            anaTuple_dataset_id,
            anaTuple_dataset_name,
            anaTuple_fileintot,
        ) in AnaTuple_map.items():
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
                customisations=self.customisations,
            )
            for prod_br in tuple(branch_set)
        ]
        return reqs

    def create_branch_map(self):
        branches = {}
        k = 0
        data_done = False
        data_sub_eras = set()
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

    def get_output_path(self, dataset_name):
        output_name = "merged_plan.json"
        return os.path.join(
            "AnaTupleFileList", self.version, self.period, dataset_name, output_name
        )

    def output(self):
        dataset_name, process_group = self.branch_data
        output_path = self.get_output_path(dataset_name)
        return self.remote_target(output_path, fs=self.fs_anaTuple)

    def run(self):
        dataset_name, process_group = self.branch_data
        AnaTupleFileList = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "AnaTupleFileList.py"
        )
        with contextlib.ExitStack() as stack:
            remote_output = self.output()

            print("Localizing inputs")
            local_inputs = [
                stack.enter_context(inp[1].localize("r")).path for inp in self.input()
            ]
            print(f"Localized {len(local_inputs)} inputs")

            job_home, remove_job_home = self.law_job_home()
            tmpFile = os.path.join(job_home, f"AnaTupleFileList_tmp.json")
            nEventsPerFile = self.setup.global_params.get("nEventsPerFile", 100_000)
            AnaTupleFileList_cmd = [
                "python3",
                AnaTupleFileList,
                "--outFile",
                tmpFile,
            ]  # , '--remove-files', 'True']
            AnaTupleFileList_cmd.extend(["--nEventsPerFile", f"{nEventsPerFile}"])
            if dataset_name == "data":
                AnaTupleFileList_cmd.extend(["--isData", "True"])
                if self.test > 0:
                    print(
                        "Don't split test by lumi if its data, its already only 1000 events"
                    )
                    AnaTupleFileList_cmd.extend(["--lumi", f"1.0"])
                else:
                    # I know this isn't clean, but I don't want to put a 'if not self.test' for the base case
                    AnaTupleFileList_cmd.extend(
                        ["--lumi", f'{self.setup.global_params["luminosity"]}']
                    )
                AnaTupleFileList_cmd.extend(
                    [
                        "--nPbPerFile",
                        f'{self.setup.global_params.get("nPbPerFile", 10_000)}',
                    ]
                )
            AnaTupleFileList_cmd.extend(local_inputs)
            ps_call(AnaTupleFileList_cmd, verbose=1)

            with remote_output.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.path
                shutil.move(tmpFile, out_local_path)


class AnaTupleFileListTask(AnaTupleFileListBuilderTask):
    def workflow_requires(self):
        return {"AnaTupleFileListBuilderTask": AnaTupleFileListBuilderTask.req(self)}

    def requires(self):
        return [AnaTupleFileListBuilderTask.req(self)]

    def output(self):
        dataset_name, process_group = self.branch_data
        return self.local_target(self.get_output_path(dataset_name))

    def run(self):
        with self.input()[0].localize("r") as input_local:
            self.output().makedirs()
            shutil.copy(input_local.path, self.output().path)


class AnaTupleMergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 24.0)
    delete_inputs_after_merge = luigi.BoolParameter(default=True)

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
        for _, (_, _, ds_branch, dataset_dependencies, _, _, _) in self.branch_map.items():
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
            process_group, ds_branch, dataset_dependencies,
            input_file_list,
            _,
            skip_future_tasks,
        ) = self.branch_data
        anaTuple_branch_map = AnaTupleFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        required_branches = { "root": {}, "json": {} }
        for prod_br, (
            anaTuple_dataset_id,
            anaTuple_dataset_name,
            anaTuple_fileintot,
        ) in anaTuple_branch_map.items():
            match = dataset_name == anaTuple_dataset_name
            if not match and process_group == "data":
                anaTuple_dataset = self.datasets[anaTuple_dataset_name]
                anaTuple_process_group = anaTuple_dataset["process_group"]
                match = anaTuple_process_group == "data"
            dependency_type = None
            if match:
                # print(f"{anaTuple_dataset_name}, {dataset_name} are the same, thus including:")
                file_name = anaTuple_fileintot.path.split("/")[-1]
                # print(input_file_list)
                # print(f"anaTuple_dataset_name/file_name  = {anaTuple_dataset_name}/{file_name}  in input_fileList? ", f"{anaTuple_dataset_name}/{file_name}" in input_file_list)
                if (
                    f"{anaTuple_dataset_name}/{file_name}" in input_file_list
                ):  # [1:] to remove the first '/' in the pathway
                    dependency_type = "root"
            elif anaTuple_dataset_name in dataset_dependencies.keys():
                dependency_type = "json"
            if dependency_type:
                if anaTuple_dataset_name not in required_branches[dependency_type]:
                    required_branches[dependency_type][anaTuple_dataset_name] = []
                required_branches[dependency_type][anaTuple_dataset_name].append(AnaTupleFileTask.req(
                    self,
                    max_runtime=AnaTupleFileTask.max_runtime._default,
                    branch=prod_br,
                    branches=(prod_br,),
                ))
        required_branches["list"] = AnaTupleFileListTask.req(
            self,
            max_runtime=AnaTupleFileListTask.max_runtime._default,
            n_cpus=AnaTupleFileListTask.n_cpus._default,
            branch=ds_branch,
            branches=(ds_branch,),
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
                raise RuntimeError(f"Dataset {dataset_name} appears multiple times in AnaTupleFileListTask branch map!")
            ds_branches[dataset_name] = ds_branch

        for ds_branch, (dataset_name, process_group) in ds_branch_map.items():
            dataset_dependencies = self.collect_extra_dependencies(dataset_name, ds_branches, process_group)
            this_dataset_dict = self.setup.getAnaTupleFileList(
                dataset_name,
                AnaTupleFileListTask.req(self, branch=ds_branch, branches=()).output(),
            )
            for this_dict in this_dataset_dict["merge_strategy"]:
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
        output_path_string = os.path.join(
            "anaTuples", self.version, self.period, dataset_name, "{}"
        )
        outputs = [output_path_string.format(out_file) for out_file in output_file_list]
        return [
            self.remote_target(out_path, fs=self.fs_anaTuple) for out_path in outputs
        ]

    def run(self):
        producer_Merge = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "MergeAnaTuples.py"
        )
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
            os.path.join(job_home, f"AnaTupleMergeTask_tmp{i}.root")
            for i in range(len(self.output()))
        ]
        with contextlib.ExitStack() as stack:
            remote_inputs = {
                "root": {
                    "index": 0,
                    "sources": [ self.input()["root"] ],
                },
                "json": {
                    "index": 1,
                    "sources": [ self.input()["root"], self.input()["json"] ],
                }
            }
            local_inputs = {}
            for key, entry in remote_inputs.items():
                print(f"Localizing {key} inputs")
                n_total = 0
                local_inputs[key] = {}
                index = entry["index"]
                for source in entry["sources"]:
                    for ds_name, files in source.items():
                        if ds_name not in local_inputs[key]:
                            local_inputs[key][ds_name] = []
                        for file_list in files:
                            local_input = stack.enter_context(file_list[index].localize("r")).path
                            local_inputs[key][ds_name].append(local_input)
                        n_total += len(files)
                print(f"Localized {n_total} {key} inputs")

            local_json_files_str = SerializeObjectToString(local_inputs["json"])
            local_root_inputs = []
            for ds_name, files in local_inputs["root"].items():
                local_root_inputs.extend(files)
            cmd = [
                "python3",
                "-u",
                producer_Merge,
                "--period",
                self.period,
                "--work-dir",
                job_home,
                "--dataset",
                dataset_name,
                "--root-outputs",
                *tmpFiles,
                "--input-reports",
                local_json_files_str,
                "--input-roots",
                *local_root_inputs
            ]
            if is_data:
                cmd.append("--is-data")
            ps_call(cmd, verbose=1)

        for outFile, tmpFile in zip(self.output(), tmpFiles):
            with outFile.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.path
                shutil.move(tmpFile, out_local_path)

        if self.delete_inputs_after_merge:
            print(f"Finished merging, lets delete remote targets")
            idx = remote_inputs["root"]["index"]
            for source in remote_inputs["root"]["sources"]:
                for ds_name, files in source.items():
                    for remote_targets in files:
                        remote_targets[idx].remove()
                        with remote_targets[idx].localize("w") as tmp_local_file:
                            tmp_local_file.touch()  # Create a dummy to avoid dependency crashes

        if remove_job_home:
            shutil.rmtree(job_home)
