import copy
import contextlib
import json
import law
import luigi
import os
import shutil
import threading
import yaml
import re

from FLAF.RunKit.run_tools import ps_call, natural_sort
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
from FLAF.run_tools.law_customizations import Task, HTCondorWorkflow, copy_param
from FLAF.Common.Utilities import SerializeObjectToString, getCustomisationSplit
from FLAF.AnaProd.anaCacheProducer import (
    combineAnaCaches,
    createAnaCacheProcessorInstances,
)


def getCustomisationSplit(customisations):
    customisation_dict = {}
    if customisations is None or len(customisations) == 0:
        return {}
    if type(customisations) == str:
        customisations = customisations.split(";")
    if type(customisations) != list:
        raise RuntimeError(f"Invalid type of customisations: {type(customisations)}")
    for customisation in customisations:
        substrings = customisation.split("=")
        if len(substrings) != 2:
            raise RuntimeError("len of substring is not 2!")
        customisation_dict[substrings[0]] = substrings[1]
    return customisation_dict


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


class AnaCacheTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 10.0)

    def create_branch_map(self):
        branches = {}
        for dataset_id, dataset_name in self.iter_datasets():
            isData = self.datasets[dataset_name]["process_group"] == "data"
            branches[dataset_id] = (dataset_name, isData)
        return branches

    def requires(self):
        return [InputFileTask.req(self)]

    def workflow_requires(self):
        return {"inputFile": InputFileTask.req(self)}

    def output(self):
        dataset_name, isData = self.branch_data
        return self.remote_target(
            "anaCache", self.period, f"{dataset_name}.yaml", fs=self.fs_anaCache
        )

    def run(self):
        dataset_name, isData = self.branch_data
        if isData:
            self.output().touch()
            return
        print(
            f"Creating anaCache for dataset {dataset_name} into {self.output().uri()}"
        )
        producer = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "anaCacheProducer.py"
        )
        dir_to_list = (
            self.datasets[dataset_name]["dirName"]
            if "dirName" in self.datasets[dataset_name]
            else dataset_name
        )
        input_files = InputFileTask.load_input_files(
            self.input()[0].path, dir_to_list, test=self.test > 0
        )
        ana_caches = []
        global_params_str = SerializeObjectToString(self.global_params)
        process_name = self.datasets[dataset_name]["process_name"]
        processors_cfg = self.setup.get_processors(process_name, stage="AnaCache")
        processors = createAnaCacheProcessorInstances(
            self.global_params, processors_cfg
        )
        if len(processors_cfg) > 0:
            processors_str = SerializeObjectToString(processors_cfg)
        n_inputs = len(input_files)

        fs_nanoAOD = self.fs_nanoAOD
        if self.datasets[dataset_name].get("fs_nanoAOD", None) is not None:
            fs_nanoAOD = self.setup.get_fs(
                f"fs_nanoAOD_{dataset_name}", self.datasets[dataset_name]["fs_nanoAOD"]
            )
        if fs_nanoAOD is None:
            raise RuntimeError(f"fs_nanoAOD is not defined for dataset {dataset_name}")

        for input_idx, input_file in enumerate(input_files):
            input_target = self.remote_target(input_file, fs=fs_nanoAOD)
            print(f"[{input_idx+1}/{n_inputs}] {input_target.uri()}")
            with input_target.localize("r") as input_local:
                cmd = [
                    "python3",
                    producer,
                    "--input-files",
                    input_local.path,
                    "--global-params",
                    global_params_str,
                    "--verbose",
                    "1",
                ]
                if len(processors_cfg) > 0:
                    cmd.extend(["--processors", processors_str])
                returncode, output, err = ps_call(
                    cmd,
                    env=self.cmssw_env,
                    catch_stdout=True,
                )
            ana_cache = json.loads(output)
            print(json.dumps(ana_cache))
            ana_caches.append(ana_cache)
        total_ana_cache = combineAnaCaches(ana_caches, processors)
        print(f"total anaCache: {json.dumps(total_ana_cache)}")
        with self.output().localize("w") as output_local:
            with open(output_local.path, "w") as file:
                yaml.dump(total_ana_cache, file)
        print(
            f"anaCache for dataset {dataset_name} is created in {self.output().uri()}"
        )


class AnaTupleFileTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 40.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def workflow_requires(self):
        input_file_task_complete = InputFileTask.req(self, branches=()).complete()
        if not input_file_task_complete:
            return {
                "anaCache": AnaCacheTask.req(self, branches=()),
                "inputFile": InputFileTask.req(self, branches=()),
            }

        branches_set = set()
        for branch_idx, (
            dataset_id,
            _,
            dataset_dependencies,
            _,
        ) in self.branch_map.items():
            branches_set.add(dataset_id)
            branches_set.update(dataset_dependencies.values())
        branches = tuple(branches_set)
        return {
            "anaCache": AnaCacheTask.req(self, branches=branches),
            "inputFile": InputFileTask.req(self, branches=branches),
        }

    def requires(self):
        dataset_id, _, dataset_dependencies, _ = self.branch_data

        def mk_req(ds_id):
            return AnaCacheTask.req(
                self,
                branch=ds_id,
                max_runtime=AnaCacheTask.max_runtime._default,
                branches=(),
            )

        core_requirement = mk_req(dataset_id)
        other_requirements = {
            ds_name: mk_req(ds_id) for ds_name, ds_id in dataset_dependencies.items()
        }
        return [core_requirement, other_requirements]

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
            dataset_dependencies = self.collect_extra_dependencies(dataset_name)

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
                    dataset_dependencies,
                    fileintot,
                )
                branch_idx += 1
        return branches

    def collect_extra_dependencies(self, dataset_name):
        other_datasets = {}
        dataset = self.datasets[dataset_name]
        processors = self.setup.get_processors(
            dataset["process_name"], stage="AnaTuple"
        )
        require_whole_process = any(
            p.get("dependency_level", {}).get("AnaTuple", "file") == "process"
            for p in processors
        )
        if require_whole_process:
            process = self.setup.base_processes[dataset["process_name"]]
            for p_dataset_name in process.get("datasets", []):
                if p_dataset_name == dataset_name:
                    continue
                p_dataset_id = self.get_dataset_id(p_dataset_name)
                other_datasets[p_dataset_name] = p_dataset_id
        return other_datasets

    @workflow_condition.output
    def output(self):
        _, dataset_name, _, input_file = self.branch_data
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
        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        try:
            dataset_id, dataset_name, dataset_dependencies, input_file = (
                self.branch_data
            )
            dataset = self.datasets[dataset_name]
            process_group = dataset["process_group"]
            producer_anatuples = os.path.join(
                self.ana_path(), "FLAF", "AnaProd", "anaTupleProducer.py"
            )

            anaCache_remotes = self.input()
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
            deepTauVersion = ""
            if self.global_params["analysis_config_area"] == "config/HH_bbtautau":
                deepTauVersion = (
                    customisation_dict["deepTauVersion"]
                    if "deepTauVersion" in customisation_dict.keys()
                    else self.global_params["deepTauVersion"]
                )

            job_home, remove_job_home = self.law_job_home()
            print(f"dataset_id: {dataset_id}")
            print(f"dataset_name: {dataset_name}")
            print(f"process_group: {process_group}")
            print(f"dataset_dependencies: {','.join(dataset_dependencies.keys())}")
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
                anaCache_main = stack.enter_context(
                    anaCache_remotes[0].localize("r")
                ).path
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
                    "--anaCache",
                    anaCache_main,
                    "--channels",
                    channels,
                    "--inFileName",
                    inFileName,
                    "--reportOutput",
                    rawReportPath,
                ]
                if len(anaCache_remotes[1]) > 0:
                    anaCache_others = {}
                    for other_name, other_remote in anaCache_remotes[1].items():
                        other_local = stack.enter_context(
                            other_remote.localize("r")
                        ).path
                        anaCache_others[other_name] = other_local
                    anaCache_others_str = SerializeObjectToString(anaCache_others)
                    anatuple_cmd.extend(["--anaCacheOthers", anaCache_others_str])
                if deepTauVersion != "":
                    anatuple_cmd.extend(
                        ["--customisations", f"deepTauVersion={deepTauVersion}"]
                    )
                if compute_unc_variations:
                    anatuple_cmd.append("--compute-unc-variations")
                if store_noncentral:
                    anatuple_cmd.append("--store-noncentral")

                centralFileName = os.path.basename(local_input)
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
            verbosity = "0"
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
        finally:
            kInit_cond.acquire()
            kInit_cond.notify_all()
            kInit_cond.release()
            thread.join()


class AnaTupleFileListBuilderTask(Task, HTCondorWorkflow, law.LocalWorkflow):

    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def _era_key(self, dataset):
        return dataset.get("eraLetter")

    def _data_branch_name(self, dataset):
        key = self._era_key(dataset)
        return f"data_{key}"

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
        era_to_branches = {}
        for br_idx, (_, ana_name, _, _) in AnaTuple_map.items():
            ana_ds = self.datasets[ana_name]
            if ana_ds["process_group"] != "data":
                continue
            key = self._era_key(ana_ds)
            era_to_branches.setdefault(key, set()).add(br_idx)
        branch_set = set()
        for _, (dataset_name, process_group) in self.branch_map.items():
            dataset = self.datasets[dataset_name]
            if process_group != "data":
                for br_idx, (_, ana_name, _, _) in AnaTuple_map.items():
                    if dataset_name == ana_name:
                        branch_set.add(br_idx)
            else:
                key = self._era_key(dataset)
                if key not in era_to_branches:
                    raise RuntimeError(
                        f"Nessun AnaTuple per data eraLetter {key} ({dataset_name})"
                    )
                branch_set |= era_to_branches[key]
        return {
            "AnaTupleFileTask": AnaTupleFileTask.req(
                self,
                branches=tuple(sorted(branch_set)),
                max_runtime=AnaTupleFileTask.max_runtime._default,
                n_cpus=AnaTupleFileTask.n_cpus._default,
                customisations=self.customisations,
            )
        }

    def requires(self):
        dataset_name, process_group = self.branch_data
        dataset = self.datasets[dataset_name]
        AnaTuple_map = AnaTupleFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        branch_set = set()
        if process_group != "data":
            for br_idx, (_, ana_name, _, _) in AnaTuple_map.items():
                if dataset_name == ana_name:
                    branch_set.add(br_idx)
        else:
            key = self._era_key(dataset)
            for br_idx, (_, ana_name, _, _) in AnaTuple_map.items():
                ana_ds = self.datasets[ana_name]
                if ana_ds["process_group"] != "data":
                    continue
                if self._era_key(ana_ds) == key:
                    branch_set.add(br_idx)
        return [
            AnaTupleFileTask.req(
                self,
                max_runtime=AnaTupleFileTask.max_runtime._default,
                branch=br,
                branches=(br,),
                customisations=self.customisations,
            )
            for br in sorted(branch_set)
        ]

    def create_branch_map(self):
        branches = {}
        k = 0
        data_seen = set()
        for dataset_id, dataset_name in self.iter_datasets():
            ds = self.datasets[dataset_name]
            process_group = ds["process_group"]
            print(f"dataset name = {dataset_name}")
            print(f"process group = {process_group}")
            if process_group == "data":
                key = self._era_key(ds)
                if key in data_seen:
                    continue
                data_seen.add(key)
                dataset_split = dataset_name.split("_")

                new_dataset_name = dataset_name
                if re.match(r"([a-zA-Z]+)\d*_(Run\d+[A-Z])(_v\d+)?", dataset_name):
                    dataset_name_strip = re.match(
                        r"([a-zA-Z]+)\d*_(Run\d+[A-Z])(_v\d+)?", dataset_name
                    )
                    new_dataset_name = (
                        f"{dataset_name_strip.group(1)}_{dataset_name_strip.group(2)}"
                    )
                    print(new_dataset_name)
                dataset_name = new_dataset_name
            branches[k] = (dataset_name, process_group)
            k += 1
        return branches

    def get_output_path(self, dataset_name):
        output_name = "merged_plan.json"
        return os.path.join(
            "AnaTupleFileList",
            self.version,
            self.period,
            dataset_name,
            output_name,
        )

    def output(self):
        dataset_name, _ = self.branch_data
        return self.remote_target(
            self.get_output_path(dataset_name),
            fs=self.fs_anaTuple,
        )

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
            job_home, _ = self.law_job_home()
            tmpFile = os.path.join(job_home, "AnaTupleFileList_tmp.json")
            nEventsPerFile = self.setup.global_params.get("nEventsPerFile", 100_000)
            cmd = [
                "python3",
                AnaTupleFileList,
                "--outFile",
                tmpFile,
                "--nEventsPerFile",
                f"{nEventsPerFile}",
            ]
            if process_group == "data":
                cmd.extend(["--isData", "True"])
                if self.test > 0:
                    cmd.extend(["--lumi", "1.0"])
                else:
                    cmd.extend(["--lumi", f'{self.setup.global_params["luminosity"]}'])
                cmd.extend(
                    [
                        "--nPbPerFile",
                        f'{self.setup.global_params.get("nPbPerFile", 10_000)}',
                    ]
                )
            cmd.extend(local_inputs)
            ps_call(cmd, verbose=1)
            with remote_output.localize("w") as tmp_local:
                shutil.move(tmpFile, tmp_local.path)


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

    def _era_key(self, dataset):
        return dataset.get("eraLetter")  # solo lettera

    def _data_branch_name(self, dataset):
        return f"data_{self._era_key(dataset)}"

    def workflow_requires(self):
        merge_complete = AnaTupleFileListTask.req(self, branches=()).complete()
        if not merge_complete:
            return {
                "AnaTupleFileListTask": AnaTupleFileListTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleFileListTask.max_runtime._default,
                    n_cpus=AnaTupleFileListTask.n_cpus._default,
                )
            }

        branch_set = set(self.branch_map.keys())
        return {
            "AnaTupleFileListTask": AnaTupleFileListTask.req(
                self,
                branches=tuple(sorted(branch_set)),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
        }

    def requires(self):
        (
            dataset_name,
            process_group,
            input_file_list,
            output_file_list,
            skip_future_tasks,
        ) = self.branch_data

        dataset = self.datasets.get(dataset_name.replace("data_", ""), {})
        anaTuple_map = AnaTupleFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        AnaTupleFileList_map = AnaTupleFileListTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()

        required_reqs = []

        for prod_br, (_, ana_name, _, ana_fileintot) in anaTuple_map.items():
            ana_ds = self.datasets[ana_name]
            match = False
            if process_group != "data":
                match = ana_name == dataset_name
            else:
                match = ana_ds["process_group"] == "data" and self._era_key(
                    ana_ds
                ) == self._era_key(dataset)

            if not match:
                continue

            file_name = ana_fileintot.path.split("/")[-1]
            if f"{ana_name}/{file_name}" not in input_file_list:
                continue

            required_reqs.append(
                AnaTupleFileTask.req(
                    self,
                    max_runtime=AnaTupleFileTask.max_runtime._default,
                    branch=prod_br,
                    branches=(prod_br,),
                )
            )

        for prod_br, (list_dataset_name, _) in AnaTupleFileList_map.items():
            if list_dataset_name == dataset_name:
                required_reqs.append(
                    AnaTupleFileListTask.req(
                        self,
                        max_runtime=AnaTupleFileListTask.max_runtime._default,
                        n_cpus=AnaTupleFileListTask.n_cpus._default,
                        branch=prod_br,
                        branches=(prod_br,),
                    )
                )

        return required_reqs

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branches = {}
        nBranch = 0

        organizer_branch_map = AnaTupleFileListTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()

        for nJob, (dataset_name, process_group) in organizer_branch_map.items():
            dataset_dict = self.setup.getAnaTupleFileList(
                dataset_name,
                AnaTupleFileListTask.req(self, branch=nJob, branches=()).output(),
            )
            for merge_dict in dataset_dict["merge_strategy"]:
                branches[nBranch] = (
                    dataset_name,
                    process_group,
                    merge_dict["inputs"],
                    merge_dict["outputs"],
                    merge_dict["n_events"] == 0,
                )
                nBranch += 1
        return branches

    @workflow_condition.output
    def output(self):
        dataset_name, process_group, _, output_file_list, _ = self.branch_data
        base = os.path.join("anaTuples", self.version, self.period, dataset_name, "{}")
        return [
            self.remote_target(base.format(out), fs=self.fs_anaTuple)
            for out in output_file_list
        ]

    def run(self):
        producer_Merge = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "MergeNtuples.py"
        )
        dataset_name, process_group, _, output_file_list, _ = self.branch_data
        isData = "1" if process_group == "data" else "0"

        input_targets = [inp[0] for inp in self.input()[:-1]]
        job_home, remove_job_home = self.law_job_home()
        tmpFiles = [
            os.path.join(job_home, f"AnaTupleMergeTask_tmp{i}.root")
            for i in range(len(output_file_list))
        ]

        with contextlib.ExitStack() as stack:
            print(f"Localizing {len(input_targets)} inputs")
            local_inputs = [
                stack.enter_context(inp.localize("r")).path for inp in input_targets
            ]
            cmd = [
                "python3",
                producer_Merge,
                "--apply-filter",
                isData,
                "--outFiles",
                *tmpFiles,
                "--outFile",
                "tmp_data.root",
            ]
            cmd.extend(local_inputs)
            ps_call(cmd, verbose=1)

        for outTarget, tmpFile in zip(self.output(), tmpFiles):
            with outTarget.localize("w") as tmp_local:
                shutil.move(tmpFile, tmp_local.path)

        if self.delete_inputs_after_merge:
            print("Finished merging, deleting input remote targets")
            for target in input_targets:
                target.remove()
                with target.localize("w") as tmp:
                    tmp.touch()

        if remove_job_home:
            shutil.rmtree(job_home)
