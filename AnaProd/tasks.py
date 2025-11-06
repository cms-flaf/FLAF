import copy
import contextlib
import json
import law
import os
import shutil
import threading
import yaml


from FLAF.RunKit.run_tools import ps_call, natural_sort
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    copy_param,
    get_param_value,
)
from FLAF.Common.Utilities import SerializeObjectToString, getCustomisationSplit
from FLAF.AnaProd.anaCacheProducer import addAnaCaches
import re


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
        for sample_id, sample_name in self.iter_samples():
            branches[sample_id] = sample_name
        return branches

    def output(self):
        sample_name = self.branch_data
        return self.local_target("input_files", f"{sample_name}.txt")

    def run(self):
        sample_name = self.branch_data
        folder_name = (
            self.samples[sample_name]["dirName"]
            if "dirName" in self.samples[sample_name]
            else sample_name
        )
        print(f"Creating inputFile for sample {sample_name} into {self.output().path}")

        fs_nanoAOD = self.fs_nanoAOD
        if self.samples[sample_name].get("fs_nanoAOD", None) is not None:
            fs_nanoAOD = self.setup.get_fs(
                f"fs_nanoAOD_{sample_name}", self.samples[sample_name]["fs_nanoAOD"]
            )
        if fs_nanoAOD is None:
            raise RuntimeError(f"fs_nanoAOD is not defined for sample {sample_name}")

        with self.output().localize("w") as out_local_file:
            input_files = []
            pattern = self.samples[sample_name].get("fileNamePattern", r".*\.root$")
            for file in natural_sort(fs_nanoAOD.listdir(folder_name)):
                if re.match(pattern, file):
                    input_files.append(file)
            with open(out_local_file.path, "w") as inputFileTxt:
                for input_line in input_files:
                    inputFileTxt.write(input_line + "\n")
        print(f"inputFile for sample {sample_name} is created in {self.output().path}")

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
        for sample_id, sample_name in self.iter_samples():
            isData = self.samples[sample_name]["process_group"] == "data"
            branches[sample_id] = (sample_name, isData)
        return branches

    def requires(self):
        return [InputFileTask.req(self)]

    def workflow_requires(self):
        return {"inputFile": InputFileTask.req(self)}

    def output(self):
        sample_name, isData = self.branch_data
        return self.remote_target(
            "anaCache", self.period, f"{sample_name}.yaml", fs=self.fs_anaCache
        )

    def run(self):
        sample_name, isData = self.branch_data
        if isData:
            self.output().touch()
            return
        print(f"Creating anaCache for sample {sample_name} into {self.output().uri()}")
        producer = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "anaCacheProducer.py"
        )
        dir_to_list = (
            self.samples[sample_name]["dirName"]
            if "dirName" in self.samples[sample_name]
            else sample_name
        )
        input_files = InputFileTask.load_input_files(
            self.input()[0].path, dir_to_list, test=self.test
        )
        ana_caches = []
        generator_name = self.samples[sample_name]["generator"] if not isData else ""
        global_params_str = SerializeObjectToString(self.global_params)
        n_inputs = len(input_files)

        fs_nanoAOD = self.fs_nanoAOD
        if self.samples[sample_name].get("fs_nanoAOD", None) is not None:
            fs_nanoAOD = self.setup.get_fs(
                f"fs_nanoAOD_{sample_name}", self.samples[sample_name]["fs_nanoAOD"]
            )
        if fs_nanoAOD is None:
            raise RuntimeError(f"fs_nanoAOD is not defined for sample {sample_name}")

        for input_idx, input_file in enumerate(input_files):
            input_target = self.remote_target(input_file, fs=fs_nanoAOD)
            print(f"[{input_idx+1}/{n_inputs}] {input_target.uri()}")
            with input_target.localize("r") as input_local:
                returncode, output, err = ps_call(
                    [
                        "python3",
                        producer,
                        "--input-files",
                        input_local.path,
                        "--global-params",
                        global_params_str,
                        "--generator-name",
                        generator_name,
                        "--verbose",
                        "1",
                    ],
                    env=self.cmssw_env,
                    catch_stdout=True,
                )
            ana_cache = json.loads(output)
            print(json.dumps(ana_cache))
            ana_caches.append(ana_cache)
        total_ana_cache = addAnaCaches(*ana_caches)
        print(f"total anaCache: {json.dumps(total_ana_cache)}")
        with self.output().localize("w") as output_local:
            with open(output_local.path, "w") as file:
                yaml.dump(total_ana_cache, file)
        print(f"anaCache for sample {sample_name} is created in {self.output().uri()}")


class AnaTupleTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 40.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def create_branch_map(self):
        input_file_task_complete = InputFileTask.req(self, branches=()).complete()
        if not input_file_task_complete:
            self.cache_branch_map = False
            if not hasattr(self, "_branches_backup"):
                self._branches_backup = copy.deepcopy(self.branches)
            return {0: ()}
        self.cache_branch_map = True
        if hasattr(self, "_branches_backup"):
            self.branches = self._branches_backup
        branch_idx = 0
        branches = {}
        for sample_id, sample_name in self.iter_samples():

            fs_nanoAOD = self.fs_nanoAOD
            if self.samples[sample_name].get("fs_nanoAOD", None) is not None:
                fs_nanoAOD = self.setup.get_fs(
                    f"fs_nanoAOD_{sample_name}", self.samples[sample_name]["fs_nanoAOD"]
                )
            dir_to_list = (
                self.samples[sample_name]["dirName"]
                if "dirName" in self.samples[sample_name]
                else sample_name
            )
            input_file_list = (
                InputFileTask.req(self, branch=sample_id, branches=(sample_id,))
                .output()
                .path
            )
            input_files = InputFileTask.load_input_files(
                input_file_list, dir_to_list, test=self.test
            )
            if fs_nanoAOD is None:
                raise RuntimeError(
                    f"fs_nanoAOD is not defined for sample {sample_name}"
                )
            for input_file in input_files:
                fileintot = self.remote_target(input_file, fs=fs_nanoAOD)
                branches[branch_idx] = (
                    sample_id,
                    sample_name,
                    self.samples[sample_name]["process_group"],
                    fileintot,
                )
                branch_idx += 1
        return branches

    def workflow_requires(self):
        input_file_task_complete = InputFileTask.req(self, branches=()).complete()
        if not input_file_task_complete:
            return {
                "anaCache": AnaCacheTask.req(self, branches=()),
                "inputFile": InputFileTask.req(self, branches=()),
            }

        branches_set = set()
        for branch_idx, (
            sample_id,
            sample_name,
            process_group,
            input_file,
        ) in self.branch_map.items():
            branches_set.add(sample_id)
        branches = tuple(branches_set)
        return {
            "anaCache": AnaCacheTask.req(self, branches=branches),
            "inputFile": InputFileTask.req(self, branches=branches),
        }

    def requires(self):
        sample_id, sample_name, process_group, input_file = self.branch_data
        return [
            AnaCacheTask.req(
                self,
                branch=sample_id,
                max_runtime=AnaCacheTask.max_runtime._default,
                branches=(),
            )
        ]

    def output(self):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")
        sample_id, sample_name, process_group, input_file = self.branch_data
        output_name = os.path.basename(input_file.path)
        json_name = f"{output_name.split('.')[0]}.json"
        output_path = os.path.join(
            "anaTuples", self.version, self.period, sample_name, "split", output_name
        )
        json_path = os.path.join(
            "anaTuples", self.version, self.period, sample_name, "split", json_name
        )
        return [
            self.remote_target(output_path, fs=self.fs_anaTuple),
            self.remote_target(json_path, fs=self.fs_anaTuple),
        ]

    def run(self):
        sample_id, sample_name, process_group, input_file = self.branch_data
        process_group = self.samples[sample_name]["process_group"]
        jsonName = f"{os.path.basename(input_file.path).split('.')[0]}.json"
        producer_anatuples = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "anaTupleProducer.py"
        )
        producer_skimtuples = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "SkimProducer.py"
        )
        thread = threading.Thread(target=update_kinit_thread)
        thread.start()
        anaCache_remote = self.input()[0]
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
        try:
            job_home, remove_job_home = self.law_job_home()
            print(
                f"sample_id = {sample_id}\nsample_name = {sample_name}\nprocess_group = {process_group}\n"
                f"input_file = {input_file.uri()}"
            )

            print("step 1: nanoAOD -> anaTupleS")
            outdir_anatuples = os.path.join(job_home, "anaTuples", sample_name)
            anaTupleDef = os.path.join(
                self.ana_path(), self.global_params["anaTupleDef"]
            )
            with input_file.localize("r") as local_input, anaCache_remote.localize(
                "r"
            ) as anaCache_input:
                inFileName = os.path.basename(input_file.path)
                print(f"inFileName {inFileName}")
                anatuple_cmd = [
                    "python3",
                    "-u",
                    producer_anatuples,
                    "--period",
                    self.period,
                    "--inFile",
                    local_input.path,
                    "--outDir",
                    outdir_anatuples,
                    "--sample",
                    sample_name,
                    "--anaTupleDef",
                    anaTupleDef,
                    "--anaCache",
                    anaCache_input.path,
                    "--channels",
                    channels,
                    "--inFileName",
                    inFileName,
                    "--jsonName",
                    jsonName,
                ]
                if deepTauVersion != "":
                    anatuple_cmd.extend(
                        ["--customisations", f"deepTauVersion={deepTauVersion}"]
                    )
                if compute_unc_variations:
                    anatuple_cmd.append("--compute-unc-variations")
                if store_noncentral:
                    anatuple_cmd.append("--store-noncentral")

                centralFileName = os.path.basename(local_input.path)
                if self.test:
                    anatuple_cmd.extend(["--nEvents", "1000"])
                # ps_call(anatuple_cmd, verbose=1) # this will be uncommented when the anatuple producer will be fully independent on cmsEnv.
                ps_call(anatuple_cmd, env=self.cmssw_env, verbose=1)

            print("step 2: anaTupleS -> skimTuplE")
            outdir_skimtuples = os.path.join(job_home, "skim", sample_name)
            outFileName = os.path.basename(input_file.path)

            # if self.test:
            print(f"outFileName is {outFileName}")
            tmpFile = os.path.join(outdir_skimtuples, outFileName)
            tmpjsonFile = os.path.join(outdir_anatuples, jsonName)
            if process_group != "data":
                skimtuple_cmd = [
                    "python",
                    producer_skimtuples,
                    "--inputDir",
                    outdir_anatuples,
                    "--centralFile",
                    centralFileName,
                    "--workingDir",
                    outdir_skimtuples,
                    "--outputFile",
                    outFileName,
                ]
                if self.test:
                    skimtuple_cmd.extend(["--test", "True"])
                ps_call(skimtuple_cmd, verbose=1)
            else:
                tmpFile = os.path.join(outdir_anatuples, centralFileName)

            nano_output = self.output()[0]
            json_output = self.output()[1]
            with nano_output.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.path
                shutil.move(tmpFile, out_local_path)
            with json_output.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.path
                shutil.move(tmpjsonFile, out_local_path)

            if remove_job_home:
                shutil.rmtree(job_home)
        finally:
            kInit_cond.acquire()
            kInit_cond.notify_all()
            kInit_cond.release()
            thread.join()


class AnaTupleFileListTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def __init__(self, *args, **kwargs):
        # kwargs['workflow'] = 'local' # This might not be the best idea, has ~100 datasets to go over and localize, probably faster on condor
        super(AnaTupleFileListTask, self).__init__(*args, **kwargs)

    def workflow_requires(self):
        input_file_task_complete = InputFileTask.req(self, branches=()).complete()
        if not input_file_task_complete:
            return {
                "anaTuple": AnaTupleTask.req(self, branches=()),
                "inputFile": InputFileTask.req(self, branches=()),
            }

        AnaTuple_map = AnaTupleTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        branch_set = set()
        for idx, (sample_name, process_group) in self.branch_map.items():
            for br_idx, (
                anaTuple_sample_id,
                anaTuple_sample_name,
                anaTuple_process_group,
                anaTuple_input_file,
            ) in AnaTuple_map.items():
                if (sample_name == anaTuple_sample_name) or (
                    process_group == "data" and anaTuple_process_group == "data"
                ):
                    branch_set.add(br_idx)

        deps = {
            "AnaTupleTask": AnaTupleTask.req(
                self,
                branches=tuple(branch_set),
                max_runtime=AnaTupleTask.max_runtime._default,
                n_cpus=AnaTupleTask.n_cpus._default,
                customisations=self.customisations,
            )
        }
        return deps

    def requires(self):
        sample_name, process_group = self.branch_data
        AnaTuple_map = AnaTupleTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        branch_set = set()
        for br_idx, (
            anaTuple_sample_id,
            anaTuple_sample_name,
            anaTuple_process_group,
            anaTuple_input_file,
        ) in AnaTuple_map.items():
            if (sample_name == anaTuple_sample_name) or (
                process_group == "data" and anaTuple_process_group == "data"
            ):
                branch_set.add(br_idx)

        reqs = [
            AnaTupleTask.req(
                self,
                max_runtime=AnaTupleTask.max_runtime._default,
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
        for sample_id, sample_name in self.iter_samples():
            process_group = self.samples[sample_name]["process_group"]
            if process_group == "data":
                if data_done:
                    continue  # Will have multiple data samples, but only need one branch
                sample_name = "data"
                data_done = True
            branches[k] = (sample_name, process_group)
            k += 1
        return branches

    def output(self):
        sample_name, process_group = self.branch_data
        output_name = "merged_plan.json"
        output_path = os.path.join(
            "AnaTupleFileList", self.version, self.period, sample_name, output_name
        )
        # This is a problem for a --remove-output task, it crashes since the 'output' is only the local or something
        # With this current state, you need to do the --remove-output command twice
        if self.local_target(output_path).exists():
            return [self.local_target(output_path)]
        return [
            self.remote_target(output_path, fs=self.fs_anaTuple),
            self.local_target(output_path),
        ]

    def run(self):
        sample_name, process_group = self.branch_data
        AnaTupleFileList = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "AnaTupleFileList.py"
        )
        with contextlib.ExitStack() as stack:
            remote_output = self.output()[0]
            local_output = self.output()[1]
            # Check if remote already exists, if so then just copy and return
            if remote_output.exists():
                print("Hey remote already exists! Don't run again, just copy to local")
                with local_output.localize("w") as tmp_local_file2:
                    with remote_output.localize("r") as tmp_local_file:
                        shutil.copy(tmp_local_file.path, tmp_local_file2.path)
                        return

            print("Localizing inputs")
            local_inputs = [
                stack.enter_context(inp[1].localize("r")).path for inp in self.input()
            ]

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
            if sample_name == "data":
                AnaTupleFileList_cmd.extend(["--isData", "True"])
                if self.test:
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

                with local_output.localize("w") as tmp_local_file2:
                    shutil.copy(tmp_local_file.path, tmp_local_file2.path)


class AnaTupleMergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)

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
                )
            }

        branch_set = set()
        for idx, sample_names in self.branch_map.items():
            branch_set.update([idx])

        anaTuple_branch_map = AnaTupleTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        anaTuple_branch_set = set()
        for idx, (
            sample_name,
            process_group,
            input_file_list,
            output_file_list,
        ) in self.branch_map.items():
            for br_idx, (
                anaTuple_sample_id,
                anaTuple_sample_name,
                anaTuple_process_group,
                anaTuple_input_file,
            ) in anaTuple_branch_map.items():
                if (sample_name == anaTuple_sample_name) or (
                    process_group == "data" and anaTuple_process_group == "data"
                ):
                    anaTuple_branch_set.add(br_idx)

        return {
            "AnaTupleFileListTask": AnaTupleFileListTask.req(
                self,
                branches=tuple(branch_set),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            ),
            "AnaTupleTask": AnaTupleTask.req(
                self,
                branches=tuple(anaTuple_branch_set),
                customisations=self.customisations,
            ),
        }

    def requires(self):
        # Need both the AnaTupleTask for the input ROOT file, and the AnaTupleFileListTask for the json structure
        sample_name, process_group, input_file_list, output_file_list = self.branch_data
        anaTuple_branch_map = AnaTupleTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        AnaTupleFileList_branch_map = AnaTupleFileListTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        required_branches = []
        for prod_br, (
            anaTuple_sample_id,
            anaTuple_sample_name,
            anaTuple_process_group,
            file_location,
        ) in anaTuple_branch_map.items():
            if anaTuple_sample_name == sample_name or (
                process_group == "data" and anaTuple_process_group == "data"
            ):
                # print(f"{anaTuple_sample_name}, {sample_name} are the same, thus including:")
                file_name = file_location.path.split("/")[-1]
                # print(input_file_list)
                # print(f"anaTuple_sample_name/file_name  = {anaTuple_sample_name}/{file_name}  in input_fileList? ", f"{anaTuple_sample_name}/{file_name}" in input_file_list)
                if (
                    f"{anaTuple_sample_name}/{file_name}" in input_file_list
                ):  # [1:] to remove the first '/' in the pathway
                    required_branches.append(
                        AnaTupleTask.req(
                            self,
                            max_runtime=AnaTupleTask.max_runtime._default,
                            branch=prod_br,
                            branches=(prod_br,),
                        )
                    )
        for prod_br, (
            Merge_sample_name,
            Merge_process_group,
        ) in AnaTupleFileList_branch_map.items():
            if Merge_sample_name == sample_name:
                # print(f"including {Merge_sample_name} which is same than {sample_name}")
                required_branches.append(
                    AnaTupleFileListTask.req(
                        self,
                        max_runtime=AnaTupleFileListTask.max_runtime._default,
                        n_cpus=AnaTupleFileListTask.n_cpus._default,
                        branch=prod_br,
                        branches=(prod_br,),
                    )
                )
        return required_branches

    def create_branch_map(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            self.cache_branch_map = False
            if not hasattr(self, "_branches_backup"):
                self._branches_backup = copy.deepcopy(self.branches)
            return {0: ()}
        self.cache_branch_map = True
        if hasattr(self, "_branches_backup"):
            self.branches = self._branches_backup
        branches = {}
        nBranch = 0
        organizer_branch_map = AnaTupleFileListTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        for nJob, (sample_name, process_group) in organizer_branch_map.items():
            # This function cannot run if the json doesn't exist yet, so the branch map breaks when the output isn't already there
            this_sample_dict = self.setup.getAnaTupleFileList(
                sample_name,
                AnaTupleFileListTask.req(self, branch=nJob, branches=()).output()[-1],
            )  # Get output[-1] for the local file version
            # We probably want to return None for a json that doesn't exist or something, right now if an expected branch is not there this entire task will fail
            for this_dict in this_sample_dict["merge_strategy"]:
                input_file_list = this_dict["inputs"]
                output_file_list = this_dict["outputs"]
                branches[nBranch] = (
                    sample_name,
                    process_group,
                    input_file_list,
                    output_file_list,
                )
                nBranch += 1
        return branches

    def output(self, force_pre_output=False):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")

        sample_name, process_group, input_file_list, output_file_list = self.branch_data
        output_path_string = os.path.join(
            "anaTuples", self.version, self.period, sample_name, "{}"
        )
        outputs = [output_path_string.format(out_file) for out_file in output_file_list]
        return [
            self.remote_target(out_path, fs=self.fs_anaTuple) for out_path in outputs
        ]

    def run(self):
        producer_Merge = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "MergeNtuples.py"
        )
        sample_name, process_group, input_file_list, output_file_list = self.branch_data
        isData = (
            "1" if process_group == "data" else "0"
        )  # ps_call needs to only pass strings????
        input_list_remote_target = [inp[0] for inp in self.input()[:-1]]
        job_home, remove_job_home = self.law_job_home()
        tmpFiles = [
            os.path.join(job_home, f"AnaTupleMergeTask_tmp{i}.root")
            for i in range(len(self.output()))
        ]
        with contextlib.ExitStack() as stack:
            print(f"Starting localize of {len(input_list_remote_target)} inputs")
            local_inputs = [
                stack.enter_context(inp.localize("r")).path
                for inp in input_list_remote_target
            ]
            Merge_cmd = [
                "python3",
                producer_Merge,
                "--apply-filter",
                isData,
                "--outFiles",
                *tmpFiles,
                "--outFile",
                "tmp_data.root",
            ]
            Merge_cmd.extend(local_inputs)
            ps_call(Merge_cmd, verbose=1)

        for outFile, tmpFile in zip(self.output(), tmpFiles):
            with outFile.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.path
                shutil.move(tmpFile, out_local_path)

        delete_after_merge = True
        if delete_after_merge:
            print(f"Finished merging, lets delete remote targets")
            for remote_target in input_list_remote_target:
                remote_target.remove()
                with remote_target.localize("w") as tmp_local_file:
                    tmp_local_file.touch()  # Create a dummy to avoid dependency crashes

        if remove_job_home:
            shutil.rmtree(job_home)


class DataMergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)

    def workflow_requires(self):
        branch_set = set()
        for idx, prod_branches in self.branch_map.items():
            branch_set.update(prod_branches)
        return {"anaTuple": AnaTupleMergeTask.req(self, branches=tuple(branch_set))}

    def requires(self):
        return [
            AnaTupleMergeTask.req(
                self,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
            )
            for prod_br in self.branch_data
        ]

    def create_branch_map(self):
        anaProd_branch_map = AnaTupleMergeTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        prod_branches = []
        for prod_br, (
            sample_id,
            sample_name,
            process_group,
            output_file_name,
            input_file_list,
        ) in anaProd_branch_map.items():
            if process_group != "data":
                continue
            prod_branches.append(prod_br)
        return {0: prod_branches}

    def output(self, force_pre_output=False):
        outFileName = "nanoHTT_0.root"
        output_path = os.path.join(
            "anaTuples", self.version, self.period, "data", outFileName
        )
        return self.remote_target(output_path, fs=self.fs_anaTuple)
        # customisation_dict = getCustomisationSplit(self.customisations)
        # deepTauVersion = customisation_dict['deepTauVersion'] if 'deepTauVersion' in customisation_dict.keys() else self.global_params['deepTauVersion']
        # print(deepTauVersion)
        # output_path = os.path.join('anaTuples', self.version, self.period, 'data', outFileName)
        # if '2p5' in deepTauVersion:
        #     return self.remote_target(output_path, fs=self.fs_anaTuple2p5)

    def run(self):
        producer_dataMerge = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "MergeNtuples.py"
        )
        with contextlib.ExitStack() as stack:
            local_inputs = [
                stack.enter_context(inp.localize("r")).path for inp in self.input()
            ]
            with self.output().localize("w") as tmp_local_file:
                tmpFile = tmp_local_file.path
                dataMerge_cmd = [
                    "python3",
                    producer_dataMerge,
                    "--outFile",
                    tmpFile,
                ]  # , '--useUproot', 'True']
                dataMerge_cmd.extend(local_inputs)
                ps_call(dataMerge_cmd, verbose=1)


# rename AnaCacheTupeTask one to AnalysisCahceTask and move to FLAF/Analysis/tasks.py


class DNNCacheTupleTask(Task, HTCondorWorkflow, law.LocalWorkflow):

    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 30.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def workflow_requires(self):
        branches_set = set()
        for branch_idx, (
            sample_name,
            process_group,
            input_file,
            ana_br_idx,
            spin,
            mass,
        ) in self.branch_map.items():
            if ana_br_idx not in branches_set:
                branches_set.add(ana_br_idx)
        return {
            "anaTuple": AnaTupleTask.req(
                self, branches=tuple(branches_set), customisations=self.customisations
            )
        }

    def requires(self):
        sample_name, process_group, input_file, ana_br_idx, spin, mass = (
            self.branch_data
        )
        return [
            AnaTupleTask.req(
                self,
                max_runtime=AnaTupleTask.max_runtime._default,
                branch=ana_br_idx,
                branches=(ana_br_idx,),
                customisations=self.customisations,
            )
        ]

    @staticmethod
    def load_sample_configs(signal_config_file):
        with open(signal_config_file, "r") as f:
            signal_config = yaml.safe_load(f)
        return signal_config

    def create_branch_map(self):
        signal_config_file = os.path.join(
            self.ana_path(), "config", "signal_samples.yaml"
        )
        signal_config = self.load_sample_configs(signal_config_file)

        branches = {}
        anaProd_branch_map = AnaTupleTask.req(self, branch=-1, branches=()).branch_map
        br_idx_final = 0
        for sample_key, sample_info in signal_config.items():
            for ana_br_idx, (
                sample_id,
                sample_name,
                process_group,
                input_file,
            ) in anaProd_branch_map.items():
                mass = sample_info["mass"]
                spin = sample_info["spin"]
                branches[br_idx_final] = (
                    sample_name,
                    process_group,
                    input_file,
                    ana_br_idx,
                    spin,
                    mass,
                )
                br_idx_final += 1
        return branches

    def output(self):
        sample_name, process_group, input_file, ana_br_idx, spin, mass = (
            self.branch_data
        )
        out_file_name = os.path.basename(self.input()[0].path)
        out_dir = os.path.join(
            "nnCacheTuple",
            self.period,
            sample_name,
            self.version,
            f"spin_{spin}",
            f"mass_{mass}",
        )
        final_file = os.path.join(out_dir, out_file_name)
        return self.remote_target(final_file, fs=self.fs_nnCacheTuple)

    def run(self):
        sample_name, process_group, input_file, ana_br_idx, spin, mass = (
            self.branch_data
        )
        unc_config = os.path.join(
            self.ana_path(), "config", self.period, "weights.yaml"
        )
        nn_interface_script = os.path.join(self.ana_path(), "AnaProd", "NNInterface.py")
        global_config = os.path.join(self.ana_path(), "config", "global.yaml")
        in_model_dir = os.path.join(self.ana_path(), "config", "nn_models")
        thread = threading.Thread(target=update_kinit_thread)
        thread.start()
        try:
            job_home, remove_job_home = self.law_job_home()
            with self.input()[0].localize("r") as local_input, self.output().localize(
                "w"
            ) as out_file:
                nn_interface_cmd = [
                    "python3",
                    nn_interface_script,
                    "--inModelDir",
                    in_model_dir,
                    "--inFile",
                    local_input.path,
                    "--outFileName",
                    out_file.path,
                    "--uncConfig",
                    unc_config,
                    "--globalConfig",
                    global_config,
                    "--period",
                    self.period,
                    "--mass",
                    str(mass),
                    "--spin",
                    str(spin),
                ]

                ps_call(nn_interface_cmd, env=self.cmssw_env, verbose=1)

            print(
                f"Finished running NNInterface for sample '{sample_name}' "
                f"(spin={spin}, mass={mass})"
            )
        finally:
            kInit_cond.acquire()
            kInit_cond.notify_all()
            kInit_cond.release()
            thread.join()


class DataCacheMergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)

    def workflow_requires(self):
        workflow_dep = {}
        for idx, prod_branches in self.branch_map.items():
            workflow_dep[idx] = AnaCacheTupleTask.req(self, branches=prod_branches)
        return workflow_dep

    def requires(self):
        prod_branches = self.branch_data
        deps = [
            AnaCacheTupleTask.req(
                self, max_runtime=AnaCacheTask.max_runtime._default, branch=prod_br
            )
            for prod_br in prod_branches
        ]
        return deps

    def create_branch_map(self):
        anaProd_branch_map = AnaCacheTupleTask.req(
            self, branch=-1, branches=()
        ).branch_map
        prod_branches = []
        for prod_br, (sample_name, process_group) in anaProd_branch_map.items():
            if process_group == "data":
                prod_branches.append(prod_br)
        return {0: prod_branches}

    def output(self, force_pre_output=False):
        outFileName = "nanoHTT_0.root"
        output_path = os.path.join(
            "anaCacheTuples", self.period, "data", self.version, outFileName
        )
        return self.remote_target(output_path, fs=self.fs_anaCacheTuple)

    def run(self):
        producer_dataMerge = os.path.join(
            self.ana_path(), "FLAF", "AnaProd", "MergeNtuples.py"
        )
        with contextlib.ExitStack() as stack:
            local_inputs = [
                stack.enter_context(inp.localize("r")).path for inp in self.input()
            ]
            with self.output().localize("w") as tmp_local_file:
                tmpFile = tmp_local_file.path
                dataMerge_cmd = ["python3", producer_dataMerge, "--outFile", tmpFile]
                dataMerge_cmd.extend(local_inputs)
                # print(dataMerge_cmd)
                ps_call(dataMerge_cmd, verbose=1)
