import law
import os
import yaml
import contextlib
import luigi
import threading
import copy
import shutil


from FLAF.RunKit.run_tools import ps_call
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    copy_param,
    get_param_value,
)
from FLAF.AnaProd.tasks import (
    AnaTupleTask,
    DataMergeTask,
    DataCacheMergeTask,
    AnaCacheTask,
    InputFileTask,
    AnaTupleFileListTask,
    AnaTupleMergeTask,
)
from FLAF.Common.Utilities import getCustomisationSplit
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    copy_param,
    get_param_value,
)
from FLAF.AnaProd.tasks import (
    AnaTupleTask,
    DataMergeTask,
    DataCacheMergeTask,
    AnaCacheTask,
    InputFileTask,
    AnaTupleFileListTask,
    AnaTupleMergeTask,
)
from FLAF.Common.Utilities import getCustomisationSplit
import importlib

unc_cfg_dict = None


def load_unc_config(unc_cfg):
    global unc_cfg_dict
    with open(unc_cfg, "r") as f:
        unc_cfg_dict = yaml.safe_load(f)
    return unc_cfg_dict


def getYear(period):
    year_dict = {
        "Run2_2016_HIPM": "2016_HIPM",
        "Run2_2016": "2016",
        "Run2_2017": "2017",
        "Run2_2018": "2018",
        "Run3_2022": "2022",
        "Run3_2022EE": "2022EE",
        "Run3_2023": "2023",
        "Run3_2023BPix": "2023BPix",
    }
    return year_dict[period]


def parseVarEntry(var_entry):
    if type(var_entry) == str:
        var_name = var_entry
        need_cache = False
    else:
        var_name = var_entry["name"]
        need_cache = var_entry.get("need_cache", False)
    return var_name, need_cache


def GetSamples(
    samples, backgrounds, signals=["GluGluToRadion", "GluGluToBulkGraviton"]
):
    global samples_to_consider
    samples_to_consider = ["data"]

    for sample_name in samples.keys():
        process_name = samples[sample_name]["process_name"]
        if process_name in signals or sample_name in backgrounds:
            samples_to_consider.append(sample_name)
    return samples_to_consider


class HistTupleProducerTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def workflow_requires(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            req_dict = {
                "AnaTupleFileListTask": AnaTupleFileListTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleFileListTask.max_runtime._default,
                    n_cpus=AnaTupleFileListTask.n_cpus._default,
                ),
                "AnaTupleMergeTask": AnaTupleMergeTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleMergeTask.max_runtime._default,
                    n_cpus=AnaTupleMergeTask.n_cpus._default,
                ),
            }
            req_dict["AnalysisCacheTask"] = []
            var_produced_by = self.setup.var_producer_map
            for var_name in self.global_params["variables"]:
                producer_to_run = var_produced_by.get(var_name, None)
                if producer_to_run is not None:
                    req_dict["AnalysisCacheTask"].append(
                        AnalysisCacheTask.req(
                            self,
                            branches=(),
                            customisations=self.customisations,
                            producer_to_run=producer_to_run,
                        )
                    )
            return req_dict

        branch_set = set()
        branch_set_cache = set()
        producer_set = set()
        for idx, (
            sample,
            br,
            need_cache_global,
            producer_list,
            input_index,
        ) in self.branch_map.items():
            branch_set.add(br)
            if need_cache_global:
                branch_set_cache.add(br)
                for producer_name in (p for p in producer_list if p is not None):
                    producer_set.add(producer_name)
        reqs = {}
        isbbtt = "HH_bbtautau" in self.global_params["analysis_config_area"].split("/")

        if len(branch_set) > 0:
            reqs["anaTuple"] = AnaTupleMergeTask.req(
                self,
                branches=tuple(branch_set),
                customisations=self.customisations,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                n_cpus=AnaTupleMergeTask.n_cpus._default,
            )

        if len(branch_set_cache) > 0:
            if isbbtt:
                reqs["anaCacheTuple"] = AnaCacheTupleTask.req(
                    self,
                    branches=tuple(branch_set_cache),
                    customisations=self.customisations,
                )
            else:
                reqs["analysisCache"] = []
                for producer_name in (p for p in producer_set if p is not None):
                    reqs["analysisCache"].append(
                        AnalysisCacheTask.req(
                            self,
                            branches=tuple(branch_set_cache),
                            customisations=self.customisations,
                            producer_to_run=producer_name,
                        )
                    )
        return reqs

    def requires(self):
        sample_name, prod_br, need_cache_global, producer_list, input_index = (
            self.branch_data
        )
        deps = (
            []
        )  # deps cannot be a set because sets are auto-sorted, creating a mismatch between main (AnaTuple) file and cache files
        isbbtt = "HH_bbtautau" in self.global_params["analysis_config_area"].split("/")

        deps.append(
            AnaTupleMergeTask.req(
                self,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
                customisations=self.customisations,
            )
        )
        caches = set()
        if need_cache_global:
            if isbbtt:
                caches.add(
                    AnaCacheTupleTask.req(
                        self,
                        max_runtime=AnaCacheTupleTask.max_runtime._default,
                        branch=prod_br,
                        branches=(prod_br,),
                        customisations=self.customisations,
                    )
                )
            else:
                for producer_name in (p for p in producer_list if p is not None):
                    caches.add(
                        AnalysisCacheTask.req(
                            self,
                            max_runtime=AnalysisCacheTask.max_runtime._default,
                            branch=prod_br,
                            branches=(prod_br,),
                            customisations=self.customisations,
                            producer_to_run=producer_name,
                        )
                    )
            if caches:
                caches_list = list(caches)
                deps.extend(caches_list)
        return deps

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

        var_produced_by = self.setup.var_producer_map

        n = 0
        branches = {}
        anaProd_branch_map = AnaTupleMergeTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()

        samples_to_consider = [
            key
            for key in self.samples.keys()
            if self.samples[key]["process_group"] != "data"
        ]
        samples_to_consider.append("data")

        # samples_to_consider = GetSamples(
        #     self.samples, self.setup.backgrounds, self.global_params["signal_types"]
        # )

        need_cache_list = [
            (var_name in var_produced_by, var_produced_by.get(var_name, None))
            for var_name in self.global_params["variables"]
        ]
        producer_list = []
        need_cache_global = any(item[0] for item in need_cache_list)
        for var_name in self.global_params["variables"]:
            need_cache = True if var_name in var_produced_by else False
            producer_to_run = (
                var_produced_by[var_name] if var_name in var_produced_by else None
            )
            need_cache_list.append(need_cache)
            producer_list.append(producer_to_run)

        for prod_br, (
            sample_name,
            sample_type,
            input_file_list,
            output_file_list,
        ) in anaProd_branch_map.items():
            if sample_name not in samples_to_consider:
                continue

            for input_index in range(len(output_file_list)):
                branches[n] = (
                    sample_name,
                    prod_br,
                    need_cache_global,
                    producer_list,
                    input_index,
                )
                n += 1
        return branches

    def output(self):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")
        sample_name, prod_br, need_cache_global, producer_list, input_index = (
            self.branch_data
        )
        input = self.input()[0][input_index]
        outFileName = f"histTuple_" + os.path.basename(input.path).split("_")[1]
        output_path = os.path.join(
            "histTuples", self.version, self.period, sample_name, outFileName
        )
        return self.remote_target(output_path, fs=self.fs_HistTuple)

    def run(self):
        sample_name, prod_br, need_cache_global, producer_list, input_index = (
            self.branch_data
        )
        # print("My inputs are")
        # print(self.input())
        # input_paths = [ x.path for x in self.input() ]
        # print(input_paths[0])
        input_file = self.input()[0][input_index]
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = customisation_dict.get(
            "channels", self.global_params["channelSelection"]
        )
        if type(channels) == list:
            channels = ",".join(channels)

        print(f"input file is {input_file.path}")
        histTupleDef = os.path.join(self.ana_path(), self.global_params["histTupleDef"])
        HistTupleProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "HistTupleProducer.py"
        )
        outFile = self.output().path
        print(f"output file is {outFile}")
        compute_unc_histograms = (
            customisation_dict.get("compute_unc_histograms") == "True"
            if "compute_unc_histograms" in customisation_dict
            else self.global_params.get("compute_unc_histograms", False)
        )
        job_home, remove_job_home = self.law_job_home()
        with contextlib.ExitStack() as stack:
            local_input = stack.enter_context((input_file).localize("r"))
            tmpFile = os.path.join(
                job_home, f"HistTupleProducerTask_{input_index}.root"
            )
            print(f"tmpfile is {tmpFile}")
            HistTupleProducer_cmd = [
                "python3",
                HistTupleProducer,
                "--inFile",
                local_input.path,
                "--outFile",
                tmpFile,
                "--dataset",
                sample_name,
                "--histTupleDef",
                histTupleDef,
                "--period",
                self.period,
                "--channels",
                channels,
            ]
            if compute_unc_histograms:
                HistTupleProducer_cmd.extend(
                    [
                        "--compute_rel_weights",
                        "True",
                        "--compute_unc_variations",
                        "True",
                    ]
                )
            if self.customisations:
                HistTupleProducer_cmd.extend([f"--customisations", self.customisations])
            if need_cache_global:
                isbbtt = "HH_bbtautau" in self.global_params[
                    "analysis_config_area"
                ].split("/")
                if isbbtt:
                    anaCache_file = self.input()[1][input_index]
                    with anaCache_file.localize("r") as local_anacache:
                        HistTupleProducer_cmd.extend(
                            ["--cacheFiles", local_anacache.path]
                        )
                ana_cache_inputs = self.input()[1:]
                local_anacache_list = [
                    stack.enter_context((inp[input_index]).localize("r")).path
                    for inp in ana_cache_inputs
                ]
                HistTupleProducer_cmd.extend(
                    ["--cacheFile", ",".join(local_anacache_list)]
                )
            ps_call(HistTupleProducer_cmd, verbose=1)

            with self.output().localize("w") as local_output:
                out_local_path = local_output.path
                shutil.move(tmpFile, out_local_path)
        if remove_job_home:
            shutil.rmtree(job_home)


class HistFromNtupleProducerTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 10.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)

    def workflow_requires(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            req_dict = {}
            req_dict["HistTupleProducerTask"] = HistTupleProducerTask.req(
                self, branches=(), customisations=self.customisations
            )
            return req_dict
        branch_set = set()
        for br_idx, (var, prod_br_list, sample_names) in self.branch_map.items():
            if var in self.global_params["variables"]:
                branch_set.update(prod_br_list)
        branches = tuple(branch_set)
        req_dict = {
            "HistTupleProducerTask": HistTupleProducerTask.req(
                self, branches=branches, customisations=self.customisations
            )
        }
        return req_dict

    def requires(self):
        var, prod_br_list, sample_name = self.branch_data
        reqs = []
        reqs.append(
            HistTupleProducerTask.req(
                self,
                max_runtime=HistTupleProducerTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
                customisations=self.customisations,
            )
            for prod_br in prod_br_list
        )
        return reqs

    def create_branch_map(self):
        branches = {}
        prod_br_list = []
        current_sample = None
        n = 0

        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            self.cache_branch_map = False
            if not hasattr(self, "_branches_backup"):
                self._branches_backup = copy.deepcopy(self.branches)
            return {0: ()}
        sample_to_branches = {}
        HistTupleBranchMap = HistTupleProducerTask.req(
            self, branches=()
        ).create_branch_map()
        for prod_br, (
            histTuple_sample_name,
            histTuple_prod_br,
            need_cache_global,
            producer_list,
            input_index,
        ) in HistTupleBranchMap.items():
            sample_to_branches.setdefault(histTuple_sample_name, []).append(prod_br)

        for sample_name, prod_br_list in sample_to_branches.items():
            for var_name in self.global_params["variables"]:
                branches[n] = (var_name, prod_br_list, sample_name)
                n += 1

        return branches

    def output(self):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")
        var, prod_br, sample_name = self.branch_data
        output_path = os.path.join(
            "hists", self.version, self.period, var, f"{sample_name}.root"
        )
        return self.remote_target(output_path, fs=self.fs_HistTuple)

    def run(self):
        var, prod_br, sample_name = self.branch_data
        job_home, remove_job_home = self.law_job_home()
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = (
            customisation_dict["channels"]
            if "channels" in customisation_dict.keys()
            else self.global_params["channelSelection"]
        )
        # Channels from the yaml are a list, but the format we need for the ps_call later is 'ch1,ch2,ch3', basically join into a string separated by comma
        if type(channels) == list:
            channels = ",".join(channels)
        # bbww does not use a deepTauVersion
        compute_unc_histograms = (
            customisation_dict["compute_unc_histograms"] == "True"
            if "compute_unc_histograms" in customisation_dict.keys()
            else self.global_params.get("compute_unc_histograms", False)
        )
        HistFromNtupleProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "HistProducerFromNTuple.py"
        )
        input_list_remote_target = [inp for inp in self.input()[0]]
        with contextlib.ExitStack() as stack:
            local_inputs = [
                stack.enter_context((inp).localize("r")).path for inp in self.input()[0]
            ]

            tmpFile = os.path.join(job_home, f"HistFromNtuple_{var}.root")

            HistFromNtupleProducer_cmd = [
                "python3",
                HistFromNtupleProducer,
                "--period",
                self.period,
                "--outFile",
                tmpFile,
                "--channels",
                channels,
                "--var",
                var,
                "--sample_name",
                sample_name,
            ]
            if compute_unc_histograms:
                HistFromNtupleProducer_cmd.extend(
                    [
                        "--compute_rel_weights",
                        "True",
                        "--compute_unc_variations",
                        "True",
                    ]
                )
            if self.customisations:
                HistFromNtupleProducer_cmd.extend(
                    [f"--customisations", self.customisations]
                )

            HistFromNtupleProducer_cmd.extend(local_inputs)
            ps_call(HistFromNtupleProducer_cmd, verbose=1)

        with (self.output()).localize("w") as tmp_local_file:
            out_local_path = tmp_local_file.path
            shutil.move(tmpFile, out_local_path)

        delete_after_merge = False  # var == self.global_config["variables"][-1] --> find more robust condition
        if delete_after_merge:
            print(f"Finished HistogramProducer, lets delete remote targets")
            for remote_target in input_list_remote_target:
                remote_target.remove()
                with remote_target.localize("w") as tmp_local_file:
                    tmp_local_file.touch()  # Create a dummy to avoid dependency crashes

        if remove_job_home:
            shutil.rmtree(job_home)


class HistMergerTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)

    def workflow_requires(self):
        branch_map = self.create_branch_map()

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
        all_samples = {}
        for br_idx, (var, prod_br_list, sample_names) in self.branch_map.items():
            all_samples[var] = prod_br_list

        new_branchset = set()
        for var in all_samples.keys():
            new_branchset.update(all_samples[var])

        return {
            "HistFromNtupleProducerTask": HistFromNtupleProducerTask.req(
                self, branches=list(new_branchset)
            )
        }

    def requires(self):
        var_name, br_indices, samples = self.branch_data
        reqs = [
            HistFromNtupleProducerTask.req(
                self,
                max_runtime=HistFromNtupleProducerTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
                customisations=self.customisations,
            )
            for prod_br in tuple(set(br_indices))
        ]

        return reqs

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
        HistFromNtupleProducerTask_branch_map = HistFromNtupleProducerTask.req(
            self, branches=()
        ).create_branch_map()
        all_samples = {}
        branches = {}
        k = 0
        for br_idx, (
            var_name,
            prod_br_list,
            current_sample,
        ) in HistFromNtupleProducerTask_branch_map.items():
            if var_name not in all_samples.keys():
                all_samples[var_name] = []
            all_samples[var_name].append((br_idx, current_sample))
        for var_name, br_list in all_samples.items():
            br_indices = []
            samples = []
            for key in br_list:
                idx, sample_name = key
                br_indices.append(idx)
                samples.append(sample_name)
            branches[k] = (var_name, br_indices, samples)
            k += 1
        return branches

    def output(self):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")
        var_name, br_indices, samples = self.branch_data
        output_path = os.path.join("merged_hists", self.version, self.period, var_name)
        output_file_name = os.path.join(output_path, f"{var_name}.root")
        return self.remote_target(output_file_name, fs=self.fs_HistTuple)

    def run(self):
        var_name, br_indices, samples = self.branch_data
        customisation_dict = getCustomisationSplit(self.customisations)

        channels = (
            customisation_dict["channels"]
            if "channels" in customisation_dict.keys()
            else self.global_params["channelSelection"]
        )
        # Channels from the yaml are a list, but the format we need for the ps_call later is 'ch1,ch2,ch3', basically join into a string separated by comma
        if type(channels) == list:
            channels = ",".join(channels)

        uncNames = ["Central"]
        unc_config = os.path.join(
            self.ana_path(), "config", self.period, f"weights.yaml"
        )
        unc_cfg_dict = load_unc_config(unc_config)
        uncs_to_exclude = (
            self.global_params["uncs_to_exclude"][self.period]
            if "uncs_to_exclude" in self.global_params.keys()
            else []
        )
        compute_unc_histograms = (
            customisation_dict["compute_unc_histograms"] == "True"
            if "compute_unc_histograms" in customisation_dict.keys()
            else self.global_params.get("compute_unc_histograms", False)
        )
        if compute_unc_histograms:
            for uncName in list(unc_cfg_dict["norm"].keys()) + list(
                unc_cfg_dict["shape"].keys()
            ):
                if uncName in uncs_to_exclude:
                    continue
                uncNames.append(uncName)

        MergerProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "HistMergerFromHists.py"
        )
        HaddMergedHistsProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "hadd_merged_hists.py"
        )
        RenameHistsProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "renameHists.py"
        )

        input_dir = os.path.join("hists", self.version, self.period, var_name)
        input_dir_remote = self.remote_dir_target(input_dir, fs=self.fs_HistTuple)
        all_datasets = []
        local_inputs = []
        with contextlib.ExitStack() as stack:
            for inp in self.input():
                sample_name = os.path.basename(inp.path)
                all_datasets.append(sample_name.strip(".root"))
                local_inputs.append(stack.enter_context(inp.localize("r")).path)
            dataset_names = ",".join(smpl for smpl in all_datasets)
            all_outputs_merged = []
            outdir_histograms = os.path.join(
                self.version, self.period, "merged", var_name, "tmp"
            )
            if len(uncNames) == 1:
                with self.output().localize("w") as outFile:
                    MergerProducer_cmd = [
                        "python3",
                        MergerProducer,
                        "--outFile",
                        outFile.path,
                        "--var",
                        var_name,
                        "--dataset_names",
                        dataset_names,
                        "--uncSource",
                        uncNames[0],
                        "--channels",
                        channels,
                        "--period",
                        self.period,
                    ]
                    MergerProducer_cmd.extend(local_inputs)
                    ps_call(MergerProducer_cmd, verbose=1)
            else:
                for uncName in uncNames:
                    final_histname = f"{var_name}_{uncName}.root"
                    tmp_outfile_merge = os.path.join(outdir_histograms, final_histname)
                    tmp_outfile_merge_remote = self.remote_target(
                        tmp_outfile_merge, fs=self.fs_histograms
                    )
                    with tmp_outfile_merge_remote.localize(
                        "w"
                    ) as tmp_outfile_merge_unc:
                        MergerProducer_cmd = [
                            "python3",
                            MergerProducer,
                            "--outFile",
                            tmp_outfile_merge_unc.path,
                            "--var",
                            var_name,
                            "--dataset_names",
                            dataset_names,
                            "--uncSource",
                            uncName,
                            "--channels",
                            channels,
                            "--period",
                            self.period,
                        ]
                        MergerProducer_cmd.extend(local_inputs)
                        ps_call(MergerProducer_cmd, verbose=1)
                        all_outputs_merged.append(tmp_outfile_merge)
            if len(uncNames) > 1:
                all_uncertainties_string = ",".join(unc for unc in uncNames)
                tmp_outFile = self.remote_target(
                    os.path.join(
                        outdir_histograms, f"all_histograms_{var_name}_hadded.root"
                    ),
                    fs=self.fs_histograms,
                )
                with contextlib.ExitStack() as stack:
                    local_merged_files = []
                    for infile_merged in all_outputs_merged:
                        tmp_outfile_merge_remote = self.remote_target(
                            infile_merged, fs=self.fs_histograms
                        )
                        local_merged_files.append(
                            stack.enter_context(
                                tmp_outfile_merge_remote.localize("r")
                            ).path
                        )
                    with self.output().localize("w") as outFile:
                        HaddMergedHistsProducer_cmd = [
                            "python3",
                            HaddMergedHistsProducer,
                            "--outFile",
                            outFile.path,
                            "--var",
                            var_name,
                        ]
                        HaddMergedHistsProducer_cmd.extend(local_merged_files)
                        ps_call(HaddMergedHistsProducer_cmd, verbose=1)


class HistProducerFileTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)

    def workflow_requires(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            req_dict = {
                "AnaTupleFileListTask": AnaTupleFileListTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleFileListTask.max_runtime._default,
                    n_cpus=AnaTupleFileListTask.n_cpus._default,
                ),
                "AnaTupleMergeTask": AnaTupleMergeTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleMergeTask.max_runtime._default,
                    n_cpus=AnaTupleMergeTask.n_cpus._default,
                ),
            }
            # Get all the producers to require for this dummy branch
            producer_set = set()
            var_produced_by = self.setup.var_producer_map
            for var_name in self.global_params["vars_to_plot"]:
                need_cache = True if var_name in var_produced_by else False
                producer_to_run = (
                    var_produced_by[var_name] if var_name in var_produced_by else None
                )
                producer_set.add(producer_to_run)
            req_dict["AnalysisCacheTask"] = [
                AnalysisCacheTask.req(
                    self,
                    branches=(),
                    customisations=self.customisations,
                    producer_to_run=producer_name,
                )
                for producer_name in list(producer_set)
                if producer_name is not None
            ]
            return req_dict

        branch_set = set()
        branch_set_cache = set()
        producer_set = set()
        for idx, (
            sample,
            br,
            var_list,
            need_cache_list,
            producer_list,
            input_index,
        ) in self.branch_map.items():
            branch_set.add(br)
            if any(need_cache_list):
                branch_set_cache.add(br)
                for producer_name in (p for p in producer_list if p is not None):
                    producer_set.add(producer_name)
        reqs = {}
        isbbtt = "HH_bbtautau" in self.global_params["analysis_config_area"].split("/")

        if len(branch_set) > 0:
            reqs["anaTuple"] = AnaTupleMergeTask.req(
                self,
                branches=tuple(branch_set),
                customisations=self.customisations,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                n_cpus=AnaTupleMergeTask.n_cpus._default,
            )
        if len(branch_set_cache) > 0:
            if isbbtt:
                reqs["anaCacheTuple"] = AnaCacheTupleTask.req(
                    self,
                    branches=tuple(branch_set_cache),
                    customisations=self.customisations,
                )
            else:
                reqs["analysisCache"] = []
                for producer_name in (p for p in producer_set if p is not None):
                    reqs["analysisCache"].append(
                        AnalysisCacheTask.req(
                            self,
                            branches=tuple(branch_set_cache),
                            customisations=self.customisations,
                            producer_to_run=producer_name,
                        )
                    )
        return reqs

    def requires(self):
        sample_name, prod_br, var_list, need_cache_list, producer_list, input_index = (
            self.branch_data
        )
        deps = []

        isbbtt = "HH_bbtautau" in self.global_params["analysis_config_area"].split("/")

        deps.append(
            AnaTupleMergeTask.req(
                self,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
                customisations=self.customisations,
            )
        )
        if any(need_cache_list):
            if isbbtt:
                deps.append(
                    AnaCacheTupleTask.req(
                        self,
                        max_runtime=AnaCacheTupleTask.max_runtime._default,
                        branch=prod_br,
                        branches=(prod_br,),
                        customisations=self.customisations,
                    )
                )
            else:
                for producer_name in (p for p in producer_list if p is not None):
                    deps.append(
                        AnalysisCacheTask.req(
                            self,
                            max_runtime=AnalysisCacheTask.max_runtime._default,
                            branch=prod_br,
                            branches=(prod_br,),
                            customisations=self.customisations,
                            producer_to_run=producer_name,
                        )
                    )
        return deps

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

        # map indicating which variable needs which cache producer
        var_produced_by = self.setup.var_producer_map

        n = 0
        branches = {}
        anaProd_branch_map = AnaTupleMergeTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        samples_to_consider = [
            key
            for key in self.samples.keys()
            if self.samples[key]["process_group"] != "data"
        ]
        samples_to_consider.append("data")
        var_list = []
        need_cache_list = []
        producer_list = []

        for var_name in self.global_params["vars_to_plot"]:
            need_cache = True if var_name in var_produced_by else False
            producer_to_run = (
                var_produced_by[var_name] if var_name in var_produced_by else None
            )
            var_list.append(var_name)
            need_cache_list.append(need_cache)
            producer_list.append(producer_to_run)
        for prod_br, (
            sample_name,
            process_group,
            input_file_list,
            output_file_list,
        ) in anaProd_branch_map.items():
            if sample_name not in samples_to_consider:
                continue
            for input_index in range(len(output_file_list)):
                branches[n] = (
                    sample_name,
                    prod_br,
                    var_list,
                    need_cache_list,
                    producer_list,
                    input_index,
                )
                n += 1
        return branches

    def output(self):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")
        sample_name, prod_br, var_list, need_cache_list, producer_list, input_index = (
            self.branch_data
        )
        input = self.input()[0][input_index]
        outFileName = os.path.basename(input.path)
        prod_dir = "prod"
        return_list = []
        for var, producer_name in zip(var_list, producer_list):
            if producer_name:
                output_path = os.path.join(
                    self.version,
                    self.period,
                    prod_dir,
                    producer_name,
                    var,
                    f"{sample_name}_{outFileName}",
                )
            else:
                output_path = os.path.join(
                    self.version,
                    self.period,
                    prod_dir,
                    var,
                    f"{sample_name}_{outFileName}",
                )
            return_list.append(self.remote_target(output_path, fs=self.fs_histograms))
        return return_list

    def run(self):
        sample_name, prod_br, var_list, need_cache_list, producer_list, input_index = (
            self.branch_data
        )
        input_file = self.input()[0][input_index]
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = (
            customisation_dict["channels"]
            if "channels" in customisation_dict.keys()
            else self.global_params["channelSelection"]
        )
        # Channels from the yaml are a list, but the format we need for the ps_call later is 'ch1,ch2,ch3', basically join into a string separated by comma
        if type(channels) == list:
            channels = ",".join(channels)
        # bbww does not use a deepTauVersion
        deepTauVersion = ""
        isbbtt = "HH_bbtautau" in self.global_params["analysis_config_area"].split("/")
        if isbbtt:
            deepTauVersion = (
                customisation_dict["deepTauVersion"]
                if "deepTauVersion" in customisation_dict.keys()
                else self.global_params["deepTauVersion"]
            )
        region = (
            customisation_dict["region"]
            if "region" in customisation_dict.keys()
            else self.global_params["region_default"]
        )
        print(f"input file is {input_file.path}")
        global_config = os.path.join(
            self.ana_path(), self.global_params["analysis_config_area"], f"global.yaml"
        )
        unc_config = os.path.join(
            self.ana_path(), "config", self.period, f"weights.yaml"
        )
        process_group = (
            self.samples[sample_name]["process_group"]
            if sample_name != "data"
            else "data"
        )
        HistProducerFile = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "HistProducerFile.py"
        )
        output_file_paths = [out.path for out in self.output()]
        print(f"output files are {output_file_paths}")
        compute_unc_histograms = (
            customisation_dict["compute_unc_histograms"] == "True"
            if "compute_unc_histograms" in customisation_dict.keys()
            else self.global_params.get("compute_unc_histograms", False)
        )
        job_home, remove_job_home = self.law_job_home()
        with input_file.localize("r") as local_input:
            ana_cache_input_counter = 1  # When we have multiple anaCache payloads, then we need to get the correct input for each one
            for var, need_cache, producer_name, output in zip(
                var_list, need_cache_list, producer_list, self.output()
            ):
                print(f"Starting var {var} with need_cache {need_cache}")
                if output.exists():
                    print(
                        f"Output for {var} {producer_name} {output} already exists! Continue"
                    )
                    ana_cache_input_counter += (
                        1  # Index to the next ana cache input for next time
                    )
                    continue
                tmpFile = os.path.join(job_home, f"HistProducerFileTask.root")
                HistProducerFile_cmd = [
                    "python3",
                    HistProducerFile,
                    "--inFile",
                    local_input.path,
                    "--outFileName",
                    tmpFile,
                    "--dataset",
                    sample_name,
                    "--uncConfig",
                    unc_config,
                    "--histConfig",
                    self.setup.hist_config_path,
                    "--sampleType",
                    process_group,
                    "--globalConfig",
                    global_config,
                    "--var",
                    var,
                    "--period",
                    self.period,
                    "--region",
                    region,
                    "--channels",
                    channels,
                ]
                if compute_unc_histograms:
                    HistProducerFile_cmd.extend(
                        [
                            "--compute_rel_weights",
                            "True",
                            "--compute_unc_variations",
                            "True",
                        ]
                    )
                    # HistProducerFile_cmd.extend(['--compute_rel_weights', 'True', '--compute_unc_variations', 'False'])
                if (deepTauVersion != "2p1") and (deepTauVersion != ""):
                    HistProducerFile_cmd.extend(["--deepTauVersion", deepTauVersion])
                if need_cache:
                    anaCache_file = self.input()[ana_cache_input_counter][input_index]
                    ana_cache_input_counter += (
                        1  # Index to the next ana cache input for next time
                    )
                    with anaCache_file.localize("r") as local_anacache:
                        HistProducerFile_cmd.extend(
                            ["--cacheFile", local_anacache.path]
                        )
                        ps_call(HistProducerFile_cmd, verbose=1)
                else:
                    ps_call(HistProducerFile_cmd, verbose=1)

                with output.localize("w") as local_output:
                    out_local_path = local_output.path
                    shutil.move(tmpFile, out_local_path)
        if remove_job_home:
            shutil.rmtree(job_home)


class HistProducerSampleTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

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
                "HistProducerFileTask": HistProducerFileTask.req(self, branches=()),
            }

        branch_set = set()
        for br_idx, (sample_name, dep_br_list, var_list) in self.branch_map.items():
            branch_set.update(dep_br_list)
        branches = tuple(branch_set)
        deps = {
            "HistProducerFileTask": HistProducerFileTask.req(
                self, branches=branches, customisations=self.customisations
            )
        }
        return deps

    def requires(self):
        sample_name, dep_br_list, var_list = self.branch_data
        reqs = [
            HistProducerFileTask.req(
                self,
                max_runtime=HistProducerFileTask.max_runtime._default,
                branch=dep_br,
                branches=(dep_br,),
                customisations=self.customisations,
            )
            for dep_br in dep_br_list
        ]
        return reqs

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
        histProducerFile_map = HistProducerFileTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        all_samples = {}
        samples_to_consider = [
            key
            for key in self.samples.keys()
            if self.samples[key]["process_group"] != "data"
        ]
        samples_to_consider.append("data")
        for n_branch, (
            sample_name,
            prod_br,
            var_list,
            need_cache_list,
            producer_list,
            input_index,
        ) in histProducerFile_map.items():
            if sample_name not in samples_to_consider:
                continue
            if sample_name not in all_samples:
                all_samples[sample_name] = []
            all_samples[sample_name].append(n_branch)
        k = 0
        for sample_name, branch_idx_list in all_samples.items():
            branches[k] = (sample_name, branch_idx_list, var_list)
            k += 1
        return branches

    def output(self):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")
        sample_name, idx_list, var_list = self.branch_data
        split_dir = "split"
        return_list = []
        for var in var_list:
            output_path = os.path.join(
                self.version, self.period, split_dir, var, f"{sample_name}.root"
            )
            return_list.append(self.remote_target(output_path, fs=self.fs_histograms))
        return return_list

    def run(self):
        sample_name, idx_list, var_list = self.branch_data
        HistProducerSample = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "HistProducerSample.py"
        )
        job_home, remove_job_home = self.law_job_home()
        with contextlib.ExitStack() as stack:
            for idx, var in enumerate(var_list):
                if (self.output()[idx]).exists():
                    print(
                        f"Output for {var} {self.output()[idx]} already exists! Continue"
                    )
                    continue
                local_inputs = [
                    stack.enter_context((inp[idx]).localize("r")).path
                    for inp in self.input()
                ]  # HistFile now has a list of outputs, get the index input
                tmpFile = os.path.join(job_home, f"HistProducerFileTask.root")
                HistProducerSample_cmd = [
                    "python3",
                    HistProducerSample,
                    "--outFile",
                    tmpFile,
                ]  # , '--remove-files', 'True']
                HistProducerSample_cmd.extend(local_inputs)
                ps_call(HistProducerSample_cmd, verbose=1)

                with (self.output()[idx]).localize("w") as tmp_local_file:
                    out_local_path = tmp_local_file.path
                    shutil.move(tmpFile, out_local_path)
        if remove_job_home:
            shutil.rmtree(job_home)


class MergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 30.0)

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

        histProducerSample_map = HistProducerSampleTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()
        all_samples = {}
        branches = {}
        for br_idx, (smpl_name, idx_list, var_list) in histProducerSample_map.items():
            for var in var_list:
                if var not in all_samples:
                    all_samples[var] = []
                all_samples[var].append(br_idx)

        new_branchset = set()
        for var in all_samples.keys():
            new_branchset.update(all_samples[var])

        return {
            "histproducersample": HistProducerSampleTask.req(
                self, branches=list(new_branchset)
            )
        }

    def requires(self):
        var, idx, branches_idx = self.branch_data
        deps = [
            HistProducerSampleTask.req(
                self,
                max_runtime=HistProducerSampleTask.max_runtime._default,
                branch=prod_br,
                customisations=self.customisations,
            )
            for prod_br in branches_idx
        ]
        return deps

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
        histProducerSample_map = HistProducerSampleTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()
        all_samples = {}
        branches = {}
        for br_idx, (smpl_name, idx_list, var_list) in histProducerSample_map.items():
            for idx, var in enumerate(var_list):
                var_key = f"{var}:{idx}"
                if var_key not in all_samples:
                    all_samples[var_key] = []
                all_samples[var_key].append(br_idx)
        k = 0
        for n, key in enumerate(all_samples.items()):
            var_key, branches_idx = key
            var = var_key.split(":")[0]  # Get the variable name without the index
            idx = var_key.split(":")[1]  # Get the index without the variable name
            branches[k] = (var, idx, branches_idx)
            k += 1
        return branches

    def output(self):
        var, idx, branches_idx = self.branch_data
        merge_dir = "merged"
        output_path = os.path.join(
            self.version, self.period, merge_dir, var, f"{var}.root"
        )
        return self.remote_target(output_path, fs=self.fs_histograms)

    def run(self):
        var, idx, branches_idx = self.branch_data
        sample_config = os.path.join(
            self.ana_path(), "FLAF", "config", self.period, f"samples.yaml"
        )
        global_config = os.path.join(
            self.ana_path(), self.global_params["analysis_config_area"], f"global.yaml"
        )
        unc_config = os.path.join(
            self.ana_path(), "config", self.period, f"weights.yaml"
        )
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = (
            customisation_dict["channels"]
            if "channels" in customisation_dict.keys()
            else self.global_params["channelSelection"]
        )
        # Channels from the yaml are a list, but the format we need for the ps_call later is 'ch1,ch2,ch3', basically join into a string separated by comma
        if type(channels) == list:
            channels = ",".join(channels)
        # bbww does not use a deepTauVersion
        deepTauVersion = ""
        isbbtt = "HH_bbtautau" in self.global_params["analysis_config_area"].split("/")
        if isbbtt:
            deepTauVersion = (
                customisation_dict["deepTauVersion"]
                if "deepTauVersion" in customisation_dict.keys()
                else self.global_params["deepTauVersion"]
            )
        region = (
            customisation_dict["region"]
            if "region" in customisation_dict.keys()
            else self.global_params["region_default"]
        )
        (
            customisation_dict["apply_btag_shape_weights"] == "True"
            if "apply_btag_shape_weights" in customisation_dict.keys()
            else self.global_params.get("apply_btag_shape_weights", False)
        )
        uncNames = ["Central"]
        unc_cfg_dict = load_unc_config(unc_config)
        uncs_to_exclude = (
            self.global_params["uncs_to_exclude"][self.period]
            if "uncs_to_exclude" in self.global_params.keys()
            else []
        )

        compute_unc_histograms = (
            customisation_dict["compute_unc_histograms"] == "True"
            if "compute_unc_histograms" in customisation_dict.keys()
            else self.global_params.get("compute_unc_histograms", False)
        )
        if compute_unc_histograms:
            for uncName in list(unc_cfg_dict["norm"].keys()) + list(
                unc_cfg_dict["shape"].keys()
            ):
                if uncName in uncs_to_exclude:
                    continue
                uncNames.append(uncName)
        print(uncNames)
        MergerProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "HistMerger.py"
        )
        HaddMergedHistsProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "hadd_merged_hists.py"
        )
        RenameHistsProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "renameHists.py"
        )

        all_inputs = []
        samples_to_consider = GetSamples(
            self.samples, self.setup.backgrounds, self.global_params["signal_types"]
        )
        samples_to_consider = [
            key
            for key in self.samples.keys()
            if self.samples[key]["process_group"] != "data"
        ]
        samples_to_consider.append("data")
        for sample_name in samples_to_consider:
            print(sample_name)
            output_path_hist_prod_sample = os.path.join(
                self.version, self.period, "split", var, f"{sample_name}.root"
            )
            all_inputs.append(
                (
                    self.remote_target(
                        output_path_hist_prod_sample, fs=self.fs_histograms
                    ),
                    sample_name,
                )
            )
        all_datasets = []
        all_outputs_merged = []

        outdir_histograms = os.path.join(
            self.version, self.period, "merged", var, "tmp"
        )

        with contextlib.ExitStack() as stack:
            local_inputs = []
            for inp, smpl in all_inputs:
                local_inputs.append(stack.enter_context(inp.localize("r")).path)
                all_datasets.append(smpl)
            dataset_names = ",".join(smpl for smpl in all_datasets)

            if len(uncNames) == 1:
                with self.output().localize("w") as outFile:
                    MergerProducer_cmd = [
                        "python3",
                        MergerProducer,
                        "--outFile",
                        outFile.path,
                        "--var",
                        var,
                        "--uncSource",
                        uncNames[0],
                        "--datasetFile",
                        dataset_names,
                        "--channels",
                        channels,
                        "--ana_path",
                        self.ana_path(),
                        "--period",
                        self.period,
                    ]  # , '--remove-files', 'True']
                    if "apply_btag_shape_weights" in customisation_dict.keys():
                        MergerProducer_cmd.append(
                            "--apply-btag-shape-weights",
                            customisation_dict["apply_btag_shape_weights"],
                        )
                    MergerProducer_cmd.extend(local_inputs)
                    ps_call(MergerProducer_cmd, verbose=1)
            else:
                for uncName in uncNames:
                    print(uncName)
                    final_histname = f"all_histograms_{var}_{uncName}.root"
                    tmp_outfile_merge = os.path.join(outdir_histograms, final_histname)
                    tmp_outfile_merge_remote = self.remote_target(
                        tmp_outfile_merge, fs=self.fs_histograms
                    )
                    with tmp_outfile_merge_remote.localize(
                        "w"
                    ) as tmp_outfile_merge_unc:
                        MergerProducer_cmd = [
                            "python3",
                            MergerProducer,
                            "--outFile",
                            tmp_outfile_merge_unc.path,
                            "--var",
                            var,
                            "--uncSource",
                            uncName,
                            "--datasetFile",
                            dataset_names,
                            "--channels",
                            channels,
                            "--ana_path",
                            self.ana_path(),
                            "--period",
                            self.period,
                        ]  # , '--remove-files', 'True']

                        MergerProducer_cmd.extend(local_inputs)
                        if "btagShape" in self.global_params["corrections"]:
                            MergerProducer_cmd.append("--apply-btag-shape-weights")
                            MergerProducer_cmd.append("True")
                        # print(MergerProducer_cmd)
                        ps_call(MergerProducer_cmd, verbose=1)
                    all_outputs_merged.append(tmp_outfile_merge)
        if len(uncNames) > 1:
            all_uncertainties_string = ",".join(unc for unc in uncNames)
            tmp_outFile = self.remote_target(
                os.path.join(outdir_histograms, f"all_histograms_{var}_hadded.root"),
                fs=self.fs_histograms,
            )
            with contextlib.ExitStack() as stack:
                local_merged_files = []
                for infile_merged in all_outputs_merged:
                    tmp_outfile_merge_remote = self.remote_target(
                        infile_merged, fs=self.fs_histograms
                    )
                    local_merged_files.append(
                        stack.enter_context(tmp_outfile_merge_remote.localize("r")).path
                    )
                with tmp_outFile.localize("w") as tmpFile:
                    HaddMergedHistsProducer_cmd = [
                        "python3",
                        HaddMergedHistsProducer,
                        "--outFile",
                        tmpFile.path,
                        "--var",
                        var,
                    ]
                    HaddMergedHistsProducer_cmd.extend(local_merged_files)
                    ps_call(HaddMergedHistsProducer_cmd, verbose=1)
            with tmp_outFile.localize("r") as tmpFile, self.output().localize(
                "w"
            ) as outFile:
                RenameHistsProducer_cmd = [
                    "python3",
                    RenameHistsProducer,
                    "--inFile",
                    tmpFile.path,
                    "--outFile",
                    outFile.path,
                    "--var",
                    var,
                    "--year",
                    getYear(self.period),
                    "--ana_path",
                    self.ana_path(),
                    "--period",
                    self.period,
                ]
                ps_call(RenameHistsProducer_cmd, verbose=1)


class AnalysisCacheTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    producer_to_run = luigi.Parameter()

    # Need to override this from HTCondorWorkflow to have separate data pathways for different cache tasks
    def htcondor_output_directory(self):
        return law.LocalDirectoryTarget(self.local_path(self.producer_to_run))

    def __init__(self, *args, **kwargs):
        # Needed to get the config and ht_condor_pathways figured out
        super(AnalysisCacheTask, self).__init__(*args, **kwargs)
        self.n_cpus = self.global_params["payload_producers"][self.producer_to_run].get(
            "n_cpus", 1
        )
        self.max_runtime = self.global_params["payload_producers"][
            self.producer_to_run
        ].get("max_runtime", 2.0)

    def workflow_requires(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            req_dict = {
                "AnaTupleFileListTask": AnaTupleFileListTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleFileListTask.max_runtime._default,
                    n_cpus=AnaTupleFileListTask.n_cpus._default,
                ),
                "AnaTupleMergeTask": AnaTupleMergeTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleMergeTask.max_runtime._default,
                    n_cpus=AnaTupleMergeTask.n_cpus._default,
                ),
            }
            # Get all the producers to require for this dummy branch
            producer_requires_set = set()
            producer_dependencies = self.global_params["payload_producers"][
                self.producer_to_run
            ]["dependencies"]
            if producer_dependencies:
                for dependency in producer_dependencies:
                    producer_requires_set.add(dependency)
            req_dict["AnalysisCacheTask"] = [
                AnalysisCacheTask.req(
                    self,
                    branches=(),
                    customisations=self.customisations,
                    producer_to_run=producer_name,
                )
                for producer_name in list(producer_requires_set)
                if producer_name is not None
            ]
            return req_dict

        workflow_dict = {}
        workflow_dict["anaTuple"] = {
            br_idx: AnaTupleMergeTask.req(
                self,
                branch=br_idx,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                n_cpus=AnaTupleMergeTask.n_cpus._default,
            )
            for br_idx, _ in self.branch_map.items()
        }
        producer_dependencies = self.global_params["payload_producers"][
            self.producer_to_run
        ]["dependencies"]
        if producer_dependencies:
            for dependency in producer_dependencies:
                workflow_dict[dependency] = {
                    br_idx: AnalysisCacheTask.req(
                        self,
                        branch=br_idx,
                        customisations=self.customisations,
                        producer_to_run=dependency,
                    )
                    for br_idx, _ in self.branch_map.items()
                }
        return workflow_dict

    def requires(self):
        producer_dependencies = self.global_params["payload_producers"][
            self.producer_to_run
        ]["dependencies"]
        requires_list = [
            AnaTupleMergeTask.req(
                self, max_runtime=AnaTupleMergeTask.max_runtime._default
            )
        ]
        if producer_dependencies:
            for dependency in producer_dependencies:
                requires_list.append(
                    AnalysisCacheTask.req(self, producer_to_run=dependency)
                )
        return requires_list

    def create_branch_map(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            self.cache_branch_map = False
            if not hasattr(self, "_branches_backup"):
                self._branches_backup = copy.deepcopy(self.branches)
            return {0: ()}
        branches = {}
        self.cache_branch_map = True
        if hasattr(self, "_branches_backup"):
            self.branches = self._branches_backup
        anaProd_branch_map = AnaTupleMergeTask.req(
            self, branch=-1, branches=()
        ).branch_map
        for br_idx, (
            sample_name,
            process_group,
            input_file_list,
            output_file_list,
        ) in anaProd_branch_map.items():
            branches[br_idx] = (sample_name, process_group, len(output_file_list))
        return branches

    def output(self):
        if len(self.branch_data) == 0:
            return self.local_target("dummy.txt")
        sample_name, process_group, nInputs = self.branch_data
        return_list = []
        for idx in range(nInputs):
            outFileName = os.path.basename(self.input()[0][idx].path)
            output_path = os.path.join(
                "AnalysisCache",
                self.version,
                self.period,
                sample_name,
                self.producer_to_run,
                outFileName,
            )
            return_list.append(
                self.remote_target(output_path, fs=self.fs_anaCacheTuple)
            )
        # return self.remote_target(output_path, fs=self.fs_AnalysisCache) # for some reason this line is not working even if I edit user_custom.yaml
        # return self.remote_target(output_path, fs=self.fs_anaCacheTuple)
        return return_list

    def run(self):
        sample_name, process_group, nInputs = self.branch_data
        unc_config = os.path.join(
            self.ana_path(), "config", self.period, f"weights.yaml"
        )
        analysis_cache_producer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "AnalysisCacheProducer.py"
        )
        global_config = os.path.join(self.ana_path(), "config", "global.yaml")
        thread = threading.Thread(target=update_kinit_thread)
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = (
            customisation_dict["channels"]
            if "channels" in customisation_dict.keys()
            else self.global_params["channelSelection"]
        )
        # Channels from the yaml are a list, but the format we need for the ps_call later is 'ch1,ch2,ch3', basically join into a string separated by comma
        if type(channels) == list:
            channels = ",".join(channels)
        thread.start()
        try:
            job_home, remove_job_home = self.law_job_home()
            print(f"At job_home {job_home}")
            for idx in range(nInputs):
                with contextlib.ExitStack() as stack:
                    # Enter a stack to maybe load the analysis cache files
                    input_file = self.input()[0][idx]
                    local_cache_files = ""
                    if len(self.input()) > 1:
                        local_cache_files = [
                            stack.enter_context(inp[idx].localize("r")).path
                            for inp in self.input()[1:]
                        ]

                    print(f"Task has cache input files {local_cache_files}")
                    output_file = self.output()[idx]
                    print(
                        f"considering sample {sample_name}, {process_group} and file {input_file.path}"
                    )
                    if output_file.exists():
                        print(f"Output file {output_file} already exists, continue")
                        continue
                    customisation_dict = getCustomisationSplit(self.customisations)
                    deepTauVersion = (
                        customisation_dict["deepTauVersion"]
                        if "deepTauVersion" in customisation_dict.keys()
                        else ""
                    )
                    tmpFile = os.path.join(job_home, f"HistProducerFileTask.root")
                    with input_file.localize("r") as local_input:
                        analysisCacheProducer_cmd = [
                            "python3",
                            analysis_cache_producer,
                            "--inFileName",
                            local_input.path,
                            "--outFileName",
                            tmpFile,
                            "--uncConfig",
                            unc_config,
                            "--globalConfig",
                            global_config,
                            "--channels",
                            channels,
                            "--producer",
                            self.producer_to_run,
                            "--workingDir",
                            job_home,
                        ]
                        if (
                            self.global_params["store_noncentral"]
                            and process_group != "data"
                        ):
                            analysisCacheProducer_cmd.extend(
                                ["--compute_unc_variations", "True"]
                            )
                        if deepTauVersion != "":
                            analysisCacheProducer_cmd.extend(
                                ["--deepTauVersion", deepTauVersion]
                            )
                        if len(self.input()) > 1:
                            analysisCacheProducer_cmd.extend(
                                ["--cacheFileNames", ",".join(local_cache_files)]
                            )
                        # Check if cmssw env is required
                        prod_env = (
                            self.cmssw_env
                            if self.global_params["payload_producers"][
                                self.producer_to_run
                            ].get("cmssw_env", False)
                            else None
                        )
                        ps_call(analysisCacheProducer_cmd, env=prod_env, verbose=1)
                    print(
                        f"Finished producing payload for producer={self.producer_to_run} with name={sample_name}, group={process_group}, file={input_file.path}"
                    )
                    with output_file.localize("w") as tmp_local_file:
                        out_local_path = tmp_local_file.path
                        shutil.move(tmpFile, out_local_path)
            if remove_job_home:
                shutil.rmtree(job_home)

        finally:
            kInit_cond.acquire()
            kInit_cond.notify_all()
            kInit_cond.release()
            thread.join()


class AnalysisCacheMergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)
    producer_to_run = luigi.Parameter()

    def workflow_requires(self):
        workflow_dep = {}
        for idx, prod_branches in self.branch_map.items():
            workflow_dep[idx] = AnalysisCacheTask.req(
                self, branches=prod_branches, producer_to_run=self.producer_to_run
            )
        return workflow_dep

    def requires(self):
        prod_branches = self.branch_data
        deps = [
            AnalysisCacheTask.req(
                self,
                max_runtime=AnaCacheTask.max_runtime._default,
                branch=prod_br,
                producer_to_run=self.producer_to_run,
            )
            for prod_br in prod_branches
        ]
        return deps

    def create_branch_map(self):
        anaProd_branch_map = AnalysisCacheTask.req(
            self, branch=-1, branches=(), producer_to_run=self.producer_to_run
        ).branch_map
        prod_branches = []
        for prod_br, (sample_name, process_group) in anaProd_branch_map.items():
            if process_group == "data":
                prod_branches.append(prod_br)
        return {0: prod_branches}

    def output(self, force_pre_output=False):
        outFileName = "merged_nano_0.root"
        output_path = os.path.join(
            "AnalysisCacheMerged",
            self.period,
            "data",
            self.version,
            self.producer_to_run,
            outFileName,
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


# must remove background samples.
class PlotTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def workflow_requires(self):
        merge_map = MergeTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()
        return {
            "merge": MergeTask.req(
                self,
                branches=tuple(merge_map.keys()),
                customisations=self.customisations,
            )
        }

    def create_branch_map(self):
        branches = {}
        merge_map = MergeTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()
        for k, (_, (var, _, _)) in enumerate(merge_map.items()):
            branches[k] = var
        return branches

    def requires(self):
        var = self.branch_data

        merge_map = MergeTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()
        merge_branch = next(br for br, (v, _, _) in merge_map.items() if v == var)

        return MergeTask.req(
            self,
            branch=merge_branch,
            customisations=self.customisations,
            max_runtime=MergeTask.max_runtime._default,
        )

    def output(self):
        var = self.branch_data
        outputs = {}
        customisation_dict = getCustomisationSplit(self.customisations)

        channels = customisation_dict.get(
            "channels", self.global_params["channelSelection"]
        )
        if isinstance(channels, str):
            channels = channels.split(",")

        base_cats = self.global_params.get("categories") or []
        boosted_cats = self.global_params.get("boosted_categories") or []
        categories = base_cats + boosted_cats
        if isinstance(categories, str):
            categories = categories.split(",")

        for ch in channels:
            for cat in categories:
                rel_path = os.path.join(
                    self.version, self.period, "plots", var, cat, f"{ch}_{var}.pdf"
                )
                outputs[f"{ch}_{cat}"] = self.remote_target(rel_path, fs=self.fs_plots)
        return outputs

    def run(self):
        var = self.branch_data
        era = self.period
        ver = self.version
        customisation_dict = getCustomisationSplit(self.customisations)

        plotter = os.path.join(self.ana_path(), "FLAF", "Analysis", "HistPlotter.py")

        def bool_flag(key, default):
            return (
                customisation_dict.get(
                    key, str(self.global_params.get(key, default))
                ).lower()
                == "true"
            )

        plot_unc = bool_flag("plot_unc", True)
        plot_wantData = bool_flag(f"plot_wantData_{var}", True)
        plot_wantSignals = bool_flag("plot_wantSignals", False)
        plot_wantQCD = bool_flag("plot_wantQCD", False)
        plot_rebin = bool_flag("plot_rebin", False)
        plot_analysis = customisation_dict.get(
            "plot_analysis", self.global_params.get("plot_analysis", "")
        )

        remote_in = (
            self.remote_target(
                os.path.join(
                    ver, era, "merged", var, "tmp", f"all_histograms_{var}_hadded.root"
                ),
                fs=self.fs_histograms,
            )
            if plot_unc
            else self.input()
        )

        with remote_in.localize("r") as local_input:
            infile = local_input.path
            print("Loading fname", infile)

            for output_key, output_target in self.output().items():
                ch, cat = output_key.split("_", 1)
                with output_target.localize("w") as local_pdf:
                    cmd = [
                        "python3",
                        plotter,
                        "--inFile",
                        infile,
                        "--outFile",
                        local_pdf.path,
                        "--bckgConfig",
                        os.path.join(
                            self.ana_path(),
                            self.global_params["analysis_config_area"],
                            "background_samples.yaml",
                        ),
                        "--globalConfig",
                        os.path.join(
                            self.ana_path(),
                            self.global_params["analysis_config_area"],
                            "global.yaml",
                        ),
                        "--sigConfig",
                        os.path.join(
                            self.ana_path(),
                            self.global_params["analysis_config_area"],
                            era,
                            "samples.yaml",
                        ),
                        "--var",
                        var,
                        "--category",
                        cat,
                        "--channel",
                        ch,
                        "--year",
                        era,
                        "--analysis",
                        plot_analysis,
                        "--ana_path",
                        self.ana_path(),
                        "--period",
                        self.period,
                    ]
                    if plot_wantData:
                        cmd.append("--wantData")
                    if plot_wantSignals:
                        cmd.append("--wantSignals")
                    if plot_wantQCD:
                        cmd += ["--wantQCD", "true"]
                    if plot_rebin:
                        cmd += ["--rebin", "true"]
                    ps_call(cmd, verbose=1)
