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
)
from FLAF.AnaProd.tasks import (
    AnaTupleFileListTask,
    AnaTupleMergeTask,
)
from FLAF.Common.Utilities import getCustomisationSplit


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

            flatten_vars = set()
            for var in self.global_params["variables"]:
                if isinstance(var, dict) and "vars" in var:
                    for v in var["vars"]:
                        flatten_vars.add(v)
                else:
                    flatten_vars.add(var)

            for var_name in flatten_vars:
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
            if self.global_params["apply_btagShape_weights"]:
                req_dict["btagShapeWeight"] = BtagShapeWeightCorrectionTask.req(
                    self,
                    branches=(),
                    customisations=self.customisations,
                )
            return req_dict

        branch_set = set()
        branch_set_cache = set()
        producer_set = set()
        for idx, (
            dataset,
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

        if len(branch_set) > 0:
            reqs["anaTuple"] = AnaTupleMergeTask.req(
                self,
                branches=tuple(branch_set),
                customisations=self.customisations,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                n_cpus=AnaTupleMergeTask.n_cpus._default,
            )

        if len(branch_set_cache) > 0:
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

        if self.global_params["apply_btagShape_weights"]:
            btag_shape_weight_branch_set = set()
            btag_shape_task_branch_map = BtagShapeWeightCorrectionTask.req(
                self, branch=-1
            ).create_branch_map()
            hist_tuple_branch_map = self.branch_map
            # filter out branches of BtagShapeWeightCorrectionTask that correspond to each sample of HistTupleProducerTask
            hist_tuple_sample_map = {}
            for idx, (
                hist_tuple_sample,
                hist_tuple_br,
                need_cache_global,
                producer_list,
                input_index,
            ) in hist_tuple_branch_map.items():
                # data doesn't have btag weights, only MC does
                # so data tasks should not even depend on BtagShapeWeightCorrectionTask
                btag_branches_for_hist_tuple_sample = [
                    idx
                    for idx, (
                        btag_sample,
                        process_group,
                        _,
                    ) in btag_shape_task_branch_map.items()
                    if btag_sample == hist_tuple_sample and process_group != "data"
                ]
                # MC samples must have exactly one BtagShapeWeightCorrectionTask per sample
                # Data samples must have exactly 0 bc they are skipped
                assert (
                    len(btag_branches_for_hist_tuple_sample) <= 1
                ), "Must be at most one BtagShapeWeightCorrectionTask branch per sample"
                if len(btag_branches_for_hist_tuple_sample) == 1:
                    btag_weight_shape_branch = btag_branches_for_hist_tuple_sample[0]
                    hist_tuple_sample_map[hist_tuple_sample] = btag_weight_shape_branch

            for sample, btag_branch in hist_tuple_sample_map.items():
                if sample == "data":
                    continue
                btag_shape_weight_branch_set.add(btag_branch)

            if len(btag_shape_weight_branch_set) > 0:
                reqs["btagShapeWeightCorr"] = BtagShapeWeightCorrectionTask.req(
                    self, branches=tuple(btag_shape_weight_branch_set)
                )

        return reqs

    def requires(self):
        dataset_name, prod_br, need_cache_global, producer_list, input_index = (
            self.branch_data
        )
        deps = {
            "anaTuple": AnaTupleMergeTask.req(
                self,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
                customisations=self.customisations,
            )
        }
        if need_cache_global:
            anaCaches = {}
            for producer_name in (p for p in producer_list if p is not None):
                if producer_name not in deps:
                    anaCaches[producer_name] = AnalysisCacheTask.req(
                        self,
                        max_runtime=AnalysisCacheTask.max_runtime._default,
                        branch=prod_br,
                        branches=(prod_br,),
                        customisations=self.customisations,
                        producer_to_run=producer_name,
                    )
            if len(anaCaches) > 0:
                deps["anaCaches"] = anaCaches

        isMC = dataset_name != "data"
        if self.global_params["apply_btagShape_weights"] and isMC:
            btag_shape_task_branch_map = BtagShapeWeightCorrectionTask.req(
                self, branch=-1
            ).create_branch_map()
            hist_tuple_producers_sample_name = (
                dataset_name  # name of the sample which HistTupleProducer is running
            )
            # in btag_branches collect all branches of BtagShapeWeightCorrectionTask that HistTupleProducer needs to run this sample
            # for some reason, passing a tuple to branches argument of BtagShapeWeightCorrectionTask is not working
            btag_branches = []
            for btag_branch_idx, (
                btag_sample_name,
                process_group,
                _,
            ) in btag_shape_task_branch_map.items():
                if (
                    process_group != "data"
                    and hist_tuple_producers_sample_name == btag_sample_name
                ):
                    btag_branches.append(btag_branch_idx)
            assert (
                len(btag_branches) <= 1
            ), "Must be at most one BtagShapeWeightCorrectionTask branch per sample"
            if len(btag_branches) == 1:
                deps["btagShapeWeightCorr"] = BtagShapeWeightCorrectionTask.req(
                    self,
                    max_runtime=BtagShapeWeightCorrectionTask.max_runtime._default,
                    branch=-1,
                    branches=tuple(btag_branches),
                )
        return deps

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        var_produced_by = self.setup.var_producer_map

        n = 0
        branches = {}
        anaProd_branch_map = AnaTupleMergeTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()

        datasets_to_consider = [
            key
            for key in self.datasets.keys()
            if self.datasets[key]["process_group"] != "data"
        ]
        datasets_to_consider.append("data")

        flatten_vars = set()
        for var in self.global_params["variables"]:
            if isinstance(var, dict) and "vars" in var:
                for v in var["vars"]:
                    flatten_vars.add(v)
            else:
                flatten_vars.add(var)

        need_cache_list = [
            (var_name in var_produced_by, var_produced_by.get(var_name, None))
            # for var_name in self.global_params["variables"]
            for var_name in flatten_vars
        ]
        producer_list = []
        need_cache_global = any(item[0] for item in need_cache_list)
        # for var_name in self.global_params["variables"]:
        for var_name in flatten_vars:
            need_cache = True if var_name in var_produced_by else False
            producer_to_run = (
                var_produced_by[var_name] if var_name in var_produced_by else None
            )
            need_cache_list.append(need_cache)
            producer_list.append(producer_to_run)

        for prod_br, (
            dataset_name,
            dataset_type,
            input_file_list,
            output_file_list,
        ) in anaProd_branch_map.items():
            if dataset_name not in datasets_to_consider:
                continue

            for input_index in range(len(output_file_list)):
                branches[n] = (
                    dataset_name,
                    prod_br,
                    need_cache_global,
                    producer_list,
                    input_index,
                )
                n += 1
        return branches

    @workflow_condition.output
    def output(self):
        dataset_name, prod_br, need_cache_global, producer_list, input_index = (
            self.branch_data
        )
        input = self.input()["anaTuple"][input_index]
        outFileName = f"histTuple_" + os.path.basename(input.path).split("_")[1]
        output_path = os.path.join(
            "histTuples", self.version, self.period, dataset_name, outFileName
        )
        return self.remote_target(output_path, fs=self.fs_HistTuple)

    def run(self):
        dataset_name, prod_br, need_cache_global, producer_list, input_index = (
            self.branch_data
        )
        input_file = self.input()["anaTuple"][input_index]
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
                dataset_name,
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
                local_anacaches = {}
                for producer_name, cache_file in self.input()["anaCaches"].items():
                    local_anacaches[producer_name] = stack.enter_context(
                        cache_file[input_index].localize("r")
                    ).path
                local_anacaches_str = ",".join(
                    f"{producer}:{path}" for producer, path in local_anacaches.items()
                )
                HistTupleProducer_cmd.extend(["--cacheFile", local_anacaches_str])

            isMC = dataset_name != "data"
            if self.global_params["apply_btagShape_weights"] and isMC:
                tc = self.input()["btagShapeWeightCorr"]["collection"]
                btag_corr_json = tc._flat_target_list[0]
                local_btag_corr_json = stack.enter_context(
                    (btag_corr_json).localize("r")
                )
                HistTupleProducer_cmd.extend(
                    [f"--btagCorrectionsJson", local_btag_corr_json.path]
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
            req_dict["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
                self,
                branches=(),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
            req_dict["HistTupleProducerTask"] = HistTupleProducerTask.req(
                self, branches=(), customisations=self.customisations
            )
            return req_dict
        branch_set = set()
        for br_idx, (var, prod_br_list, dataset_names) in self.branch_map.items():
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
        var, prod_br_list, dataset_name = self.branch_data
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

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branches = {}
        prod_br_list = []
        current_dataset = None
        n = 0

        dataset_to_branches = {}
        HistTupleBranchMap = HistTupleProducerTask.req(
            self, branches=()
        ).create_branch_map()
        for prod_br, (
            histTuple_dataset_name,
            histTuple_prod_br,
            need_cache_global,
            producer_list,
            input_index,
        ) in HistTupleBranchMap.items():
            dataset_to_branches.setdefault(histTuple_dataset_name, []).append(prod_br)

        for dataset_name, prod_br_list in dataset_to_branches.items():
            for var_name in self.global_params["variables"]:
                branches[n] = (var_name, prod_br_list, dataset_name)
                n += 1

        return branches

    @workflow_condition.output
    def output(self):
        var, prod_br, dataset_name = self.branch_data
        if isinstance(var, dict):
            var = var["name"]
        output_path = os.path.join(
            "hists", self.version, self.period, var, f"{dataset_name}.root"
        )
        return self.remote_target(output_path, fs=self.fs_HistTuple)

    def run(self):
        var, prod_br, dataset_name = self.branch_data
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

            var = var if type(var) != dict else var["name"]
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
                "--dataset_name",
                dataset_name,
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
                ),
                "HistFromNtupleProducerTask": HistFromNtupleProducerTask.req(
                    self,
                    branches=(),
                ),
            }

        branch_set = set()
        all_datasets = {}
        for br_idx, (var, prod_br_list, dataset_names) in self.branch_map.items():
            all_datasets[var] = prod_br_list

        new_branchset = set()
        for var in all_datasets.keys():
            new_branchset.update(all_datasets[var])

        return {
            "HistFromNtupleProducerTask": HistFromNtupleProducerTask.req(
                self, branches=list(new_branchset)
            )
        }

    def requires(self):
        var_name, br_indices, datasets = self.branch_data
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

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        HistFromNtupleProducerTask_branch_map = HistFromNtupleProducerTask.req(
            self, branches=()
        ).create_branch_map()
        all_datasets = {}
        branches = {}
        k = 0
        for br_idx, (
            var_name,
            prod_br_list,
            current_dataset,
        ) in HistFromNtupleProducerTask_branch_map.items():
            var_name = (
                var_name.get("name", var_name)
                if isinstance(var_name, dict)
                else var_name
            )
            if var_name not in all_datasets.keys():
                all_datasets[var_name] = []
            all_datasets[var_name].append((br_idx, current_dataset))
        for var_name, br_list in all_datasets.items():
            br_indices = []
            datasets = []
            for key in br_list:
                idx, dataset_name = key
                br_indices.append(idx)
                datasets.append(dataset_name)
            branches[k] = (var_name, br_indices, datasets)
            k += 1
        return branches

    @workflow_condition.output
    def output(self):
        var_name, br_indices, datasets = self.branch_data
        output_path = os.path.join("merged_hists", self.version, self.period, var_name)
        output_file_name = os.path.join(output_path, f"{var_name}.root")
        return self.remote_target(output_file_name, fs=self.fs_HistTuple)

    def run(self):
        var_name, br_indices, datasets = self.branch_data
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
        unc_cfg_dict = self.setup.weights_config
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
                dataset_name = os.path.basename(inp.path)
                all_datasets.append(dataset_name.strip(".root"))
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
        self.output_file_extension = self.global_params["payload_producers"][
            self.producer_to_run
        ].get("save_as", "root")

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
        requirements = {
            "anaTuple": AnaTupleMergeTask.req(
                self, max_runtime=AnaTupleMergeTask.max_runtime._default
            )
        }
        anaCaches = {}
        if producer_dependencies:
            for dependency in producer_dependencies:
                anaCaches[dependency] = AnalysisCacheTask.req(
                    self, producer_to_run=dependency
                )
        requirements["anaCaches"] = anaCaches

        return requirements

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branches = {}
        anaProd_branch_map = AnaTupleMergeTask.req(
            self, branch=-1, branches=()
        ).branch_map
        for br_idx, (
            dataset_name,
            process_group,
            input_file_list,
            output_file_list,
        ) in anaProd_branch_map.items():
            branches[br_idx] = (dataset_name, process_group, len(output_file_list))
        return branches

    @workflow_condition.output
    def output(self):
        dataset_name, process_group, nInputs = self.branch_data
        return_list = []
        for idx in range(nInputs):
            inputFilePath = self.input()["anaTuple"][idx].path
            basename, _ = os.path.splitext(inputFilePath)
            outFileName = f"{basename}.{self.output_file_extension}"
            output_path = os.path.join(
                "AnalysisCache",
                self.version,
                self.period,
                dataset_name,
                self.producer_to_run,
                outFileName,
            )
            return_list.append(
                self.remote_target(output_path, fs=self.fs_anaCacheTuple)
            )
        return return_list

    def run(self):
        dataset_name, process_group, nInputs = self.branch_data
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
                    input_file = self.input()["anaTuple"][idx]
                    if len(self.input()["anaCaches"]) > 0:
                        local_anacaches = {}
                        for producer_name, cache_files in self.input()[
                            "anaCaches"
                        ].items():
                            local_anacaches[producer_name] = stack.enter_context(
                                cache_files[idx].localize("r")
                            ).path
                        local_anacaches_str = ",".join(
                            f"{producer}:{path}"
                            for producer, path in local_anacaches.items()
                        )
                        print(f"Task has cache input files {local_anacaches_str}")
                    else:
                        local_anacaches_str = ""

                    output_file = self.output()[idx]
                    print(
                        f"considering dataset {dataset_name}, {process_group} and file {input_file.path}"
                    )
                    if output_file.exists():
                        print(f"Output file {output_file} already exists, continue")
                        continue
                    customisation_dict = getCustomisationSplit(self.customisations)
                    tmpFile = os.path.join(
                        job_home, f"AnalysisCacheTask.{self.output_file_extension}"
                    )
                    with input_file.localize("r") as local_input:
                        analysisCacheProducer_cmd = [
                            "python3",
                            analysis_cache_producer,
                            "--period",
                            self.period,
                            "--inFile",
                            local_input.path,
                            "--outFile",
                            tmpFile,
                            "--dataset",
                            dataset_name,
                            "--channels",
                            channels,
                            "--producer",
                            self.producer_to_run,
                            "--workingDir",
                            job_home,
                            "--saveAs",
                            self.output_file_extension,
                        ]
                        if (
                            self.global_params["store_noncentral"]
                            and process_group != "data"
                        ):
                            analysisCacheProducer_cmd.extend(
                                ["--compute_unc_variations", "True"]
                            )
                        if len(local_anacaches_str) > 0:
                            analysisCacheProducer_cmd.extend(
                                ["--cacheFiles", local_anacaches_str]
                            )
                        # Check if cmssw env is required
                        prod_env = (
                            self.cmssw_env
                            if self.global_params["payload_producers"][
                                self.producer_to_run
                            ].get("cmssw_env", False)
                            else None
                        )
                        isData = process_group == "data"
                        if isData:
                            analysisCacheProducer_cmd.append("--isData")

                        histTupleDef = os.path.join(
                            self.ana_path(), self.global_params["histTupleDef"]
                        )
                        analysisCacheProducer_cmd.extend(
                            ["--histTupleDef", histTupleDef]
                        )

                        ps_call(analysisCacheProducer_cmd, env=prod_env, verbose=1)
                    print(
                        f"Finished producing payload for producer={self.producer_to_run} with name={dataset_name}, group={process_group}, file={input_file.path}"
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


class HistPlotTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def workflow_requires(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            req_dict = {}
            req_dict["HistMergerTask"] = HistMergerTask.req(
                self, branches=(), customisations=self.customisations
            )
            req_dict["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
                self,
                branches=(),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
            return req_dict
        merge_map = HistMergerTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()

        branch_set = set()
        for br_idx, (var) in self.branch_map.items():
            for br, (v, _, _) in merge_map.items():
                if v == var:
                    branch_set.add(br)

        return {
            "merge": HistMergerTask.req(
                self,
                branches=tuple(branch_set),
                customisations=self.customisations,
            )
        }

    def requires(self):
        var = self.branch_data

        merge_map = HistMergerTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()
        merge_branch = next(br for br, (v, _, _) in merge_map.items() if v == var)

        return HistMergerTask.req(
            self,
            branch=merge_branch,
            customisations=self.customisations,
            max_runtime=HistMergerTask.max_runtime._default,
        )

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branches = {}
        merge_map = HistMergerTask.req(
            self, branch=-1, branches=(), customisations=self.customisations
        ).create_branch_map()
        var_dict = {}
        for var in self.global_params["variables"]:
            var_name = var if isinstance(var, str) else var["name"]
            var_dict[var_name] = var
        for k, (_, (var, _, _)) in enumerate(merge_map.items()):
            # Check if we want to plot this var in the global config
            if isinstance(var_dict[var], dict):
                if var_dict[var].get("plot_task", True):
                    branches[k] = var
            else:
                branches[k] = var
        return branches

    @workflow_condition.output
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

        custom_region_name = self.global_params.get("custom_regions")

        custom_regions = customisation_dict.get(
            custom_region_name, self.global_params[custom_region_name]
        )

        for ch in channels:
            for cat in categories:
                for custom_region in custom_regions:
                    rel_path = os.path.join(
                        self.version,
                        self.period,
                        "plots",
                        var,
                        custom_region,
                        cat,
                        f"{ch}_{var}.pdf",
                    )
                    outputs[f"{ch}:{cat}:{custom_region}"] = self.remote_target(
                        rel_path, fs=self.fs_plots
                    )
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
        plot_wantSignals = bool_flag("plot_wantSignals", True)
        plot_wantQCD = bool_flag("plot_wantQCD", False)
        plot_rebin = bool_flag("plot_rebin", True)
        plot_analysis = customisation_dict.get(
            "plot_analysis", self.global_params.get("plot_analysis", "")
        )

        with self.input().localize("r") as local_input:
            infile = local_input.path
            print("Loading fname", infile)

            # Create list of all keys and all targets
            key_list = []
            output_list = []
            for output_key, output_target in self.output().items():
                if (output_target).exists():
                    print(f"Output for {var} {output_target} already exists! Continue")
                    continue
                key_list.append(output_key)
                output_list.append(output_target)

            # Now localize all output_targets
            with contextlib.ExitStack() as stack:
                local_outputs = [
                    stack.enter_context((output).localize("w")).path
                    for output in output_list
                ]
                cmd = [
                    "python3",
                    plotter,
                    "--inFile",
                    infile,
                    "--all_outFiles",
                    ",".join(local_outputs),
                    "--globalConfig",
                    os.path.join(
                        self.ana_path(),
                        self.global_params["analysis_config_area"],
                        "global.yaml",
                    ),
                    "--var",
                    var,
                    "--all_keys",
                    ",".join(key_list),
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


class BtagShapeWeightCorrectionTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def __init__(self, *args, **kwargs):
        super(BtagShapeWeightCorrectionTask, self).__init__(*args, **kwargs)

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    def workflow_requires(self):
        merge_organization_complete = AnaTupleFileListTask.req(
            self, branches=()
        ).complete()
        if not merge_organization_complete:
            deps = {
                "AnaTupleFileListTask": AnaTupleFileListTask.req(
                    self,
                    branches=(),
                    max_runtime=AnaTupleFileListTask.max_runtime._default,
                    n_cpus=AnaTupleFileListTask.n_cpus._default,
                ),
                "BtagShape": AnalysisCacheTask.req(
                    self,
                    branches=(),
                    max_runtime=AnalysisCacheTask.max_runtime._default,
                    n_cpus=AnalysisCacheTask.n_cpus._default,
                    customisations=self.customisations,
                    producer_to_run="BtagShape",
                ),
            }
            return deps

        btag_cache_map = AnalysisCacheTask.req(
            self, branch=-1, branches=(), producer_to_run="BtagShape"
        ).create_branch_map()
        branches = [b for b in btag_cache_map.keys()]
        deps = {
            "BtagShape": AnalysisCacheTask.req(
                self,
                branches=tuple(branches),
                max_runtime=AnalysisCacheTask.max_runtime._default,
                n_cpus=AnalysisCacheTask.n_cpus._default,
                customisations=self.customisations,
                producer_to_run="BtagShape",
            )
        }
        return deps

    def requires(self):
        sample_name, process_group, list_of_br_idxes = self.branch_data
        reqs = [
            AnalysisCacheTask.req(
                self,
                max_runtime=AnalysisCacheTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
                customisations=self.customisations,
                producer_to_run="BtagShape",
            )
            for prod_br in list_of_br_idxes
        ]
        return reqs

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branches = {}
        branch_number = 0
        data_done = False
        # obtain branch map for the previos task
        # it is structured per file
        btag_cache_map = AnalysisCacheTask.req(
            self, branch=-1, branches=(), producer_to_run="BtagShape"
        ).create_branch_map()

        # restructure btag_cache_map so it maps each sample to list of branch indices from prev. task
        # i.e. for signal it will be (XtoYHto2B2Wto2B2L2Nu_MX_300_MY_125, signals) -> [0]
        # but e.g. for ttbar it will be (TTto2L2Nu, backgrounds) -> [1, 2, 3, 4, 5, ...]
        # so branch_map of BtagShapeWeightCorrectionTask will contain:
        # ---- name of sample,
        # ---- process group
        # ---- list of branch indices of the task it depends on (AnalysisCacheTask(producer_to_run="BtagShape"))
        sample_branch_map = {}
        for idx, (sample_name, process_group, num_out_files) in btag_cache_map.items():
            if (sample_name, process_group) not in sample_branch_map:
                sample_branch_map[(sample_name, process_group)] = []
            sample_branch_map[(sample_name, process_group)].append(idx)

        for (sample_name, process_group), list_of_br_idxes in sample_branch_map.items():
            if process_group == "data":
                if data_done:
                    continue  # Will have multiple data samples, but only need one branch
                sample_name = "data"
                data_done = True
                branches[branch_number] = (sample_name, process_group, list_of_br_idxes)
            branches[branch_number] = (sample_name, process_group, list_of_br_idxes)
            branch_number += 1
        return branches

    @workflow_condition.output
    def output(self):
        sample_name, _, _ = self.branch_data
        output_name = "BtagShapeWeightCorrection.json"
        output_path = os.path.join(
            "BtagShapeWeightCorrection",
            self.version,
            self.period,
            sample_name,
            output_name,
        )

        return [
            self.remote_target(output_path, fs=self.fs_anaTuple),
        ]

    def run(self):
        sample_name, _, _ = self.branch_data
        computeBtagShapeWeight = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "ComputeBtagShapeWeightCorrection.py"
        )
        with contextlib.ExitStack() as stack:
            remote_output = self.output()[0]

            local_inputs = []
            if sample_name == "data":
                local_inputs = [
                    stack.enter_context(inp.localize("r")).path for inp in self.input()
                ]
            else:
                local_inputs = [
                    stack.enter_context(inp[0].localize("r")).path
                    for inp in self.input()
                ]

            assert local_inputs, "`local_inputs` must be a non-empty list"

            btag_shape_cfg = self.global_params["payload_producers"]["BtagShape"]
            jet_multiplicities = btag_shape_cfg["jet_multiplicities"]

            job_home, remove_job_home = self.law_job_home()
            tmpFile = os.path.join(job_home, f"BtagShapeWeightCorrection_tmp.json")
            computeBtagShapeWeight_cmd = [
                "python3",
                computeBtagShapeWeight,
                "--outFile",
                tmpFile,
            ]

            computeBtagShapeWeight_cmd.append("--inputFiles")
            computeBtagShapeWeight_cmd.extend(local_inputs)

            computeBtagShapeWeight_cmd.append("--jetMultiplicities")
            computeBtagShapeWeight_cmd.extend([str(m) for m in jet_multiplicities])

            ps_call(computeBtagShapeWeight_cmd, verbose=1)

            with remote_output.localize("w") as tmp_local_file:
                out_local_path = tmp_local_file.path
                shutil.move(tmpFile, out_local_path)
