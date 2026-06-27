import law
import os
import contextlib
import luigi
import shutil
from concurrent.futures import ThreadPoolExecutor


from FLAF.RunKit.run_tools import ps_call
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    copy_param,
)
from FLAF.AnaProd.tasks import (
    AnaTupleFileListTask,
    AnaTupleMergeTask,
)
from FLAF.Common.Utilities import getCustomisationSplit, ServiceThread

# Process-local cache of the upstream AnaTupleMergeTask outputs, keyed by the resolved
# AnaTuple (version, period, customisations). HistTupleProducerTask.output() and
# AnalysisCacheTask.output() derive their own output name from the anaTuple input file name;
# resolving that per branch (instantiating AnaTupleMergeTask and building/reducing its branch
# map each time) is O(nBranches) per branch, i.e. O(nBranches^2) when all outputs are queried
# (e.g. --print-status). Building the {branch -> [targets]} map once and sharing it makes each
# lookup O(1). Safe within a process: the map is fixed by the (already produced) merge plan.
_anaTuple_outputs_cache = {}


def _dedup_variables(variables):
    """Drop duplicate variables (by name) keeping first occurrence and order. The variables
    config may list the same variable more than once; per-variable processing downstream
    (e.g. HistProducerFromNTuple's tmp_<var>.root, one HistMerger branch per variable)
    collides on duplicates. HistTupleProducer already deduplicates via a set; this keeps the
    other stages consistent."""
    seen = set()
    result = []
    for var in variables:
        name = var["name"] if isinstance(var, dict) else var
        if name in seen:
            continue
        seen.add(name)
        result.append(var)
    return result


def _anaTuple_outputs(task):
    wf = AnaTupleMergeTask.req(
        task,
        branch=-1,
        branches=(),
        customisations=task.customisations,
        max_runtime=AnaTupleMergeTask.max_runtime._default,
    )
    key = (wf.version, wf.period, wf.customisations)
    cache = _anaTuple_outputs_cache.get(key)
    if cache is None:
        cache = wf.all_branch_outputs()
        _anaTuple_outputs_cache[key] = cache
    return cache


class HistTupleProducerTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)
    # many short per-file branches: group several per HTCondor job to bound nJobs.
    tasks_per_job = copy_param(HTCondorWorkflow.tasks_per_job, 10)

    @property
    def bundle_flavours(self):
        fl = AnaTupleFileListTask.req(self, branches=())
        return ["core", ("AnaTupleFileList", fl.version)]

    def workflow_requires(self):
        reqs = super().workflow_requires()
        merge_organization_complete = AnaTupleFileListTask.req(
            self,
            branches=(),
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

            reqs.update(req_dict)
            return reqs

        branch_set = set()
        branch_set_cache = set()
        producer_set = set()
        agg_dict = {}
        # An aggregation branch map depends only on the producer, not on the requesting
        # branch, so resolve each producer's {dataset -> aggregation branches} once instead
        # of rebuilding it for every branch (which was O(nBranches) full cascade rebuilds).
        agg_dataset_branches = {}
        for idx, (
            dataset,
            br,
            producer_list,
            aggregate_list,
            input_index,
        ) in self.branch_map.items():
            branch_set.add(br)
            if len(producer_list) > 0:
                branch_set_cache.add(idx)
                for producer_name in (p for p in producer_list if p is not None):
                    producer_set.add(producer_name)

            for agg_name in aggregate_list:
                if agg_name not in agg_dataset_branches:
                    by_dataset = {}
                    aggr_task_branch_map = AnalysisCacheAggregationTask.req(
                        self,
                        branch=-1,
                        producer_to_aggregate=agg_name,
                    ).create_branch_map()
                    for aggr_br_idx, (
                        aggr_dataset_name,
                        _,
                    ) in aggr_task_branch_map.items():
                        by_dataset.setdefault(aggr_dataset_name, []).append(aggr_br_idx)
                    agg_dataset_branches[agg_name] = by_dataset
                agg_dict.setdefault(agg_name, set()).update(
                    agg_dataset_branches[agg_name].get(dataset, ())
                )

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

        reqs["aggregateCache"] = []
        for agg_name, agg_br_set in agg_dict.items():
            reqs["aggregateCache"].append(
                AnalysisCacheAggregationTask.req(
                    self,
                    branches=tuple(agg_br_set),
                    customisations=self.customisations,
                    producer_to_aggregate=agg_name,
                )
            )

        return reqs

    def requires(self):
        dataset_name, prod_br, producer_list, aggregate_list, input_index = (
            self.branch_data
        )
        # subs take care of their version; forward ana params, no version=
        deps = {
            "anaTuple": AnaTupleMergeTask.req(
                self,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                branch=prod_br,
                branches=(prod_br,),
                customisations=self.customisations,
            )
        }
        if len(producer_list) > 0:
            anaCaches = {}
            for producer_name in (p for p in producer_list if p is not None):
                if producer_name not in deps:
                    anaCaches[producer_name] = AnalysisCacheTask.req(
                        self,
                        max_runtime=AnalysisCacheTask.max_runtime._default,
                        branch=self.branch,
                        branches=(self.branch,),
                        customisations=self.customisations,
                        producer_to_run=producer_name,
                    )
            if len(anaCaches) > 0:
                deps["anaCaches"] = anaCaches

        agg_dict = {}
        if len(aggregate_list) > 0:
            anaAggs = {}
            for agg_name in aggregate_list:
                if agg_name not in agg_dict.keys():
                    agg_dict[agg_name] = set()
                aggr_task_branch_map = AnalysisCacheAggregationTask.req(
                    self,
                    branch=-1,
                    producer_to_aggregate=agg_name,
                ).create_branch_map()

                for aggr_br_idx, (
                    aggr_dataset_name,
                    _,
                ) in aggr_task_branch_map.items():
                    if aggr_dataset_name == dataset_name:
                        agg_dict[agg_name].add(aggr_br_idx)

                # The aggregates COULD multiple inputs, handle appropriately
                if agg_name not in anaAggs:
                    anaAggs[agg_name] = []
                    for br in agg_dict[agg_name]:
                        anaAggs[agg_name].append(
                            AnalysisCacheAggregationTask.req(
                                self,
                                branch=br,
                                customisations=self.customisations,
                                producer_to_aggregate=agg_name,
                            )
                        )
            if len(anaAggs) > 0:
                deps["anaAggs"] = anaAggs

        return deps

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        var_produced_by = self.setup.var_producer_map

        n = 0
        branches = {}
        anaProd_branch_map = AnaTupleMergeTask.req(
            self,
            branch=-1,
            branches=(),
        ).create_branch_map()
        if not isinstance(anaProd_branch_map, dict):
            anaProd_branch_map = {}

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

        producer_list = []
        aggregate_list = []
        for var_name in flatten_vars:
            producer_to_run = (
                var_produced_by[var_name] if var_name in var_produced_by else None
            )
            if producer_to_run is not None:
                producer_list.append(producer_to_run)

        payload_producers = self.global_params.get("payload_producers")
        if payload_producers:
            for producer_name, producer_cfg in payload_producers.items():
                if not producer_cfg.get("is_global", False):
                    continue
                if producer_cfg.get("needs_aggregation", False):
                    aggregate_list.append(producer_name)
                else:
                    if producer_name not in producer_list:
                        producer_list.append(producer_name)

        for prod_br, (
            dataset_name,
            process_group,
            ds_branch,
            dataset_dependencies,
            input_file_list,
            output_file_list,
            skip_future_tasks,
            runs,
        ) in anaProd_branch_map.items():
            if skip_future_tasks:
                continue
            if dataset_name not in datasets_to_consider:
                continue

            for input_index in range(len(output_file_list)):
                producers_to_run = []
                aggregates_to_run = []
                if payload_producers:
                    for prod in producer_list:
                        cfg = payload_producers.get(prod, None)
                        is_configurable = cfg is not None
                        if not is_configurable:
                            producers_to_run.append(prod)
                            continue

                        target_groups = cfg.get("target_groups", None)
                        applies_for_group = (
                            target_groups is None or process_group in target_groups
                        )

                        if applies_for_group:
                            producers_to_run.append(prod)

                    for agg in aggregate_list:
                        cfg = payload_producers.get(agg, None)
                        is_configurable = cfg is not None
                        if not is_configurable:
                            aggregates_to_run.append(agg)
                            continue

                        target_groups = cfg.get("target_groups", None)
                        applies_for_group = (
                            target_groups is None or process_group in target_groups
                        )

                        if applies_for_group:
                            aggregates_to_run.append(agg)

                    branches[n] = (
                        dataset_name,
                        prod_br,
                        producers_to_run,
                        aggregates_to_run,
                        input_index,
                    )
                else:
                    branches[n] = (
                        dataset_name,
                        prod_br,
                        producer_list,
                        aggregate_list,
                        input_index,
                    )
                n += 1
        return branches

    @workflow_condition.output
    def output(self):
        dataset_name, prod_br, producer_list, aggregate_list, input_index = (
            self.branch_data
        )
        # Derive the output name from the anaTuple input only (not the full requires graph:
        # anaCaches/anaAggs are irrelevant here). The anaTuple outputs are resolved once per
        # process and shared, so this is O(1) instead of O(nBranches) per branch.
        input = _anaTuple_outputs(self)[prod_br][input_index]
        input_name = os.path.basename(input.abspath)
        outFileName = (
            f"histTuple_" + os.path.basename(input.abspath).split("_", 1)[1]
            if input_name.startswith("anaTuple_")
            else input_name
        )
        output_path = os.path.join(
            self.version, "HistTuples", self.period, dataset_name, outFileName
        )
        return self.remote_target(output_path, fs=self.fs_HistTuple)

    def run(self):
        dataset_name, prod_br, producer_list, aggregate_list, input_index = (
            self.branch_data
        )
        input_file = self.input()["anaTuple"][input_index]
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = customisation_dict.get(
            "channels", self.global_params["channelSelection"]
        )
        if type(channels) == list:
            channels = ",".join(channels)

        print(f"input file is {input_file.abspath}")
        histTupleDef = os.path.join(self.ana_path(), self.global_params["histTupleDef"])
        HistTupleProducer = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "HistTupleProducer.py"
        )
        outFile = self.output().abspath
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
                local_input.abspath,
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
                "--LAWrunVersion",
                self.version,
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
            if self.user_custom:
                HistTupleProducer_cmd.extend(["--user-custom", self.user_custom])
            if len(producer_list) > 0:
                local_anacaches = {}
                for producer_name, cache_file in self.input()["anaCaches"].items():
                    local_anacaches[producer_name] = stack.enter_context(
                        cache_file.localize("r")
                    ).abspath
                local_anacaches_str = ",".join(
                    f"{producer}:{path}"
                    for producer, path in local_anacaches.items()
                    if path.endswith("root")
                )
                HistTupleProducer_cmd.extend(["--cacheFile", local_anacaches_str])

            # Localize the per-sample aggregated caches (e.g. BtagShape) and pass them as
            # "producer:localpath" so the consumer reads the law-localized file instead of
            # reconstructing a version-dependent path.
            anaAggs = self.input().get("anaAggs", {})
            local_anaaggs_str = ",".join(
                f"{producer_name}:"
                + stack.enter_context(agg_targets[0].localize("r")).abspath
                for producer_name, agg_targets in anaAggs.items()
                if agg_targets
            )
            if local_anaaggs_str:
                HistTupleProducer_cmd.extend(
                    ["--aggregatedCacheFiles", local_anaaggs_str]
                )

            ps_call(HistTupleProducer_cmd, verbose=1)

            with self.output().localize("w") as local_output:
                out_local_path = local_output.abspath
                shutil.move(tmpFile, out_local_path)
        if remove_job_home:
            shutil.rmtree(job_home)


class HistFromNtupleProducerTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 10.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)
    variables = luigi.Parameter(default="")
    # Number of input ntuple files processed per job. A job reads its chunk of files once
    # and fills the histograms for ALL active variables in a single RDataFrame event-loop
    # pass (one traversal fills every booked histogram, so the dominant per-event cost --
    # I/O + filter/define evaluation -- is shared across variables). Work is therefore split
    # by FILES, never by variable: a per-variable split re-reads the same events once per
    # variable batch, which is what made big datasets (e.g. TTtoLNu2Q with ~500 files read
    # 10x) exceed the wall-time limit. Big datasets are parallelized by chunking their files.
    n_files_per_job = luigi.IntParameter(default=20)

    @property
    def bundle_flavours(self):
        fl = AnaTupleFileListTask.req(self, branches=())
        return ["core", ("AnaTupleFileList", fl.version)]

    @property
    def active_variables(self):
        all_vars = _dedup_variables(self.global_params["variables"])
        if not self.variables:
            return all_vars
        selected = {v.strip() for v in self.variables.split(",") if v.strip()}
        result = [
            v for v in all_vars if (v["name"] if isinstance(v, dict) else v) in selected
        ]
        found = {v["name"] if isinstance(v, dict) else v for v in result}
        missing = selected - found
        if missing:
            print(f"Warning: variables not found in config: {sorted(missing)}")
        return result

    def workflow_requires(self):
        reqs = super().workflow_requires()
        merge_organization_complete = AnaTupleFileListTask.req(
            self,
            branches=(),
        ).complete()
        if not merge_organization_complete:
            reqs["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
                self,
                branches=(),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
            reqs["HistTupleProducerTask"] = HistTupleProducerTask.req(
                self,
                branches=(),
                customisations=self.customisations,
            )
            return reqs
        branch_set = set()
        for br_idx, (dataset_name, prod_br_list, chunk_id) in self.branch_map.items():
            branch_set.update(prod_br_list)
        branches = tuple(sorted(branch_set))
        reqs["HistTupleProducerTask"] = HistTupleProducerTask.req(
            self,
            branches=branches,
            customisations=self.customisations,
        )
        return reqs

    def requires(self):
        dataset_name, prod_br_list, chunk_id = self.branch_data
        return [
            HistTupleProducerTask.req(
                self,
                max_runtime=HistTupleProducerTask.max_runtime._default,
                branch=prod_br,
                customisations=self.customisations,
            )
            for prod_br in prod_br_list
        ]

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(
            self,
            branch=-1,
            branches=(),
        ).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        branches = {}

        dataset_to_branches = {}
        HistTupleBranchMap = HistTupleProducerTask.req(
            self, branches=()
        ).create_branch_map()
        for prod_br, (
            histTuple_dataset_name,
            histTuple_prod_br,
            producer_list,
            aggregate_list,
            input_index,
        ) in HistTupleBranchMap.items():
            dataset_to_branches.setdefault(histTuple_dataset_name, []).append(prod_br)

        # Each (dataset, file-chunk) is its own branch/job: a job localizes its chunk of
        # input files once and produces the histograms for ALL active variables in a single
        # event-loop pass. Variables are NOT split across jobs (the event loop reads each
        # event once and fills every booked histogram in that pass, so adding variables is
        # nearly free; splitting them would instead re-read the same events per batch). Big
        # datasets are parallelized by chunking their files into groups of n_files_per_job,
        # so no single job has to read all ~500 files of e.g. TTtoLNu2Q.
        n_files = max(1, self.n_files_per_job)
        idx = 0
        for dataset_name, prod_br_list in sorted(dataset_to_branches.items()):
            prod_br_list = sorted(prod_br_list)
            for chunk_id, start in enumerate(range(0, len(prod_br_list), n_files)):
                branches[idx] = (
                    dataset_name,
                    prod_br_list[start : start + n_files],
                    chunk_id,
                )
                idx += 1

        return branches

    @workflow_condition.output
    def output(self):
        dataset_name, prod_br_list, chunk_id = self.branch_data
        outputs = {}
        for var in self.active_variables:
            var_name = var["name"] if isinstance(var, dict) else var
            output_path = os.path.join(
                self.version,
                "Hists_split",
                self.period,
                var_name,
                dataset_name,
                f"chunk_{chunk_id}.root",
            )
            outputs[var_name] = self.remote_target(output_path, fs=self.fs_HistTuple)
        return outputs

    def run(self):
        dataset_name, prod_br_list, chunk_id = self.branch_data
        var_names = [
            v["name"] if isinstance(v, dict) else v for v in self.active_variables
        ]
        job_home, remove_job_home = self.law_job_home()
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = (
            customisation_dict["channels"]
            if "channels" in customisation_dict.keys()
            else self.global_params["channelSelection"]
        )
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
        nMT = self.n_cpus * 2 if self.effective_workflow == "htcondor" else 8

        # Determine which variables still need to be produced for this (dataset, chunk).
        outputs = self.output()
        vars_to_run = [v for v in var_names if not outputs[v].exists()]
        if not vars_to_run:
            print(
                f"All outputs already exist for dataset {dataset_name} "
                f"chunk {chunk_id}, skipping"
            )
            if remove_job_home:
                shutil.rmtree(job_home)
            return

        # Localize this chunk's input files concurrently, then run the producer once over all
        # of them. A chunk is small by construction (n_files_per_job), and the producer fills
        # every variable in a single event-loop pass, writing one <var>.root per variable --
        # so there is nothing to stage in waves or merge afterwards.
        max_dl = max(1, int(self.global_params.get("max_simultaneous_downloads", 8)))
        out_dir = os.path.join(job_home, "hists")
        os.makedirs(out_dir, exist_ok=True)

        def _localize(inp):
            file_stack = contextlib.ExitStack()
            return file_stack.enter_context(inp.localize("r")).abspath, file_stack

        with contextlib.ExitStack() as stack:
            with ThreadPoolExecutor(max_workers=max_dl) as executor:
                localized = list(executor.map(_localize, self.input()))
            local_inputs = []
            for abspath, file_stack in localized:
                stack.callback(file_stack.close)
                local_inputs.append(abspath)

            cmd = [
                "python3",
                HistFromNtupleProducer,
                "--period",
                self.period,
                "--outDir",
                out_dir,
                "--channels",
                channels,
                "--vars",
                ",".join(vars_to_run),
                "--dataset_name",
                dataset_name,
                "--LAWrunVersion",
                self.version,
                "--nMT",
                str(nMT),
            ]
            if compute_unc_histograms:
                cmd.extend(
                    [
                        "--compute_rel_weights",
                        "True",
                        "--compute_unc_variations",
                        "True",
                    ]
                )
            if self.customisations:
                cmd.extend(["--customisations", self.customisations])
            if self.user_custom:
                cmd.extend(["--user-custom", self.user_custom])
            cmd.extend(local_inputs)
            ps_call(cmd, verbose=1)

        # Upload the single per-variable output file.
        for var in vars_to_run:
            with outputs[var].localize("w") as tmp_out:
                shutil.move(os.path.join(out_dir, f"{var}.root"), tmp_out.abspath)

        if remove_job_home:
            shutil.rmtree(job_home)


class HistMergerTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 2)
    variables = luigi.Parameter(default="")

    @property
    def bundle_flavours(self):
        fl = AnaTupleFileListTask.req(self, branches=())
        return ["core", ("AnaTupleFileList", fl.version)]

    @property
    def active_variables(self):
        all_vars = _dedup_variables(self.global_params["variables"])
        if not self.variables:
            return all_vars
        selected = {v.strip() for v in self.variables.split(",") if v.strip()}
        result = [
            v for v in all_vars if (v["name"] if isinstance(v, dict) else v) in selected
        ]
        found = {v["name"] if isinstance(v, dict) else v for v in result}
        missing = selected - found
        if missing:
            print(f"Warning: variables not found in config: {sorted(missing)}")
        return result

    def workflow_requires(self):
        reqs = super().workflow_requires()
        branch_map = self.create_branch_map()
        merge_organization_complete = AnaTupleFileListTask.req(
            self,
            branches=(),
        ).complete()
        if not merge_organization_complete:
            reqs["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
                self,
                branches=(),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
            reqs["HistFromNtupleProducerTask"] = HistFromNtupleProducerTask.req(
                self,
                branches=(),
            )
            return reqs

        new_branchset = set()
        for br_idx, (var_name, hfn_br_list, dataset_names) in self.branch_map.items():
            new_branchset.update(hfn_br_list)

        reqs["HistFromNtupleProducerTask"] = HistFromNtupleProducerTask.req(
            self, branches=sorted(new_branchset)
        )
        return reqs

    def requires(self):
        var_name, br_indices, datasets = self.branch_data
        return [
            HistFromNtupleProducerTask.req(
                self,
                max_runtime=HistFromNtupleProducerTask.max_runtime._default,
                branch=br_idx,
                customisations=self.customisations,
            )
            for br_idx in br_indices
        ]

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(
            self,
            branch=-1,
            branches=(),
        ).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        hfn_branch_map = HistFromNtupleProducerTask.req(
            self, branches=()
        ).create_branch_map()
        if not hfn_branch_map:
            return {}
        # HFN branches per (dataset, file-chunk), and every chunk produces every active
        # variable. So each merger branch (one per variable) depends on all HFN branches;
        # we record the dataset name aligned with each HFN branch so the merger can map each
        # input file to its process type (multiple chunks of the same dataset are summed).
        hfn_br_indices = []
        dataset_names = []
        for br_idx, (dataset_name, _prod_br_list, _chunk_id) in sorted(
            hfn_branch_map.items()
        ):
            hfn_br_indices.append(br_idx)
            dataset_names.append(dataset_name)
        # One HistMerger branch per active variable.
        branches = {}
        for k, var in enumerate(self.active_variables):
            var_name = var["name"] if isinstance(var, dict) else var
            branches[k] = (var_name, hfn_br_indices, dataset_names)
        return branches

    @workflow_condition.output
    def output(self):
        var_name, br_indices, datasets = self.branch_data
        output_path = os.path.join(self.version, "Hists_merged", self.period, var_name)
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

        all_datasets = []
        local_inputs = []
        with contextlib.ExitStack() as stack:
            # `datasets` is aligned with self.input() (both follow requires()/br_indices
            # order); multiple file-chunks of the same dataset appear as repeated entries.
            # Each input is a tiny remote histogram file, but every davs localize pays a fixed
            # connection cost, so localizing the O(100s) of (dataset, chunk) inputs serially
            # dominated the merge. Download them concurrently -- each in its own (thread-safe)
            # ExitStack registered for cleanup; ThreadPoolExecutor.map preserves order so
            # local_inputs stays aligned with all_datasets (mapped to process types below).
            var_targets = [
                (inp[var_name], dataset_name)
                for inp, dataset_name in zip(self.input(), datasets)
            ]
            max_dl = max(
                1, int(self.global_params.get("max_simultaneous_downloads", 8))
            )

            def _localize(var_file):
                file_stack = contextlib.ExitStack()
                abspath = file_stack.enter_context(var_file.localize("r")).abspath
                return abspath, file_stack

            with ThreadPoolExecutor(max_workers=max_dl) as executor:
                localized = list(executor.map(lambda t: _localize(t[0]), var_targets))
            for (var_file, dataset_name), (abspath, file_stack) in zip(
                var_targets, localized
            ):
                stack.callback(file_stack.close)
                all_datasets.append(dataset_name)
                local_inputs.append(abspath)
            dataset_names = ",".join(smpl for smpl in all_datasets)
            all_outputs_merged = []
            if len(uncNames) == 1:
                with self.output().localize("w") as outFile:
                    MergerProducer_cmd = [
                        "python3",
                        MergerProducer,
                        "--outFile",
                        outFile.abspath,
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
                        "--LAWrunVersion",
                        self.version,
                    ]
                    MergerProducer_cmd.extend(local_inputs)
                    ps_call(MergerProducer_cmd, verbose=1)
            else:
                job_home, remove_job_home = self.law_job_home()
                for uncName in uncNames:
                    final_histname = f"{var_name}_{uncName}.root"
                    tmp_outfile_merge = os.path.join(job_home, final_histname)
                    MergerProducer_cmd = [
                        "python3",
                        MergerProducer,
                        "--outFile",
                        tmp_outfile_merge,
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
                        "--LAWrunVersion",
                        self.version,
                    ]
                    MergerProducer_cmd.extend(local_inputs)
                    ps_call(MergerProducer_cmd, verbose=1)
                    all_outputs_merged.append(tmp_outfile_merge)
                with self.output().localize("w") as outFile:
                    HaddMergedHistsProducer_cmd = [
                        "python3",
                        HaddMergedHistsProducer,
                        "--outFile",
                        outFile.abspath,
                        "--var",
                        var_name,
                    ]
                    HaddMergedHistsProducer_cmd.extend(all_outputs_merged)
                    ps_call(HaddMergedHistsProducer_cmd, verbose=1)
                if remove_job_home:
                    shutil.rmtree(job_home)


class AnalysisCacheTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    producer_to_run = luigi.Parameter()

    @property
    def bundle_flavours(self):
        fl = AnaTupleFileListTask.req(self, branches=())
        flavours = ["core", ("AnaTupleFileList", fl.version)]
        if (
            self.global_params.get("payload_producers", {})
            .get(self.producer_to_run, {})
            .get("cmssw_env", False)
        ):
            flavours.append("cmssw")
        return flavours

    # Need to override this from HTCondorWorkflow to have separate data pathways for different cache tasks
    def htcondor_output_directory(self):
        return law.LocalDirectoryTarget(self.local_path(self.producer_to_run))

    def __init__(self, *args, **kwargs):
        ana_v = kwargs.get("ana_version") or kwargs.get("anaCache_version")
        if ana_v:
            kwargs["version"] = ana_v
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
        reqs = super().workflow_requires()
        merge_organization_complete = AnaTupleFileListTask.req(
            self,
            branches=(),
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
            reqs.update(req_dict)
            return reqs

        workflow_dict = {}
        workflow_dict["anaTuple"] = {
            br_idx: AnaTupleMergeTask.req(
                self,
                branch=prod_br,
                branches=(),
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                n_cpus=AnaTupleMergeTask.n_cpus._default,
            )
            for br_idx, (
                dataset_name,
                prod_br,
                producer_list,
                aggregate_list,
                input_index,
            ) in self.branch_map.items()
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
                        branches=(),
                        customisations=self.customisations,
                        producer_to_run=dependency,
                    )
                    for br_idx, _ in self.branch_map.items()
                }
        reqs.update(workflow_dict)
        return reqs

    def requires(self):
        dataset_name, prod_br, producer_list, aggregate_list, input_index = (
            self.branch_data
        )
        producer_dependencies = self.global_params["payload_producers"][
            self.producer_to_run
        ]["dependencies"]
        # version for anaTuple handled inside AnaTupleMergeTask (from ana* copied by req or its per-task CLI); simplify no if/version.
        requirements = {
            "anaTuple": AnaTupleMergeTask.req(
                self,
                branch=prod_br,
                max_runtime=AnaTupleMergeTask.max_runtime._default,
                branches=(),
            )
        }
        anaCaches = {}
        if producer_dependencies:
            for dependency in producer_dependencies:
                anaCaches[dependency] = AnalysisCacheTask.req(
                    self,
                    producer_to_run=dependency,
                )
        requirements["anaCaches"] = anaCaches

        return requirements

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(
            self,
            branch=-1,
            branches=(),
        ).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        branches = HistTupleProducerTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        return branches

    @workflow_condition.output
    def output(self):
        dataset_name, prod_br, _, _, input_index = self.branch_data
        # Derive the output name from the anaTuple input only; the anaTuple outputs are
        # resolved once per process and shared, so this is O(1) instead of building the
        # anaTuple/anaCache requirements per branch (O(nBranches) each).
        inputFilePath = _anaTuple_outputs(self)[prod_br][input_index].abspath
        outFileNameWithoutExtension = os.path.basename(inputFilePath).split(".")[0]
        outFileName = f"{outFileNameWithoutExtension}.{self.output_file_extension}"
        output_path = os.path.join(
            self.version,
            "AnalysisCache",
            self.producer_to_run,
            self.period,
            dataset_name,
            outFileName,
        )
        return self.remote_target(output_path, fs=self.fs_anaCacheTuple)

    def run(self):
        with ServiceThread() as service_thread:
            dataset_name, prod_br, producer_list, aggregate_list, input_index = (
                self.branch_data
            )
            analysis_cache_producer = os.path.join(
                self.ana_path(), "FLAF", "Analysis", "AnalysisCacheProducer.py"
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
            job_home, remove_job_home = self.law_job_home()
            print(f"At job_home {job_home}")

            with contextlib.ExitStack() as stack:
                # Enter a stack to maybe load the analysis cache files
                input_file = self.input()["anaTuple"][input_index]
                if len(self.input()["anaCaches"]) > 0:
                    local_anacaches = {}
                    for producer_name, cache_files in self.input()["anaCaches"].items():
                        local_anacaches[producer_name] = stack.enter_context(
                            cache_files[input_index].localize("r")
                        ).abspath
                    local_anacaches_str = ",".join(
                        f"{producer}:{path}"
                        for producer, path in local_anacaches.items()
                    )
                    print(f"Task has cache input files {local_anacaches_str}")
                else:
                    local_anacaches_str = ""

                print(
                    f"considering dataset {dataset_name}, and file {input_file.abspath}"
                )
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
                        local_input.abspath,
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
                        "--LAWrunVersion",
                        self.version,
                    ]
                    if (
                        self.global_params["store_noncentral"]
                        and dataset_name != "data"
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

                    histTupleDef = os.path.join(
                        self.ana_path(), self.global_params["histTupleDef"]
                    )
                    analysisCacheProducer_cmd.extend(["--histTupleDef", histTupleDef])

                    ps_call(analysisCacheProducer_cmd, env=prod_env, verbose=1)
                print(
                    f"Finished producing payload for producer={self.producer_to_run} with name={dataset_name}, file={input_file.abspath}"
                )
                with self.output().localize("w") as tmp_local_file:
                    out_local_path = tmp_local_file.abspath
                    shutil.move(tmpFile, out_local_path)
            if remove_job_home:
                shutil.rmtree(job_home)


class HistPlotTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    variables = luigi.Parameter(default="")

    @property
    def bundle_flavours(self):
        fl = AnaTupleFileListTask.req(self, branches=())
        return ["core", ("AnaTupleFileList", fl.version)]

    @property
    def active_variables(self):
        all_vars = _dedup_variables(self.global_params["variables"])
        if not self.variables:
            return all_vars
        selected = {v.strip() for v in self.variables.split(",") if v.strip()}
        result = [
            v for v in all_vars if (v["name"] if isinstance(v, dict) else v) in selected
        ]
        found = {v["name"] if isinstance(v, dict) else v for v in result}
        missing = selected - found
        if missing:
            print(f"Warning: variables not found in config: {sorted(missing)}")
        return result

    def workflow_requires(self):
        reqs = super().workflow_requires()
        merge_organization_complete = AnaTupleFileListTask.req(
            self,
            branches=(),
        ).complete()
        if not merge_organization_complete:
            reqs["HistMergerTask"] = HistMergerTask.req(
                self,
                branches=(),
                customisations=self.customisations,
            )
            # rely on req auto-copy for ana* so FileList __init__ manages version; no if/version for org list
            reqs["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
                self,
                branches=(),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
            return reqs
        merge_map = HistMergerTask.req(
            self,
            branch=-1,
            branches=(),
            customisations=self.customisations,
        ).create_branch_map()

        branch_set = set()
        for br_idx, (var) in self.branch_map.items():
            for br, (v, _, _) in merge_map.items():
                if v == var:
                    branch_set.add(br)

        reqs["merge"] = HistMergerTask.req(
            self,
            branches=tuple(branch_set),
            customisations=self.customisations,
        )
        return reqs

    def requires(self):
        var = self.branch_data

        merge_map = HistMergerTask.req(
            self,
            branch=-1,
            branches=(),
            customisations=self.customisations,
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
        return AnaTupleFileListTask.req(
            self,
            branch=-1,
            branches=(),
        ).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        branches = {}
        merge_map = HistMergerTask.req(
            self,
            branch=-1,
            branches=(),
            customisations=self.customisations,
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
                        "Plots",
                        self.period,
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
            infile = local_input.abspath
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
                    stack.enter_context((output).localize("w")).abspath
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
                    "--LAWrunVersion",
                    self.version,
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


class AnalysisCacheAggregationTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    producer_to_aggregate = luigi.Parameter()

    @property
    def bundle_flavours(self):
        fl = AnaTupleFileListTask.req(self, branches=())
        return ["core", ("AnaTupleFileList", fl.version)]

    def __init__(self, *args, **kwargs):
        ana_v = kwargs.get("ana_version") or kwargs.get("anaCache_version")
        if ana_v:
            kwargs["version"] = ana_v
        super(AnalysisCacheAggregationTask, self).__init__(*args, **kwargs)

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return AnaTupleFileListTask.req(
            self,
            branch=-1,
            branches=(),
        ).complete()

    def workflow_requires(self):
        reqs = super().workflow_requires()
        merge_organization_complete = AnaTupleFileListTask.req(
            self,
            branches=(),
        ).complete()
        payload_producers = self.global_params["payload_producers"]
        if not merge_organization_complete:
            reqs["AnaTupleFileListTask"] = AnaTupleFileListTask.req(
                self,
                branches=(),
                max_runtime=AnaTupleFileListTask.max_runtime._default,
                n_cpus=AnaTupleFileListTask.n_cpus._default,
            )
            reqs["AnalysisCacheTask"] = AnalysisCacheTask.req(
                self,
                branches=(),
                max_runtime=AnalysisCacheTask.max_runtime._default,
                n_cpus=AnalysisCacheTask.n_cpus._default,
                customisations=self.customisations,
                producer_to_run=self.producer_to_aggregate,
            )
            return reqs

        cache_branch_set = set()
        for idx, (
            sample_name,
            list_of_producer_cache_keys,
        ) in self.branch_map.items():
            for br in list_of_producer_cache_keys:
                cache_branch_set.add(br)

        reqs["AnalysisCacheTask"] = AnalysisCacheTask.req(
            self,
            branches=tuple(cache_branch_set),
            max_runtime=AnalysisCacheTask.max_runtime._default,
            n_cpus=AnalysisCacheTask.n_cpus._default,
            customisations=self.customisations,
            producer_to_run=self.producer_to_aggregate,
        )
        return reqs

    def requires(self):
        # I don't need to check here that this producer applies to target group
        # the reason is that if its in the branch map - it already was checked
        sample_name, list_of_producer_cache_keys = self.branch_data
        reqs = [
            AnalysisCacheTask.req(
                self,
                max_runtime=AnalysisCacheTask.max_runtime._default,
                branch=prod_br,
                customisations=self.customisations,
                producer_to_run=self.producer_to_aggregate,
            )
            for prod_br in list_of_producer_cache_keys
        ]
        return reqs

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        # structure of branch map
        # ---- name of sample,
        # ---- list of branch indices of the AnalysisCacheTask(producer_to_run=producer_name)

        branches = {}
        branch_idx = 0

        payload_producers = self.global_params["payload_producers"]
        producer_cfg = payload_producers[self.producer_to_aggregate]
        # AnalysisCacheTask manages its version from anaCache_version/ana_version copied by req or per-task CLI.
        # No version forcing if here.
        producer_cache_branch_map = AnalysisCacheTask.req(
            self,
            branch=-1,
            branches=(),
            producer_to_run=self.producer_to_aggregate,
        ).create_branch_map()

        # find which branches of this producer correspond to each sample
        sample_branch_map = {}
        for producer_cache_branch_idx, (
            sample_name,
            _,
            _,
            _,
            _,
        ) in producer_cache_branch_map.items():
            if sample_name not in sample_branch_map:
                sample_branch_map[sample_name] = []
            sample_branch_map[sample_name].append(producer_cache_branch_idx)

        target_groups = producer_cfg.get("target_groups", None)

        for sample_name, list_of_producer_cache_keys in sample_branch_map.items():
            process_group = (
                self.datasets[sample_name]["process_group"]
                if sample_name != "data"
                else "data"
            )
            applies_for_group = target_groups is None or process_group in target_groups
            if applies_for_group:
                branches[branch_idx] = (sample_name, list_of_producer_cache_keys)
                branch_idx += 1

        return branches

    @workflow_condition.output
    def output(self):
        sample_name, _ = self.branch_data
        extension = self.global_params["payload_producers"][
            self.producer_to_aggregate
        ].get("save_as", "root")
        output_name = f"aggregatedCache.{extension}"
        # Remote target (like AnalysisCacheTask) so the aggregated cache is a proper shared
        # artifact that law localizes for consumers, instead of a local path consumers
        # reconstruct (which broke under version overrides).
        output_path = os.path.join(
            self.version,
            "AnalysisCacheAggregation",
            self.producer_to_aggregate,
            self.period,
            sample_name,
            output_name,
        )
        return self.remote_target(output_path, fs=self.fs_anaCacheTuple)

    def run(self):
        sample_name, _ = self.branch_data
        producers = self.global_params["payload_producers"]
        cacheAggregator = os.path.join(
            self.ana_path(), "FLAF", "Analysis", "AnalysisCacheAggregator.py"
        )
        with contextlib.ExitStack() as stack:
            local_output = self.output()
            inputs = self.input()
            local_inputs = [
                stack.enter_context(inp.localize("r")).abspath for inp in inputs
            ]
            assert local_inputs, "`local_inputs` must be a non-empty list"
            producer_cfg = producers[self.producer_to_aggregate]
            ext = producer_cfg.get("save_as", "root")
            job_home, remove_job_home = self.law_job_home()
            tmpFile = os.path.join(job_home, f"aggregatedCache_tmp.{ext}")
            aggregate_cmd = [
                "python3",
                cacheAggregator,
                "--outFile",
                tmpFile,
                "--period",
                self.period,
                "--producer",
                self.producer_to_aggregate,
                "--LAWrunVersion",
                self.version,
            ]
            aggregate_cmd.append("--inputFiles")
            aggregate_cmd.extend(local_inputs)
            ps_call(aggregate_cmd, verbose=1)

            with local_output.localize("w") as local_out:
                shutil.move(tmpFile, local_out.abspath)
            print(
                f"Created aggregated cache for producer {self.producer_to_aggregate} "
                f"and dataset {sample_name}"
            )
            if remove_job_home:
                shutil.rmtree(job_home)
