#!/usr/bin/env python3
"""The per-branch requirements of a branch task are branch tasks of one shared workflow.

A branch task resolves its branch map through its workflow (`as_workflow()`), the workflow
with the branch task's own `branches`. A requirement made as `X.req(self, branch=b,
branches=(b,))` had a workflow of its own for every b, and Task.cached_branch_map hands each
of them a copy of the whole map, which law reduces to b in place; a dict keeps its allocation
when entries are deleted, so every such requirement held a map of the whole required stage.
An AnaTupleFileListBuilderTask submission over Run3_2024 (34k requirements on a 56k-branch
AnaTupleFileTask map, ~2.6 MB each) was OOM-killed that way. Task.modify_param_values gives
every branch task `branches=()`, so a requirement is the same task whatever selection it inherits
from the task that makes it or is given. Proven here, on the real FLAF and law classes
over the scratch analysis of test_era_requires with a few hundred input files:

* the AnaTupleFileTask requirements of an AnaTupleFileListBuilderTask branch, the
  AnaTupleFileTask and AnaTupleFileListBuilderTask requirements of an AnaTupleMergeTask branch
  and the AnaTupleMergeTask and AnalysisCacheTask requirements of the HistTupleProducerTask
  branches are, per required stage, branch tasks of one workflow over its whole map, and hold
  that one branch map between them;
* each is the task (task_id) that its workflow runs as that branch, and for the Builder the
  branch task of the AnaTupleFileTask workflow that the Builder's workflow requires;
* so are the requirements a submission round resolves in the driver
  (`_stage_path_cache_input`, where the Run3_2024 submission ran out of memory), and the
  branch requirements of a workflow run over a selection of its branches (`--branches`),
  which they inherit;
* a branch task given a selection is the branch task without one (the same instance), from the
  constructor and from the command line (`--branch 1 --branches 1`, and a selection that does not
  hold the branch is ignored); a workflow keeps its selection;
* a workflow requirement names its `branches` (law's `req()` copies the requester's): before
  their gates hold, the Builder and AnalysisCacheTask over a selection require whole workflows.
"""

import json
import unittest
from unittest import mock

import luigi
import luigi.task_register

# a sibling, by its bare name (`test.<module>` would be CPython's own test package)
import test_era_requires as base  # noqa: E402

ana_prod = base.ana_prod
analysis = base.analysis
ERA = base.ERA_A

#: input files per dataset of ERA
N_FILES = {"Data_A": 300, "TTto2L2Nu": 100, "DYto2L": 200}


def significant(task, ignore=()):
    """The significant parameters of `task` (what its task_id is made of) but `ignore`."""
    return {
        name: value
        for name, value in task.to_str_params(only_significant=True).items()
        if name not in ignore
    }


class BranchRequirementsShareOneWorkflow(base._Scratch):
    def setUp(self):
        super().setUp()
        ref = ana_prod.InputFileTask(version=self.version, era=ERA)
        for dataset_id, dataset_name in ref.iter_datasets():
            files = [f"{dataset_name}_{i}.root" for i in range(N_FILES[dataset_name])]
            base.write(
                ref.req_branch(dataset_id).output().abspath,
                json.dumps({"input_files": files}),
            )
        base.clear_process_caches()

    def write_merge_plans(self):
        """One merged file per AnaTupleFileList dataset, made of all its AnaTuple files."""
        ref = ana_prod.AnaTupleFileListTask(version=self.version, era=ERA)
        for branch, (dataset_name, _) in ref.get_branch_map().items():
            source = "Data_A" if dataset_name == "data" else dataset_name
            inputs = [f"{source}/anaTupleFile_{i}" for i in range(N_FILES[source])]
            plan = [
                {
                    "inputs": inputs,
                    "outputs": ["anaTuple_0.root"],
                    "n_events": 10,
                    "runs": [],
                }
            ]
            base.write(ref.req_branch(branch).output().abspath, json.dumps(plan))
        base.clear_process_caches()

    def assert_one_workflow(self, reqs, family, n_branches, ignore=()):
        """`reqs`: branch tasks of `family` that share one workflow over its whole map
        (`n_branches` branches), each the task that workflow runs as that branch: the same
        significant parameters but those in `ignore`."""
        self.assertTrue(reqs)
        self.assertEqual({req.get_task_family() for req in reqs}, {family})
        self.assertEqual(len({id(req.as_workflow()) for req in reqs}), 1)
        self.assertEqual(len({id(req.get_branch_map()) for req in reqs}), 1)
        workflow = reqs[0].as_workflow()
        self.assertEqual(len(workflow.get_branch_map()), n_branches)
        branch_tasks = workflow.get_branch_tasks()
        for req in reqs:
            self.assertEqual(
                significant(req, ignore), significant(branch_tasks[req.branch], ignore)
            )

    def test_the_files_of_a_builder_branch(self):
        builder = ana_prod.AnaTupleFileListBuilderTask(
            version=self.version, era=ERA, workflow="htcondor"
        )
        data = [
            b for b, (name, _) in builder.get_branch_map().items() if name == "data"
        ]
        reqs = builder.req_branch(data[0]).requires()
        self.assertEqual(len(reqs), N_FILES["Data_A"])
        self.assert_one_workflow(reqs, "AnaTupleFileTask", sum(N_FILES.values()))
        # the very tasks of the AnaTupleFileTask workflow that the Builder requires
        files = builder.workflow_requires()["AnaTupleFileTask"].get_branch_tasks()
        for req in reqs:
            self.assertEqual(req.task_id, files[req.branch].task_id)

    def test_the_inputs_of_the_merge_branches(self):
        self.write_merge_plans()
        merge = ana_prod.AnaTupleMergeTask(
            version=self.version, era=ERA, workflow="htcondor"
        )
        reqs = {
            data[0]: merge.req_branch(branch).requires()
            for branch, data in merge.get_branch_map().items()
        }
        self.assertEqual(sorted(reqs), ["DYto2L", "TTto2L2Nu", "data"])
        self.assertEqual(len(reqs["DYto2L"]["root"]["DYto2L"]), N_FILES["DYto2L"])
        self.assert_one_workflow(
            reqs["DYto2L"]["root"]["DYto2L"], "AnaTupleFileTask", sum(N_FILES.values())
        )
        # one workflow for the inputs of every merge branch
        files = [
            req
            for branch_reqs in reqs.values()
            for dataset_reqs in branch_reqs["root"].values()
            for req in dataset_reqs
        ]
        self.assertEqual(len(files), sum(N_FILES.values()))
        self.assert_one_workflow(files, "AnaTupleFileTask", sum(N_FILES.values()))
        # the plans of the simulated datasets
        plans = [
            req for branch_reqs in reqs.values() for req in branch_reqs["json"].values()
        ]
        self.assertEqual(len(plans), 2)
        self.assert_one_workflow(plans, "AnaTupleFileListBuilderTask", 3)

    def test_the_inputs_of_the_hist_tuple_branches(self):
        self.write_merge_plans()
        producer = analysis.HistTupleProducerTask(
            version=self.version, era=ERA, workflow="htcondor"
        )
        reqs = [producer.req_branch(b).requires() for b in producer.get_branch_map()]
        self.assertEqual(len(reqs), 3)
        self.assert_one_workflow([r["anaTuple"] for r in reqs], "AnaTupleMergeTask", 3)
        caches = [r["anaCaches"]["P1"] for r in reqs]
        # AnalysisCacheTask.__init__ sets n_cpus from the producer after luigi has made the
        # parameters: a requirement keeps the class default (exclude_params_req), its
        # workflow hands the producer's value on to the branch tasks
        self.assert_one_workflow(caches, "AnalysisCacheTask", 3, ignore=("n_cpus",))

    def test_the_requirements_of_a_submission_round(self):
        builder = ana_prod.AnaTupleFileListBuilderTask(
            version=self.version, era=ERA, workflow="htcondor"
        )
        config = mock.Mock(input_files={})
        with mock.patch(
            "FLAF.RunKit.law_gfal.collect_existing_target_entries",
            return_value=([], 0),
        ), mock.patch(
            "FLAF.RunKit.law_gfal.collect_setup_path_cache_entries", return_value=[]
        ), mock.patch(
            "FLAF.RunKit.law_gfal.write_path_cache_file"
        ), mock.patch.object(
            base.lc, "list_target_directories"
        ):
            builder._stage_path_cache_input(
                config, [[b] for b in builder.get_branch_map()]
            )
        self.assertIn("path_cache", config.input_files)
        files = [
            task
            for task in luigi.task_register.Register._Register__instance_cache.values()
            if task.get_task_family() == "AnaTupleFileTask"
        ]
        workflows = [task for task in files if task.is_workflow()]
        self.assertEqual([w.branches for w in workflows], [()])
        reqs = [task for task in files if task.is_branch()]
        self.assertEqual(len(reqs), sum(N_FILES.values()))
        self.assert_one_workflow(reqs, "AnaTupleFileTask", sum(N_FILES.values()))

    def test_the_branch_requirements_of_a_selection(self):
        self.write_merge_plans()
        for selected in (0, 1):
            with self.subTest(selected=selected):
                cache = analysis.AnalysisCacheTask(
                    version=self.version,
                    era=ERA,
                    workflow="htcondor",
                    producer_to_run="P2",
                    branches=(selected,),
                )
                reqs = cache.workflow_requires()
                merges = list(reqs["anaTuple"].values())
                self.assertEqual([m.branch for m in merges], [selected])
                self.assertEqual(merges[0].branches, ())
                self.assert_one_workflow(merges, "AnaTupleMergeTask", 3)
                caches = list(reqs["P1"].values())
                self.assertEqual([c.branch for c in caches], [selected])
                self.assertEqual(caches[0].branches, ())
                self.assert_one_workflow(
                    caches, "AnalysisCacheTask", 3, ignore=("n_cpus",)
                )

    def test_a_branch_task_is_one_task_whatever_its_selection(self):
        files = ana_prod.InputFileTask(version=self.version, era=ERA)
        argv = ["InputFileTask", "--version", self.version, "--era", ERA]
        for b in (0, 1):
            with self.subTest(branch=b):
                other_b = 1 - b
                selection = ana_prod.InputFileTask(
                    version=self.version, era=ERA, branches=(b,)
                )
                self.assertTrue(selection.branches)
                self.assertNotEqual(selection.task_id, files.task_id)
                branch = files.req_branch(b)
                for branches in [(b,), (other_b,), (0, 2), ((0, 2),)]:
                    task = ana_prod.InputFileTask(
                        version=self.version, era=ERA, branch=b, branches=branches
                    )
                    self.assertIs(task, branch)
                self.assertIs(selection.req_branch(b), branch)
                same = ana_prod.InputFileTask.req(
                    selection, branch=b, _skip_task_excludes=True
                )
                self.assertIs(same, branch)
                # inherited from a task of another stage over a selection (Task.req's own
                # path); another instance only because InputFileTask.__init__ sets
                # workflow="local" after luigi keyed the instance on the inherited workflow
                other = ana_prod.AnaTupleFileTask(
                    version=self.version, era=ERA, branches=(other_b,)
                )
                inherited = ana_prod.InputFileTask.req(other, branch=b)
                self.assertEqual(inherited.branches, ())
                self.assertEqual(inherited.task_id, branch.task_id)
                for selected in (b, other_b):
                    with luigi.cmdline_parser.CmdlineParser.global_instance(
                        argv + ["--branch", str(b), "--branches", str(selected)]
                    ) as parser:
                        task = parser.get_task_obj()
                    self.assertIs(task, branch)
                    self.assertEqual(task.output().abspath, branch.output().abspath)
        with luigi.cmdline_parser.CmdlineParser.global_instance(
            argv + ["--branches", "1"]
        ) as parser:
            task = parser.get_task_obj()
        self.assertTrue(task.is_workflow())
        self.assertEqual(sorted(task.get_branch_map()), [1])


class WorkflowRequirementsBeforeTheGate(base._Scratch):
    """A workflow over a selection requires whole upstream workflows while its gate does not
    hold: these requirements keep their explicit `branches=()`, or they would inherit the
    selection, in the requester's numbering."""

    def assert_whole(self, reqs, keys):
        self.assertLessEqual(set(keys), set(reqs))
        tasks = [task for key in keys for task in luigi.task.flatten(reqs[key])]
        self.assertTrue(tasks)
        for task in tasks:
            self.assertTrue(task.is_workflow())
            self.assertEqual(task.branches, ())

    def test_the_workflow_requirements_before_the_gate(self):
        builder = ana_prod.AnaTupleFileListBuilderTask(
            version=self.version, era=ERA, workflow="htcondor", branches=(1,)
        )
        self.assert_whole(builder.workflow_requires(), ["anaTuple", "inputFile"])
        base.write_input_lists(self.version, ERA)
        base.clear_process_caches()
        cache = analysis.AnalysisCacheTask(
            version=self.version,
            era=ERA,
            workflow="htcondor",
            producer_to_run="P2",
            branches=(1,),
        )
        self.assert_whole(
            cache.workflow_requires(),
            ["AnaTupleFileListTask", "AnaTupleMergeTask", "AnalysisCacheTask"],
        )


if __name__ == "__main__":
    unittest.main()
