#!/usr/bin/env python3
"""The AnaTuple merge plans of an existing production, reused from a new checkout (#352).

Every branch map from AnaTupleMergeTask down is built from the local copy of the merge plans
(AnaTupleFileListTask, in the checkout's data area). A new checkout that reuses a production
(`--anaTuple-version`, fs_anaTuple at its storage) has the production's plans on fs_anaTuple
(AnaTupleFileListBuilderTask complete) but no local copy. `anaTuple_plans_ready` copies them
there. Proven here, on the real FLAF and law classes over the scratch analysis of
test_era_requires (fs_default, hence fs_anaTuple, a local directory):

* without the Builder's plans, or with only some of them, nothing is copied and the answer
  is the one before: not ready;
* with them, every plan is copied byte for byte (as AnaTupleFileListTask.run() copies it),
  over one era and over two, a bounded number at once, and the answer is ready;
* a copy that fails or is interrupted leaves no output that counts as complete, and a reader
  never sees a partial plan; concurrent callers, threads or processes, all end with the
  complete plans;
* inside a batch job nothing is copied;
* in a new checkout every stage requires, and every conditioned stage builds as its branch
  map, what it does once the plans have been copied by hand (the issue's workaround), for
  `--workflow local`, `htcondor`, `htcondor --bundle` and `crab`;
* no condition or requirement fallback asks AnaTupleFileListTask.complete() on its own.
"""

import ast
import json
import multiprocessing
import os
import shutil
import threading
import time
import unittest
from unittest import mock

# a sibling, by its bare name (`test.<module>` would be CPython's own test package)
import test_era_requires as base  # noqa: E402

ana_prod = base.ana_prod
analysis = base.analysis
ERA_A = base.ERA_A
ERA_B = base.ERA_B

_real_copy = shutil.copy


def builder_targets(version, era):
    """{dataset: (plan target, reports target)} of AnaTupleFileListBuilderTask over `era`."""
    ref = ana_prod.AnaTupleFileListBuilderTask(
        version=version, era=era, workflow="local"
    )
    targets = {}
    for branch, (dataset_name, _) in ref.get_branch_map().items():
        outputs = ref.req_branch(branch).output()
        targets[dataset_name] = (outputs["plan"], outputs["reports"])
    return targets


def plan_text(era, dataset_name):
    plan = []
    n_out = 0
    for inputs, n_outputs, n_events in base.MERGE_PLANS[(era, dataset_name)]:
        outputs = [f"anaTuple_{n_out + i}.root" for i in range(n_outputs)]
        n_out += n_outputs
        plan.append(
            {"inputs": inputs, "outputs": outputs, "n_events": n_events, "runs": []}
        )
    # large enough that a copy written in two halves is visibly partial
    return json.dumps(plan, indent=2) + "\n" + " " * 4096 + "\n"


def write_builder_plans(version, era, datasets=None):
    """The Builder's outputs of `era` on fs_anaTuple (of `datasets`, default all)."""
    for dataset_name, (plan, reports) in builder_targets(version, era).items():
        if datasets is not None and dataset_name not in datasets:
            continue
        base.write(plan.abspath, plan_text(era, dataset_name))
        base.write(reports.abspath, json.dumps({"reports": {}}))


def local_copies(version, era):
    """{dataset: local copy path} of AnaTupleFileListTask over `era`."""
    ref = ana_prod.AnaTupleFileListTask(version=version, era=era)
    return {
        dataset_name: ref.req_branch(branch).output().abspath
        for branch, (dataset_name, _) in ref.get_branch_map().items()
    }


def files_under(path):
    found = []
    for dir_path, _, file_names in os.walk(path):
        found.extend(os.path.join(dir_path, name) for name in file_names)
    return sorted(found)


def slow_copy(src, dst, *args, **kwargs):
    """shutil.copy that writes the first half, waits, then the rest."""
    with open(src) as f:
        text = f.read()
    with open(dst, "w") as f:
        f.write(text[: len(text) // 2])
        f.flush()
        time.sleep(0.05)
        f.write(text[len(text) // 2 :])
    shutil.copymode(src, dst)
    return dst


class _Reuse(base._Scratch):
    def plans_ready(self, era=ERA_A, cls=None, **kwargs):
        cls = cls or analysis.HistTupleProducerTask
        kwargs.setdefault("workflow", "local")
        return ana_prod.anaTuple_plans_ready(
            cls(version=self.version, era=era, **kwargs)
        )

    def data_area(self):
        return os.path.join(
            os.environ["ANALYSIS_DATA_PATH"], self.version, "AnaTupleFileListTask"
        )

    def assert_copied(self, era):
        copies = local_copies(self.version, era)
        self.assertTrue(copies)
        for dataset_name, path in copies.items():
            with open(path) as f:
                self.assertEqual(f.read(), plan_text(era, dataset_name), path)
        self.assertEqual(
            files_under(os.path.join(self.data_area(), era)), sorted(copies.values())
        )


class NothingToCopy(_Reuse):
    """Without the production's plans the answer is the one before, and nothing is written."""

    def test_no_builder_outputs(self):
        self.assertFalse(self.plans_ready())
        self.assertFalse(os.path.exists(self.data_area()))

    def test_some_builder_outputs(self):
        write_builder_plans(self.version, ERA_A, datasets=["data", "TTto2L2Nu"])
        self.assertFalse(self.plans_ready())
        self.assertFalse(os.path.exists(self.data_area()))

    def test_one_era_of_two_without_them(self):
        write_builder_plans(self.version, ERA_A)
        self.assertFalse(self.plans_ready(era=(ERA_A, ERA_B)))
        self.assertFalse(os.path.exists(self.data_area()))

    def test_inside_a_batch_job(self):
        write_builder_plans(self.version, ERA_A)
        with mock.patch.dict(os.environ, {"LAW_JOB_HOME": self.tmp}):
            self.assertFalse(self.plans_ready())
        self.assertFalse(os.path.exists(self.data_area()))

    def test_an_existing_copy_is_read_as_it_is(self):
        base.write_merge_plans(self.version, ERA_A)
        with mock.patch.object(ana_prod.AnaTupleFileListTask, "copy_plans") as copy:
            self.assertTrue(self.plans_ready())
        copy.assert_not_called()


class TheProductionsPlansAreCopied(_Reuse):
    def test_one_era(self):
        write_builder_plans(self.version, ERA_A)
        self.assertTrue(self.plans_ready())
        self.assert_copied(ERA_A)
        self.assertFalse(os.path.exists(os.path.join(self.data_area(), ERA_B)))

    def test_two_eras(self):
        for era in (ERA_A, ERA_B):
            write_builder_plans(self.version, era)
        self.assertTrue(self.plans_ready(era=(ERA_A, ERA_B)))
        for era in (ERA_A, ERA_B):
            self.assert_copied(era)

    def test_only_the_missing_ones(self):
        write_builder_plans(self.version, ERA_A)
        kept = local_copies(self.version, ERA_A)["DYto2L"]
        base.write(kept, "kept")
        self.assertTrue(self.plans_ready())
        with open(kept) as f:
            self.assertEqual(f.read(), "kept")

    def test_from_every_requester(self):
        # the predicate as the conditions call it, from each conditioned stage and workflow
        for cls, extra in [
            (ana_prod.AnaTupleMergeTask, {}),
            (analysis.HistTupleProducerTask, {}),
            (analysis.HistFromNtupleProducerTask, {}),
            (analysis.HistMergerTask, {}),
            (analysis.HistPlotTask, {}),
            (analysis.AnalysisCacheTask, {"producer_to_run": "P1"}),
            (analysis.AnalysisCacheAggregationTask, {"producer_to_aggregate": "P2"}),
        ]:
            for workflow in ("local", "htcondor", "crab"):
                with self.subTest(cls=cls.__name__, workflow=workflow):
                    shutil.rmtree(os.environ["ANALYSIS_DATA_PATH"], ignore_errors=True)
                    base.clear_process_caches()
                    write_builder_plans(self.version, ERA_A)
                    task = cls(
                        version=self.version, era=ERA_A, workflow=workflow, **extra
                    )
                    self.assertTrue(task.workflow_condition())
                    self.assert_copied(ERA_A)

    def test_as_the_task_copies_them(self):
        write_builder_plans(self.version, ERA_A)
        file_list = ana_prod.AnaTupleFileListTask(version=self.version, era=ERA_A)
        branch_task = file_list.req_branch(0)
        branch_task.run()
        ran = branch_task.output().abspath
        with open(ran) as f:
            ran_text = f.read()
        ran_mode = os.stat(ran).st_mode
        os.remove(ran)
        self.assertTrue(self.plans_ready())
        with open(ran) as f:
            self.assertEqual(f.read(), ran_text)
        self.assertEqual(os.stat(ran).st_mode, ran_mode)
        source = builder_targets(self.version, ERA_A)[branch_task.branch_data[0]][0]
        self.assertEqual(ran_mode, os.stat(source.abspath).st_mode)

    def test_a_bounded_number_at_once(self):
        write_builder_plans(self.version, ERA_A)
        state = {"now": 0, "max": 0}
        lock = threading.Lock()

        def counting_copy(*args, **kwargs):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.1)
            with lock:
                state["now"] -= 1
            return _real_copy(*args, **kwargs)

        with mock.patch.object(ana_prod, "PLAN_COPY_THREADS", 2):
            with mock.patch("shutil.copy", counting_copy):
                self.assertTrue(self.plans_ready())
        self.assertEqual(state["max"], 2)
        self.assert_copied(ERA_A)

    def test_a_failure_stops_the_queued_copies(self):
        # Two at once over six plans: the first plan is slow, the second fails at once. The
        # failure ends the copy then, not once the copies before it are done: the queued
        # plans are not started.
        for error in (OSError, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                shutil.rmtree(os.environ["ANALYSIS_DATA_PATH"], ignore_errors=True)
                base.clear_process_caches()
                for era in (ERA_A, ERA_B):
                    write_builder_plans(self.version, era)
                first, second = list(local_copies(self.version, ERA_A).values())[:2]
                started = []

                def copy(src, dst, *args, **kwargs):
                    output = os.path.join(
                        os.path.dirname(dst),
                        os.path.basename(dst)[1:].rsplit(".", 1)[0],
                    )
                    started.append(output)
                    if output == second:
                        raise error("copy failed")
                    time.sleep(1.0 if output == first else 0.2)
                    return _real_copy(src, dst, *args, **kwargs)

                with mock.patch.object(ana_prod, "PLAN_COPY_THREADS", 2):
                    with mock.patch("shutil.copy", copy):
                        with self.assertRaises(error):
                            self.plans_ready(era=(ERA_A, ERA_B))
                self.assertEqual(set(started[:2]), {first, second})
                self.assertLessEqual(len(started), 3)


class APartialCopyNeverCounts(_Reuse):
    def test_a_failed_copy(self):
        write_builder_plans(self.version, ERA_A)

        def failing_copy(src, dst, *args, **kwargs):
            with open(src) as f:
                text = f.read()
            with open(dst, "w") as f:
                f.write(text[: len(text) // 2])
            raise OSError("interrupted")

        with mock.patch("shutil.copy", failing_copy):
            with self.assertRaises(OSError):
                self.plans_ready()
        self.assertFalse(
            ana_prod.AnaTupleFileListTask(version=self.version, era=ERA_A).complete()
        )
        self.assertEqual(files_under(self.data_area()), [])
        base.clear_process_caches()
        self.assertTrue(self.plans_ready())
        self.assert_copied(ERA_A)

    def test_a_killed_copy(self):
        # what a copy killed before its rename leaves: a partial file beside the output
        write_builder_plans(self.version, ERA_A)
        output = local_copies(self.version, ERA_A)["DYto2L"]
        leftover = os.path.join(
            os.path.dirname(output), f".{os.path.basename(output)}.x1y2z3"
        )
        base.write(leftover, plan_text(ERA_A, "DYto2L")[:20])
        self.assertFalse(
            ana_prod.AnaTupleFileListTask(version=self.version, era=ERA_A).complete()
        )
        self.assertTrue(self.plans_ready())
        with open(output) as f:
            self.assertEqual(f.read(), plan_text(ERA_A, "DYto2L"))

    def test_a_reader_never_sees_a_partial_plan(self):
        write_builder_plans(self.version, ERA_A)
        outputs = local_copies(self.version, ERA_A)
        expected = {path: plan_text(ERA_A, name) for name, path in outputs.items()}
        seen = []
        done = threading.Event()

        def read_while_copying():
            while not done.is_set():
                for path, text in expected.items():
                    try:
                        with open(path) as f:
                            seen.append(f.read() == text)
                    except FileNotFoundError:
                        pass

        reader = threading.Thread(target=read_while_copying)
        reader.start()
        try:
            with mock.patch("shutil.copy", slow_copy):
                self.assertTrue(self.plans_ready())
        finally:
            done.set()
            reader.join()
        self.assertTrue(seen)
        self.assertTrue(all(seen))


def _ready_in_process(ana, data, version, barrier, results):
    os.environ["ANALYSIS_PATH"] = ana
    os.environ["ANALYSIS_DATA_PATH"] = data
    base.clear_process_caches()
    task = analysis.HistTupleProducerTask(version=version, era=ERA_A, workflow="local")
    barrier.wait()
    try:
        with mock.patch("shutil.copy", slow_copy):
            results.put(ana_prod.anaTuple_plans_ready(task))
    except Exception as e:  # reported to the parent, which fails the test
        results.put(repr(e))


class ConcurrentCallers(_Reuse):
    N = 4

    def test_threads(self):
        write_builder_plans(self.version, ERA_A)
        barrier = threading.Barrier(self.N)
        results = []

        def call():
            task = analysis.HistTupleProducerTask(
                version=self.version, era=ERA_A, workflow="local"
            )
            barrier.wait()
            results.append(ana_prod.anaTuple_plans_ready(task))

        with mock.patch("shutil.copy", slow_copy):
            threads = [threading.Thread(target=call) for _ in range(self.N)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(results, [True] * self.N)
        self.assert_copied(ERA_A)

    def test_processes(self):
        write_builder_plans(self.version, ERA_A)
        # spawned, not forked: a fork of a process that has run threads can deadlock
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(self.N)
        results = context.Queue()
        processes = [
            context.Process(
                target=_ready_in_process,
                args=(
                    os.environ["ANALYSIS_PATH"],
                    os.environ["ANALYSIS_DATA_PATH"],
                    self.version,
                    barrier,
                    results,
                ),
            )
            for _ in range(self.N)
        ]
        for process in processes:
            process.start()
        answers = [results.get(timeout=120) for _ in processes]
        for process in processes:
            process.join(timeout=120)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(answers, [True] * self.N)
        self.assert_copied(ERA_A)


class ANewCheckoutRequiresWhatACopiedOneDoes(_Reuse):
    """Every stage with a requirement body, in a checkout whose data area is empty and whose
    fs_anaTuple holds a production's plans, requires and maps what it does once
    AnaTupleFileListTask has copied them (the workaround of the issue)."""

    #: the stages whose branch maps are built from the plans
    PLAN_STAGES = {
        "AnaTupleMergeTask",
        "HistTupleProducerTask",
        "HistFromNtupleProducerTask",
        "HistMergerTask",
        "AnalysisCacheTask",
        "HistPlotTask",
        "AnalysisCacheAggregationTask",
    }

    def snapshot(self, stage, module, extra, params):
        task = getattr(module, stage)(
            version=self.version, era=ERA_A, **extra, **params
        )
        # the requirements first: asked before anything else has copied the plans
        reqs = base.requirement_ids(task.workflow_requires())
        branch_map = None
        if stage in self.PLAN_STAGES:
            branch_map = dict(task.get_branch_map())
        return reqs, branch_map

    def test_every_stage(self):
        write_builder_plans(self.version, ERA_A)
        copied_data = os.path.join(self.tmp, "data_copied")
        os.environ["ANALYSIS_DATA_PATH"] = copied_data
        base.clear_process_caches()
        file_list = ana_prod.AnaTupleFileListTask(version=self.version, era=ERA_A)
        for branch_task in file_list.get_branch_tasks().values():
            branch_task.run()
        base.clear_process_caches()
        expected = {}
        for stage, module, extra in base.STAGES:
            for flavour, params in base.FLAVOURS.items():
                expected[(stage, flavour)] = self.snapshot(stage, module, extra, params)
        for stage, module, extra in base.STAGES:
            for flavour, params in base.FLAVOURS.items():
                with self.subTest(stage=stage, flavour=flavour):
                    fresh = os.path.join(self.tmp, f"data_{stage}_{flavour}")
                    os.environ["ANALYSIS_DATA_PATH"] = fresh
                    base.clear_process_caches()
                    now = self.snapshot(stage, module, extra, params)
                    self.assertEqual(now, expected[(stage, flavour)])
                    if stage in self.PLAN_STAGES:
                        self.assertNotEqual(expected[(stage, flavour)][1], {0: None})


class OnePredicateDecides(unittest.TestCase):
    """No dynamic condition or requirement fallback asks AnaTupleFileListTask.complete() on
    its own: they all go through anaTuple_plans_ready, which may copy the plans first.
    """

    def test_no_direct_completeness_check(self):
        found = []
        for module in ("AnaProd", "Analysis"):
            path = os.path.join(base.flaf_repo, module, "tasks.py")
            with open(path) as f:
                tree = ast.parse(f.read())
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(function):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "complete"
                        and isinstance(node.func.value, ast.Call)
                        and "AnaTupleFileList" in ast.unparse(node.func.value.func)
                    ):
                        found.append(f"{module}/tasks.py:{node.lineno} {function.name}")
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
