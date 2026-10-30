#!/usr/bin/env python3
"""A task derived from a task of its own family through req() (cms-flaf/FLAF#356).

FLAF keeps the per-task resources (--max-runtime, --n-cpus, --tasks-per-job, --crab-memory) out
of req(), so that a task never hands its own to what it requires. law applies that exclusion
whatever the two families are, so a plain `X.req(x)` with `x` of family X silently takes the
class defaults instead of the values the command line gave `x`, and, n_cpus being significant,
can build a second task for the same outputs. Task.req refuses such a call unless it says which
of the two it means; law's own workflow <-> branch conversions say it already.

Everything here drives the real FLAF and law code, with only the analysis Setup faked.
"""

import contextlib
import importlib.util
import itertools
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

# FLAF.Common.Utilities (imported through Setup) imports ROOT at module level, but nothing
# exercised here touches it, and the unit-test CI has no ROOT. The stand-in is removed from
# sys.modules right after the import, so that an `import ROOT` in another test module of the
# same process still fails as it should.
_root_stub = None
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    _root_stub = sys.modules["ROOT"] = mock.MagicMock()

import law  # noqa: E402
import luigi  # noqa: E402

from FLAF.run_tools import law_customizations as lc  # noqa: E402

if _root_stub is not None and sys.modules.get("ROOT") is _root_stub:
    del sys.modules["ROOT"]

ERA = "Run3_2022"


class ReqGuardStage(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    """A FLAF workflow with the bases, in the order, of the analysis stages."""

    max_runtime = lc.copy_param(lc.HTCondorWorkflow.max_runtime, 5.0)
    n_cpus = lc.copy_param(lc.HTCondorWorkflow.n_cpus, 2)
    bundle_flavours = ["core"]

    def create_branch_map(self):
        return {0: 0, 1: 1}

    def run(self):
        pass


class ReqGuardOtherStage(ReqGuardStage):
    """Another family with the same parameters and defaults."""


#: non-default values of the four per-task parameters, as `law run` takes them
CLI_VALUES = {
    "max_runtime": ("--max-runtime", "9", 9.0),
    "n_cpus": ("--n-cpus", "3", 3),
    "tasks_per_job": ("--tasks-per-job", "7", 7),
    "crab_memory": ("--crab-memory", "4000", 4000),
}

_versions = itertools.count()


def defaults(cls):
    return {name: getattr(cls, name)._default for name in CLI_VALUES}


def values(task, names=CLI_VALUES):
    return {name: getattr(task, name) for name in names}


@contextlib.contextmanager
def flaf_env():
    """The analysis Setup faked, ANALYSIS_DATA_PATH in a temporary directory, submit node."""
    setup = types.SimpleNamespace(global_params={"crab": {}}, get_fs=lambda name: None)
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
        lc.Setup, "getGlobal", return_value=setup
    ), mock.patch.object(lc, "_known_eras", return_value=(ERA,)), mock.patch.dict(
        os.environ, {"ANALYSIS_DATA_PATH": tmp, "ANALYSIS_PATH": tmp}
    ):
        for var in ("LAW_JOB_HOME", "LAW_CRAB_JOB_NUMBER", "X509_USER_PROXY"):
            os.environ.pop(var, None)
        yield tmp


def reset_cli_state():
    # law caches the parsed command line per process, FLAF its prefer-cli drop set per parser
    # identity and luigi task instances by their parameters
    law.parser._reset()
    lc.Task._req_prefer_cli_drop_cache.clear()
    luigi.task_register.Register.clear_instance_cache()


class ASameFamilyReq(unittest.TestCase):
    def setUp(self):
        self.assertNotEqual(
            defaults(ReqGuardStage), {n: v[2] for n, v in CLI_VALUES.items()}
        )
        reset_cli_state()
        self.addCleanup(reset_cli_state)
        args = [
            ReqGuardStage.get_task_family(),
            "--version",
            f"vguard{next(_versions)}",
            "--era",
            ERA,
            "--workflow",
            "htcondor",
        ]
        for flag, text, _ in CLI_VALUES.values():
            args += [flag, text]
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(flaf_env())
        parser = stack.enter_context(
            luigi.cmdline_parser.CmdlineParser.global_instance(args)
        )
        self.root = parser.get_task_obj()

    def test_the_command_line_values_reach_the_root(self):
        self.assertEqual(values(self.root), {n: v[2] for n, v in CLI_VALUES.items()})

    def test_the_same_task_keeps_them(self):
        same = ReqGuardStage.req(self.root, _skip_task_excludes=True)
        self.assertEqual(values(same), values(self.root))
        self.assertEqual(same.task_id, self.root.task_id)

    def test_a_branch_keeps_max_runtime_and_n_cpus(self):
        # tasks_per_job and crab_memory are workflow-only (exclude_params_branch)
        branch = self.root.req_branch(0)
        names = ("max_runtime", "n_cpus")
        self.assertEqual(values(branch, names), values(self.root, names))

    def test_its_workflow_keeps_them_and_is_the_root(self):
        workflow = self.root.req_branch(0).req_workflow()
        names = ("max_runtime", "n_cpus")
        self.assertEqual(values(workflow, names), values(self.root, names))
        self.assertEqual(workflow.task_id, self.root.task_id)

    def test_a_plain_one_raises(self):
        calls = {
            "plain": lambda: ReqGuardStage.req(self.root),
            "a branch": lambda: ReqGuardStage.req(self.root, branch=0),
            # a law control keyword other than the three leaves the exclusion on
            "_exclude": lambda: ReqGuardStage.req(self.root, _exclude={"branches"}),
            "req_different_branching": lambda: ReqGuardStage.req_different_branching(
                self.root
            ),
            "from a branch": lambda: ReqGuardStage.req(self.root.req_branch(0)),
        }
        for name, call in calls.items():
            with self.subTest(name):
                with self.assertRaises(lc.SameFamilyReqError) as raised:
                    call()
                self.assertIn("_skip_task_excludes=True", str(raised.exception))
                self.assertIn("_skip_task_excludes=False", str(raised.exception))

    def test_a_separate_task_takes_the_class_defaults(self):
        other = ReqGuardStage.req(self.root, _skip_task_excludes=False)
        self.assertEqual(values(other), defaults(ReqGuardStage))
        # n_cpus is significant: a task of its own, which is what the caller declared
        self.assertNotEqual(other.task_id, self.root.task_id)

    def test_the_get_and_set_keywords_are_a_decision_too(self):
        for flag in ("_skip_task_excludes_get", "_skip_task_excludes_set"):
            for value in (True, False):
                with self.subTest(flag=flag, value=value):
                    ReqGuardStage.req(self.root, **{flag: value})

    def test_another_family_is_unaffected(self):
        other = ReqGuardOtherStage.req(self.root)
        self.assertEqual(values(other), defaults(ReqGuardOtherStage))

    def test_laws_and_flafs_own_conversions_pass(self):
        self.assertEqual(self.root.as_branch(1).as_workflow(), self.root)
        self.assertEqual(sorted(self.root.get_branch_tasks()), [0, 1])
        self.assertEqual(self.root.single(ERA).task_id, self.root.task_id)


if __name__ == "__main__":
    unittest.main()
