#!/usr/bin/env python3
"""What a bundle job is handed at submission, and how the HelloWorld smoke test reads its log.

* A bundle job sources env.sh from its unpacked bundles, which refuses to run without flaf_env:
  a task whose bundle flavours do not pack it is refused at submission
  (`HTCondorWorkflow._bundle_tasks`), rather than every one of its jobs failing on the worker.
* Without spooling (`--bundle --htcondor-spool False`) the job inputs are uploaded to
  `worker_files/<era>/` (`CERNHTCondorJobFileFactory._stage_and_update_jdl`). law names a copied
  input after the path it was copied from, not after its content, so one name carries the
  rendered scripts of other tasks and of edited code: the upload is keyed on the content.
* `test/test_hello_world.py` takes the staged log's path from the line law prints for a failed
  job, whose fields law separates with ", ".
"""

import contextlib
import hashlib
import importlib.util
import io
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import law  # noqa: E402

# The unit-test runner has no ROOT, and law_customizations reaches FLAF.Common.Utilities,
# which imports it at module level without using it on import. A placeholder stands in for
# that import only and is removed again.
_root_placeholder = (
    "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None
)
if _root_placeholder:
    sys.modules["ROOT"] = types.ModuleType("ROOT")
try:
    from FLAF.run_tools import law_customizations as lc  # noqa: E402
    from FLAF.test.hello_world_task import HelloWorldTask  # noqa: E402
finally:
    if _root_placeholder:
        sys.modules.pop("ROOT", None)

# the smoke-test runner is a script, loaded from its file
_spec = importlib.util.spec_from_file_location(
    "flaf_test_hello_world", os.path.join(flaf_repo, "test", "test_hello_world.py")
)
test_hello_world = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(test_hello_world)

ERA = "Run3_2022EE"

#: the bundles of the analyses' config/global.yaml (H_mumu, HH_bbtautau and HH_bbWW alike)
ANALYSIS_BUNDLES = {
    "core": {
        "hashed": True,
        "patterns": ["FLAF", "AnaProd", "Analysis", "config", "env.sh", "include"],
    },
    "soft": {"patterns": ["soft/flaf_env"]},
    "cmssw": {"patterns": ["soft/CMSSW_16_0_6"]},
}


class StubBundleTask(lc.BundleTask):
    """BundleTask without the law machinery: what decides whether it packs flaf_env."""

    bundles = ANALYSIS_BUNDLES

    @property
    def global_params(self):
        return {"bundles": self.bundles}


class StubWorkflow:
    """What `_bundle_tasks` reads of a workflow task over one era."""

    _bundle_tasks = lc.HTCondorWorkflow._bundle_tasks
    _bundle_requirements = lc.HTCondorWorkflow._bundle_requirements
    meta_era = None
    version = "v1"
    workflow = "htcondor"

    def __init__(self, bundle_flavours, bundles=ANALYSIS_BUNDLES, uses_bundles=True):
        self.bundle_flavours = bundle_flavours
        self.global_params = {"bundles": bundles}
        self.uses_bundles = uses_bundles

    def _uses_bundles(self):
        return self.uses_bundles


def stub_bundle_req(task, flavour, version, upstream_workflow):
    # luigi's metaclass owns __call__, so the instance is built without it
    bundle = object.__new__(StubBundleTask)
    bundle.flavour = flavour
    bundle.version = version
    bundle.era = (ERA,)
    bundle.bundles = task.global_params["bundles"]
    return bundle


class ABundleJobShipsTheEnvironment(unittest.TestCase):
    def setUp(self):
        ana = tempfile.mkdtemp(prefix="flaf_bundle_flavours_")
        self.addCleanup(shutil.rmtree, ana, True)
        for patch in (
            mock.patch.dict(
                os.environ,
                {
                    "ANALYSIS_PATH": ana,
                    "FLAF_PATH": os.path.join(ana, "FLAF"),
                    "FLAF_ENVIRONMENT_PATH": os.path.join(ana, "soft", "flaf_env"),
                },
            ),
            mock.patch.object(lc.BundleTask, "req", staticmethod(stub_bundle_req)),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def flavours(task):
        with contextlib.redirect_stderr(io.StringIO()):
            return [flavour for flavour, _ in task._bundle_tasks()]

    def test_a_task_shipping_core_alone_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, r"\['core'\] packs flaf_env"):
            self.flavours(StubWorkflow(["core"]))

    def test_a_task_shipping_the_soft_flavour_goes_through(self):
        self.assertEqual(
            self.flavours(StubWorkflow(["core", "soft"])), ["core", "soft"]
        )

    def test_the_flavours_of_the_framework_tasks_go_through(self):
        for flavours in (
            ["core", "soft", "inputFileList", "cmssw"],
            ["core", "soft", ("AnaTupleFileList", "v0")],
        ):
            with self.subTest(flavours=flavours):
                self.flavours(StubWorkflow(flavours))

    def test_the_hello_world_task_ships_the_environment(self):
        self.assertIn(
            "soft", self.flavours(StubWorkflow(HelloWorldTask.bundle_flavours))
        )

    def test_an_analysis_without_the_flavour_is_refused(self):
        lc.BundleTask._unconfigured_reported.discard("soft")
        bundles = {"core": ANALYSIS_BUNDLES["core"]}
        with self.assertRaisesRegex(RuntimeError, "packs flaf_env"):
            self.flavours(StubWorkflow(["core", "soft"], bundles))

    def test_an_environment_packed_by_another_flavour_goes_through(self):
        bundles = {"core": ANALYSIS_BUNDLES["core"], "all_soft": {"patterns": ["soft"]}}
        self.assertEqual(
            self.flavours(StubWorkflow(["core", "all_soft"], bundles)),
            ["core", "all_soft"],
        )

    def test_an_environment_kept_outside_every_bundle_is_refused(self):
        with mock.patch.dict(
            os.environ, {"FLAF_ENVIRONMENT_PATH": "/elsewhere/flaf_env"}
        ):
            with self.assertRaisesRegex(RuntimeError, "/elsewhere/flaf_env"):
                self.flavours(StubWorkflow(["core", "soft"]))

    def test_a_job_without_bundles_is_not_checked(self):
        self.assertEqual(
            StubWorkflow(["core"], uses_bundles=False)._bundle_requirements(), {}
        )


class WorkerFilesAreKeyedOnTheirContent(unittest.TestCase):
    """`_stage_and_update_jdl` over job files that list inputs under fixed names, each written
    in a submission directory of its own (law's mkdtemp), with a local directory standing in
    for `<version>/worker_files/<era>/` on fs_default."""

    #: the names law gives FLAF's job inputs: a hash of the source path, not of the content
    BOOTSTRAP = "bootstrap_21baff6313.sh"
    STAGEOUT = "stageout_logs_fc19ee06ca.sh"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_worker_files_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.worker_files = law.LocalDirectoryTarget(
            os.path.join(self.tmp, "v1", "worker_files", ERA)
        )
        self.worker_files.touch()

    def submit(self, inputs):
        """Stage a job file listing `inputs` ({name: content}) and a remote URL; return its
        transfer_input_files, the paths uploaded, and its lines."""
        submission = tempfile.mkdtemp(dir=self.tmp)
        paths = []
        for name, content in inputs.items():
            paths.append(os.path.join(submission, name))
            with open(paths[-1], "w") as f:
                f.write(content)
        job_file = os.path.join(submission, "job.jdl")
        with open(job_file, "w") as f:
            f.write(
                "universe = vanilla\n"
                "executable = htcondor_wrapper_28131c4dc3.sh\n"
                f"transfer_input_files = {','.join(paths + ['davs://eos.example/v1/x.tar.bz2'])}\n"
                f"initialdir = {submission}\n"
                "queue 1\n"
            )
        uploads = []
        upload = law.LocalFileTarget.copy_from_local

        def record(target, *args, **kwargs):
            uploads.append(os.path.relpath(target.abspath, self.worker_files.abspath))
            return upload(target, *args, **kwargs)

        with mock.patch.object(
            law.LocalFileTarget, "copy_from_local", autospec=True, side_effect=record
        ), contextlib.redirect_stdout(io.StringIO()):
            lc.CERNHTCondorJobFileFactory._stage_and_update_jdl(
                job_file, self.worker_files
            )
        with open(job_file) as f:
            lines = f.read().splitlines()
        (line,) = [l for l in lines if l.startswith("transfer_input_files")]
        return line.split(" = ", 1)[1].split(","), uploads, lines

    @staticmethod
    def content_of(uri):
        assert uri.startswith("file://"), uri
        with open(uri[len("file://") :]) as f:
            return f.read()

    def key(self, content):
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def test_another_content_under_the_same_name_gets_its_own_copy(self):
        """A second task, or an edit, renders another bootstrap.sh under the same name."""
        a, b = "bundle_list=core_a\n", "bundle_list=core_b\n"
        first, _, _ = self.submit({self.BOOTSTRAP: a})
        second, uploads, _ = self.submit({self.BOOTSTRAP: b})
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(uploads, [f"{self.key(b)}/{self.BOOTSTRAP}"])
        self.assertEqual(self.content_of(first[0]), a)
        self.assertEqual(self.content_of(second[0]), b)

    def test_the_same_content_is_uploaded_once(self):
        inputs = {self.BOOTSTRAP: "bundle_list=core_a\n", self.STAGEOUT: "logs/Task\n"}
        first, first_uploads, _ = self.submit(inputs)
        second, second_uploads, _ = self.submit(inputs)
        self.assertEqual(first, second)
        self.assertEqual(len(first_uploads), 2)
        self.assertEqual(second_uploads, [])

    def test_the_copy_keeps_its_name(self):
        """The job finds its inputs by the name of the transferred file."""
        bootstrap, stageout = "bundle_list=core_a\n", "logs/Task\n"
        urls, _, lines = self.submit(
            {self.BOOTSTRAP: bootstrap, self.STAGEOUT: stageout}
        )
        base = self.worker_files.uri()
        self.assertEqual(
            urls,
            [
                f"{base}/{self.key(bootstrap)}/{self.BOOTSTRAP}",
                f"{base}/{self.key(stageout)}/{self.STAGEOUT}",
                "davs://eos.example/v1/x.tar.bz2",
            ],
        )
        self.assertFalse([l for l in lines if l.startswith("initialdir")])


def law_failure_report(jobs):
    """What law prints for failed jobs (BaseRemoteWorkflowProxy._print_status_errors), with
    the fields of the HTCondor job manager of law 0.1.21, the remote host included."""
    proxy = object.__new__(lc._BundleAwareHTCondorWorkflowProxy)
    proxy.task = types.SimpleNamespace(task_id="HelloWorldTask_v1")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        proxy._print_status_errors(jobs)
    return out.getvalue()


class TheReportedLogIsTheStagedOne(unittest.TestCase):
    LOG_DIR = (
        "davs://eoshome-k.cern.ch:8444/eos/user/k/kandroso/FLAF_all/H_mumu/v1-d/logs/"
        f"HelloWorldTask/{ERA}"
    )

    def job(self, job_id, branch):
        return {
            "job_id": job_id,
            "branches": [branch],
            "status": "failed",
            "code": 60,
            "error": "job status set to 'failed' due to non-zero exit code 60",
            "extra": {
                "log": f"{self.LOG_DIR}/stdall_{branch}To{branch + 1}_{job_id}.txt",
                "mem_peak_mb": 42.0,
                "remote_host": "undefined",
            },
        }

    def test_the_path_ends_at_the_field_separator(self):
        jobs = {1: self.job("13559339.0", 0), 2: self.job("13559340.0", 1)}
        report = law_failure_report(jobs)
        self.assertIn("remote host: undefined", report)
        paths = test_hello_world.parse_reported_log_paths(report)
        self.assertEqual(paths, [job["extra"]["log"] for job in jobs.values()])
        # the check the smoke test makes of its one job's log
        self.assertEqual(os.path.dirname(paths[0]), self.LOG_DIR)
        self.assertRegex(os.path.basename(paths[0]), test_hello_world.LOG_NAME_RE)

    def test_a_path_that_ends_the_line(self):
        """law 0.1.20 printed no remote host, and the log was the last field."""
        log = f"{self.LOG_DIR}/stdall_0To1_13559339.0.txt"
        report = f"    job: 1, branches: [0], id: 13559339.0, log: {log}\n"
        self.assertEqual(test_hello_world.parse_reported_log_paths(report), [log])


if __name__ == "__main__":
    unittest.main()
