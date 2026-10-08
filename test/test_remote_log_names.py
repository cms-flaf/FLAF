"""Staged job logs: one per attempt, at the name the submit side reports.

law reuses a job's postfix (``_<first>To<last>``) when it resubmits it, so the HTCondor job
id is part of the staged name.  The name is built twice -- by stageout_logs.sh on the worker
and by the HTCondor proxy for law's messages -- and the two must agree.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

STAGEOUT = os.path.join(flaf_repo, "run_tools", "stageout_logs.sh")
BASE_URL = "davs://eos.example//logs/AnaTupleFileTask/Run3_2022"
CRAB_TAG = "1a2b3c4d"


def stage_out(env_vars):
    """Run stageout_logs.sh with a recording gfal-copy; returns the destination URL."""
    tmp = tempfile.mkdtemp()
    try:
        bin_dir = os.path.join(tmp, "bin")
        os.mkdir(bin_dir)
        record = os.path.join(tmp, "copied")
        for tool, body in (
            ("gfal-copy", f'echo "$@" > {record}'),
            ("gfal-mkdir", "true"),
        ):
            path = os.path.join(bin_dir, tool)
            with open(path, "w") as f:
                f.write(f"#!/bin/sh\n{body}\n")
            os.chmod(path, 0o755)
        with open(STAGEOUT) as f:
            script = (
                f.read()
                .replace("{{log_remote_base_url}}", BASE_URL)
                .replace("{{crab_log_tag}}", CRAB_TAG)
            )
        script_path = os.path.join(tmp, "stageout_logs.sh")
        with open(script_path, "w") as f:
            f.write(script)
        env = {
            "PATH": bin_dir + os.pathsep + os.environ["PATH"],
            "LAW_JOB_INIT_DIR": tmp,
            "X509_USER_PROXY": "/dev/null",
        }
        env.update(env_vars)
        local = env_vars.pop("_local")
        with open(os.path.join(tmp, local), "w") as f:
            f.write("log\n")
        env.pop("_local", None)
        subprocess.run(["bash", script_path], env=env, check=True, capture_output=True)
        with open(record) as f:
            return f.read().split()[-1]
    finally:
        shutil.rmtree(tmp)


class TestStageoutScript(unittest.TestCase):
    def test_htcondor_attempt_gets_its_own_name(self):
        url = stage_out(
            {
                "_local": "stdall_0To4.txt",
                "LAW_HTCONDOR_JOB_POSTFIX": "_0To4",
                "LAW_HTCONDOR_JOB_CLUSTER": "123",
                "LAW_HTCONDOR_JOB_PROCESS": "7",
            }
        )
        self.assertEqual(url, BASE_URL + "/stdall_0To4_123.7.txt")

    def test_without_postfix_the_cluster_name_is_kept(self):
        url = stage_out(
            {
                "_local": "stdall_123_7.txt",
                "LAW_HTCONDOR_JOB_CLUSTER": "123",
                "LAW_HTCONDOR_JOB_PROCESS": "7",
            }
        )
        self.assertEqual(url, BASE_URL + "/stdall_123_7.txt")

    def test_crab_name_is_unique_across_crab_tasks(self):
        # CRAB numbers the jobs of every CRAB task from 1, and the waves and retries of a
        # production are separate CRAB tasks staging into one directory.
        url = stage_out(
            {
                "_local": "stdall.txt",
                "LAW_CRAB_JOB_NUMBER": "42",
                "LAW_JOB_TASK_BRANCHES_CSV": "17,18",
            }
        )
        self.assertEqual(url, BASE_URL + f"/stdall_17_crab{CRAB_TAG}.42.txt")


try:
    from FLAF.run_tools import law_customizations as lc

    HAVE_LAW = True
except Exception:  # law/luigi not importable outside the analysis environment
    HAVE_LAW = False


class _Config:
    postfix_output_files = True
    postfix = ["_0To4", "_4To9"]


@unittest.skipUnless(HAVE_LAW, "law is not importable in this environment")
class TestReportedLogLocation(unittest.TestCase):
    def test_reported_location_matches_the_staged_name(self):
        class FakeFS:
            pass

        class FakeTarget:
            def uri(self):
                return BASE_URL

        task = mock.Mock(fs_default=FakeFS())
        task.remote_log_dir_target.return_value = FakeTarget()
        proxy = lc._BundleAwareHTCondorWorkflowProxy.__new__(
            lc._BundleAwareHTCondorWorkflowProxy
        )
        proxy.task = task
        config = _Config()
        submission = {
            5: {"log": "/afs/data/stdall_0To4.txt", "config": config},
            6: {"log": "/afs/data/stdall_4To9.txt", "config": config},
        }
        job_ids = ["123.0", RuntimeError("submission failed")]
        with mock.patch.object(lc, "WLCGFileSystem", FakeFS), mock.patch.object(
            lc.BundleAwareHTCondorWorkflowProxyBase,
            "_submit_group",
            return_value=(job_ids, submission),
        ):
            _, data = proxy._submit_group({})
        self.assertEqual(data[5]["log"], BASE_URL + "/stdall_0To4_123.0.txt")
        # A job that was not submitted has no id; it keeps the plain name.
        self.assertEqual(data[6]["log"], BASE_URL + "/stdall_4To9.txt")
        self.assertEqual(
            stage_out(
                {
                    "_local": "stdall_0To4.txt",
                    "LAW_HTCONDOR_JOB_POSTFIX": "_0To4",
                    "LAW_HTCONDOR_JOB_CLUSTER": "123",
                    "LAW_HTCONDOR_JOB_PROCESS": "0",
                }
            ),
            data[5]["log"],
        )


if __name__ == "__main__":
    unittest.main()
