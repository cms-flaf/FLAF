#!/usr/bin/env python3
"""run_tools/check_era_rename.py finds every leftover form of the period -> era rename and
nothing else."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(flaf_repo, "run_tools", "check_era_rename.py")

CLEAN = {
    "task.py": """
        class MyTask(Task):
            def store_parts(self):
                return (self.version, "MyTask", self.only_era)

            def message(self):
                return f"{self.only_era}: done"

            @property
            def only_era(self):
                return self.era[0]

            def with_eras(self, eras):
                return type(self).req(self, era=eras)

            def upstream(self):
                return Other.req(self, branches=(), _skip_task_excludes=True)

            def first(self):
                return self.era[0]  # noqa: era-scope

        def columns():
            return ["period", "isData"]

        class Builder(DataFrameBuilderBase):
            def __init__(self, era):
                self.era = era

            def sf(self):
                return f'getSF("{self.era}")'

        timeout = dict(cache_validity_period=10, local_path_cache_validity_period=5)
    """,
    "config.yaml": "patterns:\n  - data/{version}/InputFileTask/{era}\ncolumns: [ period ]\n",
}
DIRTY = {
    "task.py": """
        class MyTask(Task):
            def store_parts(self):
                return (self.version, "MyTask", self.period)

            def name(self):
                return f"{self.era}_x" + str(self.era) + "_".join(self.era)

        def run(period):
            return getattr(task, "period", None), cfg.get("period")

        class Other(HTCondorWorkflow):
            def requires(self):
                return Up.req(
                    self,
                    era=("Run3_2022",),
                ), Up.req_params(self, **{"era": self.era})

            def label(self):
                return self.era[0]
    """,
    "config.yaml": "patterns:\n  - data/{version}/InputFileTask/{period}\n",
    "run.sh": "law run Task --period Run3_2022\n",
}


def make_repo(root, files):
    subprocess.run(["git", "init", "-q", root], check=True)
    for path, text in files.items():
        with open(os.path.join(root, path), "w") as f:
            f.write(textwrap.dedent(text))
    subprocess.run(["git", "-C", root, "add", "-A"], check=True)


def check(root, *args):
    return subprocess.run(
        [sys.executable, CHECK, "--repo", root, *args], capture_output=True, text=True
    )


class CheckEraRename(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_clean_code_passes(self):
        make_repo(self.tmp.name, CLEAN)
        result = check(self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def test_every_leftover_is_reported(self):
        make_repo(self.tmp.name, DIRTY)
        result = check(self.tmp.name)
        self.assertEqual(result.returncode, 1)
        expected = [
            "task.py:4: identifier 'period'",
            "task.py:7: the era tuple formatted as a string",
            "task.py:7: str(self.era)",
            "task.py:7: join(self.era)",
            "task.py:9: identifier 'period'",
            "task.py:10: getattr(..., 'period')",
            "task.py:10: get(..., 'period')",
            "task.py:14: req(..., era=...) outside with_eras",
            "task.py:17: req_params(..., era=...) outside with_eras",
            "task.py:20: self.era[...] outside only_era",
            "config.yaml:2: '{period}'",
            "run.sh:1: '--period'",
        ]
        for line in expected:
            self.assertIn(line, result.stdout)

    def test_the_era_scope_rules_report_exactly_the_leaks(self):
        make_repo(self.tmp.name, DIRTY)
        found = [
            line
            for line in check(self.tmp.name).stdout.splitlines()
            if "with_eras" in line or "only_era" in line
        ]
        self.assertEqual(len(found), 3, found)

    def test_a_marker_silences_only_its_own_rule(self):
        files = dict(DIRTY)
        files["task.py"] = files["task.py"].replace(
            "return self.era[0]", "return self.era[0]  # noqa: era-rename"
        )
        make_repo(self.tmp.name, files)
        self.assertIn("self.era[...] outside only_era", check(self.tmp.name).stdout)

    def test_a_marked_line_and_an_excluded_file_are_skipped(self):
        files = dict(DIRTY)
        files["run.sh"] = "law run Task --period Run3_2022  # noqa: era-rename\n"
        make_repo(self.tmp.name, files)
        result = check(self.tmp.name, "--exclude", "task.py", "--exclude", "*.yaml")
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
