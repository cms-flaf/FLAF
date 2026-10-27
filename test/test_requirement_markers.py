#!/usr/bin/env python3
"""The requirement markers of test/conftest.py, as the Unit Tests workflow uses them.

A test that needs what the runner lacks is deselected only when it declares so; a module that
fails to import without a marker stops the run, and every deselected or idle test file is listed.
Each case runs pytest on a scratch test/ the way an analysis does, with `-p FLAF.test.conftest`.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)

CI_SELECTION = "not needs_root and not needs_cvmfs and not needs_proxy"

PASSING = """
def test_passes():
    pass
"""
MARKED_UNIMPORTABLE = """
import pytest

pytestmark = pytest.mark.needs_root

import module_the_runner_lacks
"""
# what an empty stand-in for ROOT, left behind by another module, gives on import
MARKED_BROKEN_ON_IMPORT = """
import pytest

pytestmark = [pytest.mark.needs_root]

raise AttributeError("module 'ROOT' has no attribute 'gROOT'")
"""
MARKED_TEST = """
import pytest


@pytest.mark.needs_cvmfs
def test_needs_cvmfs():
    assert False


def test_passes():
    pass
"""
UNMARKED_UNIMPORTABLE = """
import module_the_runner_lacks
"""
LEAVES_A_STAND_IN = """
import sys
from unittest import mock

sys.modules["ROOT"] = mock.MagicMock()


def test_passes():
    pass
"""
WITHDRAWS_ITS_STAND_IN = """
import sys
import types

previous = sys.modules.get("ROOT")
sys.modules["ROOT"] = types.ModuleType("ROOT")
if previous is None:
    del sys.modules["ROOT"]
else:
    sys.modules["ROOT"] = previous


def test_passes():
    pass
"""


class TheRequirementMarkers(unittest.TestCase):
    def run_pytest(self, files, *args):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "test"))
            for name, text in files.items():
                with open(os.path.join(tmp, "test", name), "w") as f:
                    f.write(textwrap.dedent(text))
            env = dict(
                os.environ, PYTHONPATH=flaf_parent, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "--strict-markers",
                ]
                + ["-p", "FLAF.test.conftest", *args, "test"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
            )
        return result.returncode, result.stdout + result.stderr

    def test_marked_tests_are_deselected_and_listed(self):
        rc, out = self.run_pytest(
            {
                "test_ok.py": PASSING,
                "test_unimportable.py": MARKED_UNIMPORTABLE,
                "test_broken.py": MARKED_BROKEN_ON_IMPORT,
                "test_one_marked.py": MARKED_TEST,
                "test_empty.py": "",
            },
            "-m",
            CI_SELECTION,
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("2 passed, 3 deselected", out)
        self.assertIn(
            "test/test_unimportable.py: not importable here (ModuleNotFoundError: No module "
            "named 'module_the_runner_lacks'); needs_root",
            out,
        )
        self.assertIn(
            "test/test_broken.py: not importable here (AttributeError: module 'ROOT' has no "
            "attribute 'gROOT'); needs_root",
            out,
        )
        self.assertIn("test/test_one_marked.py: 1 test(s); needs_cvmfs", out)
        self.assertIn("test files that ran no test", out)
        self.assertIn("test/test_empty.py", out)

    def test_an_unmarked_module_that_fails_to_import_stops_the_run(self):
        rc, out = self.run_pytest(
            {"test_ok.py": PASSING, "test_unmarked.py": UNMARKED_UNIMPORTABLE},
            "-m",
            CI_SELECTION,
        )
        self.assertEqual(rc, 2, out)
        self.assertIn("ERROR test/test_unmarked.py", out)

    def test_a_selected_unimportable_module_fails_with_its_import_error(self):
        rc, out = self.run_pytest(
            {"test_unimportable.py": MARKED_UNIMPORTABLE}, "-m", "needs_root"
        )
        self.assertEqual(rc, 1, out)
        self.assertIn("No module named 'module_the_runner_lacks'", out)

    def test_only_marked_tests_select_nothing(self):
        # the workflow turns this exit code into a notice
        rc, out = self.run_pytest(
            {"test_unimportable.py": MARKED_UNIMPORTABLE}, "-m", CI_SELECTION
        )
        self.assertEqual(rc, 5, out)

    def test_a_module_that_leaves_a_root_stand_in_stops_the_run(self):
        rc, out = self.run_pytest(
            {"test_leak.py": LEAVES_A_STAND_IN, "test_tidy.py": WITHDRAWS_ITS_STAND_IN},
            "-m",
            CI_SELECTION,
        )
        self.assertEqual(rc, 2, out)
        self.assertIn(
            "test/test_leak.py leaves a stand-in for ROOT in sys.modules", out
        )
        self.assertNotIn("test/test_tidy.py leaves", out)

    def test_a_misspelt_marker_is_refused(self):
        rc, out = self.run_pytest(
            {"test_typo.py": "import pytest\n\npytestmark = pytest.mark.needs_rot\n"},
            "-m",
            CI_SELECTION,
        )
        self.assertNotEqual(rc, 0, out)
        self.assertIn("'needs_rot' not found in `markers` configuration option", out)


if __name__ == "__main__":
    unittest.main()
