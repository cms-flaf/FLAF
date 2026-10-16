#!/usr/bin/env python3
"""The era parameter of every FLAF task, and the removed `period`.

`--era` takes known era names; the value is a sorted tuple of unique names, the same for a
string and a tuple, and serialised back to the same command line. Every form of the old
`period` (`--period`, `--<Task>-period`, `[<Task>] period` in the configuration, a `period=`
keyword, `{period}` in a bundle pattern) stops with the rename message, and so does a per-task
`--<Task>-era`: the eras are set once per invocation.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import luigi  # noqa: E402
import luigi.cmdline_parser  # noqa: E402

# law_customizations imports ROOT and Corrections at module level, which the unit-test runner
# does not have and nothing tested here uses: stand-ins for these imports alone.
_HEAVY_MODULES = [
    "ROOT",
    "numpy",
    "awkward",
    "uproot",
    "Corrections",
    "Corrections.Corrections",
    "Corrections.CorrectionsCore",
    "Corrections.lumi",
]


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_replaced = {
    name: sys.modules.get(name) for name in _HEAVY_MODULES if not _importable(name)
}
sys.modules.update({name: mock.MagicMock() for name in _replaced})
try:
    from FLAF.Common.Setup import UnknownEraError
    from FLAF.run_tools import law_customizations as lc
finally:
    for _name, _module in _replaced.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

FILES = {
    "config/global.yaml": """
phys_model: M
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
""",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT ]\n",
    "config/processes.yaml": "TT:\n  datasets: [ TTto2L2Nu ]\n",
    "config/Run3_2022/datasets.yaml": "TTto2L2Nu: {}\n",
    "config/Run3_2023/datasets.yaml": "TTto2L2Nu: {}\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
}


class EraParamProbe(lc.Task):
    def run(self):
        pass


class OtherEraParamProbe(lc.Task):
    def run(self):
        pass


class _ScratchAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_era_parameter_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ana = os.path.join(self.tmp, "ana")
        os.makedirs(ana)
        os.symlink(flaf_repo, os.path.join(ana, "FLAF"))
        for path, text in FILES.items():
            full = os.path.join(ana, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(text)
        env = mock.patch.dict(
            os.environ,
            {"ANALYSIS_PATH": ana, "ANALYSIS_DATA_PATH": os.path.join(self.tmp, "data")},
        )
        env.start()
        self.addCleanup(env.stop)
        # the command line is checked once per process
        checked = mock.patch.object(lc, "_era_cli_checked", False)
        checked.start()
        self.addCleanup(checked.stop)

    def from_command_line(self, *args, config=None):
        """The root task the command line builds, as `law run` does."""
        if config:
            parser = luigi.configuration.get_config()
            for section, option, value in config:
                if not parser.has_section(section):
                    parser.add_section(section)
                parser.set(section, option, value)
                self.addCleanup(parser.remove_option, section, option)
        with luigi.cmdline_parser.CmdlineParser.global_instance(list(args)) as cp:
            return cp.get_task_obj()


class EraIdentity(_ScratchAnalysis):
    def test_a_string_and_a_tuple_are_the_same_era(self):
        a = EraParamProbe(version="v", era="Run3_2022")
        b = EraParamProbe(version="v", era=("Run3_2022",))
        self.assertEqual(a.era, ("Run3_2022",))
        self.assertEqual(a.task_id, b.task_id)
        self.assertIs(a, b)

    def test_a_repeated_era_is_one_era(self):
        self.assertEqual(
            EraParamProbe(version="v", era="Run3_2022,Run3_2022").era, ("Run3_2022",)
        )

    def test_an_unknown_era_is_refused(self):
        for era in ("Run3_2099", "Processors"):
            with self.subTest(era=era):
                with self.assertRaises(UnknownEraError) as caught:
                    EraParamProbe(version="v", era=era)
                self.assertIn("Run3_2022", str(caught.exception))

    def test_no_era_is_refused(self):
        with self.assertRaises(ValueError):
            EraParamProbe(version="v", era="")

    def test_several_eras_are_not_taken_yet(self):
        with self.assertRaises(ValueError) as caught:
            EraParamProbe(version="v", era="Run3_2023,Run3_2022")
        self.assertIn("Run3_2022,Run3_2023", str(caught.exception))
        self.assertIn("not supported yet", str(caught.exception))

    def test_the_command_line_round_trips(self):
        task = EraParamProbe(version="v", era="Run3_2022")
        args = task.cli_args()
        self.assertEqual(args["--era"], "Run3_2022")
        self.assertNotIn("--period", args)
        again = self.from_command_line(
            "EraParamProbe", "--version", "v", "--era", args["--era"]
        )
        self.assertEqual(again.task_id, task.task_id)

    def test_a_requirement_gets_the_era(self):
        task = EraParamProbe(version="v", era="Run3_2023")
        other = OtherEraParamProbe.req(task)
        self.assertEqual(other.era, ("Run3_2023",))
        self.assertIsNone(other.period)
        self.assertEqual(other.setup.global_params["era"], "Run3_2023")

    def test_the_paths_carry_the_era(self):
        task = EraParamProbe(version="v", era="Run3_2023")
        self.assertEqual(task.store_parts(), ("v", "EraParamProbe", "Run3_2023"))


class PeriodIsRenamed(_ScratchAnalysis):
    def assert_renamed(self, caught):
        self.assertIn("renamed to 'era'", str(caught.exception))

    def test_the_root_period_flag(self):
        with self.assertRaises(ValueError) as caught:
            self.from_command_line("EraParamProbe", "--version", "v", "--period", "Run3_2022")
        self.assert_renamed(caught)

    def test_a_per_task_period_flag(self):
        with self.assertRaises(ValueError) as caught:
            self.from_command_line(
                "EraParamProbe",
                "--version",
                "v",
                "--era",
                "Run3_2022",
                "--OtherEraParamProbe-period",
                "Run3_2023",
            )
        self.assert_renamed(caught)

    def test_a_period_in_the_configuration(self):
        with self.assertRaises(ValueError) as caught:
            self.from_command_line(
                "EraParamProbe",
                "--version",
                "v",
                "--era",
                "Run3_2022",
                config=[("OtherEraParamProbe", "period", "Run3_2023")],
            )
        self.assert_renamed(caught)

    def test_a_period_keyword(self):
        with self.assertRaises(ValueError) as caught:
            EraParamProbe(version="v", era="Run3_2022", period="Run3_2022")
        self.assert_renamed(caught)

    def test_a_per_task_era_flag(self):
        with self.assertRaises(ValueError) as caught:
            self.from_command_line(
                "EraParamProbe",
                "--version",
                "v",
                "--era",
                "Run3_2022",
                "--OtherEraParamProbe-era",
                "Run3_2023",
            )
        self.assertIn("--OtherEraParamProbe-era", str(caught.exception))
        self.assertIn("once per invocation", str(caught.exception))

    def test_a_plain_command_line_runs(self):
        task = self.from_command_line(
            "EraParamProbe", "--version", "v", "--era", "Run3_2022"
        )
        self.assertEqual(task.era, ("Run3_2022",))


if __name__ == "__main__":
    unittest.main()
