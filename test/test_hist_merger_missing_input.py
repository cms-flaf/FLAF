"""A missing HistMergerFromHists input aborts the merge and writes no merged output.

The merger runs as a script, as HistMergerTask runs it, with stand-ins for ROOT and the FLAF
framework: none of them is installed in the unit-test environment. The inputs are empty stand-ins
for the histogram files, so the merge reads no histograms; the check on each input comes before
the one that reads it. The missing file is either the first input, or follows an existing one as
in the case of a partial re-merge.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
merger_path = os.path.join(flaf_repo, "Analysis", "HistMergerFromHists.py")

STAND_INS = {
    "ROOT.py": textwrap.dedent(
        """
        class _Input:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def IsZombie(self):
                return False

            def GetListOfKeys(self):
                return []

        class TFile:
            # creating the output is what a merge that goes on would do
            def __init__(self, path, *args):
                open(path, "w").close()

            @staticmethod
            def Open(path, *args):
                return _Input()

            def Close(self):
                pass
        """
    ),
    "stub_analysis.py": "",
    os.path.join("FLAF", "__init__.py"): "",
    os.path.join("FLAF", "Common", "__init__.py"): "",
    os.path.join("FLAF", "Common", "Utilities.py"): "",
    os.path.join("FLAF", "Common", "HistHelper.py"): textwrap.dedent(
        """
        def GetUncNameTypes(unc_cfg_dict):
            return {}

        def get_all_items_recursive(inFile):
            return {}
        """
    ),
    os.path.join("FLAF", "Common", "Setup.py"): textwrap.dedent(
        """
        class _PhysModel:
            def processes(self, process_type):
                return []

        class Setup:
            def __init__(self, *args, **kwargs):
                self.global_params = {
                    "analysis_import": "stub_analysis",
                    "categories": [],
                    "scales": [],
                    "channelSelection": [],
                }
                self.weights_config = {"norm": {}, "shape": {}}
                self.datasets = {
                    "DYto2Tau_M_50": {"process_name": "DYto2Tau"},
                    "DYto2Tau_M_10": {"process_name": "DYto2Tau"},
                }
                self.base_processes = {"DYto2Tau": {"parent_process": "DYto2Tau"}}
                self.phys_model = _PhysModel()
        """
    ),
}


class TestHistMergerMissingInput(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.stand_ins = os.path.join(self.tmp, "stand_ins")
        for rel_path, content in STAND_INS.items():
            path = os.path.join(self.stand_ins, rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        self.out_file = os.path.join(self.tmp, "merged.root")

    def run_merger(self, inputs):
        """Run the merger on (dataset, path) inputs, as HistMergerTask does."""
        dataset_names = ",".join(dataset for dataset, _ in inputs)
        env = dict(
            os.environ, PYTHONPATH=self.stand_ins, ANALYSIS_PATH=self.stand_ins
        )
        return subprocess.run(
            [
                sys.executable,
                merger_path,
                *[path for _, path in inputs],
                "--dataset_names",
                dataset_names,
                "--var",
                "lep1_pt",
                "--outFile",
                self.out_file,
                "--era",
                "Run3_2022EE",
                "--LAWrunVersion",
                "test",
            ],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_missing_input_aborts_the_merge(self):
        missing = os.path.join(self.tmp, "chunk_0.root")

        result = self.run_merger([("DYto2Tau_M_50", missing)])

        # Skipping the input used to exit 0 and write a merged file with no histograms.
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("does not exist", result.stderr)
        self.assertFalse(
            os.path.exists(self.out_file), "a merged output was written with no input"
        )

    def test_missing_input_after_an_existing_one_aborts_the_merge(self):
        existing = os.path.join(self.tmp, "chunk_0.root")
        missing = os.path.join(self.tmp, "chunk_1.root")
        open(existing, "w").close()

        result = self.run_merger(
            [("DYto2Tau_M_50", existing), ("DYto2Tau_M_10", missing)]
        )

        # The issue's case: the existing chunk was merged and the missing one skipped, exiting 0.
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(missing, result.stderr)
        self.assertFalse(
            os.path.exists(self.out_file), "a partial merged output was written"
        )


if __name__ == "__main__":
    unittest.main()
