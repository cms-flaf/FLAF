#!/usr/bin/env python3
"""The job matrix of the GitHub-Actions integration backend (.github/scripts/ci/generate_matrix.py).

The multi-era job runs its task once over every era of the analysis: its `era` is the
comma-joined list, in the order the variables give, and nothing else carries the eras. The per-era
and per-process jobs keep one era each; an active analysis without eras is refused.
"""

import contextlib
import importlib.util
import io
import json
import os
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "generate_matrix",
    os.path.join(flaf_repo, ".github", "scripts", "ci", "generate_matrix.py"),
)
generate_matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_matrix)

ERAS = ["Run3_2023", "Run3_2022", "Run3_2024"]
VARIABLES = {
    "HH_bbWW_active": "1",
    "HH_bbWW_task": "StatInference.law.tasks.ResonantLimitsAndHistPlotTask",
    "HH_bbWW_args": "--test 1000",
    "HH_bbWW_processes": "custom_CI_Signal custom_CI_Data",
    "HH_bbWW_eras": " ".join(ERAS),
    "H_mumu_active": "1",
    "H_mumu_task": "FLAF.Analysis.tasks.HistPlotTask",
    "H_mumu_processes": "custom_CI_signal",
    "H_mumu_eras": "Run3_2022",
}


def run(variables):
    """The matrices main() prints for `variables` (no GITHUB_OUTPUT file is written)."""
    out = io.StringIO()
    argv = ["generate_matrix.py", "--variables", json.dumps(variables)]
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"}
    with mock.patch.dict(os.environ, env, clear=True), mock.patch(
        "sys.argv", argv
    ), contextlib.redirect_stdout(out):
        generate_matrix.main()
    return json.loads(out.getvalue())


class MultiEraJob(unittest.TestCase):
    def test_the_multi_era_job_gets_every_era(self):
        result = run(VARIABLES)
        (job,) = result["multi_eras"]
        self.assertEqual(job["analysis"], "HH_bbWW")
        self.assertEqual(job["era"], ",".join(ERAS))
        self.assertNotIn("eras", job)

    def test_the_other_jobs_keep_one_era(self):
        result = run(VARIABLES)
        for key in ("processes", "eras"):
            for job in result[key]:
                self.assertNotIn(",", job["era"])
        self.assertEqual(
            [j["era"] for j in result["eras"] if j["analysis"] == "HH_bbWW"], ERAS
        )
        self.assertFalse(any(j["analysis"] == "H_mumu" for j in result["multi_eras"]))

    def test_an_active_analysis_without_eras_is_refused(self):
        with self.assertRaises(ValueError):
            run({**VARIABLES, "HH_bbWW_eras": "  "})


if __name__ == "__main__":
    unittest.main()
