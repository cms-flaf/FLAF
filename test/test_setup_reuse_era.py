#!/usr/bin/env python3
"""An era that reuses the MC of another era must load when the eras' global.yaml use anchors.

The configuration of an era is the text of FLAF/config, FLAF/config/<era>, config and
config/<era> parsed as one YAML document, so an era file can override part of a top-level
block with `<<: *anchor`. Setup also reads the global.yaml of the era named in
`reuse_mc_from_era` (Run3_2025 and Run3_2026 reuse Run3_2024) to get its `shared_mc`; that read
has to see the top-level layers too, or the alias is undefined there.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

# FLAF.Common.Utilities (imported through Setup) imports ROOT at module level, which nothing here
# touches. Where ROOT is missing, a stand-in serves the import only and is withdrawn again: test
# modules share one process, and a later one must still get the real ROOT or fail to import it.
_root_stand_in = None
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    _root_stand_in = sys.modules["ROOT"] = mock.MagicMock()

from FLAF.Common.Setup import Setup  # noqa: E402

if _root_stand_in is not None and sys.modules.get("ROOT") is _root_stand_in:
    del sys.modules["ROOT"]

FILES = {
    "config/global.yaml": """
phys_model: M
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
corrections: &corrections_default
  lumi: { stage: AnaTuple }
  btag: { stage: AnaTuple, tagger: particleNet }
""",
    "config/Run3_2024/global.yaml": """
corrections:
  <<: *corrections_default
  btag: { stage: AnaTuple, tagger: UParTAK4 }
""",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT ]\n",
    "config/processes.yaml": "TT:\n  datasets: [ TTto2L2Nu ]\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
}


class TestSetupReuseEra(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ana = self.tmp.name
        os.symlink(flaf_repo, os.path.join(self.ana, "FLAF"))
        files = dict(FILES)
        files["config/Run3_2025/global.yaml"] = files["config/Run3_2024/global.yaml"]
        for path, text in files.items():
            full = os.path.join(self.ana, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reusing_era_loads_with_anchors(self):
        reference = Setup(self.ana, "Run3_2024", "test")
        setup = Setup(self.ana, "Run3_2025", "test")
        self.assertEqual(setup.global_params["reuse_mc_from_era"], "Run3_2024")
        self.assertEqual(
            setup.global_params["corrections"],
            {
                "lumi": {"stage": "AnaTuple"},
                "btag": {"stage": "AnaTuple", "tagger": "UParTAK4"},
            },
        )
        # shared_mc comes from the reused era, as without anchors
        self.assertEqual(
            setup.global_params["shared_mc"], reference.global_params["shared_mc"]
        )


if __name__ == "__main__":
    unittest.main()
