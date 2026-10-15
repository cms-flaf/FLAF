#!/usr/bin/env python3
"""CRAB's Site.storageSite / Data.outLFNDirBase are a placeholder any user can submit with.

CRAB transfers nothing for FLAF, but the two values must pass the CRAB server's validation and
the task worker's Rucio resolution of a write PFN at the site. They used to be derived from
fs_default, and an fs_default on a shared account (`T3_US_FNALLPC:/store/user/lpcflaf/...`, the
HH_bbWW production area) gave an outLFNDirBase that the server refuses with HTTP 400 to everyone
but lpcflaf, while an xrootd fs_default could not be submitted at all. The LFN rule is checked
with CRAB's own `checkOutLFN`, read from the client the `crab` command runs, and the site with
its CMS storage description; where /cvmfs is not mounted those checks are skipped, never
replaced by a copy.
"""

import ast
import importlib.util
import itertools
import json
import os
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

# law_customizations imports FLAF.Common.Setup, whose Utilities imports ROOT at module level.
# The unit-test environment has no ROOT and nothing exercised here touches it, so an empty
# stand-in is provided for the import only, and withdrawn again so that no other test sees it.
_root_stand_in = None
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    _root_stand_in = sys.modules["ROOT"] = types.ModuleType("ROOT")

import law  # noqa: E402

from FLAF.run_tools import law_customizations as lc  # noqa: E402

if _root_stand_in is not None and sys.modules.get("ROOT") is _root_stand_in:
    del sys.modules["ROOT"]

#: the current production CRAB client, resolved the way /cvmfs/cms.cern.ch/common/crab does
CRAB_CVMFS_BASE = "/cvmfs/cms.cern.ch/share/cms"
CRAB_CLIENT_LATEST = os.path.join(
    CRAB_CVMFS_BASE, "crab", "1.0", "etc", "crab-prod.latest"
)

#: the CMS storage descriptions Rucio's lfn2pfn reads
SITECONF = "/cvmfs/cms.cern.ch/SITECONF"

#: what CRAB's DagmanCreator appends to outLFNDirBase before resolving the write PFN:
#: <primary dataset>/<request name>/<timestamp>/<counter> (dmwm/CRABServer 30286f7,
#: src/python/TaskWorker/Actions/DagmanCreator.py:94-127 and :682)
DIRECT_DEST_SUFFIX = "/CRAB_UserFiles/crab_SomeTask/261015_120000/0000"

#: fs_default values in use: the shared FNAL account (HH_bbWW and HH_bbtautau production),
#: its recommended xrootd form, personal CERNBox in both accepted spellings
FS_DEFAULTS = (
    "T3_US_FNALLPC:/store/user/lpcflaf/HH_bbWW/",
    "root://cmseos.fnal.gov//eos/uscms/store/user/lpcflaf/HH_bbWW/",
    "davs://eoshome-k.cern.ch:8444/eos/user/k/kandroso/HH_bbWW/",
    "T3_CH_CERNBOX:/store/user/kandroso/HH_bbWW/",
)

#: the outLFNDirBase the fs_default mapping produced for the first of FS_DEFAULTS, which the
#: CRAB server refused for the submitter kandroso
REFUSED_LFN = "/store/user/lpcflaf/HH_bbWW"


class StageoutTask(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    """A FLAF task with the base classes, in the order, of AnaTupleMergeTask / HistMergerTask."""

    bundle_flavours = ["core"]

    def create_branch_map(self):
        return {0: 0}

    def run(self):
        pass


_versions = itertools.count()


def stageout_location(fs_default):
    """`crab_stageout_location()` of a real FLAF CRAB task whose Setup has `fs_default`."""
    setup = types.SimpleNamespace(
        global_params={"fs_default": fs_default, "crab": {}},
        get_fs=lambda name: None,
    )
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
        lc.Setup, "getGlobal", return_value=setup
    ), mock.patch.dict(os.environ, {"ANALYSIS_DATA_PATH": tmp, "ANALYSIS_PATH": tmp}):
        # a fresh version per call: luigi caches task instances by parameter values
        task = StageoutTask(
            version=f"vstageout{next(_versions)}", period="Run3_2022", workflow="local"
        )
        return task.crab_stageout_location()


class ThePlaceholder(unittest.TestCase):
    def test_it_does_not_depend_on_fs_default(self):
        for fs_default in FS_DEFAULTS:
            with self.subTest(fs_default=fs_default):
                self.assertEqual(
                    stageout_location(fs_default), lc.CRAB_PLACEHOLDER_STAGEOUT
                )

    def test_it_is_a_site_and_an_lfn(self):
        site, lfn = lc.CRAB_PLACEHOLDER_STAGEOUT
        self.assertRegex(site, r"^T[0-3]_[A-Z]{2}_\w+$")
        self.assertTrue(lfn.startswith("/store/"), lfn)


class TheCrabServerRule(unittest.TestCase):
    """The placeholder against CRAB's `checkOutLFN` (skipped where /cvmfs is not mounted)."""

    @classmethod
    def setUpClass(cls):
        try:
            with open(CRAB_CLIENT_LATEST) as f:
                version = f.read().strip()
        except OSError:
            raise unittest.SkipTest(
                f"no CRAB client ({CRAB_CLIENT_LATEST}): the outLFNDirBase placeholder is "
                "not checked against CRAB's checkOutLFN"
            )
        path = os.path.join(
            CRAB_CVMFS_BASE, "crab-prod", version, "lib", "ServerUtilities.py"
        )
        with open(path) as f:
            tree = ast.parse(f.read())
        # only the function: the module's own imports need the CRAB runtime
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "checkOutLFN"
        ]
        if len(nodes) != 1:
            raise AssertionError(f"{path}: no single checkOutLFN definition")
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
        cls.check_out_lfn = staticmethod(namespace["checkOutLFN"])

    def test_the_refused_lfn_is_refused(self):
        # the rule loaded is the one behind the HTTP 400, so the acceptance below means something
        self.assertFalse(self.check_out_lfn(REFUSED_LFN, "kandroso"))
        self.assertTrue(self.check_out_lfn(REFUSED_LFN, "lpcflaf"))

    def test_every_user_may_submit_with_the_placeholder(self):
        for fs_default in FS_DEFAULTS:
            _, lfn = stageout_location(fs_default)
            for user in ("kandroso", "lpcflaf", "someone"):
                with self.subTest(fs_default=fs_default, user=user):
                    self.assertTrue(self.check_out_lfn(lfn, user), lfn)


class TheSiteConfiguration(unittest.TestCase):
    """The placeholder against the site's storage.json (skipped where /cvmfs is not mounted).

    A bare prefix maps every LFN, a rule the LFNs its pattern matches, which is how the CMS
    lfn2pfn algorithm Rucio runs for the task worker starts (dmwm/CMSRucio 5133fbe,
    src/policy/CMSRucioPolicy/algorithms/lfn2pfn.py:44-47 without rules and :53-74 with them,
    raising when no rule matches); the real Rucio call is not made here, it needs a grid proxy.
    The task worker makes it for every task, transfers or not (DagmanCreator.py:682).
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(SITECONF):
            raise unittest.SkipTest(
                f"no {SITECONF}: the placeholder site is not checked against its storage rules"
            )

    @staticmethod
    def maps_for_write(site, lfn):
        with open(os.path.join(SITECONF, site, "storage.json")) as f:
            volumes = json.load(f)
        for volume in volumes:
            for protocol in volume.get("protocols", []):
                if protocol.get("access") != "global-rw":
                    continue
                if protocol.get("prefix"):
                    return True
                if any(re.match(r["lfn"], lfn) for r in protocol.get("rules", [])):
                    return True
        return False

    def test_a_site_that_maps_user_lfns_only_is_told_apart(self):
        # T3_CH_CERNBOX maps /store/user/<x> only: with a /store/group placeholder there
        # every task ended SUBMITFAILED in the task worker
        dest = "/store/group/FLAF_crab_unused" + DIRECT_DEST_SUFFIX
        self.assertFalse(self.maps_for_write("T3_CH_CERNBOX", dest))
        dest = "/store/user/someone/FLAF_crab_unused" + DIRECT_DEST_SUFFIX
        self.assertTrue(self.maps_for_write("T3_CH_CERNBOX", dest))

    def test_the_placeholder_site_maps_the_placeholder_lfn(self):
        site, lfn = lc.CRAB_PLACEHOLDER_STAGEOUT
        self.assertTrue(
            self.maps_for_write(site, lfn + DIRECT_DEST_SUFFIX), (site, lfn)
        )


if __name__ == "__main__":
    unittest.main()
