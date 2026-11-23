#!/usr/bin/env python3
"""Where the histogram stages write, per storage key: fs_HistTuple (HistTuples and the
PreHistTupleProduction markers), fs_Hists (split and merged histograms, and the `.merged`
markers beside the split ones) and fs_StatInference, each falling back as documented
(fs_Hists -> fs_HistTuple -> fs_default, fs_StatInference -> fs_default). Proven on the real
FLAF classes over the scratch analysis of test_era_requires, with its merge plans written:
- nothing set: every histogram stage writes to fs_default, as before these keys;
- fs_HistTuple alone: the histograms stay beside the HistTuples, as before fs_Hists;
- fs_Hists: only the histograms move; the HistTuples and the plots stay where they were;
- fs_Hists without fs_HistTuple: the HistTuples stay on fs_default;
- fs_StatInference: its own; unset, fs_default's instance; it moves nothing else;
- remote stores, in the forms the docs offer (a URL, a list of URLs): every target is on the
  filesystem instance of its key.
"""

import os
import unittest
from unittest import mock

# a sibling, by its bare name (`test.<module>` would be CPython's own test package)
import test_era_requires as base  # noqa: E402

import FLAF.Common.Setup as flaf_setup  # noqa: E402
import FLAF.RunKit.law_gfal as law_gfal  # noqa: E402

analysis = base.analysis
ERA = base.ERA_A
REMOTE = "davs://storage.example:1094/store/user/flaf"


class StorageKeys(base._Scratch):
    def setUp(self):
        super().setUp()
        self.store = os.path.join(self.tmp, "store")

    def configure(self, **keys):
        """fs_<key>: <value> for each keyword, in the era's own global.yaml: a URL or a list
        as given, any other value a directory under the scratch directory."""
        paths = {
            key: (
                value
                if isinstance(value, list) or "://" in value
                else os.path.join(self.tmp, value)
            )
            for key, value in keys.items()
        }
        base.write(
            os.path.join(self.ana, "config", ERA, "global.yaml"),
            "".join(f"fs_{key}: {path}\n" for key, path in paths.items()),
        )
        base.write_input_lists(self.version, ERA)
        base.write_merge_plans(self.version, ERA)
        base.clear_process_caches()
        return paths

    def task(self, cls, **kwargs):
        return cls(version=self.version, era=ERA, workflow="local", **kwargs)

    def assert_under(self, targets, directory):
        targets = list(targets)
        self.assertTrue(targets)
        for target in targets:
            self.assertTrue(
                target.abspath.startswith(directory + os.sep),
                f"{target.abspath} is not under {directory}",
            )

    def hist_tuples(self):
        producer = self.task(analysis.HistTupleProducerTask)
        return [producer.req_branch(b).output() for b in producer.get_branch_map()]

    def pre_hist_tuple_markers(self):
        task = self.task(analysis.PreHistTupleProductionTask)
        return [task.req_branch(b).output() for b in task.get_branch_map()]

    def split_hists(self):
        producer = self.task(analysis.HistFromNtupleProducerTask)
        return [
            target
            for b in producer.get_branch_map()
            for target in producer.req_branch(b).output().values()
        ]

    def merged_hists(self):
        merger = self.task(analysis.HistMergerTask)
        merged = [merger.req_branch(b).output() for b in merger.get_branch_map()]
        external = [
            analysis.MergedHistsTask(version=self.version, era=ERA, variable=v).output()
            for v in ("v1", "v2")
        ]
        return merged + external

    def merged_markers(self):
        return [analysis._split_merged_marker(t) for t in self.split_hists()]

    def assert_layout(self, hist_tuples, hists):
        self.assert_under(self.hist_tuples(), hist_tuples)
        self.assert_under(self.pre_hist_tuple_markers(), hist_tuples)
        self.assert_under(self.split_hists(), hists)
        self.assert_under(self.merged_markers(), hists)
        self.assert_under(self.merged_hists(), hists)

    def test_nothing_set_keeps_everything_on_fs_default(self):
        self.configure()
        self.assert_layout(self.store, self.store)
        task = self.task(analysis.HistMergerTask)
        self.assertIs(task.fs_Hists, task.fs_default)
        self.assertIs(task.fs_StatInference, task.fs_default)

    def test_fs_HistTuple_alone_keeps_the_histograms_beside_the_hist_tuples(self):
        paths = self.configure(HistTuple="tuples")
        self.assert_layout(paths["HistTuple"], paths["HistTuple"])
        task = self.task(analysis.HistMergerTask)
        self.assertIs(task.fs_Hists, task.fs_HistTuple)
        self.assertIs(task.fs_StatInference, task.fs_default)

    def test_fs_Hists_moves_only_the_histograms(self):
        paths = self.configure(HistTuple="tuples", Hists="hists")
        self.assert_layout(paths["HistTuple"], paths["Hists"])
        task = self.task(analysis.HistMergerTask)
        self.assertEqual(task.fs_plots.base, self.store)
        self.assertEqual(task.fs_StatInference.base, self.store)

    def test_fs_Hists_without_fs_HistTuple(self):
        paths = self.configure(Hists="hists")
        self.assert_layout(self.store, paths["Hists"])

    def test_fs_StatInference_moves_nothing_else(self):
        paths = self.configure(StatInference="inference")
        self.assert_layout(self.store, self.store)
        task = self.task(analysis.HistMergerTask)
        self.assertEqual(task.fs_StatInference.base, paths["StatInference"])
        self.assertEqual(task.fs_default.base, self.store)

    def test_remote_stores(self):
        hists = [f"{REMOTE}/hists", "root://other.example//store/user/flaf/hists"]
        # no path-cache server and no grid proxy (a CI runner has neither): the filesystems
        # are built and never listed
        with mock.patch.object(
            flaf_setup, "local_path_cache_server", lambda: None
        ), mock.patch.object(law_gfal, "get_voms_proxy_info", lambda: {"path": None}):
            self.configure(HistTuple=f"{REMOTE}/tuples", Hists=hists)
            task = self.task(analysis.HistMergerTask)
            self.assertEqual(task.fs_HistTuple.base, [f"{REMOTE}/tuples"])
            self.assertEqual(task.fs_Hists.base, hists)
            for target in self.hist_tuples() + self.pre_hist_tuple_markers():
                self.assertIs(target.fs, task.fs_HistTuple)
                self.assertTrue(target.uri().startswith(f"{REMOTE}/tuples/"))
            for target in (
                self.split_hists() + self.merged_markers() + self.merged_hists()
            ):
                self.assertIs(target.fs, task.fs_Hists)
                # one of the list's bases, chosen per call
                uri = target.uri()
                self.assertTrue(any(uri.startswith(f"{b}/") for b in hists), uri)
            self.assertIs(task.fs_StatInference, task.fs_default)

    def test_the_resolved_settings(self):
        paths = self.configure(HistTuple="tuples")
        setup = self.task(analysis.HistMergerTask).setup
        self.assertEqual(setup.fs_setting("Hists"), paths["HistTuple"])
        self.assertEqual(setup.fs_setting("HistTuple"), paths["HistTuple"])
        self.assertEqual(setup.fs_setting("StatInference"), self.store)
        self.assertEqual(setup.fs_setting("plots"), self.store)
        self.assertEqual(setup.fs_setting("default"), self.store)


if __name__ == "__main__":
    unittest.main()
