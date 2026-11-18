#!/usr/bin/env python3
"""HistMergerFromHists in steps: one process at a time (merge_process), each written into the
output and freed, then the steps that need every process on the merged file (postprocess).

The inputs are synthetic split files (channel/region/category/<var>[_<unc>_<scale>]). The
merged output must hold the sum of each process's inputs, data only nominal; the post step
must see every process, in merge order, with data aliased under the variations, as when the
whole merge was held in memory.

Needs ROOT: run inside an analysis environment (source env.sh).
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import pytest

pytestmark = pytest.mark.needs_root

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import numpy as np
import ROOT

import FLAF.Analysis.HistMergerFromHists as HM
from FLAF.Common.HistHelper import SIGNAL_TREES_DIR

ROOT.gROOT.SetBatch(True)

VAR = "x"
CHANNELS = ["eE", "muMu"]
REGIONS = ["SR", "CR"]
CATEGORIES = ["res2b"]
UNC = "U1"
NBINS = 4
# dataset -> process; two chunks of TT_a, data is named data_obs to keep its aliases
PROCESS_OF = {"TT_a": "TT", "TT_b": "TT", "DY_a": "DY", "data": "data_obs"}


def bins(dataset, chunk, ch, reg, scale):
    """Distinct, exactly representable contents for every input histogram."""
    seed = (
        list(PROCESS_OF).index(dataset) * 1000
        + chunk * 100
        + CHANNELS.index(ch) * 10
        + REGIONS.index(reg) * 3
        + ["Central", "Up", "Down"].index(scale)
    )
    return [seed + 0.25 * b for b in range(NBINS + 2)]


def write_input(path, dataset, chunk, trees=False):
    f = ROOT.TFile(path, "RECREATE")
    for ch in CHANNELS:
        for reg in REGIONS:
            for cat in CATEGORIES:
                d = HM.Utilities.mkdir(f, f"{ch}/{reg}/{cat}")
                scales = ["Central"] if dataset == "data" else ["Central", "Up", "Down"]
                for scale in scales:
                    name = VAR if scale == "Central" else f"{VAR}_{UNC}_{scale}"
                    h = ROOT.TH1D(name, name, NBINS, 0, 1)
                    h.SetDirectory(ROOT.nullptr)
                    for b, v in enumerate(bins(dataset, chunk, ch, reg, scale)):
                        h.SetBinContent(b, v)
                        h.SetBinError(b, 0.5 * v)
                    d.WriteTObject(h, name)
    if trees:
        d = HM.Utilities.mkdir(f, SIGNAL_TREES_DIR)
        for tree_name in ["Central", f"{UNC}_Up", f"{UNC}_Down"]:
            d.cd()
            t = ROOT.TTree(tree_name, tree_name)
            v = np.zeros(1, dtype=np.float64)
            t.Branch("v", v, "v/D")
            for i in range(3 + chunk):
                v[0] = 10 * chunk + i
                t.Fill()
            t.Write()
    f.Close()


def context():
    """A MergeContext with the configuration set directly (no Setup)."""
    ctx = HM.MergeContext.__new__(HM.MergeContext)
    ctx.var = VAR
    ctx.channels = CHANNELS
    ctx.regions = REGIONS
    ctx.categories = CATEGORIES
    ctx.scales = ["Up", "Down"]
    ctx.unc_sources = ["Central", UNC]
    ctx.member_uncs = []
    ctx.unc_cfg_dict = {"norm": {}, "shape": {UNC: {"name": UNC + "_{}"}}}
    ctx.all_unc_dict = {UNC: {"name": UNC + "_{}"}}
    ctx.data_process_name = "data_obs"
    ctx.qcd_data_process = "data_obs"
    ctx.analysis_import = "Analysis.hh_bbtautau"
    ctx.estimateQCD = True
    ctx.process_of = PROCESS_OF.__getitem__
    ctx.data_driven = {}
    ctx.model_backgrounds = ["DY", "TT", "Fakes"]
    ctx.model_signals = []
    ctx.setup = types.SimpleNamespace(era="Run3_2022")
    return ctx


FAKES = {
    "estimator": "Analysis.test_fakes.estimate",
    "nuisances": {"FakesStat": "FakesStat_{channel}_{}"},
}


def fakes_module(calls, extra=None):
    """A stand-in estimator: Fakes in SR = data - simulation in CR, for every variation of
    the source; at the nominal pass also FakesStat Up/Down = 1.1 / 0.9 x the nominal."""

    def estimate(ctx):
        calls.append(
            (ctx.unc_source, list(ctx.variations), list(ctx.mc_backgrounds), ctx.era)
        )
        out = {}
        for ch in ctx.channels:
            for cat in ctx.categories:
                src = (ch, "CR", cat)
                for unc, scale in ctx.variations:
                    h = ctx.require(ctx.data_process, src, unc, scale)
                    for p in ctx.mc_backgrounds:
                        h.Add(ctx.require(p, src, unc, scale), -1)
                    out[((ch, "SR", cat), (unc, scale))] = h
                    if ctx.nominal_pass:
                        for name, factor in (("Up", 1.1), ("Down", 0.9)):
                            v = h.Clone()
                            v.Scale(factor)
                            out[((ch, "SR", cat), ("FakesStat", name))] = v
        if extra:
            extra(ctx, out)
        return out

    module = types.ModuleType("Analysis.test_fakes")
    module.estimate = estimate
    return module


def read(path):
    """{dir/name: contents} of every histogram of a file."""
    out = {}
    f = ROOT.TFile.Open(path)

    def walk(d, prefix):
        for key in d.GetListOfKeys():
            cls = ROOT.TClass.GetClass(key.GetClassName())
            name = f"{prefix}/{key.GetName()}" if prefix else key.GetName()
            if cls.InheritsFrom("TDirectory"):
                walk(d.Get(key.GetName()), name)
            elif cls.InheritsFrom("TH1"):
                h = key.ReadObj()
                out[name] = [h.GetBinContent(b) for b in range(NBINS + 2)]
            elif cls.InheritsFrom("TTree"):
                out[name] = ("tree", key.ReadObj().GetEntries())

    walk(f, "")
    f.Close()
    return out


class TestMergeInSteps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.inputs = {}  # process -> [(dataset, path)], in merge order
        for dataset, chunks in (("TT_a", 2), ("DY_a", 1), ("TT_b", 1), ("data", 1)):
            for chunk in range(chunks):
                path = os.path.join(self.tmp.name, f"{dataset}_{chunk}.root")
                write_input(path, dataset, chunk, trees=dataset == "DY_a")
                self.inputs.setdefault(PROCESS_OF[dataset], []).append((dataset, path))
        self.out = os.path.join(self.tmp.name, "merged.root")

    def merge(self, ctx):
        for process, items in self.inputs.items():
            f = HM.open_merged(self.out)
            HM.merge_process(ctx, process, items, f)
            f.Close()

    def expected_sum(self, process, ch, reg, scale):
        total = np.zeros(NBINS + 2)
        for dataset, path in self.inputs[process]:
            chunk = int(path.rsplit("_", 1)[1].split(".")[0])
            total += bins(dataset, chunk, ch, reg, scale)
        return list(total)

    def test_each_process_summed_and_written(self):
        self.merge(context())
        merged = read(self.out)
        for ch in CHANNELS:
            for reg in REGIONS:
                d = f"{ch}/{reg}/res2b"
                for process in ("TT", "DY"):
                    self.assertEqual(
                        merged[f"{d}/{process}"],
                        self.expected_sum(process, ch, reg, "Central"),
                    )
                    for scale in ("Up", "Down"):
                        self.assertEqual(
                            merged[f"{d}/{process}_{UNC}_{scale}"],
                            self.expected_sum(process, ch, reg, scale),
                        )
                # data: nominal, and (named data_obs, not "data") its aliases
                nominal = self.expected_sum("data_obs", ch, reg, "Central")
                self.assertEqual(merged[f"{d}/data_obs"], nominal)
                self.assertEqual(merged[f"{d}/data_obs_{UNC}_Up"], nominal)
        # TT_a has two chunks and TT_b one: one TT, summed over the three
        self.assertNotIn("eE/SR/res2b/TT_a", merged)
        # the signal trees of DY, concatenated per variation
        self.assertEqual(merged[f"{SIGNAL_TREES_DIR}/DY/Central"], ("tree", 3))
        self.assertEqual(merged[f"{SIGNAL_TREES_DIR}/DY/{UNC}_Up"], ("tree", 3))

    def test_postprocess_sees_every_process_in_merge_order(self):
        ctx = context()
        self.merge(ctx)
        calls = []

        def add_qcd(var, hists, channels, categories, unc, samples, scales, **kw):
            # what the estimate is given: the processes, their order, data under the keys
            # of the source (the aliases), and the source's histograms of every process
            calls.append((unc, list(samples), kw["data_process_name"]))
            hists["QCD"] = {}
            keys = (
                [("Central", "Central")]
                if unc == "Central"
                else [(unc, s) for s in scales]
            )
            for ch in channels:
                for cat in categories:
                    for unc_key in keys:
                        src = ((ch, "CR", cat), unc_key)
                        h = hists[kw["data_process_name"]][src].Clone()
                        h.SetDirectory(ROOT.nullptr)
                        for sample in samples:
                            if sample != kw["data_process_name"]:
                                h.Add(hists[sample][src], -1)
                        hists["QCD"][((ch, "SR", cat), unc_key)] = h

        fake = types.ModuleType("Analysis.QCD_estimation")
        fake.AddQCDInHistDict = add_qcd
        with mock.patch.dict(sys.modules, {"Analysis.QCD_estimation": fake}):
            f = HM.open_merged(self.out)
            HM.postprocess(ctx, list(self.inputs), f)
            f.Close()

        self.assertEqual(
            calls,
            [
                ("Central", ["TT", "DY", "data_obs"], "data_obs"),
                (UNC, ["TT", "DY", "data_obs"], "data_obs"),
            ],
        )
        merged = read(self.out)
        for ch in CHANNELS:
            data = np.array(self.expected_sum("data_obs", ch, "CR", "Central"))
            for scale, suffix in (("Central", ""), ("Up", f"_{UNC}_Up")):
                mc = sum(
                    np.array(self.expected_sum(p, ch, "CR", scale))
                    for p in ("TT", "DY")
                )
                self.assertEqual(merged[f"{ch}/SR/res2b/QCD{suffix}"], list(data - mc))

    def test_no_postprocess_without_a_data_driven_estimate(self):
        ctx = context()
        ctx.estimateQCD = False
        self.merge(ctx)
        before = read(self.out)
        f = HM.open_merged(self.out)
        HM.postprocess(ctx, list(self.inputs), f)
        f.Close()
        self.assertEqual(read(self.out), before)

    def run_estimate(self, ctx, module, processes=None):
        with mock.patch.dict(sys.modules, {"Analysis.test_fakes": module}):
            f = HM.open_merged(self.out)
            HM.postprocess(ctx, processes or list(self.inputs), f)
            f.Close()

    def data_driven_context(self):
        ctx = context()
        ctx.estimateQCD = False
        ctx.data_driven = {"Fakes": FAKES}
        return ctx

    def test_data_driven_estimate_per_source(self):
        ctx = self.data_driven_context()
        self.merge(ctx)
        calls = []
        self.run_estimate(ctx, fakes_module(calls))
        # once per source, the MC backgrounds in model order, merged ones only
        self.assertEqual(
            calls,
            [
                ("Central", [("Central", "Central")], ["DY", "TT"], "Run3_2022"),
                (UNC, [(UNC, "Up"), (UNC, "Down")], ["DY", "TT"], "Run3_2022"),
            ],
        )
        merged = read(self.out)
        for ch in CHANNELS:
            data = np.array(self.expected_sum("data_obs", ch, "CR", "Central"))
            for scale, suffix in (("Central", ""), ("Up", f"_{UNC}_Up")):
                mc = sum(
                    np.array(self.expected_sum(p, ch, "CR", scale))
                    for p in ("TT", "DY")
                )
                self.assertEqual(
                    merged[f"{ch}/SR/res2b/Fakes{suffix}"], list(data - mc)
                )
            nominal = np.array(merged[f"{ch}/SR/res2b/Fakes"])
            self.assertEqual(
                merged[f"{ch}/SR/res2b/Fakes_FakesStat_{ch}_Up"], list(nominal * 1.1)
            )
            self.assertEqual(
                merged[f"{ch}/SR/res2b/Fakes_FakesStat_{ch}_Down"], list(nominal * 0.9)
            )
            # only where the estimator writes
            self.assertNotIn(f"{ch}/CR/res2b/Fakes", merged)

    def test_no_estimate_without_data(self):
        ctx = self.data_driven_context()
        self.merge(ctx)
        calls = []
        self.run_estimate(ctx, fakes_module(calls), processes=["TT", "DY"])
        self.assertEqual(calls, [])
        self.assertNotIn("eE/SR/res2b/Fakes", read(self.out))

    def test_no_estimate_for_a_subset(self):
        # --no-data-driven: MergeContext leaves data_driven empty
        ctx = self.data_driven_context()
        ctx.data_driven = {}
        self.merge(ctx)
        before = read(self.out)
        calls = []
        self.run_estimate(ctx, fakes_module(calls))
        self.assertEqual((calls, read(self.out)), ([], before))

    def test_estimator_cannot_read_another_source(self):
        ctx = self.data_driven_context()
        self.merge(ctx)

        def read_up_at_nominal(est, out):
            if est.nominal_pass:
                est.get("TT", ("eE", "CR", "res2b"), UNC, "Up")

        with self.assertRaisesRegex(KeyError, "not a variation of the source Central"):
            self.run_estimate(ctx, fakes_module([], read_up_at_nominal))

    def test_undeclared_or_misplaced_variations_are_refused(self):
        def undeclared(est, out):
            out[(("eE", "SR", "res2b"), ("Other", "Up"))] = next(iter(out.values()))

        def nuisance_off_nominal(est, out):
            if not est.nominal_pass:
                h = next(iter(out.values()))
                out[(("eE", "SR", "res2b"), ("FakesStat", "Up"))] = h

        for extra in (undeclared, nuisance_off_nominal):
            with self.subTest(extra=extra.__name__):
                ctx = self.data_driven_context()
                self.merge(ctx)
                with self.assertRaisesRegex(
                    RuntimeError, "it may return the variations"
                ):
                    self.run_estimate(ctx, fakes_module([], extra))

    def test_returned_copies_do_not_change_the_merged_file(self):
        ctx = self.data_driven_context()
        self.merge(ctx)
        before = read(self.out)

        def scale_reads(est, out):
            est.get("TT", ("eE", "CR", "res2b")).Scale(10)

        self.run_estimate(ctx, fakes_module([], scale_reads))
        after = read(self.out)
        self.assertEqual(after["eE/CR/res2b/TT"], before["eE/CR/res2b/TT"])

    def test_an_input_of_another_process_is_refused(self):
        f = HM.open_merged(self.out)
        with self.assertRaisesRegex(RuntimeError, "is of process DY, not TT"):
            HM.merge_process(context(), "TT", self.inputs["DY"], f)
        f.Close()


if __name__ == "__main__":
    unittest.main()
