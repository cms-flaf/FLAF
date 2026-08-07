#!/usr/bin/env python3
"""Test the strict gen-level DY identification (FLAF/include/GenProcess/DY.h) on a nanoAOD file.

Runs ``gen_process::dy::identifyLHE`` over every event (checking none throw and reporting the
dilepton flavor and mass), and, with ``--tautau``, also ``identifyTauTau`` (reporting the
Z->tautau gen-filter efficiency).

Usage:
    test_DY.py --input <nanoAOD.root> [...] [--trees Events [EventsNotSelected]] \
        [--max-events N] [--tautau]
Exit code is non-zero if any event fails to be identified.

For the HLepRare skims the events are split across the ``Events`` (skim-selected) and
``EventsNotSelected`` (skim-rejected) trees; pass ``--trees Events EventsNotSelected`` to
get the unbiased gen-filter efficiency (the skim selection is correlated with the visible
tau kinematics, so ``Events`` alone is biased).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument(
        "--trees",
        nargs="+",
        default=["Events"],
        help="trees to process; use 'Events EventsNotSelected' for HLepRare skims",
    )
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--tautau",
        action="store_true",
        help="also test the tau-tau filter (DYto2Tau samples)",
    )
    args = parser.parse_args()

    import ROOT

    ROOT.gROOT.SetBatch(True)
    flaf = os.environ.get("FLAF_PATH") or os.path.join(
        os.environ["ANALYSIS_PATH"], "FLAF"
    )
    header = os.path.join(flaf, "include", "GenProcess", "DY.h")
    if not ROOT.gInterpreter.Declare(f'#include "{header}"'):
        raise RuntimeError(f"failed to declare {header}")
    ROOT.gInterpreter.Declare("""
    #include <exception>
    #include <string>
    #include <vector>
    namespace _dy_test {
        long long n_fail_lhe = 0, n_fail_tt = 0;
        std::vector<std::string> messages;
        template <typename VecF, typename VecId>
        int lheFlavor(const VecF& pt, const VecF& eta, const VecF& phi, const VecF& mass,
                      const VecId& pdgId, const VecId& status) {
            try { return gen_process::dy::identifyLHE(pt, eta, phi, mass, pdgId, status).flavor; }
            catch (const std::exception& e) {
                ++n_fail_lhe;
                if (messages.size() < 20) messages.push_back(e.what());
                return -1;
            }
        }
        template <typename VecF, typename VecId>
        float lheMass(const VecF& pt, const VecF& eta, const VecF& phi, const VecF& mass,
                      const VecId& pdgId, const VecId& status) {
            try { return gen_process::dy::identifyLHE(pt, eta, phi, mass, pdgId, status).mll; }
            catch (const std::exception&) { return -1.f; }
        }
        template <typename VecF, typename VecId, typename VecFlags, typename VecMother>
        int tautauFilter(const VecF& pt, const VecF& eta, const VecF& phi, const VecF& mass,
                         const VecId& pdgId, const VecFlags& flags, const VecMother& mother) {
            try {
                return gen_process::dy::identifyTauTau(pt, eta, phi, mass, pdgId, flags, mother)
                           .passFilter() ? 1 : 0;
            } catch (const std::exception& e) {
                ++n_fail_tt;
                if (messages.size() < 20) messages.push_back(e.what());
                return -1;
            }
        }
    }
    """)

    # Book work per tree, then trigger once. EventsNotSelected (HLepRare) may lack the
    # LHEPart branches, so the LHE flavor/mass check is only booked where they exist.
    booked = []
    for tree in args.trees:
        df = ROOT.RDataFrame(tree, list(args.input))
        if args.max_events:
            df = df.Range(args.max_events)
        cols = set(str(c) for c in df.GetColumnNames())
        item = {"tree": tree, "n": df.Count()}
        if "LHEPart_pt" in cols:
            df = df.Define(
                "dy_flavor",
                "_dy_test::lheFlavor(LHEPart_pt, LHEPart_eta, LHEPart_phi, LHEPart_mass, "
                "LHEPart_pdgId, LHEPart_status)",
            ).Define(
                "dy_mll",
                "_dy_test::lheMass(LHEPart_pt, LHEPart_eta, LHEPart_phi, LHEPart_mass, "
                "LHEPart_pdgId, LHEPart_status)",
            )
            item["hf"] = df.Histo1D(
                ("f_" + tree, "flavor", 20, -1.5, 18.5), "dy_flavor"
            )
            item["mll_min"] = df.Filter("dy_mll > 0").Min("dy_mll")
            item["mll_max"] = df.Max("dy_mll")
        if args.tautau:
            df = df.Define(
                "dy_filter",
                "_dy_test::tautauFilter(GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, "
                "GenPart_pdgId, GenPart_statusFlags, GenPart_genPartIdxMother)",
            )
            item["hfilt"] = df.Histo1D(
                ("filt_" + tree, "filter", 4, -1.5, 2.5), "dy_filter"
            )
        booked.append(item)

    hf = ROOT.TH1D("f", "flavor", 20, -1.5, 18.5)
    hfilt = ROOT.TH1D("filt", "filter", 4, -1.5, 2.5)
    n_total = 0
    mll_lo, mll_hi = [], []
    for item in booked:
        n_total += int(item["n"].GetValue())
        if "hf" in item:
            hf.Add(item["hf"].GetValue())
            mll_lo.append(item["mll_min"].GetValue())
            mll_hi.append(item["mll_max"].GetValue())
        if "hfilt" in item:
            hfilt.Add(item["hfilt"].GetValue())

    def fcount(v):
        return int(hf.GetBinContent(hf.FindBin(v)))

    n_fail_lhe = int(ROOT._dy_test.n_fail_lhe)
    print(f"input: {', '.join(args.input)}")
    print(f"events processed: {n_total}")
    print(
        f"  LHE flavor: e={fcount(11)} mu={fcount(13)} tau={fcount(15)} "
        f"(unidentified={fcount(-1)})"
    )
    if mll_lo:
        print(f"  LHE m_ll range: [{min(mll_lo):.1f}, {max(mll_hi):.1f}] GeV")
    print(
        f"  identifyLHE failures: {n_fail_lhe} ({100.0 * n_fail_lhe / max(n_total, 1):.4f}%)"
    )

    n_fail_tt = 0
    if args.tautau:
        n_pass = int(hfilt.GetBinContent(hfilt.FindBin(1)))
        n_notpass = int(hfilt.GetBinContent(hfilt.FindBin(0)))
        n_fail_tt = int(ROOT._dy_test.n_fail_tt)
        eff = 100.0 * n_pass / max(n_pass + n_notpass, 1)
        print(
            f"  tau-tau filter: pass={n_pass} fail={n_notpass} "
            f"-> efficiency {eff:.1f}%  (identifyTauTau failures={n_fail_tt})"
        )
    for msg in ROOT._dy_test.messages:
        print(f"    - {msg}")

    n_fail = n_fail_lhe + n_fail_tt
    if n_fail > 0:
        print("FAILED: some events were not identified as DY")
        return 1
    print("SUCCESS: all events identified as DY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
