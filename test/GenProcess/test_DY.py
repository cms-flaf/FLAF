#!/usr/bin/env python3
"""Test the strict gen-level DY identification (FLAF/include/GenProcess/DY.h) on a nanoAOD file.

Runs ``gen_process::dy::identifyLHE`` over every event (checking none throw and reporting the
dilepton flavor and mass), and, with ``--tautau``, also ``identifyTauTau`` (reporting the
Z->tautau gen-filter efficiency).

Usage:
    test_DY.py --input <nanoAOD.root> [...] [--tree Events] [--max-events N] [--tautau]
Exit code is non-zero if any event fails to be identified.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--tree", default="Events")
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

    df = ROOT.RDataFrame(args.tree, list(args.input))
    if args.max_events:
        df = df.Range(args.max_events)
    df = df.Define(
        "dy_flavor",
        "_dy_test::lheFlavor(LHEPart_pt, LHEPart_eta, LHEPart_phi, LHEPart_mass, "
        "LHEPart_pdgId, LHEPart_status)",
    ).Define(
        "dy_mll",
        "_dy_test::lheMass(LHEPart_pt, LHEPart_eta, LHEPart_phi, LHEPart_mass, "
        "LHEPart_pdgId, LHEPart_status)",
    )
    hf = df.Histo1D(("f", "flavor", 20, -1.5, 18.5), "dy_flavor")
    if args.tautau:
        df = df.Define(
            "dy_filter",
            "_dy_test::tautauFilter(GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, "
            "GenPart_pdgId, GenPart_statusFlags, GenPart_genPartIdxMother)",
        )
        hfilt = df.Histo1D(("filt", "filter", 4, -1.5, 2.5), "dy_filter")
    n_total = df.Count()
    mll_min = df.Filter("dy_mll > 0").Min("dy_mll")
    mll_max = df.Max("dy_mll")
    n_total = n_total.GetValue()
    hf = hf.GetValue()

    def fcount(v):
        return int(hf.GetBinContent(hf.FindBin(v)))

    n_fail_lhe = int(ROOT._dy_test.n_fail_lhe)
    print(f"input: {', '.join(args.input)}")
    print(f"events processed: {n_total}")
    print(
        f"  LHE flavor: e={fcount(11)} mu={fcount(13)} tau={fcount(15)} "
        f"(unidentified={fcount(-1)})"
    )
    print(f"  LHE m_ll range: [{mll_min.GetValue():.1f}, {mll_max.GetValue():.1f}] GeV")
    print(
        f"  identifyLHE failures: {n_fail_lhe} ({100.0 * n_fail_lhe / max(n_total, 1):.4f}%)"
    )

    n_fail_tt = 0
    if args.tautau:
        hfilt = hfilt.GetValue()
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
