#!/usr/bin/env python3
"""Test the strict gen-level TT identification (FLAF/include/GenProcess/TT.h) on a nanoAOD file.

Runs ``gen_process::tt::identify`` over every event and checks that none throw (i.e. every
event has the expected ttbar topology), and prints the W-decay-mode distribution and the
mean top and b pT.

Usage:
    test_TT.py --input <nanoAOD.root> [<nanoAOD.root> ...] [--tree Events] [--max-events N]
Exit code is non-zero if any event fails to be identified.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="nanoAOD file(s)")
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    import ROOT

    ROOT.gROOT.SetBatch(True)

    flaf = os.environ.get("FLAF_PATH") or os.path.join(
        os.environ["ANALYSIS_PATH"], "FLAF"
    )
    header = os.path.join(flaf, "include", "GenProcess", "TT.h")
    if not ROOT.gInterpreter.Declare(f'#include "{header}"'):
        raise RuntimeError(f"failed to declare {header}")
    ROOT.gInterpreter.Declare(
        """
    #include <array>
    #include <exception>
    #include <string>
    #include <vector>
    namespace _tt_test {
        long long n_fail = 0;
        std::vector<std::string> messages;
        struct Result {
            int decay_code = -1;
            std::array<float, 2> top_pt{{-1.f, -1.f}};
            std::array<float, 2> b_pt{{-1.f, -1.f}};
        };
        template <typename VecId, typename VecFlags, typename VecMother, typename VecF>
        Result identify(const VecId& pdgId, const VecFlags& statusFlags, const VecMother& mother,
                        const VecF& pt, const VecF& eta, const VecF& phi, const VecF& mass) {
            Result r;
            try {
                const auto info = gen_process::tt::identify(pdgId, statusFlags, mother, pt, eta, phi, mass);
                for (int k = 0; k < 2; ++k) {
                    if (!(info.top_p4[k].pt() > 0 && info.b_p4[k].pt() > 0))
                        throw std::runtime_error("non-positive top or b-quark pT");
                }
                int n_lep = 0;
                for (int k = 0; k < 2; ++k)
                    if (info.lep_index[k] >= 0)
                        ++n_lep;
                if (n_lep != info.nLeptonicW())
                    throw std::runtime_error("lepton indices do not match the leptonic W count");
                r.decay_code = info.nLeptonicW();
                for (int k = 0; k < 2; ++k) {
                    r.top_pt[k] = info.top_p4[k].pt();
                    r.b_pt[k] = info.b_p4[k].pt();
                }
            } catch (const std::exception& e) {
                ++n_fail;
                if (messages.size() < 20) messages.push_back(e.what());
            }
            return r;
        }
    }
    """
    )

    df = ROOT.RDataFrame(args.tree, list(args.input))
    if args.max_events:
        df = df.Range(args.max_events)
    df = df.Define(
        "tt_result",
        "_tt_test::identify(GenPart_pdgId, GenPart_statusFlags, GenPart_genPartIdxMother,"
        " GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass)",
    )
    df = df.Define("tt_code", "tt_result.decay_code")
    df = df.Define("top_pt", "tt_result.top_pt[0]")
    df = df.Define("antitop_pt", "tt_result.top_pt[1]")
    df = df.Define("b_pt", "tt_result.b_pt[0]")
    df = df.Define("bbar_pt", "tt_result.b_pt[1]")
    h = df.Histo1D(("tt_code", "TT n leptonic W", 5, -1.5, 3.5), "tt_code")
    identified = df.Filter("tt_code >= 0")
    means = {c: identified.Mean(c) for c in ["top_pt", "antitop_pt", "b_pt", "bbar_pt"]}
    n_total = df.Count()
    n_total = n_total.GetValue()
    h = h.GetValue()

    def count(c):
        return int(h.GetBinContent(h.FindBin(c)))

    n_fail = int(ROOT._tt_test.n_fail)
    print(f"input: {', '.join(args.input)}")
    print(f"events processed: {n_total}")
    print(
        f"  4Q (0 leptonic W): {count(0)}\n"
        f"  LNu2Q (1 leptonic W): {count(1)}\n"
        f"  2L2Nu (2 leptonic W): {count(2)}"
    )
    if n_total > n_fail:
        print(
            f"  mean pT: top {means['top_pt'].GetValue():.1f} GeV,"
            f" anti-top {means['antitop_pt'].GetValue():.1f} GeV,"
            f" b {means['b_pt'].GetValue():.1f} GeV,"
            f" bbar {means['bbar_pt'].GetValue():.1f} GeV"
        )
    print(f"  unidentified (threw): {n_fail} ({100.0 * n_fail / max(n_total, 1):.4f}%)")
    for msg in ROOT._tt_test.messages:
        print(f"    - {msg}")

    if n_fail > 0:
        print("FAILED: some events were not identified as TT")
        return 1
    print("SUCCESS: all events identified as TT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
