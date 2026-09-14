#!/usr/bin/env python3
"""Test the strict gen-level TT identification (FLAF/include/GenProcess/TT.h) on a nanoAOD file.

Runs ``gen_process::tt::identify`` over every event and checks that none throw (i.e. every
event has the expected ttbar topology), and prints the W-decay-mode distribution. The
overload taking ``GenPart_pt`` is used, so the pT of the last-copy top and anti-top
(``TTInfo::top_pt``) is summarised as well, and an event where either is not positive
counts as a failure.

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
        };
        template <typename VecId, typename VecFlags, typename VecMother, typename VecPt>
        Result identify(const VecId& pdgId, const VecFlags& statusFlags, const VecMother& mother,
                        const VecPt& pt) {
            Result r;
            try {
                const auto info = gen_process::tt::identify(pdgId, statusFlags, mother, pt);
                if (!(info.top_pt[0] > 0.f && info.top_pt[1] > 0.f))
                    throw std::runtime_error("non-positive last-copy top pT");
                r.decay_code = info.nLeptonicW();
                r.top_pt = info.top_pt;
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
        " GenPart_pt)",
    )
    df = df.Define("tt_code", "tt_result.decay_code")
    df = df.Define("top_pt", "tt_result.top_pt[0]")
    df = df.Define("antitop_pt", "tt_result.top_pt[1]")
    h = df.Histo1D(("tt_code", "TT n leptonic W", 5, -1.5, 3.5), "tt_code")
    identified = df.Filter("tt_code >= 0")
    top_mean = identified.Mean("top_pt")
    antitop_mean = identified.Mean("antitop_pt")
    top_max = identified.Max("top_pt")
    antitop_max = identified.Max("antitop_pt")
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
            f"  last-copy top pT: mean {top_mean.GetValue():.1f} GeV,"
            f" max {top_max.GetValue():.1f} GeV\n"
            f"  last-copy anti-top pT: mean {antitop_mean.GetValue():.1f} GeV,"
            f" max {antitop_max.GetValue():.1f} GeV"
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
