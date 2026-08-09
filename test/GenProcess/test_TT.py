#!/usr/bin/env python3
"""Test the strict gen-level TT identification (FLAF/include/GenProcess/TT.h) on a nanoAOD file.

Runs ``gen_process::tt::identify`` over every event and checks that none throw (i.e. every
event has the expected ttbar topology), and prints the W-decay-mode distribution.

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
    ROOT.gInterpreter.Declare("""
    #include <exception>
    #include <string>
    #include <vector>
    namespace _tt_test {
        long long n_fail = 0;
        std::vector<std::string> messages;
        template <typename VecId, typename VecFlags, typename VecMother>
        int decayCode(const VecId& pdgId, const VecFlags& statusFlags, const VecMother& mother) {
            try {
                return gen_process::tt::identify(pdgId, statusFlags, mother).nLeptonicW();
            } catch (const std::exception& e) {
                ++n_fail;
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
        "tt_code",
        "_tt_test::decayCode(GenPart_pdgId, GenPart_statusFlags, GenPart_genPartIdxMother)",
    )
    h = df.Histo1D(("tt_code", "TT n leptonic W", 5, -1.5, 3.5), "tt_code")
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
