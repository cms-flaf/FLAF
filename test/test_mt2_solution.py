#!/usr/bin/env python3
"""Calculate_MT2_func_withSolution must terminate.

With ROOT 6.40 (LCG_110a) the MT2 of this t̄t event (lepton + b-jet on each side, neutrinos massless)
comes out one ulp different from ROOT 6.36, the discriminant in ben_findsols is then exactly zero,
and its scan over metpy used to step by (high - low) / 10000 = 0 forever: the HH_bbWW DNN cache
hung on it. A hang inside JIT-compiled C++ cannot be interrupted from Python, so each evaluation
runs in a subprocess with a timeout.

Needs ROOT: run inside an analysis environment (source env.sh).
"""

import math
import os
import subprocess
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# lep1, lep2, jet1, jet2 as (pt, eta, phi, mass), then MET (pt, phi)
EVENT = [
    52.53080749511719, 0.13671493530273438, 2.24029541015625, 0.105712890625,
    50.09596633911133, -0.1760101318359375, -1.956329345703125, 0.105712890625,
    109.76155853271484, 0.9776611328125, 1.046630859375, 8.651389122009277,
    21.497875213623047, 1.6044921875, -2.48583984375, 4.922633171081543,
    68.05218505859375, -1.3857808113098145,
]  # fmt: skip

CODE = """
import sys, ROOT
ROOT.gROOT.SetBatch(True)
inc = sys.argv[1]
ROOT.gInterpreter.Declare('#include "%s/include/MT2.h"' % inc)
ROOT.gROOT.ProcessLine('#include "%s/include/Lester_mt2_bisect.cpp"' % inc)
ROOT.gInterpreter.Declare('''
#include "Math/Vector4D.h"
analysis::MT2Result mt2_blbl(const std::vector<double>& a) {
  using V = ROOT::Math::PtEtaPhiMVector;
  V l1(a[0], a[1], a[2], a[3]), l2(a[4], a[5], a[6], a[7]);
  V j1(a[8], a[9], a[10], a[11]), j2(a[12], a[13], a[14], a[15]), met(a[16], 0, a[17], 0);
  return analysis::Calculate_MT2_func_withSolution(l1 + j1, l2 + j2, met, 0.0, 0.0);
}''')
values = ROOT.std.vector("double")([float(x) for x in sys.argv[2:]])
r = ROOT.mt2_blbl(values)
print(r.mt2, r.pxInvisible1, r.pyInvisible1, r.pxInvisible2, r.pyInvisible2)
"""


class TestMT2Solution(unittest.TestCase):
    def test_zero_discriminant_terminates(self):
        args = [sys.executable, "-c", CODE, flaf_repo] + [repr(x) for x in EVENT]
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            self.fail("Calculate_MT2_func_withSolution did not terminate")
        self.assertEqual(result.returncode, 0, result.stderr)
        mt2, px1, py1, px2, py2 = (float(x) for x in result.stdout.split()[-5:])
        self.assertAlmostEqual(mt2, 108.200813218563, places=6)
        for value in (px1, py1, px2, py2):
            self.assertTrue(math.isfinite(value), value)


if __name__ == "__main__":
    unittest.main()
