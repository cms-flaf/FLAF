#!/usr/bin/env python3
"""Test that Setup can successfully load global.yaml for all specified eras."""

import sys
import os
from unittest import mock

ana_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
flaf_path = os.path.join(ana_path, "FLAF")
if flaf_path not in sys.path:
    sys.path.insert(0, flaf_path)
sys.path.insert(0, ana_path)

sys.modules["ROOT"] = mock.MagicMock()

from FLAF.Common.Setup import Setup


def test_setup_loading(eras):
    failed_eras = []

    print(f"Testing Setup loading for {len(eras)} eras...")
    print(f"Analysis path: {ana_path}")
    print("-" * 80)

    for era in eras:
        print(f"\nTesting era: {era}")
        try:
            setup = Setup(ana_path=ana_path, period=era, law_run_version="test")

            assert setup.global_params is not None, "global_params is None"
            assert len(setup.global_params.keys()) > 0, "global_params is empty"
            assert setup.phys_model is not None, "phys_model is None"

            print(
                f"  OK: {era} — {setup.phys_model.name}, {len(list(setup.global_params.keys()))} params"
            )
        except Exception as e:
            print(f"  FAILED: {era} — {e}")
            import traceback

            traceback.print_exc()
            failed_eras.append((era, str(e)))

    print("\n" + "=" * 80)
    if failed_eras:
        print(f"FAILED: {len(failed_eras)}/{len(eras)} eras failed:")
        for era, error in failed_eras:
            print(f"  - {era}: {error}")
        return 1
    print(f"SUCCESS: all {len(eras)} eras loaded successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: test_setup_loading.py ERA [ERA ...]", file=sys.stderr)
        sys.exit(1)
    sys.exit(test_setup_loading(sys.argv[1:]))
