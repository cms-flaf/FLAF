#!/usr/bin/env python3
"""Test that Setup can successfully load global.yaml for all specified eras.

Beyond loading, this cross-checks each era's weights.yaml against the corrections
that era actually resolves to. Config is built by concatenating YAML text and
parsing it once, so an era that declares its own `corrections:` block *replaces*
the top-level one wholesale rather than merging into it. An era that drops a
shape-weight correction that way still parses fine, and only fails much later in
HistTupleProducer, as an unguarded RDataFrame Define on a column that was never
created. Catching it here turns a mid-pipeline crash into a red PR check.
"""

import sys
import os
import re
from unittest import mock

ana_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
flaf_path = os.path.join(ana_path, "FLAF")
if flaf_path not in sys.path:
    sys.path.insert(0, flaf_path)
sys.path.insert(0, ana_path)

sys.modules["ROOT"] = mock.MagicMock()

from FLAF.Common.Setup import Setup

# weight_base_<source>{scale}_rel, as written in weights.yaml before scale expansion.
# Up/Down are accepted too in case an entry ever hard-codes one side.
SHAPE_WEIGHT_RE = re.compile(r"weight_base_(\w+?)(?:\{scale\}|Up|Down)_rel")


def shape_weight_owners():
    """Map shape-weight source name -> the correction that produces it.

    Read from Corrections rather than hard-coded, so a producer added there is
    covered by this check without touching the test.
    """
    from Corrections.Corrections import Corrections

    # _shapeWeightClasses only imports the producer modules; it does not touch
    # instance state, so an uninitialised instance is enough to call it.
    classes = Corrections.__new__(Corrections)._shapeWeightClasses()
    owners = {}
    for corr_name, _ in Corrections.shape_weight_producers:
        for source in classes[corr_name].uncSource:
            owners[source] = corr_name
    return owners


def check_shape_weight_corrections(setup, owners):
    """Return a list of weights.yaml entries whose correction is not enabled here."""
    corrections = setup.global_params.get("corrections", {}) or {}
    problems = []
    for block in ("norm", "shape"):
        entries = setup.weights_config.get(block) or {}
        for entry_name, entry in entries.items():
            expression = (entry or {}).get("expression")
            if not expression:
                continue
            for source in SHAPE_WEIGHT_RE.findall(expression):
                owner = owners.get(source)
                if owner is None:
                    problems.append(
                        f"{block}:{entry_name} uses weight_base_{source}*_rel, but no "
                        f"shape-weight producer declares source '{source}'"
                    )
                elif owner not in corrections:
                    problems.append(
                        f"{block}:{entry_name} needs weight_base_{source}*_rel, but "
                        f"correction '{owner}' is missing from this era's corrections "
                        f"block (an era-level corrections block replaces the top-level "
                        f"one wholesale, it does not merge into it)"
                    )
    return problems


def test_setup_loading(eras):
    failed_eras = []
    owners = shape_weight_owners()

    print(f"Testing Setup loading for {len(eras)} eras...")
    print(f"Analysis path: {ana_path}")
    print(f"Shape-weight sources: {owners}")
    print("-" * 80)

    for era in eras:
        print(f"\nTesting era: {era}")
        try:
            setup = Setup(ana_path=ana_path, period=era, law_run_version="test")

            assert setup.global_params is not None, "global_params is None"
            assert len(setup.global_params.keys()) > 0, "global_params is empty"
            assert setup.phys_model is not None, "phys_model is None"

            problems = check_shape_weight_corrections(setup, owners)
            if problems:
                raise AssertionError(
                    "weights.yaml references shape weights this era does not produce:\n    "
                    + "\n    ".join(problems)
                )

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
