"""ShapeWeightRegistry: which weight branches multiply into each variation.

The case that matters is the cross product. A variation of one producer must still
carry every other producer's central branch, or weight_base_<var>/weight_base_Central
retains a spurious factor of one over that other producer's central weight. While
pileup was the only non-central shape source that could not show up, because the only
non-central keys belonged to pileup itself.

No ROOT needed -- this is pure bookkeeping.
"""

import os
import sys

sys.path.append(os.environ.get("ANALYSIS_PATH", os.getcwd()))

from Corrections.CorrectionsCore import (  # noqa: E402
    ShapeWeightRegistry,
    central,
    up,
    down,
)


def pu_branch(source, scale):
    return f"weight_pu_{scale}"


def ps_branch(source, scale):
    from Corrections.CorrectionsCore import getSystName

    return f"weight_ps_{getSystName(source, scale)}"


def test_pileup_alone_reproduces_the_original_mapping():
    """With one producer the registry must reproduce what the old code built."""
    reg = ShapeWeightRegistry()
    reg.register("pu", ["pu"], pu_branch)
    assert reg.asDict() == {
        (central, central): ["weight_pu_Central"],
        ("pu", up): ["weight_pu_Up"],
        ("pu", down): ["weight_pu_Down"],
    }
    assert reg.sources == [central, "pu"]


def test_second_producer_gets_the_cross_product():
    """The regression this class exists to prevent."""
    reg = ShapeWeightRegistry()
    reg.register("pu", ["pu"], pu_branch)
    reg.register("parton_shower", ["isr", "fsr"], ps_branch)

    # A PS variation must still carry the *central* pileup weight. Without it the
    # resulting _rel branch would divide weight_pu_Central out of every event.
    assert reg.branches("isr", up) == ["weight_pu_Central", "weight_ps_isrUp"]
    assert reg.branches("fsr", down) == ["weight_pu_Central", "weight_ps_fsrDown"]

    # ...and symmetrically, a pileup variation carries the central PS weight.
    assert reg.branches("pu", up) == ["weight_pu_Up", "weight_ps_Central"]

    assert reg.branches(central, central) == [
        "weight_pu_Central",
        "weight_ps_Central",
    ]
    assert reg.sources == [central, "pu", "isr", "fsr"]


def test_no_variations_leaves_only_central():
    """return_variations=False registers no owned sources, so only the central key."""
    reg = ShapeWeightRegistry()
    reg.register("pu", [], pu_branch)
    reg.register("parton_shower", [], ps_branch)
    assert reg.asDict() == {
        (central, central): ["weight_pu_Central", "weight_ps_Central"]
    }


def test_empty_registry_matches_the_old_no_corrections_case():
    """The old code seeded {(Central, Central): []} when no producer was active."""
    reg = ShapeWeightRegistry()
    assert reg.asDict() == {(central, central): []}


def test_denominator_expression_is_unchanged_for_pileup():
    """The regression guarantee, provable without running the production chain.

    updateDenomEntry builds the denominator as " * ".join(weights_to_apply), so if the
    new code produces a character-identical expression string for every (source, scale)
    key the old code visited, the resulting sums are bit-identical by construction --
    there is no arithmetic to differ, it is literally the same C++ expression.
    """
    gen = "weight_gen"

    def old_logic(pu_applied, compute_unc_variations):
        sources = [central] + (["pu"] if pu_applied and compute_unc_variations else [])
        out = {}
        for source in sources:
            for scale in [central] if source == central else [up, down]:
                weights = [gen]
                if pu_applied:
                    weights.append(f"weight_pu_{scale}")
                out[(source, scale)] = " * ".join(weights)
        return out

    def new_logic(pu_applied, compute_unc_variations):
        reg = ShapeWeightRegistry()
        if pu_applied:
            reg.register("pu", ["pu"] if compute_unc_variations else [], pu_branch)
        out = {}
        for source in reg.sources:
            for scale in [central] if source == central else [up, down]:
                weights = [gen] + reg.branches(source, scale)
                out[(source, scale)] = " * ".join(weights)
        return out

    for pu_applied in (True, False):
        for compute_unc in (True, False):
            assert old_logic(pu_applied, compute_unc) == new_logic(
                pu_applied, compute_unc
            ), f"expression changed for pu={pu_applied}, unc={compute_unc}"


def test_duplicate_registration_raises():
    reg = ShapeWeightRegistry()
    reg.register("pu", ["pu"], pu_branch)
    try:
        reg.register("pu", ["pu"], pu_branch)
    except RuntimeError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("duplicate registration must raise")


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failures.append((name, e))
            print(f"FAIL {name}: {e}")
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED"))
    sys.exit(1 if failures else 0)
