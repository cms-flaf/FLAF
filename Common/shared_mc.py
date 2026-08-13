def shared_mc_split(era, shared_mc):
    """Residue split of one MC sample across the eras in ``shared_mc``.

    Config shape::

        shared_mc:
          split_modulus: 20
          eras:
            Run3_2024: [0, 8]
            Run3_2025: [9, 17]
            Run3_2026: [18, 19]

    Returns ``(split_mod, lo, hi, frac)``. An event is assigned to ``era`` when
    ``lo <= (event % split_mod) <= hi``. ``frac`` is that residue share.
    """
    eras = shared_mc.get("eras")
    if not eras:
        raise RuntimeError("shared_mc requires an 'eras' map of [lo, hi] ranges")
    if era not in eras:
        raise RuntimeError(f"shared_mc has no era '{era}'")
    split_mod = int(shared_mc["split_modulus"])
    if split_mod <= 0:
        raise RuntimeError("shared_mc split_modulus must be positive")
    rng = eras[era]
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise RuntimeError(f"shared_mc era '{era}' must be a [lo, hi] range")
    lo, hi = int(rng[0]), int(rng[1])
    if not (0 <= lo <= hi < split_mod):
        raise RuntimeError(
            f"shared_mc era '{era}' range [{lo}, {hi}] is outside "
            f"[0, {split_mod - 1}]"
        )
    frac = (hi - lo + 1) / float(split_mod)
    return split_mod, lo, hi, frac


def shared_mc_in_era(event, split_mod, lo, hi):
    return lo <= (int(event) % split_mod) <= hi


def shared_mc_in_era_expr(split_mod, lo, hi):
    """C++ expression: 1 if this event is assigned to the era range, else 0."""
    residue = f"(static_cast<unsigned long long>(event) % {int(split_mod)}ULL)"
    return f"static_cast<int>({residue} >= {int(lo)}ULL && {residue} <= {int(hi)}ULL)"
