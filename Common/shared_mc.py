def shared_mc_split(era, shared_mc):
    """Luminosity split of one MC sample across the years in ``shared_mc``.

    Returns ``(this_year, split_mod, thresh_24, frac)`` where ``this_year`` is
    the two-digit suffix of ``era`` (``Run3_2024`` → ``24``), ``frac`` is this
    year's luminosity share, and an event is assigned to 2024 when
    ``(event % split_mod) < thresh_24``.
    """
    this_year = str(era.split("_")[-1])[-2:]
    years = shared_mc["years"]
    if this_year not in years:
        raise RuntimeError(f"shared_mc has no year '{this_year}' for era {era}")
    if "24" not in years:
        raise RuntimeError("shared_mc requires year '24'")
    lumi_sum = sum(float(ycfg["luminosity"]) for ycfg in years.values())
    if lumi_sum <= 0:
        raise RuntimeError("shared_mc year luminosities must be positive")
    split_mod = int(shared_mc.get("split_modulus", 1000000))
    if split_mod <= 0:
        raise RuntimeError("shared_mc split_modulus must be positive")
    frac_24 = float(years["24"]["luminosity"]) / lumi_sum
    thresh_24 = int(frac_24 * split_mod)
    frac = float(years[this_year]["luminosity"]) / lumi_sum
    return this_year, split_mod, thresh_24, frac


def shared_mc_in_24(event, split_mod, thresh_24):
    return (int(event) % split_mod) < thresh_24
