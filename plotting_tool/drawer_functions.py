import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import ROOT
import matplotlib.ticker as ticker
import yaml
import re
import matplotlib.colors as mcolors
from HelpersForHistograms import *

hep.style.use("CMS")

lumi_dict = {
    "Run3_2022": "7.9804",
    "Run3_2022EE": "26.6717",
    "Run3_2023": "18.063",
    "Run3_2023BPix": "9.693",
}


def get_bin_edges_widths(hist):
    nbins = hist.GetNbinsX()
    bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, nbins + 2)])
    bin_widths = np.array([hist.GetBinWidth(i) for i in range(1, nbins + 1)])
    return bin_edges, bin_widths


def get_hist_arrays(hist, divide_by_bin_width=False, scale=1.0):
    bin_edges, bin_widths = get_bin_edges_widths(hist)
    nbins = hist.GetNbinsX()
    vals = (
        np.array([hist.GetBinContent(i + 1) for i in range(nbins)], dtype=float) * scale
    )
    errs = (
        np.array([hist.GetBinError(i + 1) for i in range(nbins)], dtype=float) * scale
    )
    if divide_by_bin_width:
        vals = np.divide(
            vals, bin_widths, out=np.zeros_like(vals), where=bin_widths != 0
        )
        errs = np.divide(
            errs, bin_widths, out=np.zeros_like(errs), where=bin_widths != 0
        )
    return vals, errs, bin_edges, bin_widths


def integral(hist, divide_by_bin_width=False):
    vals, _, _, bin_widths = get_hist_arrays(hist, divide_by_bin_width)
    if divide_by_bin_width:
        return float(np.sum(vals * bin_widths))
    return float(np.sum(vals))


def compute_total_mc_and_stat_err(
    mc_hists, divide_by_bin_width=False
):  # should add the option for pre-fit unc integration .. maybe it's not so trivial..
    if not mc_hists:
        return None, None
    first_hist = next(iter(mc_hists.values()))
    nbins = first_hist.GetNbinsX()
    total_vals = np.zeros(nbins, dtype=float)
    total_errs2 = np.zeros(nbins, dtype=float)
    for h in mc_hists.values():
        vals, errs, _, _ = get_hist_arrays(h, divide_by_bin_width)
        total_vals += vals
        total_errs2 += errs**2
    return total_vals, np.sqrt(total_errs2)


def choose_reference_binning(
    histograms_dict,
):  # need to chose the reference binning when multiple hists are overlapped
    for name, h in histograms_dict.items():
        if h is None:
            continue
        return h
    return None


def order_mc_contributions(mc_hists, divide_by_bin_width=False):

    names = list(mc_hists.keys())
    # where to define stack order? need to check --> before it was in the order of the samples in the input.yaml file .. now we need to figure out

    # stack_order_cfg = inputs_cfg.get("stack_order", []) if isinstance(inputs_cfg, dict) else []
    # in_order = [n for n in stack_order_cfg if n in names]
    # in_order[::-1]

    in_order = []
    remaining = [n for n in names if n not in in_order]
    remaining_reversed = list(reversed(remaining))
    # order remaining by integral
    # remaining_sorted_by_integral = list(sorted(remaining, key=lambda n: integral(mc_hists[n], divide_by_bin_width)))
    # remaining_sorted_by_integral_reversed = list(sorted(remaining, key=lambda n: integral(mc_hists[n], divide_by_bin_width)), reverse=True)
    return in_order + remaining_reversed


def draw_mc_stack(
    ax, mc_hists, processes_dict, bin_edges, divide_by_bin_width, page_cfg_dict
):
    if not mc_hists:
        return None, None
    order = order_mc_contributions(mc_hists, divide_by_bin_width)
    mc_vals, mc_labels, mc_colors = [], [], []

    for name in order:
        h = mc_hists[name]
        vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width)
        mc_vals.append(vals)
        cfg = processes_dict[name]
        mc_labels.append(cfg.get("name", name))
        mc_colors.append(cfg.get("color_mplhep", "gray"))

    total_mc_vals, total_mc_errs = compute_total_mc_and_stat_err(
        mc_hists, divide_by_bin_width
    )

    hep.histplot(
        mc_vals,
        bins=bin_edges,
        stack=True,
        histtype="fill",
        label=mc_labels,
        facecolor=mc_colors,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    hep.histplot(
        total_mc_vals,
        bins=bin_edges,
        histtype="step",
        color="black",
        linewidth=0.5,
        ax=ax,
    )

    bkg_unc_cfg = page_cfg_dict.get("bkg_unc_hist", {})
    unc_hatch = "//" if bkg_unc_cfg.get("fill_style") == 3013 else None
    unc_alpha = bkg_unc_cfg.get("alpha", 0.35)

    y_up = total_mc_vals + total_mc_errs
    y_dn = total_mc_vals - total_mc_errs
    y_dn = np.maximum(y_dn, 0.0)

    ax.fill_between(
        bin_edges[:-1],
        y_dn,
        y_up,
        step="post",
        facecolor="none",
        edgecolor="black",
        hatch=unc_hatch,
        alpha=unc_alpha,
        linewidth=0.8,
        label=bkg_unc_cfg.get("legend_title", "Bkg. unc."),
    )

    return total_mc_vals, total_mc_errs


def draw_signals(
    ax, signal_hists, processes_dict, bin_edges, divide_by_bin_width, wantSignal
):
    if not wantSignal or not signal_hists:
        return
    for name, h in signal_hists.items():
        cfg = processes_dict[name]
        scale = cfg.get("scale", 1.0)
        vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width, scale)
        label = cfg.get("name", name)
        if scale != 1.0:
            label += f"x{scale}"
        hep.histplot(
            vals,
            bins=bin_edges,
            histtype="step",
            label=label,
            color=cfg.get("color_mplhep", "red"),
            linestyle="--",
            linewidth=1.5,
            ax=ax,
        )


def draw_data(
    ax, data_hist, bin_edges, divide_by_bin_width, wantData=True, blind_region=[]
):
    if not wantData or data_hist is None:
        return None, None
    vals, errs, _, _ = get_hist_arrays(data_hist, divide_by_bin_width)

    if blind_region:
        if len(blind_region) == 2:
            x_min = blind_region[0]
            x_max = blind_region[1]
            mask = (bin_edges[:-1] >= x_min) & (bin_edges[:-1] < x_max)
            vals[mask] = 0.0
            errs[mask] = 0.0
        # to be expanded

    hep.histplot(
        vals,
        bins=bin_edges,
        yerr=errs,
        histtype="errorbar",
        label="Data",
        color="black",
        ax=ax,
    )
    return vals, errs


def draw_ratio(
    ax_ratio,
    bin_edges,
    data_vals,
    data_errs,
    total_mc_vals,
    total_mc_errs,
    x_label,
    blind_region,
):
    if data_vals is None or total_mc_vals is None:
        return

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            data_vals,
            total_mc_vals,
            out=np.zeros_like(data_vals),
            where=total_mc_vals != 0,
        )
        ratio_err = np.abs(
            np.array(
                np.divide(
                    data_errs,
                    total_mc_vals,
                    out=np.zeros_like(data_errs),
                    where=total_mc_vals != 0,
                )
            )
        )
        mc_rel_unc = np.divide(
            total_mc_errs,
            total_mc_vals,
            out=np.zeros_like(total_mc_errs),
            where=total_mc_vals != 0,
        )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    y_up = 1.0 + mc_rel_unc
    y_dn = np.maximum(1.0 - mc_rel_unc, 0.0)

    mask = np.ones_like(ratio, dtype=bool)
    if blind_region and len(blind_region) == 2:
        x_min, x_max = blind_region
        mask = ~((bin_centers >= x_min) & (bin_centers <= x_max))
        blind_mask = ~mask
        ratio[blind_mask] = 0.0
        y_dn[blind_mask] = 0.0
        y_up[blind_mask] = 0.0

    ax_ratio.fill_between(
        bin_centers,
        y_dn,
        y_up,
        where=y_dn > 0,
        step="mid",
        facecolor="ghostwhite",
        edgecolor="black",
        hatch="//",
        alpha=0.5,
        zorder=1,
    )

    ax_ratio.errorbar(
        bin_centers[mask],
        ratio[mask],
        yerr=ratio_err[mask],
        fmt=".",
        color="black",
        markersize=10,
        zorder=2,
    )

    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    delta = 0.5
    if len(ratio[mask]):
        delta = np.abs(ratio[mask] - 1).mean()

    y_max = round(1 + delta, 2)
    y_min = round(1 - delta, 2)
    print(f"y_max = {y_max}")
    print(f"y_min = {y_min}")
    ax_ratio.set_ylim(y_min * 0.9, y_max * 1.1)
    # ax_ratio.yaxis.set_ticks(np.arange(y_min*0.85, y_max*1.15, 1.))
    ax_ratio.set_ylabel("Data/MC")
    ax_ratio.set_xlabel(x_label)
    # for item in ax_ratio.get_yticklabels():
    # item.set_fontsize(10)
    # ax_ratio.set_ylim(0.9,1.1)
    ax_ratio.set_ylabel("Data/MC")
    ax_ratio.set_xlabel(x_label)


def plot_histogram_from_config(
    variable,
    histograms_dict,
    phys_model_dict,
    processes_dict,
    axes_cfg_dict,
    page_cfg_dict,
    page_cfg_custom_dict,
    filename_base,
    period,
    stacked=True,
    compare_mode=False,
    compare_vars_mode=False,
    wantLogX=False,
    wantLogY=False,
    wantData=False,
    wantSignal=False,
    wantRatio=False,
    category=None,
    channel=None,
    group_minor_contributions=False,
    # signal_scale=1.0,         # if you want to scale signals
    # scale_dict=None,         # if you want to plot individual signals
    minor_fraction=0.001,  # percentage for other contributions
):
    hist_cfg = axes_cfg_dict.get(variable, {})
    blind_region = hist_cfg.get("blind_region", [])
    divide_by_bin_width = bool(hist_cfg.get("divide_by_bin_width", False))

    canvas_size = page_cfg_dict["page_setup"].get("canvas_size", [1000, 800])
    ratio_plot = bool(wantData and wantRatio and stacked and not compare_mode)

    fig = plt.figure(figsize=(canvas_size[0] / 100, canvas_size[1] / 100))
    gs = fig.add_gridspec(
        2 if ratio_plot else 1,
        1,
        height_ratios=[3, 1] if ratio_plot else [2],
        hspace=0.05 if ratio_plot else 0.25,
    )
    ax = fig.add_subplot(gs[0])
    fig.subplots_adjust(top=0.85)
    ax_ratio = fig.add_subplot(gs[1], sharex=ax) if ratio_plot else None

    mc_hists = {}
    mc_hists_withMinor = {}
    data_vals = data_errs = total_mc_vals = total_mc_errs = None
    y_max_comp = None
    if compare_mode:
        linestyle_cycle = ["-", "--", ":", "-."]
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        regions = list(histograms_dict.keys())

        first_region = next(iter(histograms_dict.values()))
        ref_hist = choose_reference_binning(first_region)
        if ref_hist is None:
            print(
                "[plot_histogram_from_config] Nessun istogramma valido per il binning in compare_mode."
            )
            return

        _, _, bin_edges, _ = get_hist_arrays(ref_hist, False)

        all_plotted_vals = []

        for i, region in enumerate(regions):
            hist_dict = histograms_dict[region]

            mc_hists_region = {
                k: h
                for k, h in hist_dict.items()
                if k in phys_model_dict.get("backgrounds", [])
            }

            plot_vals = None
            plot_label = ""

            if mc_hists_region:
                total_mc_vals_region, _ = compute_total_mc_and_stat_err(
                    mc_hists_region, divide_by_bin_width
                )
                plot_vals = total_mc_vals_region
                plot_label = f"Total MC: {region}"
            elif hist_dict.get("data") is not None and wantData:
                plot_vals, _, _, _ = get_hist_arrays(
                    hist_dict["data"], divide_by_bin_width
                )
                plot_label = f"Data: {region}"
            elif any(k in phys_model_dict.get("signals", []) for k in hist_dict):
                s_name = next(
                    k
                    for k in hist_dict.keys()
                    if k in phys_model_dict.get("signals", [])
                )
                h = hist_dict[s_name]
                scale = processes_dict.get(s_name, {}).get("scale", 1.0)
                plot_vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width, scale)
                s_label = processes_dict.get(s_name, {}).get("name", s_name)
                if scale != 1.0:
                    s_label += f"x{scale}"
                plot_label = f"{s_label}: {region}"

            if plot_vals is None:
                continue

            all_plotted_vals.append(plot_vals)

            linestyle = linestyle_cycle[i % len(linestyle_cycle)]
            color = color_cycle[i % len(color_cycle)]

            hep.histplot(
                plot_vals,
                bins=bin_edges,
                histtype="step",
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=plot_label,
                ax=ax,
            )

        if all_plotted_vals:
            max_vals = [
                np.max(v[np.isfinite(v)]) for v in all_plotted_vals if v.size > 0
            ]
            y_max_comp = np.max(max_vals) if max_vals else 0

    elif compare_vars_mode:
        linestyle_cycle = ["-", "--", ":", "-.", ":", "-"]
        color_cycle = [
            "blue",
            "green",
            "red",
            "cyan",
        ]  # ['cornflowerblue','pink','orange','green','yellow']#'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        var_styles = {
            var: (
                linestyle_cycle[i % len(linestyle_cycle)],
                color_cycle[i % len(color_cycle)],
            )
            for i, var in enumerate(histograms_dict.keys())
        }
        first_hist = next(iter(histograms_dict.values()))
        _, _, bin_edges, _ = get_hist_arrays(first_hist, False)
        linewidth = len(color_cycle) / 2

        alpha = 0.5
        all_plotted_vals = []

        for var, total_hist in histograms_dict.items():
            if total_hist is None:
                continue

            values = np.array(
                [total_hist.GetBinContent(i + 1) for i in range(total_hist.GetNbinsX())]
            )
            all_plotted_vals.append(values)
            style = var_styles.get(var)
            hep.histplot(
                values,
                bins=bin_edges,
                histtype="step",
                color=style[1],
                linewidth=linewidth,
                label=var,
                ax=ax,
                linestyle=style[0],
                alpha=alpha,
            )
            linewidth -= 0.2 if linewidth > 0 else 0.1
            alpha += 0.25 / len(color_cycle)

        if all_plotted_vals:
            max_vals = [
                np.max(v[np.isfinite(v)]) for v in all_plotted_vals if v.size > 0
            ]
            y_max_comp = np.max(max_vals) if max_vals else 0

    else:
        mc_hists, signal_hists, data_hist = {}, {}, None
        for contrib, hist in histograms_dict.items():
            if hist is None:
                continue
            if contrib == "data":
                data_hist = hist
            elif contrib in phys_model_dict["signals"]:
                signal_hists[contrib] = hist
            elif contrib in phys_model_dict["backgrounds"]:
                mc_hists[contrib] = hist
            else:
                print(f"ref not found for {contrib}")

        ref_hist = None
        if mc_hists:
            ref_hist = next(iter(mc_hists.values()))
        elif data_hist is not None:
            ref_hist = data_hist
        elif signal_hists:
            ref_hist = next(iter(signal_hists.values()))
        else:
            ref_hist = choose_reference_binning(histograms_dict)

        if ref_hist is None:
            print("[plot_histogram_from_config] Nessun istogramma valido trovato.")
            return

        _, _, bin_edges, _ = get_hist_arrays(ref_hist, False)

        total_mc_vals = total_mc_errs = None

        mc_hists_withMinor = mc_hists
        mc_order_withMinor = []

        if group_minor_contributions:
            integrals = {c: mc_hists[c].Integral() for c in mc_hists}
            total = sum(integrals.values())
            threshold = minor_fraction * total
            minor_contribs = [c for c, val in integrals.items() if val < threshold]

            major_contribs = [
                c for c in histograms_dict.keys() if c not in minor_contribs
            ]
            print(major_contribs)

            objsToMerge = ROOT.TList()
            other_hist = mc_hists[minor_contribs[0]]
            for minor_contrib in minor_contribs[1:]:
                objsToMerge.Add(mc_hists[minor_contrib])
            other_hist.Merge(objsToMerge)
            mc_hists_withMinor = {
                c: mc_hists[c] for c in mc_hists if c not in minor_contribs
            }
            mc_hists_withMinor["Other"] = other_hist
            mc_order_withMinor = ["Other"] + major_contribs

        if mc_hists_withMinor and stacked:
            total_mc_vals, total_mc_errs = draw_mc_stack(
                ax,
                mc_hists_withMinor,
                processes_dict,
                bin_edges,
                divide_by_bin_width,
                page_cfg_dict,
            )
        elif mc_hists_withMinor and not stacked:
            total_mc_vals, total_mc_errs = compute_total_mc_and_stat_err(
                mc_hists_withMinor, divide_by_bin_width
            )
            hep.histplot(
                total_mc_vals,
                bins=bin_edges,
                histtype="fill",
                facecolor="gray",
                alpha=0.35,
                ax=ax,
            )
            for name, h in mc_hists_withMinor.items():
                vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width)
                cfg = processes_dict[name]
                hep.histplot(
                    vals,
                    bins=bin_edges,
                    histtype="step",
                    label=cfg.get("title", name),
                    color=cfg.get("color_mplhep", "black"),
                    linewidth=2,
                    ax=ax,
                )

        draw_signals(
            ax, signal_hists, processes_dict, bin_edges, divide_by_bin_width, wantSignal
        )

        data_vals = data_errs = None
        if data_hist is not None and wantData:
            data_vals, data_errs = draw_data(
                ax, data_hist, bin_edges, divide_by_bin_width, wantData, blind_region
            )

    ax.set_ylabel(hist_cfg.get("y_title", "Events"), fontsize=28)
    if not ratio_plot:
        ax.set_xlabel(hist_cfg.get("x_title", variable), fontsize=28)
    else:
        ax.get_xaxis().set_visible(False)

    ax.set_yscale("log" if wantLogY else "linear")
    ax.set_xscale("log" if wantLogX else "linear")

    y_max = None
    if compare_mode or compare_vars_mode:
        y_max = y_max_comp
    elif mc_hists_withMinor:
        if total_mc_vals is None:
            total_mc_vals, _ = compute_total_mc_and_stat_err(
                mc_hists_withMinor, divide_by_bin_width
            )
        y_max = (
            np.max(total_mc_vals)
            if total_mc_vals is not None and len(total_mc_vals)
            else None
        )
    elif signal_hists:
        first_signal = next(iter(signal_hists.values()))
        vals, _, _, _ = get_hist_arrays(first_signal, False)
        y_max = np.max(vals) if vals is not None and len(vals) else None

    if y_max is not None and np.isfinite(y_max) and y_max > 0:
        max_factor = (
            hist_cfg.get("max_y_sf", 1.2)
            if not wantLogY
            else (10 ** (hist_cfg.get("max_y_sf", 1.0)))
        )
        ax.set_ylim(top=y_max * max_factor)
        if wantLogY:
            y_min_log = max(0.1, y_max * 1e-4)
            ax.set_ylim(bottom=y_min_log)

    ax.set_xlim(bin_edges[0] * 0.99, bin_edges[-1] * 1.01)

    legend_cfg = page_cfg_dict.get("legend_mplhep", {})
    ax.legend(
        loc="upper right",
        facecolor=legend_cfg.get("fill_color_mplhep", "white"),
        frameon=bool(legend_cfg.get("border_size", 0) == 0),
        fontsize=legend_cfg.get("text_size", 0.02) * 60,
        framealpha=0.0,
        ncol=legend_cfg.get("ncols", 2),
        handleheight=1.5,
        labelspacing=0.5,
    )

    if ratio_plot:
        if data_vals is not None and total_mc_vals is not None:
            draw_ratio(
                ax_ratio,
                bin_edges,
                data_vals,
                data_errs,
                total_mc_vals,
                total_mc_errs,
                x_label=hist_cfg.get("x_title", variable),
                blind_region=blind_region,
            )
    text_box_names = page_cfg_dict["page_setup"].get("text_boxes_mplhep", [])
    text_box_cfg = {name: page_cfg_dict.get(name, {}) for name in text_box_names}
    try:
        resolved_positions = resolve_text_positions(text_box_cfg)
    except NameError:
        print("Warning: resolve_text_positions not defined. Using default positions.")
        resolved_positions = {}

    for name in text_box_names:
        cfg = text_box_cfg[name]
        pos = resolved_positions.get(name, [0.02, 1.05])
        if cfg.get("type") == "cms_mplhep":
            hep.cms.label(
                label="Preliminary",
                data="data" in histograms_dict,
                ax=ax,
                loc=0,
                com=cfg.get("com", "13.6 TeV"),
                lumi=cfg.get("lumi", lumi_dict.get(period, "")),
                # lumi=round(cfg.get("lumi", lumi_dict.get(period, "")),1),
                year=period.split("_")[1] if "_" in period else "Unknown",
                fontsize=cfg.get("text_size", 16),
            )
        else:
            text_content = cfg.get("text", "")
            text_content = text_content.format(
                category=category, channel=channel, variable=variable
            )
            ax.text(
                pos[0],
                pos[1],
                text_content,
                transform=ax.transAxes,
                fontsize=cfg.get("text_size", 14),
                ha="left",
                va="top",
            )

    plt.savefig(f"{filename_base}.pdf", bbox_inches="tight")
    print(f"Plot saved to {filename_base}.pdf")
    plt.savefig(f"{filename_base}.png", bbox_inches="tight")
    print(f"Plot saved to {filename_base}.png")
    plt.close()
