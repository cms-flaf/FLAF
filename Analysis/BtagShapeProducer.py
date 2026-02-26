import numpy as np
import Analysis.hh_bbww as analysis
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os
import importlib


class BtagShapeProducer:
    def __init__(self, cfg, payload_name, *args):
        self.payload_name = payload_name
        self.cfg = cfg
        self.weight_module = importlib.import_module(cfg["weight_module"])

    def prepare_dfw(self, dfw, dataset):
        self.vars_to_save = [
            "weight_total",
        ]
        self.vars_to_save.extend(self.cfg["bins"].keys())

        total_weight = self.weight_module.GetWeight(None, None, None)
        dfw.df = dfw.df.Define("weight_total", f"return {total_weight}")

        cols = dfw.df.GetColumnNames()
        btag_cols = [
            c for c in cols if "weight_bTagShape" in c and c.endswith("_double")
        ]
        self.entry_names_noBtag = []
        for c in btag_cols:
            syst = c.split("_")[-2]
            branch_name = f"weight_bTagShape_{syst}_double"

            json_record_name = f"weight_noBtag_{syst}"
            if json_record_name not in self.entry_names_noBtag:
                self.entry_names_noBtag.append(json_record_name)
                self.vars_to_save.append(json_record_name)
            dfw.df = dfw.df.Define(
                json_record_name,
                f"return {branch_name} != 0.0 ? {total_weight} / {branch_name} : 0.0",
            )
        for bin_name, bin_def in self.cfg["bins"].items():
            dfw.df = dfw.df.Define(bin_name, f"return {bin_def}")
        return dfw

    def run(self, array, keep_all_columns=False):
        res = {}
        weights_total = array["weight_total"]
        for bin_name in self.cfg["bins"]:
            mask = array[bin_name]
            res[f"weight_total_{bin_name}"] = float(np.sum(weights_total[mask]))
            for syst_noBtag in self.entry_names_noBtag:
                weights_noBtag = array[syst_noBtag]
                res[f"{syst_noBtag}_{bin_name}"] = float(np.sum(weights_noBtag[mask]))
        return res

    def combine(
        self,
        *,
        final_dict,
        new_dict,
    ):
        if final_dict is None:
            final_dict = {key: new_dict[key] for key in new_dict.keys()}
        else:
            for key in final_dict.keys():
                final_dict[key] += new_dict[key]
        return final_dict

    def create_dfw(
        self,
        *,
        df,
        setup,
        dataset_name,
        histTupleDef,
        unc_cfg_dict,
        uncName,
        uncScale,
        final_weight_name,
        df_is_central,
        isData,
    ):
        histTupleDef.Initialize()
        histTupleDef.analysis_setup(setup)

        dfw = histTupleDef.GetDfw(df, setup, dataset_name)
        histTupleDef.DefineWeightForHistograms(
            dfw=dfw,
            isData=isData,
            uncName=uncName,
            uncScale=uncScale,
            unc_cfg_dict=unc_cfg_dict,
            hist_cfg_dict=setup.hists,
            global_params=setup.global_params,
            final_weight_name=final_weight_name,
            df_is_central=df_is_central,
        )

        return dfw
