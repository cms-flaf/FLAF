import numpy as np
import Analysis.hh_bbww as analysis
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os
import Analysis.hh_bbww as analysis


class BtagShapeProducer:
    def __init__(self, cfg, payload_name, *args):
        self.payload_name = payload_name
        self.cfg = cfg
        if len(self.cfg["jet_multiplicities"]) == 0:
            raise RuntimeError(
                f"Illegal `jet_multiplicities` {self.cfg["jet_multiplicities"]}."
            )
        self.lep_category_definitions = {cat: int(val) for cat, val in self.cfg["lepton_categories"].items()}
        self.vars_to_save = []
        self.vars_to_save = [
            "ncentralJet",
            "weight_noBtag",
            "weight_total",
        ]
        self.vars_to_save.extend(self.lep_category_definitions.keys())

    def prepare_dfw(self, dfw, dataset):
        total_weight = analysis.GetWeight(None, None, None)
        dfw.df = dfw.df.Define("weight_total", f"return {total_weight}")
        dfw.df = dfw.df.Define(
            "weight_noBtag", f"return weight_bTagShape_Central != 0.0 ? {total_weight} / weight_bTagShape_Central : 0.0"
        )
        for lep_cat_name, lep_cut_val in self.lep_category_definitions.items():
            dfw.df = dfw.df.Define(lep_cat_name, f"return channelId == {lep_cut_val};")
        return dfw

    def run(self, array, keep_all_columns=False):
        res = {}
        # data does not have weight_bTagShape_Central branch
        lepton_category_names = self.lep_category_definitions.keys()
        ncentralJet = array["ncentralJet"]
        weights_noBtag = array["weight_noBtag"]
        weights_total = array["weight_total"]
        # here there also should be a loop over categories: [eE, eMu, muMu] for DL and [e, mu] for SL
        for cat in lepton_category_names:
            category_mask = array[cat]
            category_dict = {}
            for jet_multiplicity in self.cfg["jet_multiplicities"]:
                # calculate number of events and total btag shape weight
                events_with_jet_multiplicity_mask = (
                    ncentralJet == jet_multiplicity
                )
                mask = np.logical_and(category_mask, events_with_jet_multiplicity_mask)
                category_dict[f"weight_noBtag_ncentralJet_{jet_multiplicity}"] = float(
                    np.sum(weights_noBtag[mask])
                )
                category_dict[f"weight_total_ncentralJet_{jet_multiplicity}"] = float(
                    np.sum(weights_total[mask])
                )
            res[cat] = category_dict
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
        Utilities.InitializeCorrections(setup, dataset_name, stage="AnalysisCache")
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