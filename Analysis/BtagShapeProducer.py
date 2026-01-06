import numpy as np
import Analysis.hh_bbww as analysis
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os
import Analysis.hh_bbww as analysis


class BtagShapeProducer:
    def __init__(self, cfg, payload_name):
        assert "isData" in cfg, "BtagShapeProducer config must contain `isData` field."
        self.payload_name = payload_name
        self.cfg = cfg
        if len(self.cfg["jet_multiplicities"]) == 0:
            raise RuntimeError(f"Illegal `jet_multiplicities` {self.cfg["jet_multiplicities"]}.")
        self.vars_to_save = []
        if not cfg["isData"]:
            self.vars_to_save = [
                "ncentralJet",
                "weight_noBtag",
                "weight_total"
            ]
            self.vars_to_save.extend(self.cfg["lepton_categories"])

    def prepare_dfw(self, dfw):
        # 1. what arguments do I pass to GetWeight?
        # it's a string representing a formula for the total weight
        # in terms of branches of the tree
        if not self.cfg["isData"]:
            total_weight = analysis.GetWeight(
                None, None, None, apply_btag_shape_weights=False
            )
            dfw.df = dfw.df.Define("weight_noBtag", f"return {total_weight}")
            dfw.df = dfw.df.Define(
                "weight_total", f"return {total_weight} * weight_bTagShape_Central"
            )
            dfw.df = dfw.df.Define("e", f"return channelId == 1;")
            dfw.df = dfw.df.Define("mu", f"return channelId == 2;")
            dfw.df = dfw.df.Define("eE", f"return channelId == 11;")
            dfw.df = dfw.df.Define("eMu", f"return channelId == 12;")
            dfw.df = dfw.df.Define("muMu", f"return channelId == 22;")
        return dfw

    def run(self, array, keep_all_columns=False):
        res = {}
        # data does not have weight_bTagShape_Central branch
        if self.cfg["isData"]:
            for cat in self.cfg["lepton_categories"]:
                category_dict = {}
                for jet_multiplicity in self.cfg["jet_multiplicities"]:
                    category_dict[f"weight_noBtag_ncentralJet_{jet_multiplicity}"] = 1.0
                    category_dict[f"weight_total_ncentralJet_{jet_multiplicity}"] = 1.0
                res[cat] = category_dict
            return res

        ncentralJet = array["ncentralJet"]
        weights_noBtag = array["weight_noBtag"]
        weights_total = array["weight_total"]
        # here there also should be a loop over categories: [eE, eMu, muMu] for DL and [e, mu] for SL
        for cat in self.cfg["lepton_categories"]:
            category_mask = array[cat]
            category_dict = {}
            for jet_multiplicity in self.cfg["jet_multiplicities"]:
                # calculate number of events and total btag shape weight
                events_with_jet_multiplicity_mask = (
                    array["ncentralJet"] == jet_multiplicity
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
