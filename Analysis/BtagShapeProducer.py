import numpy as np
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os


class BtagShapeProducer:
    def __init__(self, cfg, payload_name, *args):
        self.payload_name = payload_name
        self.cfg = cfg

    def prepare_dfw(self, dfw, dataset):
        self.vars_to_save = [
            "weight_total",
            "weight_noSF",
        ]
        self.vars_to_save.extend(self.cfg["bins"].keys())

        total_weight = "final_weight"
        dfw.df = dfw.df.Define("weight_total", f"return {total_weight}")

        cols = dfw.df.GetColumnNames()
        col_names = {str(c) for c in cols}
        btag_cols = [
            c for c in cols if "weight_bTagShape" in c and c.endswith("_double")
        ]

        # The yield with no b-tag shape SF applied at all, sum(w0). It is the numerator of
        # the renormalisation and is the same for every variation, so it is stored once
        # rather than per systematic. final_weight carries exactly one b-tag SF, always
        # under the name weight_bTagShape_Central -- in a JES pass Corrections/btag.py
        # aliases the jes-varied SF to that name -- so dividing it out is correct for
        # every pass, central and shape alike.
        applied_sf = "weight_bTagShape_Central"
        if applied_sf not in col_names:
            raise RuntimeError(
                f"{applied_sf} is not defined, so the b-tag shape SF that final_weight "
                "carries cannot be divided out and no normalisation can be derived. "
                "This producer needs the btag correction in 'shape' mode at the "
                "AnalysisCache stage."
            )
        dfw.df = dfw.df.Define(
            "weight_noSF",
            f"return {applied_sf} != 0.0 ? {total_weight} / {applied_sf} : 0.0",
        )
        self.entry_names_withSF = []
        for c in btag_cols:
            syst = c.split("_")[-2]

            # The yield this variation actually produces, sum(w0 * SF_syst), which is the
            # denominator of the renormalisation: the factor has to be 1/<SF_syst>.
            # Summing final_weight / SF_syst instead gives <1/SF_syst>, which Jensen's
            # inequality makes larger -- and larger for the up and the down variation
            # alike. Combine reads that common mode as a quadratic shape term, and the
            # up and down impacts then land on the same side of nominal.
            #
            # final_weight already carries SF_central, so the relative branch turns it
            # into w0 * SF_syst. Central and the JES passes have no _rel branch, being
            # the SF that is applied already, so there final_weight is the sum wanted.
            rel_branch = f"weight_bTagShape_{syst}_rel"
            json_record_name = f"weight_withSF_{syst}"
            if json_record_name not in self.entry_names_withSF:
                self.entry_names_withSF.append(json_record_name)
                self.vars_to_save.append(json_record_name)
                dfw.df = dfw.df.Define(
                    json_record_name,
                    (
                        f"return {total_weight} * {rel_branch}"
                        if rel_branch in col_names
                        else f"return {total_weight}"
                    ),
                )
        for bin_name, bin_def in self.cfg["bins"].items():
            dfw.df = dfw.df.Define(bin_name, f"return {bin_def}")
        return dfw

    def run(self, array, keep_all_columns=False):
        res = {}
        weights_total = array["weight_total"]
        weights_noSF = array["weight_noSF"]
        for bin_name in self.cfg["bins"]:
            mask = array[bin_name]
            res[f"weight_total_{bin_name}"] = float(np.sum(weights_total[mask]))
            res[f"weight_noSF_{bin_name}"] = float(np.sum(weights_noSF[mask]))
            for syst_withSF in self.entry_names_withSF:
                weights_withSF = array[syst_withSF]
                res[f"{syst_withSF}_{bin_name}"] = float(np.sum(weights_withSF[mask]))
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
