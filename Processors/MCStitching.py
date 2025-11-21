import yaml
import os
import sys

from FLAF.Common.CrossSectionDB import CrossSectionDB

class MCStitcher:
    """
    Processor for stitching MC samples using set of orthogonal bins with known cross-sections.
    If additional quantities that are not stored in nanoAOD are needed, one should inherit from this class and overload the defineVariables method.
    """

    def __init__(self, *, global_params, processor_entry, stage, verbose=0):
        self.global_params = global_params
        self.processor_entry = processor_entry
        self.verbose = verbose
        self.bins = []

        if stage not in ["AnaCache", "AnaTuple"]:
            raise RuntimeError(f"Unsupported stage: {stage}")

        config_path = os.path.join(os.environ["ANALYSIS_PATH"], processor_entry["config"])
        if not os.path.exists(config_path):
            raise RuntimeError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.xs_db = CrossSectionDB.Load(os.environ["ANALYSIS_PATH"], global_params["crossSectionsFile"])

        if "crossSections" in cfg:
            for entry_name, entry in cfg["crossSections"].items():
                self.xs_db.addEntry(entry_name, entry)
        self.bins = []
        for bin_number, bin_entry in enumerate(cfg["bins"]):
            for key in [ "name", "selection", "crossSection" ]:
                if key not in bin_entry:
                    msg = f"MCStitcher: missing '{key}' for bin "
                    if "name" in bin_entry:
                        msg += f"'{bin_entry['name']}'"
                    else:
                        msg += f"number {bin_number}"
                    raise RuntimeError(msg)
            value = self.xs_db.evaluateExpression(bin_entry["crossSection"], entry_name=f'MCStitching/{bin_entry["name"]}')
            if verbose > 0:
                print(f"[MCStitcher] Bin '{bin_entry['name']}': selection = {bin_entry['selection']}, crossSection = {value}", file=sys.stderr)
            bin_entry["crossSectionValue"] = value
            self.bins.append(bin_entry)

        if len(self.bins) == 0:
            raise RuntimeError("MCStitcher: no bins defined in configuration")

        totalCrossSectionFromBins = sum( bin["crossSectionValue"] for bin in self.bins )
        if "totalCrossSection" in cfg:
            totalCrossSectionFromConfig = self.xs_db.evaluateExpression(cfg["totalCrossSection"], entry_name="MCStitching/totalCrossSection")
            if abs(totalCrossSectionFromBins - totalCrossSectionFromConfig) / totalCrossSectionFromConfig > 0.001:
                raise RuntimeError(f"MCStitcher: sum of bin cross-sections ({totalCrossSectionFromBins}) does not match total cross-section ({totalCrossSectionFromConfig})")
        self.totalCrossSection = totalCrossSectionFromBins

    def defineVariables(self, df):
        """Define any additional variables needed for stitching.
        """
        return df

    def onAnaCache_initializeDenomEntry(self):
        return { bin["name"]: [] for bin in self.bins }

    def onAnaCache_updateDenomEntry(self, entry, df, output_branch_name, weights_to_apply):
        weight_formula = '*'.join(weights_to_apply) if len(weights_to_apply) > 0 else '1.0'
        for bin in self.bins:
            bin_selection = bin["selection"]
            df_bin = df.Filter(bin_selection)
            df_bin = df_bin.Define(output_branch_name, weight_formula)
            entry[bin["name"]].append(df_bin.Sum(output_branch_name))
        return entry

    def onAnaCache_materializeDenomEntry(self, entry):
        for bin in self.bins:
            bin_name = bin["name"]
            entry[bin_name] = [ x.GetValue() if type(x) != float else x for x in entry[bin_name]  ]
        return entry

    def onAnaCache_finalizeDenomEntry(self, entry):
        for bin in self.bins:
            bin_name = bin["name"]
            entry[bin_name] = sum(entry[bin_name])
        return entry

    def onAnaCache_combineAnaCaches(self, entries):
        cmb_entry = { bin["name"]: 0.0 for bin in self.bins }
        for entry in entries:
            for bin in self.bins:
                bin_name = bin["name"]
                cmb_entry[bin_name] += entry[bin_name]
        return cmb_entry

    def onAnaTuple_defineCrossSection(self, df, crossSectionBranch, xs_db, dataset_name, dataset_entry):
        xs_expression = ""
        for bin_cfg in self.bins:
            bin_selection = bin_cfg["selection"]
            bin_xs = bin_cfg["crossSectionValue"]
            xs_expression += f"if({bin_selection}) return {bin_xs};\n"
        xs_expression += f'throw std::runtime_error("No bin matched in MCStitcher for dataset {dataset_name}");'
        print(f"Cross-section expression for {dataset_name}:")
        print(xs_expression)
        print("-" * 16)
        return df.Define(crossSectionBranch, xs_expression)

    def onAnaTuple_defineDenominator(self, df, denomBranch, processor_name, dataset_name, source_name, scale_name, ana_caches):
        denom = {}
        for bin in self.bins:
            bin_name = bin["name"]
            denom[bin_name] = 0.
            for ana_cache in ana_caches.values():
                denom[bin_name] += ana_cache["denominator"][source_name][scale_name][processor_name][bin_name]
        denom_expression = ""
        for bin_cfg in self.bins:
            bin_selection = bin_cfg["selection"]
            bin_name = bin_cfg["name"]
            bin_denom = denom[bin_name]
            denom_expression += f"if({bin_selection}) return {bin_denom};\n"
        print(f"Denominator expression for {dataset_name} source={source_name} scale={scale_name}:")
        denom_expression += f'throw std::runtime_error("No bin matched in MCStitcher for dataset {dataset_name}");'
        print(denom_expression)
        print("-" * 16)
        return df.Define(denomBranch, denom_expression)
