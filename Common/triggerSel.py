import yaml


class Triggers:
    dict_legtypes = {"Electron": "Leg::e", "Muon": "Leg::mu", "Tau": "Leg::tau"}

    def __init__(self, triggerFile, deltaR_matching=0.4):
        with open(triggerFile, "r") as stream:
            self.trigger_dict = yaml.safe_load(stream)
        self.deltaR_matching = deltaR_matching

    def ApplyTriggers(self, df, offline_legs, isData=False, applyTriggerFilter=False):
        hltBranches = []
        matchedObjectsBranches = []
        for path, path_dict in self.trigger_dict.items():
            path_key = "path"
            if "path" not in path_dict:
                path_key += "_data" if isData else "_MC"
            if isData:
                for p in path_dict[path_key]:
                    if p not in df.GetColumnNames():
                        df = df.Define(p, "false")
            or_paths = " || ".join(f"({p})" for p in path_dict[path_key])
            or_paths = f" ( { or_paths } ) "
            for leg_id, leg_tuple in enumerate(path_dict["legs"]):
                leg_dict_offline = leg_tuple["offline_obj"]
                for obj in offline_legs:
                    offline_cut = leg_dict_offline["cut"].format(obj=obj,muon_pt=leg_dict_offline.get("muon_pt","pt_nano"))
                    var_name_offline = f"{obj}_offlineCut_{leg_id+1}_{path}"
                    df = df.Define(var_name_offline, offline_cut)
                    leg_dict_online = leg_tuple["online_obj"]
                    var_name_online = f"{obj}_onlineCut_{leg_id+1}_{path}"
                    cut_vars = []
                    cuts = (
                        leg_dict_online["cuts"]
                        if "cuts" in leg_dict_online
                        else [{"cut": leg_dict_online["cut"]}]
                    )
                    for cut_idx, online_cut in enumerate(cuts):
                        preCondition = online_cut.get("preCondition", "true")
                        cut_var_name = f"{obj}_onlineCut_{leg_id+1}_{path}_{cut_idx}"
                        df = df.Define(
                            cut_var_name,
                            f"!({preCondition}) || (({preCondition}) && ({online_cut['cut']}))",
                        )
                        cut_vars.append(cut_var_name)
                    df = df.Define(var_name_online, " && ".join(cut_vars))
                    matching_var = f"{obj}_Matching_{leg_id+1}_{path}"
                    df = df.Define(
                        matching_var,
                        f"""FindMatching({var_name_offline}, {var_name_online}, {obj}_p4, TrigObj_p4, {self.deltaR_matching} )""",
                    )
                    matchedObjectsBranches.append(matching_var)
            for obj in offline_legs:
                matching_var_bool = f"{obj}_HasMatching_{path}"
                matching_var_bool_str = " || ".join(
                    f"{obj}_Matching_{i+1}_{path} > -1"
                    for i in range(len(path_dict["legs"]))
                )
                df = df.Define(matching_var_bool, matching_var_bool_str)
                df = df.Redefine(matching_var_bool, f"Any({matching_var_bool}>0)")
                matchedObjectsBranches.append(matching_var_bool)
            fullPathSelection = f"{or_paths}"
            hltBranch = f"HLT_{path}"
            hltBranches.append(hltBranch)
            df = df.Define(hltBranch, fullPathSelection)
        total_or_string = " || ".join(hltBranches)
        if applyTriggerFilter:
            df = df.Filter(total_or_string, "trigger application")
        hltBranches.extend(matchedObjectsBranches)
        return df, hltBranches
