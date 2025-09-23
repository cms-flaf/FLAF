import os
import re
import yaml
import json
import copy

from FLAF.RunKit.envToJson import get_cmsenv
from FLAF.RunKit.law_wlcg import WLCGFileSystem
from FLAF.RunKit.grid_tools import gfal_ls


def select_items(all_items, filters):
    def name_match(name, pattern):
        if pattern[0] == "^":
            return re.match(pattern, name) is not None
        return name == pattern

    selected_items = {c for c in all_items}
    excluded_items = set()
    keep_prefix = "keep "
    drop_prefix = "drop "
    used_filters = set()
    for item_filter in filters:
        if item_filter.startswith(keep_prefix):
            keep = True
            items_from = excluded_items
            items_to = selected_items
            prefix = keep_prefix
        elif item_filter.startswith(drop_prefix):
            keep = False
            items_from = selected_items
            items_to = excluded_items
            prefix = drop_prefix
        else:
            raise RuntimeError(f'Unsupported filter = "{item_filter}".')
        pattern = item_filter[len(prefix) :]
        if len(pattern) == 0:
            raise RuntimeError(f"Filter with an empty pattern expression.")

        to_move = [item for item in items_from if name_match(item, pattern)]
        if len(to_move) > 0:
            used_filters.add(item_filter)
            for column in to_move:
                items_from.remove(column)
                items_to.add(column)

    unused_filters = set(filters) - used_filters
    if len(unused_filters) > 0:
        print("Unused filters: " + " ".join(unused_filters))

    return list(sorted(selected_items))


def select_processes(samples, phys_model, processes):
    new_samples = {}
    selected_processes = []
    if "backgrounds" not in phys_model or "signals" not in phys_model:
        raise RuntimeError(
            f"Phys model must contain 'backgrounds' and 'signals' keys. {phys_model}"
        )
    for bkg_process in phys_model.get("backgrounds", []):
        if bkg_process not in processes:
            raise RuntimeError(
                f"Background process '{bkg_process}' not found in processes configuration."
            )
        for bkg_name in processes.get(bkg_process, {}).get("datasets", []):
            if bkg_name not in samples.keys():
                raise RuntimeError(
                    f"Background sample '{bkg_name}' not found in samples configuration."
                )
            new_samples[bkg_name] = samples[bkg_name]
            new_samples[bkg_name]["process_name"] = bkg_process
            new_samples[bkg_name]["process_group"] = "backgrounds"
            selected_processes.append(bkg_name)
    for sig_process in phys_model.get("signals", []):
        if sig_process not in processes:
            raise RuntimeError(
                f"Signal process '{sig_process}' not found in processes configuration."
            )
        for sig_name in processes.get(sig_process, {}).get("datasets", []):
            if sig_name not in samples.keys():
                raise RuntimeError(
                    f"Signal sample '{sig_name}' not found in samples configuration."
                )
            new_samples[sig_name] = samples[sig_name]
            new_samples[sig_name]["process_name"] = sig_process
            new_samples[sig_name]["process_group"] = "signals"
            selected_processes.append(sig_name)
    for data_process in phys_model.get("data", []):
        if data_process not in processes:
            raise RuntimeError(
                f"Data process '{data_process}' not found in processes configuration."
            )
        for data_name in processes.get(data_process, {}).get("datasets", []):
            if data_name not in samples.keys():
                raise RuntimeError(
                    f"Data sample '{data_name}' not found in samples configuration."
                )
            new_samples[data_name] = samples[data_name]
            new_samples[data_name]["process_name"] = data_process
            new_samples[data_name]["process_group"] = "data"
            selected_processes.append(data_name)

    return new_samples, selected_processes


def load_parameters(*sources, keys_to_ignore=None):
    all_keys = set()
    output = {}
    keys_to_ignore = keys_to_ignore or set()
    for source in sources:
        all_keys.update(source.keys())
    for key in all_keys:
        if key not in keys_to_ignore:
            for source in sources:
                if key in source:
                    output[key] = source[key]
                    break
    return output


def apply_customisations(config_dict, customisations):
    if customisations is None or len(customisations) == 0:
        return
    if type(customisations) == str:
        customisations = customisations.split(";")
    if type(customisations) != list:
        raise RuntimeError(f"Invalid type of customisations: {type(customisations)}")
    for customisation in customisations:
        substrings = customisation.split("=")
        if len(substrings) != 2:
            raise RuntimeError("len of substring is not 2!")
        value = substrings[-1]
        key_entries = substrings[0].split(".")
        cfg_entry = config_dict
        for key in key_entries[:-1]:
            cfg_entry = cfg_entry[key]
        entry_type = type(cfg_entry[key_entries[-1]])
        cfg_entry[key_entries[-1]] = entry_type(value)
        #     if key in config_dict.keys():
        #         cfg_entry = cfg_entry[key]
        # if key_entries[-1] in config_dict.keys():
        #     entry_type = type(cfg_entry[key_entries[-1]])
        #     cfg_entry[key_entries[-1]] = entry_type(value)


class Setup:
    _global_instances = {}

    def __init__(self, ana_path, period, sample="", customisations=None):
        self.ana_path = ana_path
        self.period = period

        user_config_path = os.path.join(ana_path, "config", "user_custom.yaml")
        with open(user_config_path, "r") as f:
            user_config = yaml.safe_load(f)
        sample_config_path = os.path.join(
            ana_path, "FLAF", "config", period, "samples.yaml"
        )

        self.analysis_config_area = os.path.join(
            ana_path, user_config["analysis_config_area"]
        )
        ana_global_config_path = os.path.join(self.analysis_config_area, "global.yaml")
        ana_sample_config_path = os.path.join(
            self.analysis_config_area, period, "samples.yaml"
        )
        weights_config_path = os.path.join(ana_path, "config", period, "weights.yaml")

        with open(sample_config_path, "r") as f:
            sample_config = yaml.safe_load(f)

        if os.path.exists(ana_sample_config_path):
            with open(ana_sample_config_path, "r") as f:
                ana_sample_config = yaml.safe_load(f)
        else:
            ana_sample_config = {}

        if os.path.exists(ana_global_config_path):
            with open(ana_global_config_path, "r") as f:
                ana_global_config = yaml.safe_load(f)
        else:
            ana_global_config = {}

        with open(weights_config_path, "r") as f:
            weights_config = yaml.safe_load(f)

        self.weights_config = weights_config

        self.global_params = load_parameters(
            user_config,
            ana_sample_config.get("GLOBAL", {}),
            ana_global_config,
            sample_config.get("GLOBAL", {}),
        )
        apply_customisations(self.global_params, customisations)

        self.phys_model = None  # Init nones so we can check later if exists
        self.processes = None
        if "phys_model" not in self.global_params:
            raise RuntimeError("'phys_model' not defined in global configuration.")
        phys_models_path = os.path.join(ana_path, "config", "phys_models.yaml")
        processes_path = os.path.join(ana_path, "config", period, "processes.yaml")

        with open(phys_models_path, "r") as f:
            phys_models = yaml.safe_load(f)

        with open(processes_path, "r") as f:
            processes = yaml.safe_load(f)

        phys_model_name = self.global_params.get("phys_model", "BaseModel")
        self.phys_model = phys_models.get(phys_model_name)
        self.processes = {}
        for key, item in processes.items():
            if item.get("is_meta_process", False):
                new_process_names_for_model = []
                meta_setup = item["meta_setup"]
                dataset_name_pattern = meta_setup["dataset_name_pattern"]
                candidates = {}
                plot_color_idx = 0  # Used for indexing signal 'to_plot' colors
                for dataset in item["datasets"]:
                    cand_key = re.match(dataset_name_pattern, dataset).groups()
                    if len(cand_key) != len(meta_setup["parameters"]):
                        raise RuntimeError(
                            f"Dataset '{dataset}' does not match pattern '{dataset_name_pattern}'."
                        )
                    if not cand_key in candidates:
                        candidates[cand_key] = []
                    candidates[cand_key].append(dataset)
                for cand_key, datasets in candidates.items():
                    new_process = copy.deepcopy(item)
                    proc_name = meta_setup["process_name"]
                    plot_name = meta_setup["name_pattern"]
                    for i, param in enumerate(meta_setup["parameters"]):
                        proc_name = proc_name.replace(f"${{{param}}}", str(cand_key[i]))
                        plot_name = plot_name.replace(f"${{{param}}}", str(cand_key[i]))
                    new_process["process_name"] = proc_name
                    new_process["datasets"] = datasets
                    new_process["name"] = plot_name
                    new_process["to_plot"] = (
                        int(cand_key[0]) in new_process["meta_setup"]["to_plot"]
                    )
                    new_process["color"] = "kBlack"
                    if new_process["to_plot"]:
                        new_process["color"] = new_process["meta_setup"]["plot_color"][
                            plot_color_idx
                        ]
                        plot_color_idx += 1
                    del new_process["meta_setup"]
                    del new_process["is_meta_process"]
                    self.processes[proc_name] = new_process
                    new_process_names_for_model.append(proc_name)

                for group in self.phys_model:
                    if key in self.phys_model[group]:
                        key_index = self.phys_model[group].index(key)
                        for new_process_name in reversed(new_process_names_for_model):
                            self.phys_model[group].insert(key_index, new_process_name)
                        self.phys_model[group].remove(key)

            else:
                self.processes[key] = item

        # create payload -> what producer delivers it
        # will be used to check if cache is needed
        self.var_producer_map = {}
        payload_producers_cfg = self.global_params.get("payload_producers")
        if payload_producers_cfg:
            for producer_name, producer_config in payload_producers_cfg.items():
                columns_delivered = producer_config.get("columns")
                if columns_delivered:
                    for col in columns_delivered:
                        self.var_producer_map[f"{producer_name}_{col}"] = producer_name

        samples = load_parameters(
            ana_sample_config, sample_config, keys_to_ignore={"GLOBAL"}
        )
        for sample_name, sample_entry in samples.items():
            if type(sample_entry) != dict:
                raise RuntimeError(
                    f'Invalid sample definition period="{period}", sample_name="{sample_name}"'
                )

        # Return samples (with process_name key) and selected list to keep similar format to old method
        # Honestly should only return samples later after version 1
        samples, selected_samples = select_processes(
            samples, self.phys_model, self.processes
        )

        if sample:  # If you define a user sample, then only that sample is selected
            selected_samples = select_items(
                selected_samples, ["drop ^.*", f"keep {sample}"]
            )

        self.samples = {key: samples[key] for key in selected_samples}

        self.fs_dict = {}

        # self.hist_config_path = os.path.join(ana_path, 'config', 'plot','histograms.yaml')
        self.hist_config_path = os.path.join(
            self.analysis_config_area, "plot", "histograms.yaml"
        )
        # self.hist_config_path = os.path.join(self.analysis_config_area, 'plot','histograms.yaml')
        self.hists_ = None

        self.cmssw_env_ = None

        self.signal_samples = [
            key
            for key in self.samples
            if self.samples[key]["process_group"] == "signals"
        ]

        self.anaTupleFiles = {}

    def _create_fs_instance(self, path_or_paths):
        path_to_check = None
        if isinstance(path_or_paths, list):
            if not path_or_paths:
                raise ValueError("List of paths cannot be empty.")
            path_to_check = path_or_paths[0]  # Use the first to determine the FS type
        elif isinstance(path_or_paths, str):
            path_to_check = path_or_paths
        else:
            raise TypeError(
                f"Unsupported path type: {type(path_or_paths)}. Expected str or list of str."
            )

        if path_to_check.startswith("/"):
            return path_to_check
        else:
            return WLCGFileSystem(path_or_paths)

    def get_fs(self, fs_name, custom_paths=None):
        fs_instance = None

        if fs_name in self.fs_dict:
            return self.fs_dict[fs_name]

        if custom_paths is not None:
            try:
                fs_instance = self._create_fs_instance(custom_paths)
                self.fs_dict[fs_name] = fs_instance  # cache it
                return fs_instance
            except (TypeError, ValueError) as e:
                print(f"Error = {e}.")
                fs_instance = None

        if fs_instance is None:
            full_fs_name = f"fs_{fs_name}"
            if full_fs_name in self.global_params:
                param_value = self.global_params[full_fs_name]
                try:
                    fs_instance = self._create_fs_instance(param_value)
                except (TypeError, ValueError) as e:
                    print(f"Error = {e}.")
                    raise RuntimeError(
                        f"Invalid FS configuration for '{fs_name}' in global_params."
                    )
            else:
                if fs_name == "default":
                    raise RuntimeError(
                        f"No default file system defined in global_params or via custom_paths. "
                        f'Please define "fs_default" in your configuration or provide a custom_paths for "default".'
                    )
                fs_instance = self.get_fs("default")

            # if nothing works, nothing works
            if fs_instance is None:
                raise RuntimeError(f"Could not determine file system for '{fs_name}'.")
            self.fs_dict[fs_name] = fs_instance

            return self.fs_dict[fs_name]

    @property
    def cmssw_env(self):
        if self.cmssw_env_ is None:
            self.cmssw_env_ = get_cmsenv(cmssw_path=os.getenv("FLAF_CMSSW_BASE"))
            for var in [
                "HOME",
                "FLAF_PATH",
                "ANALYSIS_PATH",
                "ANALYSIS_DATA_PATH",
                "X509_USER_PROXY",
                "FLAF_CMSSW_BASE",
                "FLAF_CMSSW_ARCH",
            ]:
                if var in os.environ:
                    self.cmssw_env_[var] = os.environ[var]
            if "PYTHONPATH" not in self.cmssw_env_:
                self.cmssw_env_["PYTHONPATH"] = self.ana_path
            else:
                self.cmssw_env_["PYTHONPATH"] = (
                    f'{self.ana_path}:{self.cmssw_env["PYTHONPATH"]}'
                )
        return self.cmssw_env_

    @property
    def hists(self):
        if self.hists_ is None:
            with open(self.hist_config_path, "r") as f:
                self.hists_ = yaml.safe_load(f)
        return self.hists_


    @staticmethod
    def getGlobal(ana_path, period, sample, customisations=None):
        key = (ana_path, period, sample, customisations)
        if key not in Setup._global_instances:
            Setup._global_instances[key] = Setup(
                ana_path, period, sample, customisations
            )
        return Setup._global_instances[key]

    def getAnaTupleFileList(self, sample_name, remote_file):
        if sample_name in self.anaTupleFiles.keys():
            return self.anaTupleFiles[sample_name]
        else:
            with remote_file.localize("r") as f:
                with open(f.path, "r") as this_file:
                    json_dict = json.load(this_file)
                    sample_dict = json_dict
                    self.anaTupleFiles[sample_name] = sample_dict
            return self.anaTupleFiles[sample_name]
