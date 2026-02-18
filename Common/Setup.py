import os
import re
import yaml
import json
import copy

from FLAF.RunKit.envToJson import get_cmsenv
from FLAF.RunKit.law_wlcg import WLCGFileSystem
from FLAF.Common.Utilities import create_processor_instances


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


class Config:
    def __init__(self, name, paths, file_names, special_items_prefix="."):
        self.name = name
        yaml_str = ""
        self.considered_paths = []
        for path in paths:
            for file_name in file_names:
                full_path = os.path.join(path, file_name)
                self.considered_paths.append(full_path)
                if os.path.exists(full_path):
                    with open(full_path, "r") as f:
                        yaml_str += f.read() + "\n"
        if len(yaml_str) == 0:
            raise RuntimeError(
                f"No configuration files {file_names} found in paths {paths}."
            )
        config_dict = yaml.safe_load(yaml_str)
        if special_items_prefix is not None:
            config_dict = {
                k: v
                for k, v in config_dict.items()
                if not k.startswith(special_items_prefix)
            }
        self.config_dict = config_dict

    def __getitem__(self, key):
        value = self.config_dict.get(key, None)
        if value is None:
            raise KeyError(f'Key "{key}" not found in {self.name} configuration.')
        return value

    def __setitem__(self, key, value):
        self.config_dict[key] = value

    def __contains__(self, key):
        return key in self.config_dict

    def get(self, key, default=None):
        return self.config_dict.get(key, default)

    def keys(self):
        return self.config_dict.keys()

    def items(self):
        return self.config_dict.items()

    def values(self):
        return self.config_dict.values()


class PhysicsModel:
    allowed_process_types = set(["backgrounds", "signals", "data"])
    other_attributes = set(["name"])

    def __init__(self, name, model_dict):
        self.name = name
        extra_keys = (
            model_dict.keys()
            - PhysicsModel.allowed_process_types
            - PhysicsModel.other_attributes
        )
        if len(extra_keys) > 0:
            raise RuntimeError(
                f"Physics model '{name}' contains invalid process types: {extra_keys}"
            )
        self._processes = {}
        for key in PhysicsModel.allowed_process_types:
            items = model_dict.get(key, [])
            for item in items:
                if item in self._processes:
                    raise RuntimeError(
                        f"Process '{item}' is defined multiple times in physics model '{name}'."
                    )
                self._processes[item] = key
        self._base_processes = {}

    def processes(self, process_type=None):
        if process_type is None:
            return list(self._processes.keys())
        return [
            proc for proc, p_type in self._processes.items() if p_type == process_type
        ]

    def process_type(self, process_name):
        if process_name not in self._processes:
            raise RuntimeError(
                f"Process '{process_name}' not found in physics model '{self.name}'."
            )
        return self._processes[process_name]

    def replace_process(
        self, old_process_name, new_process_names, ignore_missing=False
    ):
        if old_process_name not in self._processes:
            if ignore_missing:
                return
            raise RuntimeError(
                f"Process '{old_process_name}' not found in physics model '{self.name}'."
            )
        process_type = self._processes[old_process_name]
        del self._processes[old_process_name]
        for new_process_name in new_process_names:
            if new_process_name in self._processes:
                raise RuntimeError(
                    f"Process '{new_process_name}' already exists in physics model '{self.name}'."
                )
            self._processes[new_process_name] = process_type

    def select_processes(self, filters):
        orig = set(self._processes.keys())
        selected = set(select_items(orig, filters))
        if len(selected) == 0:
            orig_base = []
            for bases in self._base_processes.values():
                orig_base.extend(bases)
            selected_base = set(select_items(set(orig_base), filters))
            if len(selected_base) == 0:
                raise RuntimeError(
                    f"No processes selected in physics model '{self.name}' with filters: {filters}"
                )
            selected = set()
            for process_name, base_process_names in self._base_processes.items():
                if any(base in selected_base for base in base_process_names):
                    selected.add(process_name)
            if len(selected) != 1:
                raise RuntimeError(
                    f"Multiple processes selected via base processes in physics model '{self.name}' with filters: {filters}"
                )
        else:
            selected_base = None
        to_remove = set(self._processes.keys()) - selected
        for process_name in to_remove:
            del self._processes[process_name]
            if process_name in self._base_processes:
                del self._base_processes[process_name]
        if selected_base is not None:
            for process_name, base_process_names in self._base_processes.items():
                new_bases = [
                    base for base in base_process_names if base in selected_base
                ]
                self._base_processes[process_name] = new_bases

    def set_base_processes(self, process_name, base_process_names):
        if process_name not in self._processes:
            raise RuntimeError(
                f"Process '{process_name}' not found in physics model '{self.name}'."
            )
        self._base_processes[process_name] = base_process_names

    def base_processes(self, process_name):
        if process_name not in self._processes:
            raise RuntimeError(
                f"Process '{process_name}' not found in physics model '{self.name}'."
            )
        if process_name not in self._base_processes:
            raise RuntimeError(f"Base processes for '{process_name}' are not set.")
        return self._base_processes[process_name]


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

    def __init__(
        self,
        ana_path,
        period,
        law_run_version,
        custom_process_selection=None,
        custom_dataset_selection=None,
        custom_model_selection=None,
        customisations=None,
    ):
        self.ana_path = ana_path
        self.period = period
        self.law_run_version = law_run_version

        self.config_path_order = [
            os.path.join(ana_path, "FLAF", "config"),
            os.path.join(ana_path, "FLAF", "config", period),
            os.path.join(ana_path, "config"),
            os.path.join(ana_path, "config", period),
        ]

        self.global_params = Config(
            "global", self.config_path_order, ["global.yaml", "user_custom.yaml"]
        )

        apply_customisations(self.global_params, customisations)

        phys_models = Config(
            "phys_models", self.config_path_order, ["phys_models.yaml"]
        )
        phys_model_name = self.global_params["phys_model"]
        if custom_model_selection is not None:
            phys_model_name = custom_model_selection
        print(f"Using physics model: {phys_model_name}")
        self.phys_model = PhysicsModel(phys_model_name, phys_models[phys_model_name])

        processes_config = Config(
            "processes", self.config_path_order, ["processes.yaml"]
        )
        processes = {}
        for key, item in processes_config.items():
            if item.get("is_meta_process", False):
                new_process_names_for_model = []
                meta_setup = item["meta_setup"]
                dataset_name_pattern = meta_setup["dataset_name_pattern"]
                candidates = {}
                plot_color_idx = 0  # Used for indexing signal 'to_plot' colors
                for dataset in item["datasets"]:
                    cand_key = tuple(
                        str(tmp_key)
                        for tmp_key in re.match(dataset_name_pattern, dataset).groups()
                    )
                    if len(cand_key) != len(meta_setup["parameters"]):
                        raise RuntimeError(
                            f"Dataset '{dataset}' does not match pattern '{dataset_name_pattern}'."
                        )
                    if not cand_key in candidates:
                        candidates[cand_key] = []
                    candidates[cand_key].append(dataset)

                tmp_to_plot = []
                for local_to_plot in item["meta_setup"]["to_plot"]:
                    x = tuple(str(local_item) for local_item in local_to_plot)
                    tmp_to_plot.append(x)

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
                    new_process["to_plot"] = cand_key in tmp_to_plot
                    new_process["color"] = "kBlack"
                    if new_process["to_plot"]:
                        new_process["color"] = new_process["meta_setup"]["plot_color"][
                            plot_color_idx
                        ]
                        plot_color_idx += 1
                    new_process["channels"] = (
                        self.global_params["channelSelection"]
                        if "channels" not in meta_setup.keys()
                        else meta_setup["channels"]
                    )
                    del new_process["meta_setup"]
                    del new_process["is_meta_process"]
                    processes[proc_name] = new_process
                    new_process_names_for_model.append(proc_name)

                self.phys_model.replace_process(
                    key, new_process_names_for_model, ignore_missing=True
                )
            else:
                processes[key] = item

        def collect_base_processes(p_name, parent_name=None):
            if p_name not in processes:
                if parent_name is not None:
                    msg = f"Process '{p_name}' defined as sub_process of '{parent_name}' not found in processes configuration."
                else:
                    msg = f"Process '{p_name}' defined in physics model '{phys_model_name}' not found in processes configuration."
                raise RuntimeError(msg)
            process = processes[p_name]
            if "sub_processes" in process:
                if "datasets" in process:
                    raise RuntimeError(
                        f"Process '{p_name}' cannot have both 'datasets' and 'sub_processes' defined."
                    )
                base_processes = []
                for sub_process in process["sub_processes"]:
                    base_processes.extend(collect_base_processes(sub_process, p_name))
                return base_processes
            else:
                return [p_name]

        for process_name in self.phys_model.processes():
            base_processes = collect_base_processes(process_name)
            self.phys_model.set_base_processes(process_name, base_processes)
        if custom_process_selection is not None:
            if type(custom_process_selection) == str:
                custom_process_selection = custom_process_selection.split(",")
            filters = ["drop ^.*"] + [
                f"keep {pattern}" for pattern in custom_process_selection
            ]
            self.phys_model.select_processes(filters)
        self.base_processes = {}
        self.parent_processes = {}
        for process_name in self.phys_model.processes():
            for b_process_name in self.phys_model.base_processes(process_name):
                b_process = processes[b_process_name]
                b_process["parent_process"] = process_name
                self.base_processes[b_process_name] = b_process
                self.parent_processes[process_name] = processes[process_name]

        all_datasets = Config("datasets", self.config_path_order, ["datasets.yaml"])
        active_datasets = {}
        for process_name, process in self.base_processes.items():
            for dataset_name in process.get("datasets", []):
                if dataset_name not in all_datasets.keys():
                    raise RuntimeError(
                        f"Dataset '{dataset_name}' for process '{process_name}' not found in datasets configuration."
                    )
                active_datasets[dataset_name] = all_datasets[dataset_name]
                active_datasets[dataset_name]["process_name"] = process_name
                active_datasets[dataset_name]["process_group"] = (
                    self.phys_model.process_type(process["parent_process"])
                )
        if custom_dataset_selection is not None:
            if type(custom_dataset_selection) == str:
                custom_dataset_selection = custom_dataset_selection.split(",")
            filters = ["drop ^.*"] + [
                f"keep {pattern}" for pattern in custom_dataset_selection
            ]
            selected_datasets = select_items(set(active_datasets.keys()), filters)
            if len(selected_datasets) == 0:
                raise RuntimeError(
                    f"No datasets selected with a custom selection: {custom_dataset_selection}"
                )
            active_datasets = {key: active_datasets[key] for key in selected_datasets}
        self.datasets = active_datasets

        for process_name, process in (
            self.base_processes | self.parent_processes
        ).items():
            if "datasets" in process:
                process["datasets"] = [
                    ds for ds in process["datasets"] if ds in self.datasets
                ]

        # create payload -> what producer delivers it
        # will be used to check if cache is needed
        self.var_producer_map = {}
        for producer_name, producer_config in self.global_params.get(
            "payload_producers", {}
        ).items():
            columns_delivered = producer_config.get("columns")
            if columns_delivered:
                for col in columns_delivered:
                    self.var_producer_map[f"{producer_name}_{col}"] = producer_name

        self.weights_config = Config(
            "weights", self.config_path_order, ["weights.yaml"]
        )
        self.fs_dict = {}
        self.hists_ = None
        self.cmssw_env_ = None
        self.anaTupleFiles = {}
        self.processors_cache = {}
        self.fs_das_ = None

    def get_processors(self, process_name, stage, create_instances=False):
        key = (process_name, stage, create_instances)
        if key not in self.processors_cache:
            if process_name in self.base_processes:
                process = self.base_processes[process_name]
            elif process_name in self.parent_processes:
                process = self.parent_processes[process_name]
            else:
                raise RuntimeError(
                    f"Process '{process_name}' not found in base processes."
                )
            all_processors = process.get("processors", {})
            stage_processors = []
            for entry in all_processors:
                if stage in entry["stages"]:
                    stage_processors.append(entry)
            if create_instances:
                processors = create_processor_instances(
                    self.global_params, stage_processors, stage
                )
                value = (stage_processors, processors)
            else:
                value = stage_processors
            self.processors_cache[key] = value
        return self.processors_cache[key]

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
            cfg = self.global_params.get("WLCGFileSystem", {})
            cache_validity = cfg.get("localPathCacheValidity", 600)
            host = cfg.get("remotePathCacheHost", None)
            port = cfg.get("remotePathCachePort", None)
            verbose = cfg.get("verbose", 0)
            return WLCGFileSystem(
                path_or_paths,
                local_path_cache_validity_period=cache_validity,
                path_cache_host=host,
                path_cache_port=port,
                verbose=verbose,
            )

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
    def fs_das(self):
        if self.fs_das_ is None:
            self.fs_das_ = WLCGFileSystem("DAS")
        return self.fs_das_

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
            self.hists_ = Config(
                "histograms", self.config_path_order, ["plot/histograms.yaml"]
            )
        return self.hists_

    @staticmethod
    def getGlobal(
        ana_path,
        period,
        law_run_version,
        custom_process_selection=None,
        custom_dataset_selection=None,
        custom_model_selection=None,
        customisations=None,
    ):
        key = (
            ana_path,
            period,
            law_run_version,
            custom_process_selection,
            custom_dataset_selection,
            custom_model_selection,
            customisations,
        )
        if key not in Setup._global_instances:
            Setup._global_instances[key] = Setup(
                ana_path,
                period,
                law_run_version,
                custom_process_selection=custom_process_selection,
                custom_dataset_selection=custom_dataset_selection,
                custom_model_selection=custom_model_selection,
                customisations=customisations,
            )
        return Setup._global_instances[key]

    def getAnaTupleFileList(self, dataset_name, remote_file):
        if dataset_name in self.anaTupleFiles.keys():
            return self.anaTupleFiles[dataset_name]
        else:
            with remote_file.localize("r") as f:
                with open(f.path, "r") as this_file:
                    json_dict = json.load(this_file)
                    dataset_dict = json_dict
                    self.anaTupleFiles[dataset_name] = dataset_dict
            return self.anaTupleFiles[dataset_name]
