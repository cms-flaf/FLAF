import os
import re
import yaml
import json

from FLAF.RunKit.envToJson import get_cmsenv
from FLAF.RunKit.law_wlcg import WLCGFileSystem
from FLAF.RunKit.grid_tools import gfal_ls

def select_items(all_items, filters):
    def name_match(name, pattern):
        if pattern[0] == '^':
            return re.match(pattern, name) is not None
        return name == pattern

    selected_items = { c for c in all_items }
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
        pattern = item_filter[len(prefix):]
        if len(pattern) == 0:
            raise RuntimeError(f'Filter with an empty pattern expression.')

        to_move = [ item for item in items_from if name_match(item, pattern) ]
        if len(to_move) > 0:
            used_filters.add(item_filter)
            for column in to_move:
                items_from.remove(column)
                items_to.add(column)

    unused_filters = set(filters) - used_filters
    if len(unused_filters) > 0:
        print("Unused filters: " + " ".join(unused_filters))

    return list(sorted(selected_items))

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
    if customisations is None or len(customisations) == 0: return
    if type(customisations) == str:
        customisations = customisations.split(';')
    if type(customisations) != list:
        raise RuntimeError(f'Invalid type of customisations: {type(customisations)}')
    for customisation in customisations:
        substrings = customisation.split('=')
        if len(substrings) != 2 :
            raise RuntimeError("len of substring is not 2!")
        value = substrings[-1]
        key_entries = substrings[0].split('.')
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

        user_config_path = os.path.join(ana_path, 'config', 'user_custom.yaml')
        with open(user_config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        sample_config_path = os.path.join(ana_path, 'FLAF', 'config', period, 'samples.yaml')
        self.analysis_config_area= os.path.join(ana_path, user_config['analysis_config_area'])
        ana_global_config_path = os.path.join(self.analysis_config_area, 'global.yaml')
        ana_sample_config_path = os.path.join(self.analysis_config_area, period, 'samples.yaml')
        weights_config_path = os.path.join(ana_path, 'config', period, 'weights.yaml')

        with open(sample_config_path, 'r') as f:
            sample_config = yaml.safe_load(f)

        if os.path.exists(ana_sample_config_path):
            with open(ana_sample_config_path, 'r') as f:
                ana_sample_config = yaml.safe_load(f)
        else:
            ana_sample_config = {}

        if os.path.exists(ana_global_config_path):
            with open(ana_global_config_path, 'r') as f:
                ana_global_config = yaml.safe_load(f)
        else:
            ana_global_config = {}

        with open(weights_config_path, 'r') as f:
            weights_config = yaml.safe_load(f)

        self.weights_config = weights_config

        self.global_params = load_parameters(user_config, ana_sample_config.get('GLOBAL', {}),
                                             ana_global_config, sample_config.get('GLOBAL', {}))
        apply_customisations(self.global_params, customisations)

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

        samples = load_parameters(ana_sample_config, sample_config, keys_to_ignore={'GLOBAL'})
        for sample_name, sample_entry in samples.items():
            if type(sample_entry) != dict:
                raise RuntimeError(f'Invalid sample definition period="{period}", sample_name="{sample_name}"')
        selected_samples = select_items(samples.keys(), self.global_params.get('sampleSelection', []))
        if sample: # If you define a user sample, then only that sample is selected
            selected_samples = select_items(selected_samples, ['drop ^.*', f'keep {sample}'])
        self.samples = { key : samples[key] for key in selected_samples }

        self.fs_dict = {}

        #self.hist_config_path = os.path.join(ana_path, 'config', 'plot','histograms.yaml')
        self.hist_config_path = os.path.join(self.analysis_config_area, 'plot', 'histograms.yaml')
        # self.hist_config_path = os.path.join(self.analysis_config_area, 'plot','histograms.yaml')
        self.hists_ = None

        self.background_config_path = os.path.join(self.analysis_config_area, 'background_samples.yaml')
        self.backgrounds_ = None

        self.cmssw_env_ = None

        self.signal_samples = [ key for key in self.samples if self.samples[key]['sampleType'] in self.global_params['signal_types'] ]

        self.anaTupleFiles = {}


    def get_fs(self, fs_name):
        if fs_name not in self.fs_dict:
            full_fs_name = f'fs_{fs_name}'
            if full_fs_name in self.global_params:
                name_cand = None
                if type(self.global_params[full_fs_name]) == str:
                    name_cand = self.global_params[full_fs_name]
                elif len(self.global_params[full_fs_name]) == 1:
                    name_cand = self.global_params[full_fs_name][0]
                if name_cand is not None and name_cand.startswith('/'):
                    self.fs_dict[fs_name] = name_cand
                else:
                    self.fs_dict[fs_name] = WLCGFileSystem(self.global_params[full_fs_name])
            else:
                if fs_name == 'default':
                    raise RuntimeError(f'No default file system defined')
                self.fs_dict[fs_name] = self.get_fs('default')
        return self.fs_dict[fs_name]

    @property
    def cmssw_env(self):
        if self.cmssw_env_ is None:
            self.cmssw_env_ = get_cmsenv(cmssw_path=os.getenv("FLAF_CMSSW_BASE"))
            for var in [ 'HOME', 'FLAF_PATH', 'ANALYSIS_PATH', 'ANALYSIS_DATA_PATH', 'X509_USER_PROXY',
                         'FLAF_CMSSW_BASE', 'FLAF_CMSSW_ARCH' ]:
                if var in os.environ:
                    self.cmssw_env_[var] = os.environ[var]
            if 'PYTHONPATH' not in self.cmssw_env_:
                self.cmssw_env_['PYTHONPATH'] = self.ana_path
            else:
                self.cmssw_env_['PYTHONPATH'] = f'{self.ana_path}:{self.cmssw_env["PYTHONPATH"]}'
        return self.cmssw_env_

    @property
    def hists(self):
        if self.hists_ is None:
            with open(self.hist_config_path, 'r') as f:
                self.hists_ = yaml.safe_load(f)
        return self.hists_

    @property
    def backgrounds(self):
        if self.backgrounds_ is None:
            with open(self.background_config_path, 'r') as f:
                self.backgrounds_ = yaml.safe_load(f)
        return self.backgrounds_

    @staticmethod
    def getGlobal(ana_path, period, sample, customisations=None):
        key = (ana_path, period, sample, customisations)
        if key not in Setup._global_instances:
            Setup._global_instances[key] = Setup(ana_path, period, sample, customisations)
        return Setup._global_instances[key]



    def getAnaTupleFileList(self, sample_name, remote_file):
        if sample_name in self.anaTupleFiles.keys():
            return self.anaTupleFiles[sample_name]
        else:
            with remote_file.localize('r') as f:
                with open(f.path, 'r') as this_file:
                    json_dict = json.load(this_file)
                    sample_dict = json_dict
                    self.anaTupleFiles[sample_name] = sample_dict
            return self.anaTupleFiles[sample_name]
            
