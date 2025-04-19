import law
import os
import yaml
import contextlib
import luigi
import threading


from FLAF.RunKit.run_tools import ps_call
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
from FLAF.run_tools.law_customizations import Task, HTCondorWorkflow, copy_param,get_param_value
from FLAF.AnaProd.tasks import AnaTupleTask, DataMergeTask, AnaCacheTupleTask, DataCacheMergeTask, AnaCacheTask

import importlib

unc_cfg_dict = None
def load_unc_config(unc_cfg):
    global unc_cfg_dict
    with open(unc_cfg, 'r') as f:
        unc_cfg_dict = yaml.safe_load(f)
    return unc_cfg_dict


def getYear(period):
    year_dict = {
        'Run2_2016_HIPM':'2016_HIPM',
        'Run2_2016':'2016',
        'Run2_2017':'2017',
        'Run2_2018':'2018',
        'Run3_2022':'2022',
        'Run3_2022EE':'2022EE',
        'Run3_2023':'2023',
        'Run3_2023BPix':'2023BPix',
    }
    return year_dict[period]

def parseVarEntry(var_entry):
    if type(var_entry) == str:
        var_name = var_entry
        need_cache = False
    else:
        var_name = var_entry['name']
        need_cache = var_entry.get('need_cache', False)
    return var_name, need_cache

def GetSamples(samples, backgrounds, signals=['GluGluToRadion','GluGluToBulkGraviton']):
    global samples_to_consider
    samples_to_consider = ['data']

    for sample_name in samples.keys():
        sample_type = samples[sample_name]['sampleType']
        if sample_type in signals or sample_name in backgrounds:
            samples_to_consider.append(sample_name)
    return samples_to_consider

def getCustomisationSplit(customisations):
    customisation_dict = {}
    if customisations is None or len(customisations) == 0: return {}
    if type(customisations) == str:
        customisations = customisations.split(';')
    if type(customisations) != list:
        raise RuntimeError(f'Invalid type of customisations: {type(customisations)}')
    for customisation in customisations:
        substrings = customisation.split('=')
        if len(substrings) != 2 :
            raise RuntimeError("len of substring is not 2!")
        customisation_dict[substrings[0]] = substrings[1]
    return customisation_dict


class HistProducerFileTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    cacheClass = None
    cacheDataClass = None
    def __init__(self, *args, **kwargs):
        super(HistProducerFileTask, self).__init__(*args, **kwargs)

        if self.cacheClass == None:
            self.use_ana_cache = 'analysis_cache_import' in self.setup.global_params
            if self.use_ana_cache:
                file, className, dataClassName = self.setup.global_params['analysis_cache_import'].split(':')
                cacheModule = importlib.import_module(file)
                HistProducerFileTask.cacheClass = getattr(cacheModule, className)
                HistProducerFileTask.cacheDataClass = getattr(cacheModule, dataClassName)

    def workflow_requires(self):
        need_data = False
        need_data_cache = False
        branch_set = set()
        branch_set_cache = set()
        for idx, (sample, br, var, need_cache) in self.branch_map.items():
            if sample == 'data':
                need_data = True
                if need_cache:
                    need_data_cache = True
            else:
                branch_set.add(br)
                if need_cache:
                    branch_set_cache.add(br)
        reqs = {}

        isbbtt = 'HH_bbtautau' in self.global_params['analysis_config_area'].split('/')

        if len(branch_set) > 0:
            reqs['anaTuple'] = AnaTupleTask.req(self, branches=tuple(branch_set),customisations=self.customisations)
        if len(branch_set_cache) > 0:
            if isbbtt:
                reqs['anaCacheTuple'] = AnaCacheTupleTask.req(self, branches=tuple(branch_set_cache),customisations=self.customisations)
            else:
                reqs['analysisCache'] = self.cacheClass.req(self, branches=tuple(branch_set_cache),customisations=self.customisations)
        if need_data:
            reqs['dataMergeTuple'] = DataMergeTask.req(self, branches=(),customisations=self.customisations)
        if need_data_cache:
            if isbbtt:
                reqs['dataCacheMergeTuple'] = DataCacheMergeTask.req(self, branches=(),customisations=self.customisations)
            else:
                reqs['analysisCacheMerge'] = self.cacheClass.req(self, branches=(),customisations=self.customisations)
        return reqs

    def requires(self):
        sample_name, prod_br, var, need_cache = self.branch_data
        deps = []

        isbbtt = 'HH_bbtautau' in self.global_params['analysis_config_area'].split('/')

        if sample_name =='data':
            deps.append(DataMergeTask.req(self, max_runtime=DataMergeTask.max_runtime._default, branch=prod_br, branches=(prod_br,),customisations=self.customisations))
            if need_cache:
                if isbbtt:
                    deps.append(DataCacheMergeTask.req(self, max_runtime=DataCacheMergeTask.max_runtime._default, branch=prod_br, branches=(prod_br,),customisations=self.customisations))
                else:
                    deps.append(self.cacheDataClass.req(self, max_runtime=self.cacheDataClass.max_runtime._default, branch=prod_br, branches=(prod_br,),customisations=self.customisations))
        else:
            deps.append(AnaTupleTask.req(self, max_runtime=AnaTupleTask.max_runtime._default, branch=prod_br, branches=(prod_br,),customisations=self.customisations))
            if need_cache:
                if isbbtt:
                    deps.append(AnaCacheTupleTask.req(self, max_runtime=AnaCacheTupleTask.max_runtime._default, branch=prod_br, branches=(prod_br,),customisations=self.customisations))
                else:
                    deps.append(self.cacheClass.req(self, max_runtime=self.cacheClass.max_runtime._default, branch=prod_br, branches=(prod_br,),customisations=self.customisations))
        return deps

    def create_branch_map(self):
        n = 0
        branches = {}
        anaProd_branch_map = AnaTupleTask.req(self, branch=-1, branches=()).create_branch_map()
        samples_to_consider = GetSamples(self.samples, self.setup.backgrounds,self.global_params['signal_types'] )
        for var_entry in self.global_params['vars_to_plot']:
            var_name, need_cache = parseVarEntry(var_entry)
            for prod_br,(sample_id, sample_name, sample_type, input_file) in anaProd_branch_map.items():
                isData = self.samples[sample_name]['sampleType'] == 'data'
                if sample_name not in samples_to_consider or isData: continue
                branches[n] = (sample_name, prod_br, var_name, need_cache)
                n += 1
            branches[n] = ('data', 0, var_name, need_cache)
            n += 1
        return branches

    def output(self):
        sample_name, prod_br, var, need_cache = self.branch_data
        outFileName = os.path.basename(self.input()[0].path)
        prod_dir = 'prod'
        output_path = os.path.join(self.version, self.period, prod_dir, var, f'{sample_name}_{outFileName}')
        return self.remote_target(output_path,  fs=self.fs_histograms)

    def run(self):
        sample_name, prod_br, var, need_cache = self.branch_data
        input_file = self.input()[0]
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = customisation_dict['channels'] if 'channels' in customisation_dict.keys() else self.global_params['channelSelection']
        #Channels from the yaml are a list, but the format we need for the ps_call later is 'ch1,ch2,ch3', basically join into a string separated by comma
        if type(channels) == list:
            channels = ','.join(channels)
        #bbww does not use a deepTauVersion
        deepTauVersion = ''
        isbbtt = 'HH_bbtautau' in self.global_params['analysis_config_area'].split('/')
        if isbbtt: deepTauVersion = customisation_dict['deepTauVersion'] if 'deepTauVersion' in customisation_dict.keys() else self.global_params['deepTauVersion']
        region = customisation_dict['region'] if 'region' in customisation_dict.keys() else self.global_params['region_default']
        print(f'input file is {input_file.path}')
        global_config = os.path.join(self.ana_path(), self.global_params['analysis_config_area'], f'global.yaml')
        unc_config = os.path.join(self.ana_path(), 'config',self.period, f'weights.yaml')
        sample_type = self.samples[sample_name]['sampleType'] if sample_name != 'data' else 'data'
        HistProducerFile = os.path.join(self.ana_path(), 'FLAF', 'Analysis', 'HistProducerFile.py')
        print(f'output file is {self.output().path}')
        compute_unc_histograms = customisation_dict['compute_unc_histograms']=='True' if 'compute_unc_histograms' in customisation_dict.keys() else self.global_params.get('compute_unc_histograms', False)
        with input_file.localize("r") as local_input, self.output().localize("w") as local_output:
            HistProducerFile_cmd = [ 'python3', HistProducerFile,
                                    '--inFile', local_input.path, '--outFileName',local_output.path,
                                    '--dataset', sample_name, '--uncConfig', unc_config,
                                    '--histConfig', self.setup.hist_config_path, '--sampleType', sample_type, '--globalConfig', global_config, '--var', var, '--period', self.period, '--region', region, '--channels', channels]
            if compute_unc_histograms:
                HistProducerFile_cmd.extend(['--compute_rel_weights', 'True', '--compute_unc_variations', 'True'])
                #HistProducerFile_cmd.extend(['--compute_rel_weights', 'True', '--compute_unc_variations', 'False'])
            if (deepTauVersion!="2p1") and (deepTauVersion!=''):
                HistProducerFile_cmd.extend([ '--deepTauVersion', deepTauVersion])
            if need_cache:
                anaCache_file = self.input()[1]
                print(anaCache_file)
                with anaCache_file.localize("r") as local_anacache:
                    HistProducerFile_cmd.extend(['--cacheFile', local_anacache.path])
                    ps_call(HistProducerFile_cmd, verbose=1)
            else:
                ps_call(HistProducerFile_cmd, verbose=1)



class HistProducerSampleTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 2.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def workflow_requires(self):
        branch_set = set()
        for br_idx, (sample_name, dep_br_list, var) in self.branch_map.items():
            branch_set.update(dep_br_list)
        branches = tuple(branch_set)
        deps = { "HistProducerFileTask": HistProducerFileTask.req(self, branches=branches,customisations=self.customisations) }
        return deps


    def requires(self):
        sample_name, dep_br_list, var = self.branch_data
        reqs = [
                HistProducerFileTask.req(self, max_runtime=HistProducerFileTask.max_runtime._default,
                                                 branch=dep_br, branches=(dep_br,),customisations=self.customisations)
                for dep_br in dep_br_list
            ]
        return reqs


    def create_branch_map(self):
        branches = {}
        histProducerFile_map = HistProducerFileTask.req(self,branch=-1, branches=()).create_branch_map()
        all_samples = {}
        samples_to_consider = GetSamples(self.samples, self.setup.backgrounds,self.global_params['signal_types'] )
        for n_branch, (sample_name, prod_br, var, need_cache)  in histProducerFile_map.items():
            if sample_name not in samples_to_consider: continue
            if sample_name not in all_samples:
                all_samples[sample_name] = {}
            if var not in all_samples[sample_name]:
                all_samples[sample_name][var]=[]
            all_samples[sample_name][var].append(n_branch)
        k = 0
        for sample_name, sample_entry in all_samples.items():
            for var, branch_idx_list in sample_entry.items():
                branches[k] = (sample_name, branch_idx_list, var)
                k += 1
        return branches

    def output(self):
        sample_name, idx_list, var  = self.branch_data
        split_dir = 'split'
        output_path = os.path.join(self.version, self.period, split_dir, var, f'{sample_name}.root')
        return self.remote_target(output_path,  fs=self.fs_histograms)

    def run(self):
        sample_name, idx_list, var  = self.branch_data
        HistProducerSample = os.path.join(self.ana_path(), 'FLAF', 'Analysis', 'HistProducerSample.py')
        with contextlib.ExitStack() as stack:
            local_inputs = [stack.enter_context(inp.localize('r')).path for inp in self.input()]
            with self.output().localize("w") as tmp_local_file:
                HistProducerSample_cmd = ['python3', HistProducerSample,'--outFile', tmp_local_file.path]#, '--remove-files', 'True']
                HistProducerSample_cmd.extend(local_inputs)
                ps_call(HistProducerSample_cmd,verbose=1)


class MergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 30.0)

    def workflow_requires(self):
        histProducerSample_map = HistProducerSampleTask.req(self,branch=-1, branches=(),customisations=self.customisations).create_branch_map()
        all_samples = {}
        branches = {}
        for br_idx, (smpl_name, idx_list, var) in histProducerSample_map.items():
            if var not in all_samples:
                all_samples[var] = []
            all_samples[var].append(br_idx)

        new_branchset = set()
        for var in all_samples.keys():
            new_branchset.update(all_samples[var])

        return { "histproducersample": HistProducerSampleTask.req(self, branches=list(new_branchset)) }

    def requires(self):
        var, branches_idx = self.branch_data
        deps = [HistProducerSampleTask.req(self, max_runtime=HistProducerSampleTask.max_runtime._default, branch=prod_br,customisations=self.customisations) for prod_br in branches_idx ]
        return deps

    def create_branch_map(self):
        histProducerSample_map = HistProducerSampleTask.req(self,branch=-1, branches=(),customisations=self.customisations).create_branch_map()
        all_samples = {}
        branches = {}
        for br_idx, (smpl_name, idx_list, var) in histProducerSample_map.items():
            if var not in all_samples:
                all_samples[var] = []
            all_samples[var].append(br_idx)
        k=0
        for n, key in enumerate(all_samples.items()):
            var, branches_idx = key
            branches[k] = (var, branches_idx)
            k+=1
        return branches

    def output(self):
        var, branches_idx = self.branch_data
        merge_dir = 'merged'
        output_path = os.path.join(self.version, self.period, merge_dir, var, f'{var}.root')
        return self.remote_target(output_path,  fs=self.fs_histograms)

    def run(self):
        var, branches_idx = self.branch_data
        sample_config = os.path.join(self.ana_path(), 'FLAF', 'config', self.period, f'samples.yaml')
        global_config = os.path.join(self.ana_path(), self.global_params['analysis_config_area'], f'global.yaml')
        unc_config = os.path.join(self.ana_path(), 'config', self.period, f'weights.yaml')
        customisation_dict = getCustomisationSplit(self.customisations)
        channels = customisation_dict['channels'] if 'channels' in customisation_dict.keys() else self.global_params['channelSelection']
        #Channels from the yaml are a list, but the format we need for the ps_call later is 'ch1,ch2,ch3', basically join into a string separated by comma
        if type(channels) == list:
            channels = ','.join(channels)
        #bbww does not use a deepTauVersion
        deepTauVersion = ''
        isbbtt = 'HH_bbtautau' in self.global_params['analysis_config_area'].split('/')
        if isbbtt: deepTauVersion = customisation_dict['deepTauVersion'] if 'deepTauVersion' in customisation_dict.keys() else self.global_params['deepTauVersion']
        region = customisation_dict['region'] if 'region' in customisation_dict.keys() else self.global_params['region_default']
        customisation_dict['apply_btag_shape_weights']=='True' if 'apply_btag_shape_weights' in customisation_dict.keys() else self.global_params.get('apply_btag_shape_weights', False)
        uncNames = ['Central']
        unc_cfg_dict = load_unc_config(unc_config)
        uncs_to_exclude = self.global_params['uncs_to_exclude'][self.period] if "uncs_to_exclude" in self.global_params.keys() else []

        compute_unc_histograms = customisation_dict['compute_unc_histograms']=='True' if 'compute_unc_histograms' in customisation_dict.keys() else self.global_params.get('compute_unc_histograms', False)
        if compute_unc_histograms:
            for uncName in list(unc_cfg_dict['norm'].keys())+unc_cfg_dict['shape']:
                if uncName in uncs_to_exclude: continue
                uncNames.append(uncName)
        print(uncNames)
        MergerProducer = os.path.join(self.ana_path(), 'FLAF', 'Analysis', 'HistMerger.py')
        HaddMergedHistsProducer = os.path.join(self.ana_path(), 'FLAF', 'Analysis', 'hadd_merged_hists.py')
        RenameHistsProducer = os.path.join(self.ana_path(), 'FLAF', 'Analysis', 'renameHists.py')


        output_path_hist_prod_sample_data = os.path.join(self.version, self.period, 'split', var, f'data.root')
        all_inputs = [(self.remote_target(output_path_hist_prod_sample_data, fs=self.fs_histograms),'data')]
        samples_to_consider = GetSamples(self.samples, self.setup.backgrounds,self.global_params['signal_types'] )
        for sample_name in self.samples.keys():
            if sample_name not in samples_to_consider: continue
            output_path_hist_prod_sample = os.path.join(self.version, self.period, 'split', var, f'{sample_name}.root')
            all_inputs.append((self.remote_target(output_path_hist_prod_sample, fs=self.fs_histograms),sample_name))
        all_datasets=[]
        all_outputs_merged = []

        outdir_histograms = os.path.join(self.version, self.period, 'merged', var, 'tmp')

        with contextlib.ExitStack() as stack:
            local_inputs = []
            for inp, smpl in all_inputs:
                local_inputs.append(stack.enter_context(inp.localize('r')).path)
                all_datasets.append(smpl)
            dataset_names = ','.join(smpl for smpl in all_datasets)

            if len(uncNames)==1:
                with self.output().localize("w") as outFile:
                    MergerProducer_cmd = ['python3', MergerProducer,'--outFile', outFile.path, '--var', var, '--uncSource', uncNames[0], '--datasetFile', dataset_names, '--channels',channels, '--ana_path', self.ana_path(), '--period', self.period]#, '--remove-files', 'True']
                    if 'apply_btag_shape_weights' in customisation_dict.keys():
                        MergerProducer_cmd.append('--apply-btag-shape-weights', customisation_dict['apply_btag_shape_weights'])
                    MergerProducer_cmd.extend(local_inputs)
                    ps_call(MergerProducer_cmd,verbose=1)
            else:
                for uncName in uncNames:
                    print(uncName)
                    final_histname = f'all_histograms_{var}_{uncName}.root'
                    tmp_outfile_merge = os.path.join(outdir_histograms,final_histname)
                    tmp_outfile_merge_remote = self.remote_target(tmp_outfile_merge, fs=self.fs_histograms)
                    with tmp_outfile_merge_remote.localize("w") as tmp_outfile_merge_unc:
                        MergerProducer_cmd = ['python3', MergerProducer,'--outFile', tmp_outfile_merge_unc.path, '--var', var, '--uncSource', uncName, '--datasetFile', dataset_names, '--channels',channels, '--ana_path', self.ana_path(), '--period', self.period]#, '--remove-files', 'True']

                        MergerProducer_cmd.extend(local_inputs)
                        if 'btagShape' in self.global_params['corrections']:
                            MergerProducer_cmd.append('--apply-btag-shape-weights')
                            MergerProducer_cmd.append('True')
                        # print(MergerProducer_cmd)
                        ps_call(MergerProducer_cmd,verbose=1)
                    all_outputs_merged.append(tmp_outfile_merge)
        if len(uncNames) > 1:
            all_uncertainties_string = ','.join(unc for unc in uncNames)
            tmp_outFile = self.remote_target( os.path.join(outdir_histograms,f'all_histograms_{var}_hadded.root'), fs=self.fs_histograms)
            with contextlib.ExitStack() as stack:
                local_merged_files = []
                for infile_merged in all_outputs_merged:
                    tmp_outfile_merge_remote = self.remote_target(infile_merged, fs=self.fs_histograms)
                    local_merged_files.append(stack.enter_context(tmp_outfile_merge_remote.localize('r')).path)
                with tmp_outFile.localize("w") as tmpFile:
                    HaddMergedHistsProducer_cmd = ['python3', HaddMergedHistsProducer,'--outFile', tmpFile.path, '--var', var]
                    HaddMergedHistsProducer_cmd.extend(local_merged_files)
                    ps_call(HaddMergedHistsProducer_cmd,verbose=1)
            with tmp_outFile.localize("r") as tmpFile, self.output().localize("w") as outFile:
                RenameHistsProducer_cmd = ['python3', RenameHistsProducer,'--inFile', tmpFile.path, '--outFile', outFile.path, '--var', var, '--year', getYear(self.period), '--ana_path', self.ana_path(), '--period', self.period]
                ps_call(RenameHistsProducer_cmd,verbose=1)
