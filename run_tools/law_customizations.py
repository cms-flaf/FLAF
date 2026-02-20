import copy
import law
import luigi
import math
import os
import tempfile

from FLAF.RunKit.run_tools import natural_sort
from FLAF.RunKit.crabLaw import update_kinit
from FLAF.RunKit.law_wlcg import WLCGFileTarget, WLCGDirectoryTarget
from FLAF.Common.Setup import Setup

law.contrib.load("htcondor")


def copy_param(ref_param, new_default):
    param = copy.deepcopy(ref_param)
    param._default = new_default
    return param


def get_param_value(cls, param_name):
    try:
        param = getattr(cls, param_name)
        return param.task_value(cls.__name__, param_name)
    except:
        return None


class Task(law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """

    version = luigi.Parameter()
    prefer_params_cli = ["version", "tasks_per_job"]
    period = luigi.Parameter()
    customisations = luigi.Parameter(default="")
    test = luigi.IntParameter(default=-1)
    dataset = luigi.Parameter(default="")
    process = luigi.Parameter(default="")
    model = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.setup = Setup.getGlobal(
            os.getenv("ANALYSIS_PATH"),
            self.period,
            self.version,
            custom_process_selection=self.process if len(self.process) > 0 else None,
            custom_dataset_selection=self.dataset if len(self.dataset) > 0 else None,
            custom_model_selection=self.model if len(self.model) > 0 else None,
            customisations=self.customisations,
        )
        self._dataset_id_name_list = None
        self._dataset_id_name_dict = None
        self._dataset_name_id_dict = None

    def store_parts(self):
        return (self.version, self.__class__.__name__, self.period)

    @property
    def cmssw_env(self):
        return self.setup.cmssw_env

    @property
    def datasets(self):
        return self.setup.datasets

    @property
    def global_params(self):
        return self.setup.global_params

    @property
    def fs_default(self):
        return self.setup.get_fs("default")

    @property
    def fs_nanoAOD(self):
        return self.setup.get_fs("nanoAOD")

    @property
    def fs_anaCache(self):
        return self.setup.get_fs("anaCache")

    @property
    def fs_anaTuple(self):
        return self.setup.get_fs("anaTuple")

    @property
    def fs_HistTuple(self):
        return self.setup.get_fs("HistTuple")

    @property
    def fs_anaCacheTuple(self):
        return self.setup.get_fs("anaCacheTuple")

    @property
    def fs_nnCacheTuple(self):
        return self.setup.get_fs("nnCacheTuple")

    @property
    def fs_histograms(self):
        return self.setup.get_fs("histograms")

    @property
    def fs_plots(self):
        return self.setup.get_fs("plots")

    def ana_path(self):
        return os.getenv("ANALYSIS_PATH")

    def ana_data_path(self):
        return os.getenv("ANALYSIS_DATA_PATH")

    def local_path(self, *path):
        parts = (self.ana_data_path(),) + self.store_parts() + path
        return os.path.join(*parts)

    def local_target(self, *path):
        return law.LocalFileTarget(self.local_path(*path))

    def remote_target(self, *path, fs=None):
        fs = fs or self.fs_default
        path = os.path.join(*path)
        if type(fs) == str:
            path = os.path.join(fs, path)
            return law.LocalFileTarget(path)
        return WLCGFileTarget(path, fs)

    def remote_dir_target(self, *path, fs=None):
        fs = fs or self.fs_default
        path = os.path.join(*path)
        if type(fs) == str:
            path = os.path.join(fs, path)
            return law.LocalDirectoryTarget(path)
        return WLCGDirectoryTarget(path, fs)

    def law_job_home(self):
        if "LAW_JOB_HOME" in os.environ:
            return os.environ["LAW_JOB_HOME"], False
        os.makedirs(self.local_path(), exist_ok=True)
        return tempfile.mkdtemp(dir=self.local_path()), True

    def _create_dataset_mappings(self):
        if self._dataset_id_name_list is None:
            self._dataset_id_name_list = []
            self._dataset_id_name_dict = {}
            self._dataset_name_id_dict = {}
            for dataset_id, dataset_name in enumerate(
                natural_sort(self.datasets.keys())
            ):
                self._dataset_id_name_list.append((dataset_id, dataset_name))
                self._dataset_id_name_dict[dataset_id] = dataset_name
                self._dataset_name_id_dict[dataset_name] = dataset_id

    def iter_datasets(self):
        self._create_dataset_mappings()
        for dataset_id, dataset_name in self._dataset_id_name_list:
            yield dataset_id, dataset_name

    def get_dataset_name(self, dataset_id):
        self._create_dataset_mappings()
        if dataset_id not in self._dataset_id_name_dict:
            raise KeyError(f"dataset id '{dataset_id}' not found")
        return self._dataset_id_name_dict[dataset_id]

    def get_dataset_id(self, dataset_name):
        self._create_dataset_mappings()
        if dataset_name not in self._dataset_name_id_dict:
            raise KeyError(f"dataset name '{dataset_name}' not found")
        return self._dataset_name_id_dict[dataset_name]

    def get_nano_version(self, dataset_name):
        dataset = self.datasets[dataset_name]
        isData = dataset["process_group"] == "data"
        version_label = "data" if isData else "mc"
        return self.global_params.get("nanoAODVersions", {}).get(
            version_label, "HLepRare"
        )

    def get_fs_nanoAOD(self, dataset_name):
        if dataset_name not in self.datasets:
            raise KeyError(f"dataset name '{dataset_name}' not found")
        dataset = self.datasets[dataset_name]

        folder_name = dataset.get("dirName", dataset_name)

        if "fs_nanoAOD" in dataset:
            return (
                self.setup.get_fs(f"fs_nanoAOD_{dataset_name}", dataset["fs_nanoAOD"]),
                folder_name,
                True,
            )

        nano_version = self.get_nano_version(dataset_name)
        if nano_version == "HLepRare":
            return self.fs_nanoAOD, folder_name, True
        das_cfg = dataset.get("nanoAOD", {})
        das_ds_name = None
        if isinstance(das_cfg, dict):
            if nano_version in das_cfg:
                das_ds_name = das_cfg[nano_version]
        elif isinstance(das_cfg, str):
            das_ds_name = das_cfg

        if das_ds_name is not None:
            return self.setup.fs_das, das_ds_name, False

        raise RuntimeError(
            f"Unable to identify the file source for dataset {dataset_name}"
        )


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """

    max_runtime = law.DurationParameter(
        default=12.0,
        unit="h",
        significant=False,
        description="maximum runtime, default unit is hours",
    )
    n_cpus = luigi.IntParameter(default=1, description="number of cpus")
    poll_interval = copy_param(law.htcondor.HTCondorWorkflow.poll_interval, 2)
    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="transfer job logs to the output directory",
    )

    def htcondor_check_job_completeness(self):
        return False

    def htcondor_poll_callback(self, poll_data):
        update_kinit(verbose=0)
        return True

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    # def htcondor_log_directory(self):
    #     # the directory where HTCondor should store job logs, it can be the same as the output directory
    #     path = ("logs",) + self.store_parts()
    #     return self.remote_dir_target(*path)

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return os.path.join(os.getenv("ANALYSIS_PATH"), "FLAF", "bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        ana_path = os.getenv("ANALYSIS_PATH")
        # render_variables are rendered into all files sent with a job
        config.render_variables["analysis_path"] = ana_path
        # force to run on CC7, https://batchdocs.web.cern.ch/local/submit.html
        config.custom_content.append(
            ("requirements", 'TARGET.OpSysAndVer =?= "AlmaLinux9"')
        )

        # maximum runtime
        config.custom_content.append(
            ("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1)
        )
        config.custom_content.append(("RequestCpus", self.n_cpus))
        return config
