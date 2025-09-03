import copy
import law
import luigi
import math
import os
import tempfile

from FLAF.RunKit.run_tools import natural_sort
from FLAF.RunKit.crabLaw import update_kinit
from FLAF.RunKit.law_wlcg import WLCGFileTarget
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


# def get_param_value(cls, param_name):
#     param = getattr(cls, param_name)
#     return param.task_value(cls.__name__, param_name)


class Task(law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """

    version = luigi.Parameter()
    prefer_params_cli = ["version"]
    period = luigi.Parameter()
    customisations = luigi.Parameter(default="")
    test = luigi.BoolParameter(default=False)
    sample = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.setup = Setup.getGlobal(
            os.getenv("ANALYSIS_PATH"), self.period, self.sample, self.customisations
        )

    def store_parts(self):
        return (self.__class__.__name__, self.version, self.period)

    @property
    def cmssw_env(self):
        return self.setup.cmssw_env

    @property
    def samples(self):
        return self.setup.samples

    @property
    def global_params(self):
        return self.setup.global_params

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
        fs = fs or self.setup.fs_default
        path = os.path.join(*path)
        if type(fs) == str:
            path = os.path.join(fs, path)
            return law.LocalFileTarget(path)
        return WLCGFileTarget(path, fs)

    def law_job_home(self):
        if "LAW_JOB_HOME" in os.environ:
            return os.environ["LAW_JOB_HOME"], False
        os.makedirs(self.local_path(), exist_ok=True)
        return tempfile.mkdtemp(dir=self.local_path()), True

    def poll_callback(self, poll_data):
        update_kinit(verbose=0)

    def iter_samples(self):
        for sample_id, sample_name in enumerate(natural_sort(self.samples.keys())):
            yield sample_id, sample_name


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

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

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
