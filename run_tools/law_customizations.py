import copy
import importlib
import law
import luigi
import math
import os
import re
import shutil
import subprocess
import tempfile

from law.parser import global_cmdline_values

from FLAF.RunKit.run_tools import natural_sort
from FLAF.RunKit.kinit import update_kinit
from FLAF.RunKit.law_wlcg import WLCGFileSystem, WLCGFileTarget, WLCGDirectoryTarget
from FLAF.Common.Setup import Setup

law.contrib.load("htcondor")
law.contrib.load("cms")


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

    # --- Per-class caches for luigi/law reflection. luigi.Task.get_params() rebuilds the
    # parameter list with dir(cls) + isinstance on every call, and law's req_params() filters
    # parameters with fnmatch on every .req() call. Both results are constant for a given
    # class (and class pair), but recomputing them dominates CPU when building or printing
    # large task graphs (thousands of .req()/instantiations). Memoizing them is transparent
    # (the cached values are exactly what luigi/law would have produced).
    _get_params_cache = {}
    _req_copy_names_cache = {}
    _req_prefer_cli_drop_cache = {}

    @classmethod
    def get_params(cls):
        cached = Task._get_params_cache.get(cls)
        if cached is None:
            cached = super(Task, cls).get_params()
            Task._get_params_cache[cls] = cached
        return cached

    @classmethod
    def req(cls, inst, **kwargs):
        # Law control kwargs (prefixed with "_", e.g. _exclude/_prefer_cli) change which
        # parameters are copied; defer those rare calls to law's full implementation.
        if any(key.startswith("_") for key in kwargs):
            return super(Task, cls).req(inst, **kwargs)
        params = {name: getattr(inst, name) for name in cls._req_copy_names(inst)}
        params.update(kwargs)
        for name in cls._req_prefer_cli_drop():
            params.pop(name, None)
        return cls(**params)

    @classmethod
    def _req_copy_names(cls, inst):
        # Names of the parameters req_params() copies from inst (common parameters minus the
        # excluded ones), constant per (cls, type(inst)). Derived from law's own req_params
        # (with prefer-cli removal disabled, which we re-apply per call) so the exclusion is
        # exactly law's; computed once and cached.
        key = (cls, type(inst))
        names = Task._req_copy_names_cache.get(key)
        if names is None:
            names = tuple(cls.req_params(inst, _prefer_cli=[]).keys())
            Task._req_copy_names_cache[key] = names
        return names

    @classmethod
    def _req_prefer_cli_drop(cls):
        # Parameters that req_params() drops because they are preferably taken from the CLI.
        # Keyed on the CLI parser identity so a None -> real-parser transition is picked up.
        prefer = cls.prefer_params_cli
        if not prefer:
            return ()
        parser = luigi.cmdline_parser.CmdlineParser.get_instance()
        key = (cls, id(parser))
        cached = Task._req_prefer_cli_drop_cache.get(key)
        if cached is None:
            drop = set()
            if parser is not None:
                prefix = cls.get_task_family() + "_"
                present = {
                    k[len(prefix) :]
                    for k in global_cmdline_values().keys()
                    if k.startswith(prefix)
                }
                drop = set(prefer) & present
            cached = tuple(drop)
            Task._req_prefer_cli_drop_cache[key] = cached
        return cached

    version = luigi.Parameter()
    prefer_params_cli = [
        "version",
        "anaTuple_version",
        "anaCache_version",
        "ana_version",
        "tasks_per_job",
    ]
    # tasks_per_job is a per-task tuning knob: each task keeps its own default (or an
    # explicit CLI value) instead of inheriting the requesting task's value via .req().
    exclude_params_req = law.Task.exclude_params_req | {"tasks_per_job"}
    period = luigi.Parameter()
    customisations = luigi.Parameter(default="")
    test = luigi.IntParameter(default=-1)
    dataset = luigi.Parameter(default="")
    process = luigi.Parameter(default="")
    model = luigi.Parameter(default="")
    user_custom = luigi.Parameter(default="")

    # Convenience parameters for using centrally produced AnaTuples/AnaCaches.
    anaTuple_version = luigi.Parameter(
        default="",
        significant=False,
        description="If set, forces version for upstream AnaTuple/AnaProd tasks "
        "(InputFileTask, AnaTuple*List*, AnaTupleMerge, ...).",
    )

    anaCache_version = luigi.Parameter(
        default="",
        significant=False,
        description="If set, forces version for AnalysisCacheTask/AnalysisCacheAggregationTask (central BtagShape etc.).",
    )

    ana_version = luigi.Parameter(
        default="",
        significant=False,
        description="If set, combines --anaTuple-version and --anaCache-version (single flag for both).",
    )

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        user_custom_file = None
        if self.user_custom:
            user_custom_file = self._resolve_user_custom_path(self.user_custom)
        self.setup = Setup.getGlobal(
            os.getenv("ANALYSIS_PATH"),
            self.period,
            self.version,
            custom_process_selection=self.process if len(self.process) > 0 else None,
            custom_dataset_selection=self.dataset if len(self.dataset) > 0 else None,
            custom_model_selection=self.model if len(self.model) > 0 else None,
            customisations=self.customisations,
            user_custom_file=user_custom_file,
        )
        self._dataset_id_name_list = None
        self._dataset_id_name_dict = None
        self._dataset_name_id_dict = None

    @staticmethod
    def _resolve_user_custom_path(user_custom):
        from FLAF.Common.Setup import resolve_user_custom_path

        return resolve_user_custom_path(user_custom)

    def _stage_user_custom_input(self, config):
        """Ship user_custom yaml as a job input for remote workers (bundle/CRAB)."""
        if not self.user_custom:
            return
        path = self.user_custom
        if not os.path.isabs(path):
            path = os.path.join(os.getenv("ANALYSIS_PATH") or "", path)
        if not path or not os.path.isfile(path):
            return
        from law.job.base import JobInputFile

        # share=True, increment=False keeps a stable basename when possible; resolve
        # still accepts law's hashed names if increment is forced elsewhere.
        config.input_files["user_custom"] = JobInputFile(
            path=path, copy=True, share=True, render=False, increment=False
        )

    def _stage_path_cache_input(self, config):
        """Dump the submit-process path cache and ship it with the CRAB job."""
        from law.job.base import JobInputFile
        from FLAF.RunKit.law_gfal import (
            SHIPPED_PATH_CACHE_BASENAME,
            collect_setup_path_cache_entries,
            write_path_cache_file,
        )

        out_dir = self.local_path()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, SHIPPED_PATH_CACHE_BASENAME)
        write_path_cache_file(path, collect_setup_path_cache_entries(self.setup))
        config.input_files["path_cache"] = JobInputFile(
            path=path, copy=True, share=True, render=False, increment=False
        )

    # Process-local memoization of create_branch_map results, shared across task
    # instances. The same branch map is otherwise rebuilt many times during task
    # initialization because every `X.req(...).create_branch_map()` constructs a fresh
    # instance and so bypasses law's per-instance branch-map cache (`_branch_map`). The
    # downstream maps form a cascade (e.g. AnalysisCacheAggregation -> AnalysisCacheTask
    # -> HistTupleProducer -> AnaTupleMerge), and `workflow_requires`/`requires` rebuild
    # it once per branch, which is O(nBranches) redundant full rebuilds and dominates the
    # loading time of post-anaTuple tasks. Within a single law process the inputs that
    # determine a branch map (config + merge plans + completed upstream outputs) are
    # stable, so memoizing by the map-determining parameters is safe.
    _branch_map_cache = {}

    def _branch_map_cache_key(self):
        return (
            type(self).__name__,
            self.version,
            self.period,
            self.customisations,
            self.dataset,
            self.process,
            self.model,
            self.test,
            self.user_custom,
            self.anaTuple_version,
            self.anaCache_version,
            self.ana_version,
            getattr(self, "producer_to_run", None),
            getattr(self, "producer_to_aggregate", None),
            getattr(self, "variables", None),
            getattr(self, "n_files_per_job", None),
        )

    def cached_branch_map(self, build_fn):
        """Return ``build_fn()`` memoized per map-determining parameter signature.

        Only populated maps are cached: an empty result means an upstream task is not
        ready yet (e.g. the merge plan does not exist), which must stay dynamic so the
        map is rebuilt once the upstream completes.
        """
        key = self._branch_map_cache_key()
        cached = Task._branch_map_cache.get(key)
        if cached is None:
            cached = build_fn()
            if cached:
                Task._branch_map_cache[key] = cached
        # Return a shallow copy: law's get_branch_map() mutates the returned dict in place
        # (`_reduce_branch_map` does `del branch_map[b]` to filter to the requested
        # `branches`), which would otherwise corrupt the shared cached map for other
        # instances. The branch-data values are immutable tuples, so a shallow copy is safe.
        return dict(cached)

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
        if isinstance(fs, law.LocalFileSystem):
            return law.LocalFileTarget(path, fs=fs)
        return WLCGFileTarget(path, fs)

    def remote_dir_target(self, *path, fs=None):
        fs = fs or self.fs_default
        path = os.path.join(*path)
        if type(fs) == str:
            path = os.path.join(fs, path)
            return law.LocalDirectoryTarget(path)
        return WLCGDirectoryTarget(path, fs)

    def remote_log_dir_target(self):
        # Remote directory where job logs are staged. Include the producer name when the
        # task has one (AnalysisCacheTask: producer_to_run; AnalysisCacheAggregationTask:
        # producer_to_aggregate) so per-producer logs of the same task do not collide.
        parts = [self.version, "logs", self.__class__.__name__, self.period]
        producer = getattr(self, "producer_to_run", None) or getattr(
            self, "producer_to_aggregate", None
        )
        if producer:
            parts.append(producer)
        return self.remote_dir_target(*parts)

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
            return self.setup.fs_rucio, das_ds_name, False

        raise RuntimeError(
            f"Unable to identify the file source for dataset {dataset_name}"
        )


class BundleTask(Task):
    flavour = luigi.Parameter(
        description="bundle flavour (core, cmssw, inputFileList, AnaTupleFileList)"
    )

    def requires(self):
        bundle_cfg = self.global_params.get("bundles", {}).get(self.flavour, {})
        task_req_cfg = bundle_cfg.get("task_requires")
        if task_req_cfg is None:
            return {}
        mod = importlib.import_module(task_req_cfg["module"])
        task_cls = getattr(mod, task_req_cfg["class"])
        return {"source": task_cls.req(self, branches=())}

    def output(self):
        return self.remote_target(
            self.version, "bundles", self.period, f"{self.flavour}.tar.bz2"
        )

    def run(self):
        bundle_cfg = self.global_params.get("bundles", {}).get(self.flavour)
        if not bundle_cfg:
            raise RuntimeError(
                f"Bundle flavour '{self.flavour}' not configured in bundles section of global.yaml"
            )

        patterns = bundle_cfg.get("patterns", [])
        ana_path = os.getenv("ANALYSIS_PATH")

        formatted_patterns = [
            p.format(version=self.version, period=self.period) for p in patterns
        ]

        os.makedirs(self.local_path(), exist_ok=True)

        # Source the "FLAF" and "Corrections" patterns from FLAF_PATH / CORRECTIONS_PATH.
        # env.sh always sets these (to the submodule copies in production, or to the edited
        # top-level copies in a FLAF_all workspace when flaf_dev.sh is used), so dev edits
        # are packaged into the tarballs transparently. The layout *inside* the tarball
        # stays canonical (FLAF/, Corrections/ at the top) so worker bootstrap is unaffected.
        flaf_base = os.getenv("FLAF_PATH") or os.path.join(ana_path, "FLAF")
        corr_base = os.getenv("CORRECTIONS_PATH") or os.path.join(
            ana_path, "Corrections"
        )

        def _get_bundle_source(pat: str) -> str:
            p = pat.replace("\\", "/")
            if p == "FLAF" or p.startswith("FLAF/"):
                rel = p[5:] if p.startswith("FLAF/") else ""
                return os.path.join(flaf_base, rel) if rel else flaf_base
            if p == "Corrections" or p.startswith("Corrections/"):
                rel = p[12:] if p.startswith("Corrections/") else ""
                return os.path.join(corr_base, rel) if rel else corr_base
            return os.path.join(ana_path, pat)

        print(f"bundle[{self.flavour}]: creating archive from {ana_path}")
        with self.output().localize("w") as tmp:
            with tempfile.TemporaryDirectory() as staging:
                found_any = False
                for pattern in formatted_patterns:
                    full_path = _get_bundle_source(pattern)
                    # Resolve top-level symlinks so the staging copy uses real content,
                    # but symlinks *within* the directory are preserved as symlinks.
                    # This prevents --dereference from following CVMFS symlinks inside flaf_env.
                    real_path = os.path.realpath(full_path)
                    if not os.path.exists(real_path):
                        print(
                            f"bundle[{self.flavour}]: warning: '{pattern}' not found, skipping"
                        )
                        continue
                    found_any = True
                    dest = os.path.join(staging, pattern)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    if os.path.isdir(real_path):
                        shutil.copytree(real_path, dest, symlinks=True)
                    else:
                        shutil.copy2(real_path, dest)

                if not found_any:
                    raise RuntimeError(
                        f"No files found for bundle flavour '{self.flavour}'"
                    )

                # CMSSW analysis customisations (HHbtag, ClassicSVfit, …) are installed as
                # absolute AFS symlinks under soft/CMSSW_*/src. On CRAB those targets do not
                # exist. Materialize any absolute symlink that points outside the staging
                # tree so the tarball is self-contained. Relative / internal links stay.
                if self.flavour == "cmssw":
                    n_mat = self._materialize_external_symlinks(staging)
                    if n_mat:
                        print(
                            f"bundle[cmssw]: materialized {n_mat} external symlink(s)"
                        )

                subprocess.run(
                    [
                        "tar",
                        "--exclude=*/__pycache__",
                        "--exclude=*.pyc",
                        "--exclude=*.pyo",
                        "-cjf",
                        tmp.abspath,
                        "-C",
                        staging,
                        ".",
                    ],
                    check=True,
                )
        print(f"bundle[{self.flavour}]: done")

    @staticmethod
    def _materialize_external_symlinks(root: str) -> int:
        """Replace absolute external symlinks under *root* with real file/dir copies.

        Returns the number of symlinks replaced. Relative symlinks and absolute ones
        that already resolve inside *root* are left unchanged.
        """
        root_real = os.path.realpath(root)
        n = 0
        # Collect first so we do not walk into trees we just replaced.
        external = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                path = os.path.join(dirpath, name)
                if not os.path.islink(path):
                    continue
                target = os.readlink(path)
                if not os.path.isabs(target):
                    continue
                # Resolve once; skip broken links with a warning.
                try:
                    resolved = os.path.realpath(path)
                except OSError:
                    print(f"bundle[cmssw]: warning: broken symlink {path} -> {target}")
                    continue
                if not os.path.exists(resolved):
                    print(
                        f"bundle[cmssw]: warning: dangling symlink {path} -> {target}"
                    )
                    continue
                # Already points inside the staging tree → fine to keep.
                if resolved == root_real or resolved.startswith(root_real + os.sep):
                    continue
                external.append((path, resolved))

        for path, resolved in external:
            os.unlink(path)
            if os.path.isdir(resolved):
                shutil.copytree(resolved, path, symlinks=True)
            else:
                shutil.copy2(resolved, path)
            n += 1
            print(f"bundle[cmssw]: materialized {path} <- {resolved}")
        return n


class CERNHTCondorJobFileFactory(law.htcondor.HTCondorJobFileFactory):
    """HTCondor job file factory that stages transfer_input_files to EOS and uses protocol URLs.

    When config._worker_files_remote_dir is set (a WLCGDirectoryTarget), every file listed in
    transfer_input_files is uploaded to that remote directory and its path in the JDL is replaced
    with the corresponding remote URL.  This lets CERN HTCondor fetch input files from EOS via the
    protocol layer instead of trying to read /eos POSIX paths, which the batch system does not
    support.
    """

    def create(self, **kwargs):
        worker_files_dir = kwargs.get("_worker_files_remote_dir")
        job_file, c = super().create(**kwargs)
        self._stage_and_update_jdl(job_file, worker_files_dir)
        return job_file, c

    @staticmethod
    def _stage_and_update_jdl(job_file, worker_files_dir=None):
        with open(job_file) as f:
            content = f.read()

        lines = content.split("\n")
        new_lines = []
        updated = False

        for line in lines:
            line_key = line.lower().split("=")[0].strip() if "=" in line else ""

            if line_key == "transfer_input_files" and worker_files_dir is not None:
                key, _, value = line.partition(" = ")
                value = value.strip()
                quoted = value.startswith('"') and value.endswith('"')
                if quoted:
                    value = value[1:-1]
                local_paths = [p.strip() for p in value.split(",") if p.strip()]
                remote_urls = []
                for local_path in local_paths:
                    if "://" in local_path:
                        remote_urls.append(local_path)
                        continue
                    basename = os.path.basename(local_path)
                    remote_file = worker_files_dir.child(basename, type="f")
                    if not remote_file.exists():
                        print(f"worker_files: uploading {basename}")
                        remote_file.copy_from_local(local_path)
                    remote_urls.append(remote_file.uri())
                line = f'{key} = {",".join(remote_urls)}'
                updated = True

            elif line_key == "initialdir":
                updated = True
                continue

            elif line_key == "x509userproxy":
                key, _, proxy_path = line.partition(" = ")
                proxy_path = proxy_path.strip()
                if "://" not in proxy_path and not proxy_path.startswith("/tmp/"):
                    tmp_proxy = f"/tmp/{os.environ.get('USER', 'law')}_voms.proxy"
                    shutil.copy2(proxy_path, tmp_proxy)
                    os.chmod(tmp_proxy, 0o600)
                    line = f"{key} = {tmp_proxy}"
                    updated = True

            new_lines.append(line)

        if updated:
            with open(job_file, "w") as f:
                f.write("\n".join(new_lines))


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
    priority = luigi.IntParameter(
        default=0,
        description="job priority among your HTCondor jobs. Accepted values from -20 (lowest) to 20 (highest). Default 0.",
    )
    bundle = luigi.BoolParameter(
        default=False,
        significant=False,
        description="download pre-built bundle archives on workers instead of accessing AFS; "
        "tasks declare which flavours they need via bundle_flavours. Always on for --workflow crab.",
    )
    htcondor_spool = luigi.BoolParameter(
        default=True,
        significant=False,
        description="pass -spool to condor_submit so input files (including the x509 proxy) are "
        "read locally on the submit host and transferred to the schedd, avoiding any "
        "shared-filesystem dependency for the proxy path",
    )

    htcondor_job_kwargs_submit = [
        "htcondor_pool",
        "htcondor_scheduler",
        "htcondor_spool",
    ]
    bundle_flavours = []

    def _flaf_root(self):
        # FLAF source root, respecting the dev overlay: flaf_dev.sh sets FLAF_PATH to
        # the top-level FLAF_all/FLAF, while the analysis env.sh sets it to the pinned
        # submodule (ANALYSIS_PATH/FLAF).  Job-input scripts shipped to workers must
        # come from here so that, in overlay mode, non-bundle jobs run the edited
        # bootstrap/stageout scripts (and, via them, the edited FLAF) rather than the
        # stale submodule copies.  Falls back to ANALYSIS_PATH/FLAF if FLAF_PATH unset.
        return os.getenv("FLAF_PATH") or os.path.join(
            os.getenv("ANALYSIS_PATH"), "FLAF"
        )

    def _uses_bundles(self):
        """Whether this submission should ship and unpack code bundles on the worker.

        Bundles are optional for HTCondor (shared AFS is available) but required for CRAB
        (WLCG workers have no AFS mount).
        """
        if not self.bundle_flavours:
            return False
        if getattr(self, "effective_workflow", None) == "crab":
            return True
        return bool(self.bundle)

    def _bundle_requirements(self):
        """Return BundleTask requirements for configured flavours (empty if unused)."""
        if not self._uses_bundles():
            return {}
        bundles = []
        for item in self.bundle_flavours:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                flavour, bversion = item
                bundles.append(BundleTask.req(self, flavour=flavour, version=bversion))
            else:
                bundles.append(BundleTask.req(self, flavour=item))
        return {"bundles": bundles}

    def _apply_bundle_render_variables(self, config):
        """Set bootstrap render variables for bundle download (or clear them)."""
        if not self._uses_bundles():
            config.render_variables["bundle_list"] = ""
            return
        if not isinstance(self.fs_default, WLCGFileSystem):
            raise RuntimeError(
                "bundle / crab workflows require fs_default to be a remote filesystem "
                "(davs://, root://, ...)"
            )
        bundle_parts = []
        for item in self.bundle_flavours:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                flavour, bversion = item
            else:
                flavour = item
                bversion = self.version
            bundle_url = self.remote_target(
                bversion, "bundles", self.period, f"{flavour}.tar.bz2"
            ).uri()
            bundle_parts.append(f"{flavour}:{bundle_url}")
        config.render_variables["bundle_list"] = " ".join(bundle_parts)

    def _apply_bootstrap_path_render_variables(self, config):
        """Set analysis_path / FLAF_PATH / CORRECTIONS_PATH / token-server for bootstrap.sh."""
        ana_path = os.getenv("ANALYSIS_PATH")
        # Bundle (and always-on CRAB) jobs unpack code on the worker and must not point back
        # at AFS. Non-bundle HTCondor jobs source the shared workspace and forward overlay paths.
        flaf_path = ""
        corrections_path = ""
        if self._uses_bundles():
            config.render_variables["analysis_path"] = "NONE"
        else:
            config.render_variables["analysis_path"] = ana_path
            flaf_path = os.getenv("FLAF_PATH", "") or ""
            corrections_path = os.getenv("CORRECTIONS_PATH", "") or ""
        config.render_variables["flaf_path"] = flaf_path
        config.render_variables["corrections_path"] = corrections_path
        # Rucio account for workers: CRAB pilots have USER=cmsplt01, which is not a Rucio
        # account. Bake the submitter account so bootstrap can export RUCIO_ACCOUNT.
        config.render_variables["rucio_account"] = (
            os.environ.get("RUCIO_ACCOUNT") or os.environ.get("USER") or ""
        )

        runTokenServer = self.global_params.get("runTokenServer", None)
        if runTokenServer and not self._uses_bundles():
            config.render_variables["run_token_server_host"] = runTokenServer["host"]
            config.render_variables["run_token_server_port"] = str(
                runTokenServer["port"]
            )
            config.input_files["get_token_script"] = os.path.join(
                self._flaf_root(), "run_tools", "get_run_token.py"
            )
        else:
            config.render_variables["run_token_server_host"] = ""
            config.render_variables["run_token_server_port"] = ""

    def _log_remote_base_url(self):
        # Must match remote_log_dir_target() (used by --print-status and the
        # HTCondor submit proxy) so producer sub-paths stay consistent.
        if isinstance(self.fs_default, WLCGFileSystem):
            return self.remote_log_dir_target().uri()
        return ""

    def workflow_requires(self):
        return self._bundle_requirements()

    def htcondor_check_job_completeness(self):
        return False

    def htcondor_poll_callback(self, poll_data):
        update_kinit(verbose=0)
        return True

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    def htcondor_log_directory(self):
        return None

    def htcondor_stageout_file(self):
        return os.path.join(self._flaf_root(), "run_tools", "stageout_logs.sh")

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return os.path.join(self._flaf_root(), "bootstrap.sh")

    def htcondor_job_file_factory_cls(self):
        return CERNHTCondorJobFileFactory

    def htcondor_job_config(self, config, job_num, branches):
        self._apply_bootstrap_path_render_variables(config)
        self._stage_user_custom_input(config)

        # force to run on AlmaLinux9, https://batchdocs.web.cern.ch/local/submit.html
        config.custom_content.append(
            ("requirements", 'TARGET.OpSysAndVer =?= "AlmaLinux9"')
        )

        # maximum runtime
        config.custom_content.append(
            ("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1)
        )
        config.custom_content.append(("RequestCpus", self.n_cpus))
        config.custom_content.append(("priority", self.priority))

        # Forward the x509 proxy so HTCondor can delegate credentials to the execution node.
        proxy_path = os.environ.get("X509_USER_PROXY", "")
        if proxy_path and os.path.isfile(proxy_path):
            config.custom_content.append(("x509userproxy", proxy_path))

        # Expose the per-job postfix so the stageout script can build the log filename dynamically.
        config.custom_content.append(
            ("environment", '"LAW_HTCONDOR_JOB_POSTFIX=$(law_job_postfix)"')
        )

        log_remote_base_url = self._log_remote_base_url()
        config.render_variables["log_remote_base_url"] = log_remote_base_url

        # Redirect the sandbox log copy to /dev/null only when stageout will
        # actually upload it; otherwise keep the file so HTCondor transfers it
        # back to the submit node for local debugging.
        if log_remote_base_url:
            config.output_files["stdall.txt"] = "/dev/null"

        self._apply_bundle_render_variables(config)
        if self._uses_bundles() and not self.htcondor_spool:
            config._worker_files_remote_dir = self.remote_dir_target(
                self.version, "worker_files", self.period
            )

        return config

    def htcondor_job_file(self):
        from law.job.base import JobInputFile

        original = law.util.law_src_path("job", "law_job.sh")
        custom = os.path.join(
            os.getenv("ANALYSIS_DATA_PATH"), "law_job_no_print_deps.sh"
        )
        if not os.path.exists(custom) or os.path.getmtime(original) > os.path.getmtime(
            custom
        ):
            with open(original) as f:
                content = f.read()
            content = re.sub(r'\bdeps_depth="[0-9]+"', 'deps_depth="0"', content)
            with open(custom, "w") as f:
                f.write(content)
            os.chmod(custom, 0o755)
        return JobInputFile(path=custom, copy=True, share=True, render_job=True)


# Custom proxy subclass so that the "log" location recorded in job submission data
# (used by law for "first log file: ..." messages at submit time, stored job json,
# and "task failed" diagnostics) points at the *remote* staged logs location for
# bundle runs instead of the local AFS path under ANALYSIS_DATA_PATH.
# The basename computation (stdall, stdall_Cluster_Proc, or stdall<postfix>) is
# the same one used by stageout_logs.sh, so the URI will match the uploaded file.
#
# Use the stable extension point: obtain the base proxy class from whatever
# the current law version has configured on HTCondorWorkflow.workflow_proxy_cls.

BundleAwareHTCondorWorkflowProxyBase = HTCondorWorkflow.workflow_proxy_cls


class _BundleAwareHTCondorWorkflowProxy(BundleAwareHTCondorWorkflowProxyBase):
    def _submit_group(self, *args, **kwargs):
        job_ids, submission_data = super()._submit_group(*args, **kwargs)

        # Compute the remote log base directly from the *task*.  Note that `self`
        # here is the workflow *proxy*, which does not carry fs_default / version /
        # period / remote_dir_target — those live on `self.task`.  (PR #267 instead
        # read the `log_remote_base_url` render variable off each job config; that
        # never produced a value the line below doesn't, since the render variable
        # is set under the identical WLCG-fs_default condition with the identical
        # computation — so it was removed.)  We stage logs remotely precisely when
        # stdall.txt is redirected, i.e. for a WLCG fs_default.
        task = getattr(self, "task", None)
        base = ""
        try:
            if task is not None and isinstance(
                getattr(task, "fs_default", None), WLCGFileSystem
            ):
                base = task.remote_log_dir_target().uri()
        except Exception:
            base = ""

        if not base:
            return job_ids, submission_data

        for job_num, data in list(submission_data.items()):
            if isinstance(job_num, Exception) or not isinstance(data, dict):
                continue
            log = data.get("log")
            if log:
                basename = os.path.basename(str(log))
                remote_log = base.rstrip("/") + "/" + basename
                data = dict(data)
                data["log"] = remote_log
                submission_data[job_num] = data
        return job_ids, submission_data


HTCondorWorkflow.workflow_proxy_cls = _BundleAwareHTCondorWorkflowProxy
# law's workflow metaclass records, at *class creation* time, whether a class set
# `workflow_proxy_cls` in its body (stored as `_defined_workflow_proxy`).  Only
# such classes are considered by `find_workflow_cls()` when a task resolves which
# workflow (and therefore which proxy) to use.  Because we patch
# `workflow_proxy_cls` here — *after* the class was created — the flag is still
# False, so multi-workflow tasks (e.g. HelloWorldTask(Task, HTCondorWorkflow,
# LocalWorkflow)) would silently fall back to law's base HTCondorWorkflowProxy and
# our `_submit_group` override (remote log path rewrite) would never run.  Flip the
# flag so this class is recognised as the "htcondor" workflow provider.
HTCondorWorkflow._defined_workflow_proxy = True


class FLAFCrabJobFileFactory(law.cms.CrabJobFileFactory):
    """CrabJobFileFactory for FLAF: no CRAB-side product/log stageout.

    Analysis products and job logs are written by FLAF itself (remote targets via
    gfal + ``stageout_logs.sh``). CRAB is used only as a batch backend, so we force:

    - ``General.transferOutputs = False``
    - ``General.transferLogs = False``
    - no ``JobType.outputFiles``
    - ``JobType.disableAutomaticOutputCollection = True`` (law default)

    ``Site.storageSite`` / ``Data.outLFNDirBase`` remain required by the CRAB client
    for a valid config and the submit-time write check, but FLAF never places analysis
    outputs there.

    Also strips deprecated ``JobType.sendPythonFolder`` (rejected by modern CRAB).
    """

    def create(self, **kwargs):
        # Prevent law from promoting custom_log_file into CRAB JobType.outputFiles
        # (which would set transferOutputs=True and duplicate FLAF log stageout).
        kwargs = dict(kwargs)
        kwargs["output_files"] = []
        # Keep a local log file name for the law job script if transfer_logs requested,
        # but do not register it as a CRAB output.
        custom_log = kwargs.get("custom_log_file")

        job_file, c = super().create(**kwargs)

        if hasattr(c, "crab"):
            c.crab.General.transferOutputs = False
            c.crab.General.transferLogs = False
            if getattr(c.crab, "JobType", None) is not None:
                c.crab.JobType.sendPythonFolder = None
                c.crab.JobType.outputFiles = None
                c.crab.JobType.disableAutomaticOutputCollection = True
        c.output_files = []
        if custom_log:
            c.custom_log_file = custom_log

        try:
            self._rewrite_crab_job_file(job_file)
        except Exception as exc:
            print(f"WARNING: could not post-process crab job file {job_file}: {exc}")
        return job_file, c

    @staticmethod
    def _rewrite_crab_job_file(job_file):
        """Rewrite the generated CRAB cfg to drop output transfer and deprecated keys."""
        with open(job_file) as f:
            lines = f.readlines()

        new_lines = []
        skip_list = False
        for ln in lines:
            stripped = ln.strip()

            # Skip deprecated option entirely.
            if "sendPythonFolder" in ln:
                continue

            # Force no CRAB-side transfers (FLAF owns remote I/O).
            if "General.transferOutputs" in ln:
                new_lines.append("cfg.General.transferOutputs = False\n")
                continue
            if "General.transferLogs" in ln:
                new_lines.append("cfg.General.transferLogs = False\n")
                continue

            # Drop JobType.outputFiles (single line or multi-line list).
            if "JobType.outputFiles" in ln:
                if stripped.endswith("[") or ("[" in stripped and "]" not in stripped):
                    skip_list = True
                continue
            if skip_list:
                if "]" in stripped:
                    skip_list = False
                continue

            if "JobType.disableAutomaticOutputCollection" in ln:
                new_lines.append(
                    "cfg.JobType.disableAutomaticOutputCollection = True\n"
                )
                continue

            new_lines.append(ln)

        with open(job_file, "w") as f:
            f.writelines(new_lines)


# Require VOMS + MyProxy before submit. The CRAB server retrieves the user proxy
# from myproxy.cern.ch (>= ~5 days remaining). A local VOMS proxy alone is not
# enough: the client may accept the task, then the server returns SUBMITFAILED.
# Do not fall back to interactive delegation or a law.cfg password file.
_FLAFCrabWorkflowProxyBase = law.cms.CrabWorkflow.workflow_proxy_cls


class _FLAFCrabWorkflowProxy(_FLAFCrabWorkflowProxyBase):
    def setup_job_manager(self):
        """Require a valid VOMS proxy and a MyProxy credential (>= 5 days)."""
        proxy = os.environ.get("X509_USER_PROXY", "")
        if not proxy or not os.path.isfile(proxy):
            raise RuntimeError(
                "CRAB submission requires a valid VOMS proxy (X509_USER_PROXY). "
                "Run: voms-proxy-init --voms cms -valid 192:00"
            )
        if not law.wlcg.check_vomsproxy_validity(proxy_file=proxy):
            raise RuntimeError(
                f"VOMS proxy at {proxy} is missing or expired; run "
                "`voms-proxy-init --voms cms -valid 192:00`"
            )
        kwargs = {"proxy": proxy}

        min_myproxy_seconds = 5 * 24 * 3600

        # MyProxy usernames may be either the DN (`myproxy-init -d`) or a SHA1 of
        # the DN (law encode_username=True / some crab helpers). Accept either form.
        for encode in (False, True):
            try:
                info = (
                    law.wlcg.get_myproxy_info(encode_username=encode, silent=True) or {}
                )
            except Exception:
                info = {}
            if info.get("username") and info.get("timeleft", 0) >= min_myproxy_seconds:
                kwargs["myproxy_username"] = info["username"]
                return kwargs

        raise RuntimeError(
            "CRAB requires a MyProxy credential valid for at least 5 days "
            "(CRAB server retrieves it from myproxy.cern.ch). "
            "Run once interactively:\n"
            "  myproxy-init -d -n -s myproxy.cern.ch\n"
            "  # verify: myproxy-info -d -s myproxy.cern.ch  (timeleft >= 5 days)\n"
            "See docs/workflow/crab.md for the CRAB-retriever form."
        )


_EOSHOME_FS_RE = re.compile(
    r"^davs://eoshome-[a-z0-9]+\.cern\.ch(?::\d+)?/eos/user/[a-z0-9]/([^/]+)(/.*)?$",
    re.IGNORECASE,
)


def _crab_stageout_from_fs_spec(fs_spec):
    """Map ``fs_default`` to CRAB ``(storageSite, outLFNDirBase)``.

    Accepted forms (same as storage docs):

    - ``T3_CH_CERNBOX:/store/user/<user>/...``
    - ``davs://eoshome-<initial>.cern.ch:.../eos/user/<initial>/<user>/...``
      → ``T3_CH_CERNBOX`` + ``/store/user/<user>/...``
    """
    if isinstance(fs_spec, (list, tuple)):
        if not fs_spec:
            raise RuntimeError("fs_default is empty; CRAB needs a remote filesystem")
        fs_spec = fs_spec[0]
    if not isinstance(fs_spec, str) or not fs_spec.strip():
        raise RuntimeError("fs_default must be a string (or list of strings)")
    spec = fs_spec.strip().rstrip("/")

    if "://" not in spec and ":" in spec:
        site, lfn = spec.split(":", 1)
        site, lfn = site.strip(), lfn.strip()
        if site and lfn.startswith("/"):
            return site, lfn

    m = _EOSHOME_FS_RE.match(spec)
    if m:
        user, rest = m.group(1), m.group(2) or ""
        return "T3_CH_CERNBOX", f"/store/user/{user}{rest}"

    raise RuntimeError(
        "CRAB derives Site.storageSite and Data.outLFNDirBase from fs_default. "
        "Use a WLCG site path (T3_CH_CERNBOX:/store/user/<you>/...) or a CERN "
        f"EOS davs://eoshome-... URL. Got: {fs_spec}"
    )


class CrabWorkflow(law.cms.CrabWorkflow):
    """CRAB (WLCG) remote workflow, built on law.contrib.cms.CrabWorkflow.

    CRAB is only the batch backend. **All analysis products and logs use FLAF remote
    I/O** (``fs_default`` / gfal via task targets and ``stageout_logs.sh``). CRAB
    ``transferOutputs`` / ``transferLogs`` / ``JobType.outputFiles`` are forced off so
    nothing is duplicated onto CRAB's stageout area.

    ``Site.storageSite`` / ``Data.outLFNDirBase`` are derived from ``fs_default``
    (submit-time write check only). Site white/black lists come from the ``crab:``
    section of ``global.yaml``. Memory is ``2 GB * n_cpus``.

    CRAB workers have no AFS, so code is always shipped via the existing BundleTask
    mechanism (same as ``--bundle`` on HTCondor). Tasks must declare ``bundle_flavours``.

    Config (``global.yaml`` / user_custom YAML)::

        crab:
          whitelist: [T2_CH_CERN]
          # blacklist: [T2_US_MIT]
    """

    # Re-declare in the class body so law's metaclass sets _defined_workflow_proxy=True
    # and find_workflow_cls('crab') resolves to *this* class (not law.cms.CrabWorkflow).
    workflow_proxy_cls = _FLAFCrabWorkflowProxy

    poll_interval = copy_param(law.cms.CrabWorkflow.poll_interval, 5)
    # When True, law names the worker log ``stdall.txt`` and FLAF stageout_logs.sh
    # uploads it to fs_default. CRAB itself never transfers this file.
    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="enable FLAF remote log stageout (stdall.txt via stageout_logs.sh); "
        "CRAB transferLogs stays off",
    )

    def _crab_cfg(self):
        return self.global_params.get("crab") or {}

    def _ensure_crab_pset(self, n_threads):
        """Write a minimal CRAB PSet with numberOfThreads matching JobType.numCores."""
        n_threads = max(1, int(n_threads))
        out_dir = self.local_path()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"crab_PSet_threads{n_threads}.py")
        content = f"""# Auto-generated by FLAF for CRAB (threads must match JobType.numCores).
import FWCore.ParameterSet.Config as cms

process = cms.Process("LAW")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring([""]))
process.output = cms.OutputModule(
    "PoolOutputModule", fileName=cms.untracked.string("out.root")
)
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False),
    numberOfThreads=cms.untracked.uint32({n_threads}),
    numberOfStreams=cms.untracked.uint32(0),
)
process.out = cms.EndPath(process.output)
"""
        if (not os.path.exists(path)) or open(path).read() != content:
            with open(path, "w") as f:
                f.write(content)
        return path

    def crab_stageout_location(self):
        """Return (storageSite, outLFNDirBase) derived from ``fs_default``.

        FLAF does **not** store analysis outputs here (CRAB transferOutputs is forced
        off; products go to ``fs_default``). CRAB still requires these fields and runs
        a submit-time write check against them.
        """
        return _crab_stageout_from_fs_spec(self.global_params.get("fs_default"))

    def crab_output_directory(self):
        return law.LocalDirectoryTarget(self.local_path())

    def crab_request_name(self, submit_jobs):
        # CRAB: no dots, max 100 characters.
        import uuid

        parts = [
            self.task_family.replace(".", "_"),
            str(self.version).replace(".", "_"),
            str(self.period).replace(".", "_"),
            uuid.uuid4().hex[:8],
        ]
        name = "_".join(parts)
        return re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:100]

    def crab_bootstrap_file(self):
        from law.job.base import JobInputFile

        return JobInputFile(
            path=os.path.join(self._flaf_root(), "bootstrap.sh"),
            copy=True,
            share=True,
            render_job=True,
        )

    def crab_stageout_file(self):
        from law.job.base import JobInputFile

        return JobInputFile(
            path=os.path.join(self._flaf_root(), "run_tools", "stageout_logs.sh"),
            copy=True,
            share=True,
            render_job=True,
        )

    def crab_workflow_requires(self):
        # Always require bundles for CRAB (no AFS on WLCG workers).
        if not self.bundle_flavours:
            raise RuntimeError(
                f"{self.__class__.__name__}: --workflow crab requires bundle_flavours "
                "on the task (code/environment shipped via BundleTask)"
            )
        return self._bundle_requirements()

    def crab_check_job_completeness(self):
        return False

    def crab_poll_callback(self, poll_data):
        update_kinit(verbose=0)
        return True

    def crab_job_file_factory_cls(self):
        return FLAFCrabJobFileFactory

    def crab_job_file(self):
        # Same deps_depth=0 patch as HTCondor: avoid huge print_deps on the worker.
        from law.job.base import JobInputFile

        original = law.util.law_src_path("job", "law_job.sh")
        custom = os.path.join(
            os.getenv("ANALYSIS_DATA_PATH"), "law_job_no_print_deps.sh"
        )
        if not os.path.exists(custom) or os.path.getmtime(original) > os.path.getmtime(
            custom
        ):
            with open(original) as f:
                content = f.read()
            content = re.sub(r'\bdeps_depth="[0-9]+"', 'deps_depth="0"', content)
            with open(custom, "w") as f:
                f.write(content)
            os.chmod(custom, 0o755)
        return JobInputFile(path=custom, copy=True, share=True, render_job=True)

    def crab_job_config(self, config, job_nums, branches=None):
        # law 0.1.20 calls crab_job_config(config, list(keys), list(values)); the base
        # signature documents a single submit_jobs arg, but the call site passes two lists.
        if not self.bundle_flavours:
            raise RuntimeError(
                f"{self.__class__.__name__}: --workflow crab requires bundle_flavours"
            )

        self._apply_bootstrap_path_render_variables(config)
        self._apply_bundle_render_variables(config)
        self._stage_user_custom_input(config)
        self._stage_path_cache_input(config)

        log_remote_base_url = self._log_remote_base_url()
        config.render_variables["log_remote_base_url"] = log_remote_base_url

        # Cores + memory. CRAB requires JobType.numCores == PSet numberOfThreads.
        # Memory is 2 GB per CPU from the existing n_cpus parameter.
        n_cpus = max(1, int(getattr(self, "n_cpus", 1) or 1))
        mem = n_cpus * 2000
        pset_path = self._ensure_crab_pset(n_cpus)
        config.crab.JobType.psetName = pset_path
        config.crab.JobType.numCores = n_cpus
        config.crab.JobType.maxMemoryMB = mem

        # Runtime limit (hours → minutes). CRAB jobs must download/unpack bundles before
        # the payload starts, so enforce a floor (default 60 min) even when the task's
        # max_runtime is tiny (e.g. HelloWorld 0.1 h would otherwise be 6 min).
        max_runtime = getattr(self, "max_runtime", None)
        if max_runtime is not None and float(max_runtime) > 0:
            try:
                cfg_floor = int(self._crab_cfg().get("min_runtime_min", 60))
                minutes = max(int(math.floor(float(max_runtime) * 60)), cfg_floor)
                config.crab.JobType.maxJobRuntimeMin = minutes
            except Exception:
                # Older CRAB clients may not support maxJobRuntimeMin; ignore if rejected later.
                pass

        # Site white/black lists come from global.yaml ``crab:`` (no CLI overrides).
        # CRAB requires a whitelist when jobs use synthetic userInputFiles (no
        # inputDataset), which is always the case for law CRAB workflows.
        whitelist = list(self._crab_cfg().get("whitelist") or [])
        blacklist = list(self._crab_cfg().get("blacklist") or [])
        if not whitelist and not blacklist:
            raise RuntimeError(
                "CRAB requires crab.whitelist (or crab.blacklist) in global.yaml "
                "under the crab: section. Example:\n"
                "  crab:\n    whitelist: [T2_CH_CERN]"
            )
        if whitelist:
            config.crab.Site.whitelist = [str(s) for s in whitelist]
            config.crab.Site.ignoreGlobalBlacklist = True
            config.crab.Data.ignoreLocality = True
        elif blacklist:
            config.crab.Site.blacklist = [str(s) for s in blacklist]

        return config
