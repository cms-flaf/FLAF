import copy
import hashlib
import importlib
import law
import luigi
import math
import os
import re
import shutil
import sys
import subprocess
import tempfile
import threading
import time

from collections import Counter, OrderedDict

from law.parser import global_cmdline_values

from FLAF.RunKit.run_tools import natural_sort, on_batch_node, timed_call_wrapper
from FLAF.RunKit.kinit import update_kinit
from FLAF.RunKit.law_gfal import require_fresh_negatives
from FLAF.run_tools.crab_sites import SiteStats, processing_sites, resolve_whitelist
from FLAF.run_tools.crab_watchdog import (
    HEARTBEAT_DIR,
    Heartbeat,
    StallWatchdog,
    watchdog_config,
)
from FLAF.RunKit.law_wlcg import WLCGFileSystem, WLCGFileTarget, WLCGDirectoryTarget
from FLAF.Common.Setup import Setup
from FLAF.AnaProd.CostModel import pack_units

law.contrib.load("htcondor")
law.contrib.load("cms")


# law resolves the files it ships with every job -- law_job.sh, the CRAB wrapper and PSet,
# the HTCondor wrapper -- as `rel_path(__file__, ...)`, and `law.util.rel_path` strips the
# file name from the anchor only when a stat confirms it is a file. One failed stat of the
# software tree (an AFS token lapsing, an EOS mount blinking) makes law treat its own module
# as a directory, and the submission dies copying `.../job.py/crab/crab_wrapper.sh` -- raised
# inside law's submit(), where nothing catches it, so the whole workflow fails (DSProd lost a
# production this way on 2026-08-31). A module file is never a directory.
def _strict_rel_path(anchor, *paths):
    anchor = os.path.abspath(os.path.expandvars(os.path.expanduser(str(anchor))))
    if anchor.endswith((".py", ".pyc")) or os.path.isfile(anchor):
        anchor = os.path.dirname(anchor)
    return os.path.normpath(os.path.join(anchor, *map(str, paths)))


# Every law module imports the function by name, so each binding is replaced; this runs after
# the contrib packages above have been loaded.
_law_rel_path = law.util.rel_path
for _module in list(sys.modules.values()):
    if getattr(_module, "__name__", "").split(".")[0] == "law" and (
        getattr(_module, "rel_path", None) is _law_rel_path
    ):
        _module.rel_path = _strict_rel_path


def submitted_task_family():
    """The task family a batch job was submitted to run, or None if it cannot be told.

    law's job script runs ``law run <Task> --branch(es) ...`` on the worker, so the root task
    of the command line is the submitted one.
    """
    parser = luigi.cmdline_parser.CmdlineParser.get_instance()
    root_task = getattr(getattr(parser, "known_args", None), "root_task", None)
    if not root_task:
        return None
    return root_task.rsplit(".", 1)[-1]


#: law's own job sources, relative to the law package, that a job file is built from
_LAW_JOB_SOURCES = (
    ("job", "law_job.sh"),
    ("contrib", "cms", "crab", "crab_wrapper.sh"),
    ("contrib", "cms", "crab", "PSet.py"),
    ("contrib", "htcondor", "htcondor_wrapper.sh"),
)

#: FLAF's own job sources, relative to the FLAF root
_FLAF_JOB_SOURCES = (("bootstrap.sh",), ("run_tools", "stageout_logs.sh"))


def flaf_root():
    """FLAF source root, respecting the dev overlay (see HTCondorWorkflow._flaf_root)."""
    return os.getenv("FLAF_PATH") or os.path.join(os.getenv("ANALYSIS_PATH"), "FLAF")


def job_source_paths():
    """The files a job file is built from, resolved without a stat."""
    law_dir = os.path.dirname(os.path.abspath(law.__file__))
    paths = [os.path.join(law_dir, *parts) for parts in _LAW_JOB_SOURCES]
    paths += [os.path.join(flaf_root(), *parts) for parts in _FLAF_JOB_SOURCES]
    return paths


def missing_job_source(retries=5, delay=3.0):
    """The first job source that cannot be read, or None if all can.

    No cause is diagnosed here: `os.path.isfile` answers False for a missing path, a refused
    one and a failing mount alike (see `job_source_error`).
    """
    for path in job_source_paths():
        for attempt in range(retries + 1):
            if os.path.isfile(path):
                break
            if attempt < retries:
                time.sleep(delay)
        else:
            return path
    return None


def job_source_error(path):
    """What the storage answers for `path`, in words a message can be acted on."""
    try:
        os.stat(path)
    except OSError as e:
        return f"[Errno {e.errno}] {e.strerror}"
    # the probe and this stat are seconds apart, so a path that blipped is answered for again
    return "stat succeeds now -- not a regular file, or the path came back"


_JOB_SOURCE_HINT = (
    "ENOENT points at the software tree or its mount, EACCES/EPERM at the credential that "
    "storage is reached with: `klist -f` shows the Kerberos expiry and the renewable window "
    "(fixed 7 days after the original kinit; a running production only ever renews it), "
    "`tokens` the AFS token."
)


def wait_for_job_sources():
    """Refuse to build a job file against a software tree that cannot be read.

    The proxies probe the same sources before law is handed a submission round, so reaching
    this raise means the tree went away inside one submission.
    """
    path = missing_job_source()
    if path is not None:
        raise RuntimeError(
            f"{path} is not readable ({job_source_error(path)}), so no job file can be "
            f"built. {_JOB_SOURCE_HINT}"
        )


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
        "parallel_jobs",
        "poll_interval",
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


# Files up to this size are hashed by content when a bundle flavour is `hashed`; larger ones
# by size and modification time (see BundleTask.source_hash).
HASH_CONTENT_MAX_BYTES = 1024 * 1024


class BundleTask(Task):
    flavour = luigi.Parameter(
        description="bundle flavour (core, cmssw, inputFileList, AnaTupleFileList)"
    )
    # The workflow the submission was started with, forwarded to the tasks whose output is
    # packed so that they are the very instances the rest of the graph depends on. Without
    # it they would be requested with the default workflow, and an incomplete one would be
    # run a second time — locally. Insignificant, so it never affects the bundle's id.
    upstream_workflow = luigi.Parameter(default=law.NO_STR, significant=False)

    def requires(self):
        """Every task whose output this bundle packs.

        `task_requires` takes one entry or a list of them: a bundle that packs the outputs of
        several tasks has to wait for all of them, or it is built from whatever happens to be
        on disk when the first producer finishes.
        """
        reqs = {}
        for entry in self.task_requires_entries():
            mod = importlib.import_module(entry["module"])
            task_cls = getattr(mod, entry["class"])
            reqs[entry["class"]] = task_cls.req(
                self, branches=(), workflow=self.upstream_workflow
            )
        self._warn_about_unrequired_producers(reqs)
        return reqs

    def task_requires_entries(self):
        cfg = self.bundle_cfg().get("task_requires")
        if cfg is None:
            return []
        return list(cfg) if isinstance(cfg, (list, tuple)) else [cfg]

    def _warn_about_unrequired_producers(self, reqs):
        """A packed `data/<version>/<Task>/<period>` directory whose task is not required
        would be bundled while its jobs are still running."""
        for pattern in self.bundle_patterns():
            parts = pattern.strip("/").split("/")
            if len(parts) < 3 or parts[0] != "data" or parts[-1] != str(self.period):
                continue
            producer = parts[-2]
            if producer in reqs:
                continue
            key = (self.flavour, producer)
            if key in BundleTask._missing_producer_reported:
                continue
            BundleTask._missing_producer_reported.add(key)
            print(
                f"bundle[{self.flavour}]: warning: '{pattern}' holds the output of "
                f"{producer}, which is not listed in task_requires — the bundle can be "
                "built before those jobs have finished",
                file=sys.stderr,
            )

    _source_hash_cache = {}
    _unconfigured_reported = set()
    _missing_producer_reported = set()

    def bundle_cfg(self):
        return self.global_params.get("bundles", {}).get(self.flavour) or {}

    def bundle_patterns(self):
        return [
            p.format(version=self.version, period=self.period)
            for p in self.bundle_cfg().get("patterns", [])
        ]

    def bundle_source(self, pattern):
        """Where a pattern is read from.

        "FLAF" and "Corrections" come from FLAF_PATH / CORRECTIONS_PATH. env.sh always sets
        these (to the submodule copies in production, or to the edited top-level copies in a
        FLAF_all workspace when flaf_dev.sh is used), so dev edits are packaged transparently.
        The layout *inside* the tarball stays canonical (FLAF/, Corrections/ at the top) so
        worker bootstrap is unaffected.
        """
        ana_path = os.getenv("ANALYSIS_PATH")
        p = pattern.replace("\\", "/")
        if p == "FLAF" or p.startswith("FLAF/"):
            rel = p[5:] if p.startswith("FLAF/") else ""
            base = os.getenv("FLAF_PATH") or os.path.join(ana_path, "FLAF")
            return os.path.join(base, rel) if rel else base
        if p == "Corrections" or p.startswith("Corrections/"):
            rel = p[12:] if p.startswith("Corrections/") else ""
            base = os.getenv("CORRECTIONS_PATH") or os.path.join(
                ana_path, "Corrections"
            )
            return os.path.join(base, rel) if rel else base
        return os.path.join(ana_path, pattern)

    def source_hash(self):
        """Digest of the state of what this flavour packs, so that an edit yields a
        different bundle.

        A bundle's output is otherwise just a path, and law treats an existing one as
        complete forever: jobs would keep unpacking the code and configs of whenever it was
        first built and rebuild their branch map from those. Small files are read, large ones are
        identified by size and modification time: hashing ~150 MB of models on every
        submission costs half a minute, while the code and configs that actually get edited
        are a few MB. Pure stat would not do for those — AFS timestamps have a one-second
        granularity, so a same-size rewrite within the same second would go unnoticed.
        Touching a large file without changing it only causes a harmless rebuild. Flavours
        carrying a large immutable payload (an installed environment, a CMSSW release) opt
        out with `hashed: false`, the default.
        """
        key = (self.flavour, self.version, self.period)
        if key not in BundleTask._source_hash_cache:
            digest = hashlib.sha256()
            for pattern in self.bundle_patterns():
                digest.update(f"\n{pattern}\n".encode())
                source = os.path.realpath(self.bundle_source(pattern))
                if not os.path.exists(source):
                    digest.update(b"<absent>")
                    continue
                for path, rel in self._iter_bundle_files(source):
                    digest.update(rel.encode())
                    if os.path.islink(path):
                        digest.update(os.readlink(path).encode())
                        continue
                    info = os.lstat(path)
                    if info.st_size > HASH_CONTENT_MAX_BYTES:
                        digest.update(f"{info.st_size}:{info.st_mtime_ns}".encode())
                        continue
                    with open(path, "rb") as f:
                        digest.update(f.read())
            BundleTask._source_hash_cache[key] = digest.hexdigest()[:12]
        return BundleTask._source_hash_cache[key]

    @staticmethod
    def _iter_bundle_files(source):
        """(path, relative path) of everything packed from *source*, in a stable order."""
        if not os.path.isdir(source):
            yield source, os.path.basename(source)
            return
        for dir_path, dir_names, file_names in os.walk(source, followlinks=False):
            dir_names[:] = sorted(d for d in dir_names if d != "__pycache__")
            for name in sorted(file_names):
                if name.endswith((".pyc", ".pyo")):
                    continue
                path = os.path.join(dir_path, name)
                yield path, os.path.relpath(path, source)

    def packs_environment(self):
        """Whether this flavour packs flaf_env.

        Decided on the paths alone: a stat that fails while the storage blinks must not flip
        the bundle's name.
        """
        env = os.path.abspath(os.environ["FLAF_ENVIRONMENT_PATH"])
        for pattern in self.bundle_patterns():
            source = os.path.abspath(self.bundle_source(pattern))
            if env == source or env.startswith(source + os.sep):
                return True
        return False

    def output(self):
        name = self.flavour
        if self.bundle_cfg().get("hashed", False):
            name = f"{self.flavour}_{self.source_hash()}"
        elif self.packs_environment():
            # An unhashed bundle is complete once it exists. Named after the environment it
            # packs, a rebuilt one (another installation script, other pins) is packed anew
            # instead of shipping the old packages with a job script rendered from the new.
            environment_id = os.environ.get("FLAF_ENVIRONMENT_ID")
            if not environment_id:
                raise RuntimeError(
                    f"bundle flavour '{self.flavour}' packs flaf_env, but FLAF_ENVIRONMENT_ID, "
                    "which names it, is not set: source the analysis env.sh"
                )
            name = f"{self.flavour}_{environment_id}"
        return self.remote_target(
            self.version, "bundles", self.period, f"{name}.tar.bz2"
        )

    def run(self):
        bundle_cfg = self.bundle_cfg()
        if not bundle_cfg:
            raise RuntimeError(
                f"Bundle flavour '{self.flavour}' not configured in bundles section of global.yaml"
            )

        ana_path = os.getenv("ANALYSIS_PATH")
        formatted_patterns = self.bundle_patterns()

        os.makedirs(self.local_path(), exist_ok=True)

        print(f"bundle[{self.flavour}]: creating archive from {ana_path}")
        with self.output().localize("w") as tmp:
            with tempfile.TemporaryDirectory() as staging:
                found_any = False
                for pattern in formatted_patterns:
                    full_path = self.bundle_source(pattern)
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


#: law's per-branch run loop in law_job.sh (law 0.1.21): one `law run --branch=b` process
#: per branch, and the job stops at the first branch that fails
_LAW_JOB_BRANCH_LOOP_RE = re.compile(
    r"^    for b in \$\{LAW_JOB_TASK_BRANCHES\}; do\n(?:.*\n)*?^    done\n",
    re.MULTILINE,
)

#: what FLAF runs instead: law 0.1.20's grouped run, all branches of a job in one law process
#: (`--branches=... --workflow=local` for more than one), so a failing branch does not keep
#: its group-mates from running, and no dependency tree is printed (`--print-deps=0`)
_LAW_JOB_GROUPED_RUN = r"""    local branch_param="branch"
    local workflow_param=""
    if [ "${LAW_JOB_TASK_N_BRANCHES}" != "1" ]; then
        branch_param="branches"
        workflow_param="--workflow=local"
    fi

    _law_job_section "run task ${branch_param} ${LAW_JOB_TASK_BRANCHES_CSV}"

    # build the full command
    local cmd="${law_exe} run ${LAW_JOB_TASK_MODULE}.${LAW_JOB_TASK_CLASS} ${LAW_JOB_TASK_PARAMS} --${branch_param}=${LAW_JOB_TASK_BRANCHES_CSV} ${workflow_param} --workers=${LAW_JOB_WORKERS}"
    echo "cmd: ${cmd}"
    echo

    _law_job_subsection "dependency tree"
    eval "LAW_LOG_LEVEL=INFO ${cmd} --print-deps=0"
    local law_ret="$?"
    if [ "${law_ret}" != "0" ]; then
        >&2 echo "dependency tree for ${branch_param} ${LAW_JOB_TASK_BRANCHES_CSV} failed (exit code ${law_ret}), stop job"
        _law_job_call_hook law_hook_job_failed "50" "${law_ret}"
        _law_job_finalize "50" "${law_ret}"
        return "$?"
    fi

    echo
    _law_job_subsection "execute attempt 1"
    export LAW_JOB_ATTEMPT="1"
    date +"%d/%m/%Y %T.%N (%Z)"
    eval "${cmd}"
    law_ret="$?"
    echo "task exit code: ${law_ret}"
    date +"%d/%m/%Y %T.%N (%Z)"

    if [ "${law_ret}" != "0" ] && [ "${LAW_JOB_AUTO_RETRY}" = "yes" ]; then
        echo
        _law_job_subsection "execute attempt 2"
        export LAW_JOB_ATTEMPT="2"
        date +"%d/%m/%Y %T.%N (%Z)"
        eval "${cmd}"
        law_ret="$?"
        echo "task exit code: ${law_ret}"
        date +"%d/%m/%Y %T.%N (%Z)"
    fi

    if [ "${law_ret}" != "0" ]; then
        >&2 echo "execution of ${branch_param} ${LAW_JOB_TASK_BRANCHES_CSV} failed (exit code ${law_ret}), stop job"
        _law_job_call_hook law_hook_job_failed "60" "${law_ret}"
        _law_job_finalize "60" "${law_ret}"
        return "$?"
    fi
"""

#: law_job.sh path -> (file name, content) of the script generated from it in this process
_grouped_law_job_scripts = {}


def grouped_law_job_script():
    """law's job script with its per-branch run loop replaced by FLAF's grouped run.

    The file is named after its content, so a different law, or a change of the block here,
    gives a new file instead of reusing a stale one. law's script is read once per process: a
    read of the software tree that fails later -- its storage blinking -- does not fail the
    submission.
    """
    original = law.util.law_src_path("job", "law_job.sh")
    if original not in _grouped_law_job_scripts:
        with open(original) as f:
            content, n = _LAW_JOB_BRANCH_LOOP_RE.subn(
                lambda _: _LAW_JOB_GROUPED_RUN, f.read()
            )
        if n != 1:
            raise RuntimeError(
                f"{original} (law {law.__version__}) has {n} per-branch run loops where "
                "FLAF expects exactly one, so the grouped run cannot be put in its place"
            )
        digest = hashlib.sha256(content.encode()).hexdigest()[:12]
        _grouped_law_job_scripts[original] = (f"law_job_flaf_{digest}.sh", content)
    name, content = _grouped_law_job_scripts[original]
    custom = os.path.join(os.getenv("ANALYSIS_DATA_PATH"), name)
    if not os.path.exists(custom):
        tmp = f"{custom}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(content)
        os.chmod(tmp, 0o755)
        os.replace(tmp, custom)
    return custom


class CERNHTCondorJobFileFactory(law.htcondor.HTCondorJobFileFactory):
    """HTCondor job file factory that stages transfer_input_files to EOS and uses protocol URLs.

    When config._worker_files_remote_dir is set (a WLCGDirectoryTarget), every file listed in
    transfer_input_files is uploaded to that remote directory and its path in the JDL is replaced
    with the corresponding remote URL.  This lets CERN HTCondor fetch input files from EOS via the
    protocol layer instead of trying to read /eos POSIX paths, which the batch system does not
    support.
    """

    def create(self, **kwargs):
        wait_for_job_sources()
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

    # Resource requests are per-task decisions: without this, a requiring task's own
    # max_runtime / n_cpus (e.g. a 2 h / 1 CPU plot task) would be copied through req()
    # onto everything it requires, silently capping the production jobs upstream.
    # Workflow <-> branch conversion is unaffected (law passes _skip_task_excludes there),
    # so CLI-given per-task values still reach that task's branches.
    exclude_params_req = law.htcondor.HTCondorWorkflow.exclude_params_req | {
        "max_runtime",
        "n_cpus",
    }

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
        return flaf_root()

    def _refuse_inline_on_worker(self):
        """Refuse to run this producer inside a batch job submitted for another task.

        A job runs ``law run <Task> --branch ...``, and luigi runs, in the job's own slot, any
        requirement it reads as incomplete there. On a worker that reading can be wrong -- a
        listing blink, a stale entry in the shipped path-cache snapshot, inputs removed after
        a merge -- and the job then rebuilds an upstream product inside a slot sized for
        another task, overwriting a file other jobs may be reading (DSProd: a production job
        built its own gridpack). Called at the top of the expensive producers' run(); a
        command line whose root task cannot be told is let through.
        """
        if not on_batch_node():
            return
        family = submitted_task_family()
        if family is None or family == self.get_task_family():
            return
        raise RuntimeError(
            f"{self.get_task_family()} branch {getattr(self, 'branch', None)} is not "
            f"complete as seen from this {family} job, which will not build it inline. "
            "Either its output is really missing -- produce it first -- or the storage could "
            "not be read from the worker, in which case a retry succeeds."
        )

    def _uses_bundles(self):
        """Whether this submission should ship and unpack code bundles on the worker.

        Bundles are optional for HTCondor (shared AFS is available) but required for CRAB
        (WLCG workers have no AFS mount).
        """
        # A worker already runs from the unpacked bundle. The --bundle flag is forwarded
        # into the worker command line (insignificant params are serialized too), and a
        # grouped job evaluates workflow_requires() there — without this guard it would
        # require BundleTask and, on a transient false-incomplete, rebuild and overwrite
        # the live tarball other jobs are downloading.
        if on_batch_node():
            return False
        if not self.bundle_flavours:
            return False
        if getattr(self, "effective_workflow", None) == "crab":
            return True
        return bool(self.bundle)

    def _bundle_tasks(self):
        """(flavour, BundleTask) for each flavour this task needs.

        A flavour the analysis does not configure is skipped rather than fatal: FLAF asks for
        the flavours it knows about, and an analysis that keeps e.g. its environment inside
        another bundle simply has fewer of them.
        """
        configured = self.global_params.get("bundles", {})
        tasks = []
        for item in self.bundle_flavours:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                flavour, bversion = item
            else:
                flavour, bversion = item, self.version
            if flavour not in configured:
                if flavour not in BundleTask._unconfigured_reported:
                    BundleTask._unconfigured_reported.add(flavour)
                    print(
                        f"bundle: flavour '{flavour}' is not configured in global.yaml, skipping",
                        file=sys.stderr,
                    )
                continue
            tasks.append(
                (
                    flavour,
                    BundleTask.req(
                        self,
                        flavour=flavour,
                        version=bversion,
                        upstream_workflow=getattr(self, "workflow", law.NO_STR)
                        or law.NO_STR,
                    ),
                )
            )
        return tasks

    def _bundle_requirements(self):
        """Return BundleTask requirements for configured flavours (empty if unused)."""
        if not self._uses_bundles():
            return {}
        return {"bundles": [task for _, task in self._bundle_tasks()]}

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
        # Ask the task for its own output: the file name carries a content hash for the
        # flavours that use one, so this must not be rebuilt by hand here.
        bundle_parts = [
            f"{flavour}:{task.output().uri()}" for flavour, task in self._bundle_tasks()
        ]
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
        harvest = getattr(self.workflow_proxy, "harvest_job_durations", None)
        if harvest is not None:
            harvest()
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

        # maximum runtime, extended on every resubmission: a job that was removed at the
        # wall would be removed again if it were given exactly the same budget.
        runtime_factor, memory_factor = self._retry_resource_factors(job_num)
        config.custom_content.append(
            (
                "+MaxRuntime",
                int(math.floor(self.max_runtime * runtime_factor * 3600)) - 1,
            )
        )
        request_memory = getattr(self, "cost_params", None) and self.cost_params().get(
            "request_memory_mb"
        )
        if request_memory:
            config.custom_content.append(
                ("RequestMemory", int(request_memory * memory_factor))
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

    def _retry_resource_factors(self, job_num):
        """(runtime, memory) multipliers for the current attempt of *job_num*.

        Law passes a single job number when submitting one job at a time and the whole
        list when it submits a group through one shared job file; a group shares its
        resource request, so it is sized for its most-retried member.
        """
        if getattr(self, "cost_params", None) is None:
            return 1.0, 1.0
        proxy = getattr(self, "workflow_proxy", None)
        # An explicit --tasks-per-job opts out of cost-aware scheduling as a whole, the
        # per-attempt escalation included.
        if not getattr(proxy, "_cost_scheduling_enabled", lambda: False)():
            return 1.0, 1.0
        params = self.cost_params()
        attempts = getattr(getattr(proxy, "job_data", None), "attempts", None) or {}
        job_nums = job_num if isinstance(job_num, (list, tuple, set)) else [job_num]
        attempt = max((attempts.get(n, 0) for n in job_nums), default=0)
        if attempt <= 0:
            return 1.0, 1.0
        cap = float(params["retry_max_factor"])
        return (
            min(float(params["retry_runtime_factor"]) ** attempt, cap),
            min(float(params["retry_memory_factor"]) ** attempt, cap),
        )

    def htcondor_job_file(self):
        from law.job.base import JobInputFile

        return JobInputFile(
            path=grouped_law_job_script(), copy=True, share=True, render_job=True
        )


# Custom proxy subclass so that the "log" location recorded in job submission data
# (used by law for "first log file: ..." messages at submit time, stored job json,
# and "task failed" diagnostics) points at the *remote* staged logs location for
# bundle runs instead of the local AFS path under ANALYSIS_DATA_PATH.
# The basename computation (stdall, stdall_Cluster_Proc, or stdall<postfix>_Cluster.Proc)
# is the same one used by stageout_logs.sh, so the URI will match the uploaded file.
#
# Use the stable extension point: obtain the base proxy class from whatever
# the current law version has configured on HTCondorWorkflow.workflow_proxy_cls.

BundleAwareHTCondorWorkflowProxyBase = HTCondorWorkflow.workflow_proxy_cls


class LawProxyState:
    """Workflow-proxy state of law that the FLAF proxies read and change.

    law 0.1.21 keeps the skip-verdict cache and the retry counters in ``_skip_jobs`` /
    ``_job_retries``, outside its API. They are named for that release alone, the one
    ``run_tools/mk_flaf_env.sh`` installs.
    """

    @property
    def _cost_skip_jobs(self):
        return self._skip_jobs

    @property
    def _cost_job_retries(self):
        return self._job_retries


class SubmissionGuards(LawProxyState):
    """Remote-workflow-proxy mixin: what both FLAF proxies check before a submission round.

    * A software tree that cannot be read. The job file is built inside law's submit(), and
      the error raised there for an unreadable source is caught nowhere between it and luigi,
      so one blink of the storage the tree lives on fails the whole workflow and ends the
      driver (DSProd, twice on consecutive days). The round is skipped instead -- the offered
      retries are parked, so no attempt is spent on it -- and the next poll tries again.
    * A resumed workflow whose jobs come back in large numbers for missing outputs. law
      retries a job it had recorded as finished whose outputs are gone ("unknown job id" -- a
      recorded-finished job keeps no job id) and a live one reported finished without them
      ("initially missing task outputs"), so a storage outage during that check, or outputs
      removed after use, would resubmit most of a production (DSProd: 8300 jobs). The run
      stops instead and says why.
    """

    #: share of a resumed workflow's jobs that may come back for missing outputs in one go
    max_lost_fraction = 0.1
    min_lost_jobs = 2

    #: how long submission rounds may be skipped for an unreadable software tree
    max_skip_minutes = 30.0

    #: law's error for a still-live job whose outputs are missing on a resumed run
    missing_outputs_error = "initially missing task outputs"

    _resumed_jobs = None
    _resumed_attempts = None
    _lost_outputs_judged = False
    _skipping_since = None

    def _snapshot_resumed_jobs(self):
        """The job data a resumed run starts from, taken before the first poll changes it."""
        if self._submitted and self._resumed_jobs is None:
            self._resumed_jobs = copy.deepcopy(dict(self.job_data.jobs))
            self._resumed_attempts = dict(self.job_data.attempts)

    def _lost_output_candidates(self, job_nums):
        """Of `job_nums`, the jobs that came back for missing outputs: recorded as finished
        when this run started, or reported finished by a live job without them."""
        before = self._resumed_jobs or {}
        finished = self.job_manager.FINISHED
        return [
            job_num
            for job_num in job_nums
            if (before.get(job_num) or {}).get("status") == finished
            or (self.job_data.jobs.get(job_num) or {}).get("error")
            == self.missing_outputs_error
        ]

    def _restore_resumed(self, job_nums):
        """Put jobs back as this run loaded them (entry and attempts); returns what was
        there before, for `_put_back`."""
        replaced = {}
        jobs = self._resumed_jobs or {}
        attempts = self._resumed_attempts or {}
        for job_num in job_nums:
            replaced[job_num] = (
                self.job_data.jobs.get(job_num),
                self.job_data.attempts.get(job_num),
            )
            if job_num in jobs:
                self.job_data.jobs[job_num] = jobs[job_num]
            if job_num in attempts:
                self.job_data.attempts[job_num] = attempts[job_num]
            else:
                self.job_data.attempts.pop(job_num, None)
        return replaced

    def _put_back(self, replaced):
        for job_num, (data, attempts) in replaced.items():
            if data is not None:
                self.job_data.jobs[job_num] = data
            if attempts is None:
                self.job_data.attempts.pop(job_num, None)
            else:
                self.job_data.attempts[job_num] = attempts

    def dump_job_data(self):
        """Until the first retry generation of a resumed run has been judged, write the jobs
        that came back for missing outputs as they were loaded.

        law dumps its rewrite of them (a retry, one more attempt) before it hands them to
        submit(), where they are judged; a driver ending in between would leave the next run
        retries it does not recognise, and nothing to judge.
        """
        if (
            not self._submitted
            or self._resumed_jobs is None
            or self._lost_outputs_judged
        ):
            return super(SubmissionGuards, self).dump_job_data()
        failed = (self.job_manager.RETRY, self.job_manager.FAILED)
        pending = self._lost_output_candidates(
            [
                n
                for n, d in self.job_data.jobs.items()
                if (d or {}).get("status") in failed
            ]
        )
        if not pending:
            return super(SubmissionGuards, self).dump_job_data()
        replaced = self._restore_resumed(pending)
        try:
            return super(SubmissionGuards, self).dump_job_data()
        finally:
            self._put_back(replaced)

    def _too_many_lost(self, n_lost):
        return n_lost >= self.min_lost_jobs and n_lost > self.max_lost_fraction * len(
            self.job_data
        )

    def _stop_on_mass_lost_outputs(self, retry_jobs):
        """Raise instead of resubmitting, when most of a resumed workflow lost its outputs.

        Judged once, on the first retry generation of a resumed run: that is where law puts
        every job whose outputs went missing, and a later genuine failure must not be counted.
        Called before anything can park that generation -- the wave gate, or a skipped round
        -- since a parked mass retry would later be released where this no longer sees it.
        """
        if not self._submitted or self._lost_outputs_judged or not retry_jobs:
            return
        candidates = self._lost_output_candidates(retry_jobs)
        self._lost_outputs_judged = True
        if not self._too_many_lost(len(candidates)):
            return
        # law judged from what it gathered while the workflow was scheduled, through cached
        # existence answers: a live job may have finished since, and a branch may be complete
        # by its task's own rule (e.g. inputs merged and replaced by markers). Only a job whose
        # branches are incomplete on a fresh look counts as lost; the others are marked
        # skippable, so that law's submit() passes them by and its next poll books them done.
        require_fresh_negatives()
        skip_jobs = self._skip_jobs
        lost = []
        for job_num in candidates:
            if all(self.task.as_branch(b).complete() for b in retry_jobs[job_num]):
                skip_jobs[job_num] = True
            else:
                lost.append(job_num)
        if not self._too_many_lost(len(lost)):
            return
        # law has already rewritten these jobs as retries and counted the attempt; write them
        # as they were, so that the next run finds and judges them again
        self._restore_resumed(candidates)
        self.dump_job_data()
        raise RuntimeError(
            f"{len(lost)} of the {len(self.job_data)} jobs of this resumed workflow came back "
            f"for missing outputs (more than {self.max_lost_fraction:.0%}), and their outputs "
            "are still missing on a fresh look, so this run would redo a large part of the "
            "workflow. Nothing was submitted, and the submission file was left as it was.\n"
            "  - if the storage was unreachable while the outputs were checked, run again once "
            "it is back;\n"
            "  - if they were removed on purpose after being used (e.g. merged inputs), the "
            "task that used them is what should run, not this workflow: check why it was "
            "scheduled;\n"
            "  - to redo the work deliberately, run again with --ignore-submission."
        )

    def _park_retries(self, retry_jobs):
        """Move a retry generation in front of the unsubmitted backlog, and dump.

        `unsubmitted_jobs` is where a held job has to wait: it is dumped to disk and counted by
        `JobData.__len__`, so a killed driver finds it again and the poll loop's job-count
        snapshot stays intact. In front, because law's submit() fills a round from it in dict
        order, and a retry queued behind a large backlog would not be reached for hours.
        """
        parked = OrderedDict()
        for job_num, branches in (retry_jobs or {}).items():
            if self._can_skip_job(job_num, branches):
                continue
            self.job_data.jobs.pop(job_num, None)
            parked[job_num] = branches
        if parked:
            parked.update(self.job_data.unsubmitted_jobs)
            self.job_data.unsubmitted_jobs = parked
        if retry_jobs or parked:
            self.dump_job_data()
        return parked

    def _skip_submission_round(self, retry_jobs):
        """True when the job sources cannot be read and this round must not reach law.

        Probed once rather than waited out, because this runs inside the poll loop. Skipping is
        bounded in time: the job data may live on other storage than the software tree, so a
        permanent outage of the tree alone would otherwise loop for ever.
        """
        if not (retry_jobs or self.job_data.unsubmitted_jobs):
            return False
        missing = missing_job_source(retries=1, delay=1.0)
        if missing is None:
            self._skipping_since = None
            return False
        reason = f"{missing} is not readable ({job_source_error(missing)})"
        if getattr(self.task, "no_poll", False):
            raise RuntimeError(
                f"{reason}, so no job file can be built, and with --no-poll nothing would "
                f"submit this round later. {_JOB_SOURCE_HINT}"
            )
        now = time.monotonic()
        if self._skipping_since is None:
            self._skipping_since = now
        elif now - self._skipping_since > self.max_skip_minutes * 60:
            raise RuntimeError(
                f"{reason}, and submission rounds have been skipped for more than "
                f"{self.max_skip_minutes:.0f} minutes. {_JOB_SOURCE_HINT}"
            )
        self.task.publish_message(
            f"{reason}; skipping this submission round -- nothing is lost, the next poll "
            f"submits it. {_JOB_SOURCE_HINT}"
        )
        self._park_retries(retry_jobs)
        return True

    def poll(self):
        self._snapshot_resumed_jobs()
        return super(SubmissionGuards, self).poll()


class _BundleAwareHTCondorWorkflowProxy(
    SubmissionGuards, LawProxyState, BundleAwareHTCondorWorkflowProxyBase
):
    """HTCondor proxy with remote log paths and cost-aware job composition.

    law groups branches into jobs with ``iter_chunks(sorted(branch_map), tasks_per_job)``:
    fixed-size and contiguous.  When the per-branch cost is heavy-tailed -- as it is for
    AnaTuple production, where a handful of dilepton-skim files cost twenty times what the
    rest do, and where expensive branches are adjacent because they belong to the same
    dataset -- that packs the expensive work together into jobs that overrun the wall
    clock, get removed, and are retried with exactly the same grouping.

    A task opts in by providing ``branch_cost_map()``; then jobs are built to a target
    duration instead, and the durations of finished single-branch jobs are recorded to
    refine the estimates of the next run.
    """

    def __init__(self, *args, **kwargs):
        super(_BundleAwareHTCondorWorkflowProxy, self).__init__(*args, **kwargs)
        self._cost_repack_pending = True
        self._cost_poll_started = False
        self._cost_max_job_num = 0
        self._cost_own_jobs = set()
        self._job_started_at = {}
        self._job_harvested = set()
        self._apply_cost_parallel_jobs()

    def _cost_scheduling_enabled(self):
        if not callable(getattr(self.task, "branch_cost_map", None)):
            return False
        return not _cli_has_tasks_per_job(self.task.get_task_family())

    def process_resources(self, force=False):
        # luigi asks for the job resources while scheduling, and law answers by walking its
        # default grouping, recording every job whose branches all exist as a finished job
        # (`_can_skip_job`).  Cost-aware packing replaces that grouping, so those records
        # would remain as finished jobs that never existed and inflate the counts of the
        # whole run.  FLAF tasks declare no job resources, so nothing else is lost.
        if self._cost_scheduling_enabled():
            return {}
        return super(_BundleAwareHTCondorWorkflowProxy, self).process_resources(
            force=force
        )

    def _apply_cost_parallel_jobs(self):
        """Bound the queue footprint by default.

        Beyond queue hygiene this is what creates submission waves, and therefore the
        opportunity to re-pack the work that has not been submitted yet with the better
        estimates that the finished jobs provide.
        """
        if not self._cost_scheduling_enabled() or _cli_has_param(
            "parallel-jobs", self.task.get_task_family()
        ):
            return
        if self.poll_data.n_parallel != self.n_parallel_max:
            return
        n_parallel = int(self.task.cost_params().get("parallel_jobs") or 0)
        if n_parallel > 0:
            self._set_parallel_jobs(n_parallel)

    def _next_job_num(self):
        """A job number never used before.

        Numbers must never be recycled: ``_can_skip_job`` caches its verdict per number
        and ``job_retries`` / ``attempts`` are keyed by it, so reusing one for a different
        set of branches would silently apply stale bookkeeping.  The live dicts alone are
        not a safe source for the maximum -- law moves a job out of ``jobs`` when a retry
        cannot be submitted, and re-packing drops entries from ``unsubmitted_jobs`` -- so
        the high-water mark is kept on the proxy and fed from every dict that has ever
        been keyed by a job number.
        """
        nums = (
            list(self.job_data.jobs.keys())
            + list(self.job_data.unsubmitted_jobs.keys())
            + list(self.job_data.attempts.keys())
            + list(self._cost_job_retries.keys())
            + [self._cost_max_job_num]
        )
        self._cost_max_job_num = max(nums) + 1
        return self._cost_max_job_num

    def _cost_repack_unsubmitted(self):
        """Re-group the not-yet-submitted branches into jobs of bounded duration."""
        unsubmitted = self.job_data.unsubmitted_jobs
        if not unsubmitted:
            return
        branches = sorted({b for group in unsubmitted.values() for b in group})
        if not branches:
            return
        task = self.task
        try:
            costs = task.branch_cost_map()
            params = task.cost_params()
            capacity = task.cost_capacity_seconds()
        except Exception as e:
            print(f"cost-aware packing unavailable, keeping the current grouping: {e}")
            return
        default = (params["default_file_seconds"], "default")
        units = [(b,) + tuple(costs.get(b, default)) for b in branches]
        groups = pack_units(
            units, capacity, params["max_units_per_job"], params["tier_safety"]
        )
        if not groups:
            return
        # The packing covers every branch, so that a restarted production keeps the jobs a
        # fresh one would have; only the branches still missing are submitted.
        existing = self._get_existing_branches()
        n_packed = len(groups)
        groups = [g for g in ([b for b in g if b not in existing] for g in groups) if g]
        n_done = sum(1 for b in branches if b in existing)
        for job_num in list(unsubmitted.keys()):
            unsubmitted.pop(job_num, None)
            self._cost_skip_jobs.pop(job_num, None)
            self._cost_job_retries.pop(job_num, None)
            self.job_data.attempts.pop(job_num, None)
        job_num = self._next_job_num()
        for group in groups:
            unsubmitted[job_num] = sorted(group)
            job_num += 1
        self._cost_max_job_num = job_num - 1
        total = sum(costs.get(b, default)[0] for b in branches)
        message = (
            f"cost-aware packing: {len(branches)} branch(es), "
            f"{law.util.human_duration(seconds=int(total))} of estimated work "
            f"-> {n_packed} job(s) of at most "
            f"{law.util.human_duration(seconds=int(capacity))}"
        )
        if n_done:
            message += (
                f"; {n_done} branch(es) already produced, "
                f"{len(groups)} job(s) to submit"
            )
        self.task.publish_message(message)

    def _cost_repack_once(self):
        """Re-group what is still unsubmitted, at most once per process.

        law's poll() snapshots the total job count before its loop and derives both the
        acceptance threshold and the end-of-loop test from that snapshot, so the number of
        jobs must not change once polling has started: fewer than the snapshot and the
        loop can never reach the threshold, more and it returns as soon as the snapshot is
        met, leaving the extra jobs unharvested.  Both entry points below therefore run
        strictly before that snapshot is taken.
        """
        if not self._cost_scheduling_enabled() or not self._cost_repack_pending:
            return
        if self._cost_poll_started:
            return
        self._cost_repack_unsubmitted()
        self._cost_repack_pending = False

    def poll(self):
        # A resumed workflow never calls submit() before polling (law guards it with
        # `if not self._submitted`), and that is exactly the run that has measurements
        # from its predecessor to act on.  The snapshot is taken inside super().poll(),
        # so re-grouping here is still ahead of it.
        self._cost_repack_once()
        self._cost_poll_started = True
        return super(_BundleAwareHTCondorWorkflowProxy, self).poll()

    def submit(self, retry_jobs=None):
        self._stop_on_mass_lost_outputs(retry_jobs)
        if self._skip_submission_round(retry_jobs):
            return OrderedDict()
        self._cost_repack_once()
        new_submission_data = super(_BundleAwareHTCondorWorkflowProxy, self).submit(
            retry_jobs=retry_jobs
        )
        # Durations are only trusted for jobs this process submitted; recording them here
        # rather than in _submit_group covers the batch path as well.
        for job_num in new_submission_data or {}:
            if not isinstance(job_num, Exception):
                self._cost_own_jobs.add(job_num)
        return new_submission_data

    def harvest_job_durations(self):
        """Feed the durations of finished jobs back into the cost model.

        Durations are measured between the first poll that saw a job running and the
        first that saw it finished; the poll interval is negligible next to the
        multi-hour jobs this matters for, and jobs too short to carry any signal are
        discarded by the model.  The measurements land in the ``job`` tier, which the
        packer trusts without a safety margin, so only unambiguous samples are taken:

        * jobs this process submitted -- a job already running when the workflow was
          restarted would be timed from the restart, not from its real start;
        * single-branch jobs -- law skips a job only when *every* branch is complete, so
          a group may contain branches that were already done and return long before its
          nominal event count would suggest.

        Both errors are one-directional (the rate comes out too low) and would persist in
        a store that is shared across eras and across every later run of the version.
        """
        if not self._cost_scheduling_enabled():
            return
        now = time.time()
        samples = []
        for job_num, data in self.job_data.jobs.items():
            if job_num not in self._cost_own_jobs:
                continue
            status = data.get("status")
            if status == self.job_manager.RUNNING:
                self._job_started_at.setdefault(job_num, now)
            elif status == self.job_manager.FINISHED:
                if job_num in self._job_harvested:
                    continue
                self._job_harvested.add(job_num)
                started = self._job_started_at.pop(job_num, None)
                branches = data.get("branches") or []
                if started is not None and len(branches) == 1:
                    samples.append((branches, now - started))
        if not samples:
            return
        try:
            self.task.record_job_durations(samples)
        except Exception as e:
            print(f"cost model: unable to record job durations: {e}")

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

        # job_ids and submission_data are in the same order, as in law's own rewrite.
        for i, (job_id, (job_num, data)) in enumerate(
            zip(job_ids, list(submission_data.items()))
        ):
            if isinstance(job_num, Exception) or not isinstance(data, dict):
                continue
            log = data.get("log")
            if log:
                basename = os.path.basename(str(log))
                # Same rule as stageout_logs.sh: a resubmitted job keeps its postfix, so the
                # HTCondor job id is added to give every attempt its own log.
                config = data.get("config")
                if (
                    not isinstance(job_id, Exception)
                    and config is not None
                    and config.postfix_output_files
                    and config.postfix
                ):
                    cluster, process = str(job_id).split(".")
                    basename = f"stdall{config.postfix[i]}_{cluster}.{process}.txt"
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
    """

    def create(self, **kwargs):
        wait_for_job_sources()
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
        """Rewrite the generated CRAB cfg to drop output transfer."""
        with open(job_file) as f:
            lines = f.readlines()

        new_lines = []
        skip_list = False
        for ln in lines:
            stripped = ln.strip()

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


#: the payload's own exception in a CRAB job's stdout. CMSSW's wrapper prefixes the payload
#: stream with "== CMSSW:", and the LAST such line is the one that ended the job: a log also
#: carries harmless earlier ones.
_payload_error_cre = re.compile(
    r"^(?:==\s*CMSSW:\s*)?((?:\w+\.)*\w*(?:Error|Exception)):\s*(\S.*)$"
)

#: how much of one error line is printed
_payload_error_chars = 600

#: where the grid CAs live when a host has them; the schedd's certificate may need them
_grid_ca_path = "/etc/grid-security/certificates"

#: how much of a job's stdout is kept while looking for the payload's own error
_max_log_bytes = 4 * 1024 * 1024


def payload_error(text):
    """The last exception line of a job's stdout, trimmed to one line, or None."""
    found = None
    for line in text.splitlines():
        match = _payload_error_cre.match(line.strip())
        if match:
            found = f"{match.group(1)}: {match.group(2)}"
    if found and len(found) > _payload_error_chars:
        found = found[:_payload_error_chars] + " ..."
    return found


def fetch_job_stdout(url, max_bytes=_max_log_bytes, timeout=30.0, deadline=60.0):
    """The tail of a CRAB job's stdout from the scheduler, read with the run's grid proxy.

    The schedd serves it over HTTPS with client-certificate authentication, and the proxy the
    submission already needs is that certificate. Only the tail is wanted -- the exception that
    ended the job is at the end -- so it is asked for with a range request, and both the bytes
    and the wall clock are bounded: this runs inside a poll, and law waits on the query without
    a timeout of its own, so a slow transfer would hold up the status of every CRAB task.
    `timeout` bounds one socket operation, `deadline` the whole transfer.
    """
    import ssl
    import urllib.request

    proxy = os.environ.get("X509_USER_PROXY", "")
    if not proxy or not os.path.exists(proxy):
        raise RuntimeError("no X509_USER_PROXY to authenticate with")
    context = ssl.create_default_context()
    if os.path.isdir(_grid_ca_path):
        context.load_verify_locations(capath=_grid_ca_path)
    context.load_cert_chain(proxy, proxy)
    request = urllib.request.Request(url, headers={"Range": f"bytes=-{int(max_bytes)}"})
    started = time.monotonic()
    chunks, size, read = [], 0, 0
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        ranged = response.status == 206
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            read += len(chunk)
            chunks.append(chunk)
            size += len(chunk)
            while size > max_bytes and len(chunks) > 1:
                size -= len(chunks.pop(0))
            elapsed = time.monotonic() - started
            if elapsed > deadline or (not ranged and read > 8 * max_bytes):
                raise RuntimeError(
                    f"stdout is still arriving after {read // (1024 * 1024)} MB and "
                    f"{elapsed:.0f} s"
                    + ("" if ranged else ", and the server would not send just its end")
                )
    return b"".join(chunks).decode("utf-8", "replace")


class CrabTaskRefused(Exception):
    """A task the CRAB server will never run, carrying the reason it gave."""

    def __init__(self, state, warnings, proj_dir):
        self.state = state
        self.warnings = list(warnings or [])
        self.proj_dir = proj_dir
        reason = "; ".join(self.warnings) or "no reason given by the server"
        super(CrabTaskRefused, self).__init__(
            f"the CRAB server refused {os.path.basename(str(proj_dir))} ({state}): {reason}"
        )


class CrabTaskNotScheduledYet(Exception):
    """A task the CRAB server has accepted but not yet handed to a scheduler."""

    def __init__(self, state):
        self.state = state
        super(CrabTaskNotScheduledYet, self).__init__(
            f"the task is {state}: accepted by the CRAB server, not yet on a scheduler"
        )


class CrabTaskSubmitFailed(Exception):
    """A task the CRAB server failed to submit, carrying law's report of its jobs (failed)."""

    def __init__(self, state, message, proj_dir, result):
        self.state = state
        self.message = message
        self.proj_dir = proj_dir
        self.result = result
        super(CrabTaskSubmitFailed, self).__init__(
            f"the CRAB server failed to submit {os.path.basename(str(proj_dir))} ({state}): "
            f"{message or 'the server gave no failure message'}"
        )


class FLAFCrabJobManager(law.cms.CrabJobManager):
    """CRAB job manager that rides out a status response it cannot read, recognises a task
    the server refused, failed to submit or has not scheduled yet, keeps the CRAB client out
    of the AFS home, feeds the per-site job record, reports why jobs failed and applies the
    stall watchdog.

    ``crab status`` occasionally returns output with no "Status on the CRAB server" line
    at all. law then raises, and because a group failure is mapped onto every job of the
    CRAB task, one such response becomes one error per job (4763 identical errors in a
    single poll of the DSProd production). Worse, law skips the whole poll iteration on
    any query error: no status line, no resubmission of retry jobs, and any other task's
    good data discarded with it — ``poll_fails`` consecutive occurrences kill the
    workflow.

    The condition is transient, so the query is simply retried. If it still cannot be
    read, the task's jobs are reported as pending — what law itself does when a freshly
    submitted task has no per-job information yet — and the fact is published once, for
    the task, instead of once per job. A task that stays unreadable for
    ``max_unreadable_polls`` consecutive polls stops the run: a production that quietly
    stalls is worse than one that stops.

    A condition that must end the run is recorded in ``stop_reason`` and raised by
    ``CrabWorkflow.crab_poll_callback``, never from here: law runs queries in a thread pool
    and ``get_async_result_silent`` turns an exception into the *result*, which the poll
    loop counts as one more failed query while every other task loses that poll's status.
    """

    #: attempts, and the pause between them, before a status response is given up on
    query_retries = 3
    query_retry_delay = 15.0

    #: consecutive unreadable polls of one task that are tolerated before the run stops
    max_unreadable_polls = 10

    #: in-flight site counts of a project not queried for this long stop counting
    in_flight_stale_seconds = 3600.0

    #: freshly failed jobs whose stdout is read for the payload's own error, per CRAB task
    #: and poll (law queries each task separately). A wave that fails by the hundred fails
    #: for a handful of reasons, and one line each is what is wanted -- not one HTTP fetch
    #: per job while the poll waits.
    max_failure_reports = 5

    #: how much of a job's stdout is kept while looking for its error
    max_log_bytes = _max_log_bytes

    #: server statuses of a task that will never produce a job. `SUBMITREFUSED` is set by
    #: the CRAB TaskWorker when it rejects the request outright -- an unknown site name in
    #: the whitelist, say -- and it is absorbing: `crab resubmit` and `crab kill` refuse a
    #: task in it. law cannot read its status, so its jobs are reported failed here, which
    #: law retries into a fresh task.
    terminal_server_states = ("SUBMITREFUSED",)

    #: server statuses of a task the TaskWorker or the schedd failed to submit. law reports
    #: its jobs failed itself and retries them into a fresh task, which is what recovers a
    #: one-off failure; only how often that may happen is bounded here.
    failed_server_states = ("SUBMITFAILED",)

    #: statuses meaning "accepted, but not on a scheduler yet". Every task enters the CRAB
    #: database as `WAITING` and is promoted later. law reports `WAITING on command SUBMIT`,
    #: the form of every new task, as pending without a bound, and cannot read the other
    #: commands at all, which would count against `max_unreadable_polls`; both are
    #: classified here, so a backlogged TaskWorker -- exactly when a task lingers -- costs
    #: no retry, and a task that never leaves the state still stops the run.
    pending_server_states = ("WAITING",)

    #: polls a task may spend unscheduled before the run is stopped (five hours at the
    #: default interval): a task that never leaves `WAITING` would otherwise be polled for
    #: ever with every job pending and nothing said
    max_unscheduled_polls = 60

    #: how often the wait is repeated in the log while it lasts
    unscheduled_report_every = 12

    #: distinct submissions of this run the server may refuse before the run is stopped. A
    #: refusal is a verdict on what was sent: the first can be a stale site list, which is
    #: dropped here, so a second one on a freshly read list is a configuration fault, and
    #: retrying would spend every branch's attempts on the same verdict.
    max_refused_submissions = 2

    #: distinct submissions of this run the server may fail to submit before the run is
    #: stopped. One can be the TaskWorker's bad luck; a second is a cause that will not go
    #: away (a MyProxy credential the TaskWorker cannot retrieve, say), and law would spend
    #: every branch's attempts submitting into it.
    max_failed_submissions = 2

    #: per-site record to feed, injected by CrabWorkflow.crab_create_job_manager; None
    #: disables harvesting
    site_stats = None

    #: cached CRIC site list to drop when a submission is refused, injected the same way
    site_cache_path = None

    #: stall watchdog, injected the same way; None disables it
    watchdog = None

    def __init__(self, *args, **kwargs):
        super(FLAFCrabJobManager, self).__init__(*args, **kwargs)
        #: proj_dir -> number of consecutive polls whose response could not be read
        self._unreadable = {}
        #: proj_dir -> consecutive polls the task has been accepted but not scheduled
        self._unscheduled = {}
        #: sandbox env with HOME moved off AFS, built once per manager
        self._flaf_env = None
        self._stats_lock = threading.Lock()
        self._stats_seen = set()
        #: proj_dir -> (timestamp, Counter of jobs still pending/running per site)
        self._in_flight = {}
        #: keys already reported, so a status that repeats every poll is printed once
        self._noted = set()
        #: project dirs this run submitted: only a refusal or a failed submission of one of
        #: them says anything about the configuration this run is using
        self._submitted_projects = set()
        #: project dirs of this run's submissions the server refused, counted once each
        self._refused_projects = set()
        #: project dirs of this run's submissions the server failed to submit, the same way
        self._failed_projects = set()
        #: why the run must stop, read and raised by the poll callback
        self.stop_reason = None
        #: log URLs whose payload error was already printed -- the URL carries the attempt,
        #: so a job that fails again is reported again while a poll that repeats is not
        self._reported_logs = set()

    @property
    def cmssw_env(self):
        """The sandbox env with the CRAB client kept out of the AFS home.

        CRAB rewrites its task cache ``~/.crab3`` (via ``~/.crab3.<pid>``) on every
        command, status polls included — with ``$HOME`` on AFS a multi-day production
        dies with PermissionError the moment the AFS token lapses, presenting as a
        status-query failure for every job at once. So every crab invocation gets a home
        of its own under the local tmp; ``--proxy`` is passed explicitly on every
        command, so ``~/.globus`` from the real home is never needed.

        A ``crab`` wrapper on PATH additionally runs every subcommand except ``submit``
        from that home, so ``crab.log`` does not land wherever law happens to run.
        ``submit`` must keep its directory: law runs it with cwd = the job-file directory
        and the generated config names ``scriptExe``/``inputFiles`` relative to it, which
        CRAB resolves against the cwd.
        """
        if self._flaf_env is None:
            # never mutate the base env: law caches it process-wide per sandbox
            env = dict(law.cms.CrabJobManager.cmssw_env.fget(self))
            home = os.path.join(tempfile.gettempdir(), f"flaf_crab_home_{os.getuid()}")
            bin_dir = os.path.join(home, "bin")
            os.makedirs(bin_dir, exist_ok=True)
            wrapper = os.path.join(bin_dir, "crab")
            content = (
                "#!/bin/bash\n"
                "# Written by FLAF (run_tools/law_customizations.py). Keeps crab.log\n"
                "# out of the working area; submit must keep its cwd (the generated\n"
                "# config names scriptExe/inputFiles relative to it).\n"
                'case "$1" in\n'
                "  submit) ;;\n"
                '  *) cd "$HOME" || exit 1 ;;\n'
                "esac\n"
                'exec /cvmfs/cms.cern.ch/common/crab "$@"\n'
            )
            try:
                current = open(wrapper).read()
            except OSError:
                current = None
            if current != content:
                tmp = f"{wrapper}.tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    f.write(content)
                os.chmod(tmp, 0o755)
                os.replace(tmp, wrapper)
            env["HOME"] = home
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
            self._flaf_env = env
        return self._flaf_env

    @classmethod
    def server_status(cls, out):
        """The `Status on the CRAB server` value, matched the way law matches it.

        `query_server_status_cre` is anchored `^...$` and compiled without `re.MULTILINE`,
        and law applies it per line; searching the whole response with it finds nothing.
        """
        for line in (out or "").replace("\r", "").split("\n"):
            match = cls.query_server_status_cre.match(line.strip())
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def server_state(cls, out):
        """Just the state of the server status, without the `on command SUBMIT` half."""
        status = cls.server_status(out)
        return (status or "").split(" on command ")[0].strip().upper()

    @classmethod
    def server_failure(cls, out):
        """The `Failure message from server` value, matched per line as law matches it."""
        for line in (out or "").replace("\r", "").split("\n"):
            match = cls.query_server_failure_cre.match(line.strip())
            if match:
                return match.group(1).strip()
        return None

    @classmethod
    def has_per_job_data(cls, out):
        """Whether a response carries a scheduler status and the per-job JSON line, as law
        matches them: without both, law reports every job of the task from the server status
        alone."""
        lines = (out or "").replace("\r", "").split("\n")
        return any(
            cls.query_scheduler_status_cre.match(line) for line in lines
        ) and any(cls.query_json_line_cre.match(line) for line in lines)

    @classmethod
    def server_warnings(cls, out):
        """The `Warning:` lines of a status response -- where a refusal states its reason.

        A refused task carries no `Failure message from server`: the TaskWorker uploads the
        reason as a task warning, which the client prints as `Warning:`.
        """
        return [
            line.split(":", 1)[1].strip()
            for line in (out or "").replace("\r", "").split("\n")
            if line.strip().startswith("Warning:")
        ]

    def submit(self, *args, **kwargs):
        """Submit, and remember the task that came of it.

        A resumed run re-polls the tasks of earlier runs, refused ones included; counting
        those stopped a corrected DSProd production on its first poll, before it could
        resubmit their branches (2026-09-13).
        """
        job_ids = super(FLAFCrabJobManager, self).submit(*args, **kwargs)
        for job_id in job_ids or []:
            proj_dir = getattr(job_id, "proj_dir", None)
            if proj_dir:
                self._submitted_projects.add(str(proj_dir))
        return job_ids

    @classmethod
    def parse_query_output(cls, out, proj_dir, job_ids, skip_transfers=False):
        """Parse a status response, and say what it looked like when that fails.

        law's error names the server status it ended up with ("but got 'None'") but never
        the output it read, so an unreadable response cannot be diagnosed after the fact.
        Attach the head of it — the status lines live in the first few lines, and the
        per-job JSON that follows is megabytes, so a slice is enough.

        Tasks without per-job information are classified on the server status: a refused
        task, which law cannot read; a task not scheduled yet, which law either reports
        pending without a bound (`WAITING on command SUBMIT`) or cannot read (any other
        command); and a task the server failed to submit, which law reports failed. The
        last two are raised only when the response has no per-job JSON, so a task that
        still publishes it keeps its real job states.
        """
        try:
            result = super(FLAFCrabJobManager, cls).parse_query_output(
                out, proj_dir, job_ids, skip_transfers=skip_transfers
            )
        except Exception as exc:
            state = cls.server_state(out)
            if state in cls.terminal_server_states:
                raise CrabTaskRefused(state, cls.server_warnings(out), proj_dir)
            if state in cls.pending_server_states:
                raise CrabTaskNotScheduledYet(state)
            head = [
                line[:200]
                for line in (out or "").replace("\r", "").split("\n")[:12]
                if not line.startswith("{")
            ]
            shown = "\n      ".join(head) or "<no output>"
            raise Exception(
                f"{exc}\n    first lines of what crab returned ({len(out or '')} bytes):"
                f"\n      {shown}"
            )
        state = cls.server_state(out)
        if state in cls.pending_server_states and not cls.has_per_job_data(out):
            raise CrabTaskNotScheduledYet(state)
        if state in cls.failed_server_states and not cls.has_per_job_data(out):
            raise CrabTaskSubmitFailed(state, cls.server_failure(out), proj_dir, result)
        return result

    def query(self, proj_dir, job_ids=None, *args, **kwargs):
        proj_dir = str(proj_dir)
        last_error = None
        for attempt in range(self.query_retries + 1):
            try:
                result = super(FLAFCrabJobManager, self).query(
                    proj_dir, job_ids=job_ids, *args, **kwargs
                )
            except CrabTaskRefused as exc:
                # terminal: retrying the query, and waiting between attempts, can only
                # repeat it
                return self._refused(exc, proj_dir, job_ids)
            except CrabTaskNotScheduledYet as exc:
                # not an error at all, so neither the delay nor the unreadable count applies
                return self._not_scheduled_yet(exc, proj_dir, job_ids)
            except CrabTaskSubmitFailed as exc:
                # law's own verdict, read without error: nothing to retry
                return self._submit_failed(exc, proj_dir)
            except Exception as exc:
                # law raises before parsing when the client exits non-zero, with the output
                # it read inside the message: a refusal must be recognised there too
                state = self.server_state(str(exc))
                if state in self.terminal_server_states:
                    return self._refused(
                        CrabTaskRefused(
                            state, self.server_warnings(str(exc)), proj_dir
                        ),
                        proj_dir,
                        job_ids,
                    )
                last_error = exc
                if attempt < self.query_retries:
                    time.sleep(self.query_retry_delay)
                continue
            self._unreadable.pop(proj_dir, None)
            self._unscheduled.pop(proj_dir, None)
            self._apply_watchdog(result)
            self._harvest_site_stats(proj_dir, result)
            self.report_failures(result)
            return result

        n = self._unreadable.get(proj_dir, 0) + 1
        self._unreadable[proj_dir] = n
        if n > self.max_unreadable_polls:
            self.stop_reason = (
                f"the status of {os.path.basename(proj_dir)} has been unreadable for {n} "
                f"consecutive polls; last error: {last_error}"
            )
        else:
            print(
                f"could not read the status of {os.path.basename(proj_dir)} "
                f"({n}/{self.max_unreadable_polls} consecutive), keeping its jobs pending: "
                f"{last_error}"
            )
        return self._all_pending(proj_dir, job_ids, last_error)

    def _all_pending(self, proj_dir, job_ids, error):
        """Every job of the project reported pending -- what law does for a task with no
        jobs yet. Without a readable crab.log there is nothing to degrade to, so `error` is
        raised then."""
        if job_ids is None:
            job_ids = self._job_ids_from_proj_dir(proj_dir)
        if job_ids is None:
            raise error
        return {
            job_id: self.job_status_dict(job_id=job_id, status=self.PENDING)
            for job_id in job_ids
        }

    def _not_scheduled_yet(self, exc, proj_dir, job_ids):
        """A task the server has accepted but not handed to a scheduler: its jobs are pending.

        Not an error, so neither the retry delay nor `max_unreadable_polls` applies -- but
        bounded, because a task that never leaves this status would otherwise stall the
        production in silence.
        """
        self._unreadable.pop(proj_dir, None)
        n = self._unscheduled.get(proj_dir, 0) + 1
        self._unscheduled[proj_dir] = n
        if n > self.max_unscheduled_polls:
            self.stop_reason = (
                f"{os.path.basename(proj_dir)} has been {exc.state} for {n} consecutive "
                "polls without reaching a scheduler. The CRAB server accepted it, so this is "
                "not a configuration fault; the TaskWorker is the place to look."
            )
        elif n == 1 or n % self.unscheduled_report_every == 0:
            print(
                f"{os.path.basename(proj_dir)}: {exc} ({n} polls); its jobs stay pending"
            )
        return self._all_pending(proj_dir, job_ids, exc)

    def _note_once(self, key, message):
        """Print `message` the first time `key` produces it: a status repeats every poll."""
        if key not in self._noted:
            self._noted.add(key)
            print(message)

    def _invalidate_site_cache(self):
        """Drop the cached site list, so the next submission asks CRIC again."""
        path = self.site_cache_path
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # the report says the list was dropped, so a failure to drop it must not be silent
            print(f"could not drop the cached site list {path}: {exc}")

    def _refusal_report(self, exc, ours=True):
        """Everything an operator needs to act, in one message.

        The server names only the first site it objected to, so the list it was given
        matters as much as the objection.
        """
        whose = (
            "this submission"
            if ours
            else "a submission left by an earlier run (nothing this run sent)"
        )
        lines = [
            f"the CRAB server refused {whose} ({exc.state}). It will never run, so its "
            "jobs are reported failed and law will submit them as a new task.",
            f"  project:  {exc.proj_dir}",
        ]
        for warning in exc.warnings or ["<the server gave no reason>"]:
            lines.append(f"  server:   {warning}")
        if self.site_cache_path:
            lines.append(
                f"  sites:    whitelist globs are expanded from {self.site_cache_path} when "
                "a site is excluded; it was dropped so the next submission re-reads CRIC"
            )
        lines.append(
            "  check:    the whitelist must contain only CMS Processing Site Names -- CRIC "
            "'?json&preset=site-names', rows with type 'psn'"
        )
        return "\n".join(lines)

    def _refused(self, exc, proj_dir, job_ids):
        """Report the jobs of a refused task as failed, so law resubmits them as a new task.

        `code` is left None on purpose: `_harvest_site_stats` charges a site only for a
        failure carrying a job-level code, and a task the server never scheduled ran nowhere.
        """
        self._unreadable.pop(proj_dir, None)
        ours = str(proj_dir) in self._submitted_projects
        if ours:
            self._refused_projects.add(str(proj_dir))
        # the likeliest reason for a refusal is a site name CRAB does not know, and the
        # whitelist may have been expanded from a cached site list
        self._invalidate_site_cache()
        self._note_once(("refused", proj_dir), self._refusal_report(exc, ours))
        if len(self._refused_projects) >= self.max_refused_submissions:
            # recorded, not raised (see the class docstring); the jobs are still reported
            # failed below, so the state law sees stays consistent whichever way the run ends
            self.stop_reason = (
                f"{len(self._refused_projects)} submissions made by this run have been "
                "refused by the server, so the next one would be too: this is a "
                "configuration fault, not bad luck.\n" + self._refusal_report(exc, ours)
            )
        if job_ids is None:
            job_ids = self._job_ids_from_proj_dir(proj_dir)
        if job_ids is None:
            raise exc
        return {
            job_id: self.job_status_dict(
                job_id=job_id, status=self.FAILED, code=None, error=str(exc)
            )
            for job_id in job_ids
        }

    def _submit_failed(self, exc, proj_dir):
        """Return law's report of a task the server failed to submit: every job failed, which
        law retries into a new task. `code` is None there, so no site is charged.

        Counted like a refusal: a second failed submission of this run stops it.
        """
        self._unreadable.pop(proj_dir, None)
        ours = str(proj_dir) in self._submitted_projects
        if ours:
            self._failed_projects.add(str(proj_dir))
        whose = (
            "this submission"
            if ours
            else "a submission left by an earlier run (nothing this run sent)"
        )
        report = (
            f"the CRAB server failed {whose} ({exc.state}); law submits its jobs again as "
            f"a new task.\n  project:  {proj_dir}\n  server:   "
            f"{exc.message or '<the server gave no failure message>'}"
        )
        self._note_once(("submit failed", proj_dir), report)
        if len(self._failed_projects) >= self.max_failed_submissions:
            # recorded, not raised (see the class docstring); law's failed jobs are returned
            # below either way
            self.stop_reason = (
                f"{len(self._failed_projects)} submissions made by this run have failed on "
                "the CRAB server, so the next one would too: law would spend every "
                "branch's attempts on it.\n" + report
            )
        return exc.result

    def _apply_watchdog(self, result):
        """Turn a stalled job into a failed one, on this poll's fresh status.

        Rewriting the status here rather than editing law's job data is what makes the
        standard retry path do the work: law sees a failed job on this very iteration,
        counts the attempt, and hands the branches back to the wave gate like any other
        failure, while the number of jobs law is polling does not change.

        `code` stays None: `_harvest_site_stats` skips a failure without a job-level code,
        and a verdict is the watchdog's own action, so the site is recorded once per branch,
        explicitly, below.
        """
        watchdog = self.watchdog
        if watchdog is None or not watchdog.enabled:
            return
        for job_id, reason in watchdog.verdicts(result).items():
            data = result.get(job_id)
            if not isinstance(data, dict) or data.get("status") != self.RUNNING:
                # it finished in the seconds since the verdict was formed; resubmitting a
                # branch that is already done is the worst false positive available here
                continue
            site = ((data.get("extra") or {}).get("site_history") or [None])[-1]
            data["status"] = self.FAILED
            data["code"] = None
            data["error"] = reason
            watchdog.forget(job_id)
            msg = f"watchdog: failing job {job_id} -- {reason}"
            if site:
                msg += f" (last site {site})"
            print(msg)
            if site and self.site_stats is not None:
                # once per job id, and the watchdog's max_per_branch bounds the verdicts per
                # branch: a branch that hangs wherever it lands is the branch's problem, and
                # charging each of its stalls to another site would poison the baseline
                with self._stats_lock:
                    key = (str(job_id), "watchdog")
                    if key not in self._stats_seen:
                        self._stats_seen.add(key)
                        self.site_stats.record(site, False)
                        self.site_stats.save()

    def _harvest_site_stats(self, proj_dir, result):
        """Record what CRAB itself said about each job, per site.

        Keyed by the per-attempt job id straight from the parsed query result: law's poll
        syncs per-job ``extra`` (which carries ``site_history``) onto ``job_data``
        positionally, so with several live CRAB projects the site info there can be
        attached to the wrong job — the record here never goes through that path. A
        retried job lands in a new CRAB task and therefore has a new job id, so each
        attempt counts once. law's own bookkeeping cannot reach the record either: a status
        response only carries what happened to the job, and a job that ended without a
        job-level error code (killed, held, or never started) says nothing about the site
        and is skipped — counted, a mass kill drove every site's baseline to ~100 % and the
        quarantine could no longer fire (DSProd, 8285 such failures in one poll).

        Jobs still in flight are counted too — not as outcomes, but as part of what was
        sent to a site, which is the denominator its failure rate is measured against.
        """
        if self.site_stats is None or not result:
            return
        in_flight = Counter()
        now = time.time()
        with self._stats_lock:
            for job_id, data in result.items():
                if not isinstance(data, dict):
                    continue
                history = (data.get("extra") or {}).get("site_history") or []
                if not history:
                    continue
                site = history[-1]
                status = data.get("status")
                if status == self.FINISHED:
                    ok = True
                elif status == self.FAILED and data.get("code") is not None:
                    ok = False
                elif status == self.FAILED:
                    continue
                else:
                    in_flight[site] += 1
                    continue
                key = (str(job_id), ok)
                if key in self._stats_seen:
                    continue
                self._stats_seen.add(key)
                self.site_stats.record(site, ok)
            cutoff = now - self.in_flight_stale_seconds
            self._in_flight[proj_dir] = (now, in_flight)
            self._in_flight = {
                p: (t, c) for p, (t, c) in self._in_flight.items() if t >= cutoff
            }
            combined = Counter()
            for _, counts in self._in_flight.values():
                combined.update(counts)
            self.site_stats.set_in_flight(combined, source=id(self))
            self.site_stats.save()

    def report_failures(self, result):
        """Print why each freshly failed job failed, in its payload's own words.

        CRAB's exit code is a label rather than a diagnosis -- 4197 failures of one DSProd
        production all carried exit 5, "Error while running CMSSW" -- while the exception
        that ended the job sits in its stdout on the schedd, whose URL law already records.
        Best effort by construction: a diagnostic that raised, or that made a poll wait on a
        slow web server, would cost more than the message is worth; when the stdout cannot
        be read or carries no exception, that is said, once per attempt.
        """
        failures = [
            (job_id, data)
            for job_id, data in (result or {}).items()
            if isinstance(data, dict) and data.get("status") == self.FAILED
            # without a job-level code this is a kill or law's own bookkeeping, and there
            # is no payload stdout to read
            and data.get("code") is not None
            and (data.get("extra") or {}).get("log_file")
            and data["extra"]["log_file"] not in self._reported_logs
        ]
        if not failures:
            return
        shown = failures[: self.max_failure_reports]
        for job_id, data in shown:
            url = data["extra"]["log_file"]
            self._reported_logs.add(url)
            try:
                site = (data.get("extra") or {}).get("site_history") or []
                where = f" at {site[-1]}" if site else ""
                print(
                    f"crab job {job_id.crab_num} of "
                    f"{os.path.basename(str(job_id.proj_dir))} failed{where} with exit "
                    f"code {data.get('code')}: {self._payload_error(url)}"
                )
            except Exception as exc:
                # a raise here would cost the whole poll -- law turns an exception from
                # `query` into the poll's result -- to print one line
                print(f"could not report a failed job ({exc}); its stdout is at {url}")
        if len(failures) > len(shown):
            print(
                f"... and {len(failures) - len(shown)} more failed job(s) of this task whose "
                f"reason was not fetched (max_failure_reports={self.max_failure_reports}); "
                "their stdout is linked from the job data"
            )

    def _payload_error(self, url):
        """The last exception line of a job's stdout, or why it could not be read."""
        try:
            text = fetch_job_stdout(url, max_bytes=self.max_log_bytes)
        except Exception as exc:
            return f"could not read its stdout ({exc}); see {url}"
        error = payload_error(text)
        if not error:
            return f"its stdout carries no exception; see {url}"
        return error


_FLAFCrabWorkflowProxyBase = law.cms.CrabWorkflow.workflow_proxy_cls


_CRAB_DEFAULT_PARALLEL_JOBS = 5000
_CRAB_DEFAULT_REFILL_FRACTION = 0.2
_CRAB_DEFAULT_POLL_INTERVAL = 5  # minutes

#: how long a retry held back by the wave gate may wait before it goes out on its own,
#: whatever the wave size. Waiting for a wave that a handful of retries cannot fill costs a
#: full job length per retry generation: over a 4800-job DSProd production the parked
#: retries waited 11.35 h at the median, ~10.5 h of the 68.4 h it took to reach 99.4 %.
_CRAB_DEFAULT_RETRY_RELEASE_MINUTES = 45


def _cli_has_param(name, task_family=None):
    """True when the user passed ``--<name>`` (or ``--<task_family>-<name>``) on the CLI.

    The match is exact: an option addressed to one task must not silently disable the yaml
    value or the CRAB default for every other task in the graph. Unlike ``--tasks-per-job``
    (excluded from ``req`` and therefore the root task's alone), the bare form of the
    parameters checked here is copied through ``req`` to every task the root requires, so
    it counts for all of them, and so does the root task's prefixed form, which ``req``
    copies just the same; another task's prefixed form reaches that task because ``Task``
    lists these parameters in ``prefer_params_cli``.
    """
    parser = luigi.cmdline_parser.CmdlineParser.get_instance()
    tokens = list(getattr(parser, "cmdline_args", None) or [])
    root_task = getattr(getattr(parser, "known_args", None), "root_task", None) or ""
    root_family = root_task.rsplit(".", 1)[-1]
    wanted = set()
    for variant in (name.replace("_", "-"), name.replace("-", "_")):
        wanted.add(f"--{variant}")
        for family in (task_family, root_family):
            if family:
                wanted.add(f"--{family}-{variant}")
    return any(tok.split("=", 1)[0] in wanted for tok in tokens)


def _cli_has_tasks_per_job(task_family):
    """True when the user pinned the group size for *task_family* explicitly.

    Cost-aware packing then steps aside: an operator who asks for a specific
    ``--tasks-per-job`` gets it, which keeps the previous behaviour available as an
    escape hatch if an estimate ever misbehaves.  The match is exact, so setting the
    option for one task does not silently change how another one is scheduled: the bare
    ``--tasks-per-job`` belongs to the root task only, since the parameter is excluded
    from ``req`` and never reaches the tasks it requires.
    """
    parser = luigi.cmdline_parser.CmdlineParser.get_instance()
    tokens = list(getattr(parser, "cmdline_args", None) or [])
    root_task = getattr(getattr(parser, "known_args", None), "root_task", None) or ""
    wanted = set()
    for name in ("tasks-per-job", "tasks_per_job"):
        if task_family:
            wanted.add(f"--{task_family}-{name}")
            if root_task.rsplit(".", 1)[-1] == task_family:
                wanted.add(f"--{name}")
    return any(tok.split("=", 1)[0] in wanted for tok in tokens)


class _FLAFCrabWorkflowProxy(SubmissionGuards, _FLAFCrabWorkflowProxyBase):
    def __init__(self, *args, **kwargs):
        super(_FLAFCrabWorkflowProxy, self).__init__(*args, **kwargs)
        #: start of the release window of the retries the wave gate is holding back, or
        #: None while it holds none (see `_update_retry_release_clock`)
        self._retry_parked_since = None
        self._apply_crab_parallel_jobs()
        self._apply_crab_poll_interval()
        # read once here, so that a setting that is not a number stops the run at the start
        # rather than at the first retry, deep inside a production
        self._crab_refill_fraction()
        self._crab_retry_release_minutes()

    def run(self):
        # law judges which outputs exist from what it gathered while luigi scheduled the
        # workflow, through cached existence answers -- and a CRAB worker cannot reach the
        # path-cache server, so an output written while no driver was polling (a restarted
        # driver, a workflow waiting for its turn) still reads as absent there. Gather it
        # again, with every "absent" resting on a listing taken from here on: otherwise the
        # first poll of a resumed run sends finished branches back to the grid.
        require_fresh_negatives()
        self._existing_branches = None
        self._skip_jobs.clear()
        return super(_FLAFCrabWorkflowProxy, self).run()

    def _crab_number(self, key, default):
        """A finite number from the `crab:` config; a value that is not one is an error, not
        a silent default (`waited >= nan` is never true, for one)."""
        raw = self.task._crab_cfg().get(key, default)
        try:
            value = math.nan if isinstance(raw, bool) else float(raw)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            raise ValueError(f"crab.{key} must be a number, got {raw!r}")
        return value

    def _crab_refill_fraction(self):
        return min(
            max(
                self._crab_number("refill_fraction", _CRAB_DEFAULT_REFILL_FRACTION), 0.0
            ),
            1.0,
        )

    def _crab_retry_release_minutes(self):
        return max(
            self._crab_number(
                "retry_release_minutes", _CRAB_DEFAULT_RETRY_RELEASE_MINUTES
            ),
            0.0,
        )

    def _apply_crab_parallel_jobs(self):
        """CRAB default is 5000 jobs in flight; yaml then CLI override.

        Multi-workflow tasks inherit HTCondor's unlimited ``parallel_jobs``, so
        the CrabWorkflow class default never wins. Apply the CRAB default here.
        """
        if _cli_has_param("parallel-jobs", self.task.get_task_family()):
            return
        yaml_n = self.task._crab_cfg().get("parallel_jobs")
        if yaml_n is not None:
            self._set_parallel_jobs(int(yaml_n))
            return
        if self.poll_data.n_parallel == self.n_parallel_max:
            self._set_parallel_jobs(_CRAB_DEFAULT_PARALLEL_JOBS)

    def _apply_crab_poll_interval(self):
        """CRAB default is a 5-minute poll; yaml then CLI override.

        Same MRO trap as ``parallel_jobs``: multi-workflow tasks inherit HTCondor's
        2-minute ``poll_interval``, so the CrabWorkflow class default never wins. Each
        poll is one multi-MB ``crab status --json`` per live CRAB task, so the HTCondor
        cadence doubles both the server load and the exposure to an unreadable response.

        A value equal to the HTCondor default is indistinguishable from the inherited
        one and is treated as unset — pin an explicit 2 on the CLI or in
        ``crab.poll_interval``.
        """
        if _cli_has_param("poll-interval", self.task.get_task_family()):
            return
        yaml_v = self.task._crab_cfg().get("poll_interval")
        if yaml_v is not None:
            self.task.poll_interval = float(yaml_v)
            return
        htcondor_default = float(HTCondorWorkflow.poll_interval._default)
        if float(self.task.poll_interval) == htcondor_default:
            self.task.poll_interval = _CRAB_DEFAULT_POLL_INTERVAL

    def _parked_retries(self):
        """The job numbers of retries the wave gate is holding back in `unsubmitted_jobs`.

        `job_data.attempts` is law's per-job retry counter and part of the submission
        file, and law increments it before a retry ever reaches `submit`, so it tells a
        parked retry from a never-submitted branch -- across a restart too, where both
        arrive in the same `unsubmitted_jobs` mapping.
        """
        return set(self.job_data.unsubmitted_jobs) & set(self.job_data.attempts)

    def _update_retry_release_clock(self, after_release=False):
        """Keep a release window running exactly while the wave gate holds a retry back.

        The window starts when the first retry is parked and is not moved by later ones, so
        the oldest parked retry waits at most one window. It is checked on every round, not
        only where this proxy parks a generation itself: a resumed run reads parked retries
        from the submission file while law hands it an empty retry generation on every poll.
        Only the timestamp lives in memory, so a restarted driver delays a release by at most
        one window and never loses a job. `after_release` starts a fresh window for the
        retries a release could not take, rather than an expired one that would open the gate
        on every poll.
        """
        if not self._parked_retries():
            self._retry_parked_since = None
        elif after_release or self._retry_parked_since is None:
            self._retry_parked_since = time.monotonic()

    def _parked_retries_are_due(self):
        """Whether the oldest parked retry has waited out its release window."""
        if self._retry_parked_since is None:
            return False
        waited = time.monotonic() - self._retry_parked_since
        return waited >= self._crab_retry_release_minutes() * 60

    def _should_submit_crab_group(self, n_backlog, n_retry):
        """Whether to submit now, or hold jobs back so they accumulate into one CRAB task.

        Creating a CRAB task is expensive and a task holds only a few thousand jobs, so a
        production is submitted in waves of at least ``refill_fraction * parallel_jobs``
        jobs. Jobs are held back only while such a wave is still **achievable**: once the
        work left in the whole production — running plus waiting — can no longer fill
        one, waiting can only delay it, so whatever is waiting goes out immediately,
        however little that is. That covers the tail of a large production and every
        small production (which can never fill a wave and so is never batched at all),
        while a trickle of retries early on still accumulates.

        Waiting work is counted in two parts. ``n_backlog`` is what sits in
        ``unsubmitted_jobs`` -- never-submitted branches plus the retries an earlier poll
        parked there -- and only it is measured against the wave size; ``n_retry`` is the
        generation of retries this poll offers, which has not waited for anything yet.
        Gating on free slots alone let a handful of retries out as their own CRAB task
        whenever the production did not fill ``parallel_jobs``: with 3270 of 5000 slots
        taken, 1730 were free, so the gate was open from the first poll onwards.

        A wave that is never reached must not park a retry for ever, though: a retry that
        has been parked for ``crab.retry_release_minutes`` goes out however small the
        wave it makes.
        """
        n_parallel = self.poll_data.n_parallel
        if n_parallel >= self.n_parallel_max:
            # unlimited parallelism: keep law's own behaviour
            return True
        n_waiting = n_backlog + n_retry
        if n_waiting <= 0:
            return True
        n_active = self.poll_data.n_active
        min_wave = self._crab_refill_fraction() * n_parallel
        # a full-sized wave, and the room to run it
        if min(n_backlog, n_parallel - n_active) >= min_wave:
            return True
        # even if every job still running were to fail, the next wave could not reach
        # the bar
        if n_active + n_waiting < min_wave:
            return True
        return self._parked_retries_are_due()

    def submit(self, retry_jobs=None):
        # before anything can park the retries (a skipped round, the wave gate): a mass
        # retry parked here would be released later where the brake no longer sees it
        self._stop_on_mass_lost_outputs(retry_jobs)
        if self._skip_submission_round(retry_jobs):
            return OrderedDict()
        retry_jobs = retry_jobs or OrderedDict()
        # A --no-poll invocation resubmits failures exactly once and then returns, so
        # a parked job would not be offered again until someone runs the task anew —
        # never hold anything back there.
        if getattr(self.task, "no_poll", False):
            return super(_FLAFCrabWorkflowProxy, self).submit(retry_jobs or None)
        # before the gate is consulted, so that retries parked by an earlier poll or by a
        # previous driver are on the clock too, not only a generation parked right here
        self._update_retry_release_clock()
        if self._should_submit_crab_group(
            len(self.job_data.unsubmitted_jobs), len(retry_jobs)
        ):
            # law's submit() fills the wave from `unsubmitted_jobs` whatever opened the gate,
            # so a release on the timer takes the never-submitted backlog with it: the CRAB
            # task is being created either way.
            submitted = super(_FLAFCrabWorkflowProxy, self).submit(retry_jobs or None)
            # whatever law had no free slot for keeps waiting, on a fresh window
            self._update_retry_release_clock(after_release=True)
            return submitted

        # Park retries in front of the backlog, so the next eligible wave picks them up as
        # one larger CRAB task instead of a task for a handful of jobs now.
        if self._park_retries(retry_jobs):
            self._update_retry_release_clock()
        return OrderedDict()

    def setup_job_manager(self):
        """Require a working CRAB sandbox, a valid VOMS proxy and a MyProxy credential CRAB
        can read.

        law builds the CMSSW sandbox it runs ``crab`` in lazily, inside every submission
        attempt; a failure there is swallowed per job — each one is stored with
        ``dummy_job_id``, polled as "unknown job id", retried, and the workflow only dies
        when the retry tolerance is exceeded, half an hour later, with the real cause
        nowhere in the log. Building it here (law calls this once, before the first
        submission or poll) turns that into a single actionable error.
        """
        try:
            self.job_manager.cmssw_env
        except Exception as exc:
            raise RuntimeError(
                "could not set up the CMSSW sandbox that law runs `crab` in "
                "(job.crab_sandbox_name in law.cfg): "
                f"{exc}\nThe sandbox dumps its environment with bare `python`, which "
                "modern CMSSW does not ship — check that `python` on PATH resolves to a "
                "python3 (flaf_env provides one; see docs/workflow/crab.md)."
            ) from exc
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
        kwargs = {"proxy_file": proxy}
        # CRABClient names the credential sha1(DN) and looks under no other name, so one
        # stored under the plain DN -- what a bare `myproxy-init -d` leaves behind -- is
        # invisible to the TaskWorker and must not satisfy this gate: the task would be
        # accepted and then fail on the server with SUBMITFAILED.
        try:
            info = law.wlcg.get_myproxy_info(encode_username=True, silent=True) or {}
        except Exception:
            info = {}
        # law submits with `crab submit --proxy <file>`, which makes CRABClient skip its own
        # delegation and renewal, so nothing in a run tops this credential up. 5 days is
        # the TaskWorker's own minimum.
        if info.get("username") and info.get("timeleft", 0) >= 5 * 24 * 3600:
            kwargs["myproxy_username"] = info["username"]
            return kwargs
        raise RuntimeError(
            "CRAB requires a MyProxy credential valid for at least 5 days, stored under the "
            "SHA1 of your DN with the CRAB TaskWorker retrieval policy (the TaskWorker "
            "retrieves it from myproxy.cern.ch). Create it with the CRAB client, in a shell "
            "with the analysis env.sh sourced:\n"
            "  cmsEnv crab createmyproxy --days 30   # asks for the GRID certificate passphrase\n"
            "A bare `myproxy-init` is not enough: it stores the credential under the plain "
            "DN and without the retrieval policy, so CRAB never sees it. See "
            "docs/workflow/crab.md for a passphrase-free stop-gap."
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


#: CRAB's own resource limits (CRABClient ServerUtilities.MAX_MEMORY_PER_CORE and
#: MAX_MEMORY_SINGLE_CORE): the client refuses a task above max(MAX_MEMORY_SINGLE_CORE,
#: numCores * MAX_MEMORY_PER_CORE), so more than 2500 MB per core is bought with cores. The
#: 5000 MB single-core figure is the allowance on *resubmit*, never applied at submit.
CRAB_MB_PER_CORE = 2500
CRAB_MB_SINGLE_CORE = 3000

#: the only values `JobType.numCores` accepts (CRABClient JobType/CMSSWConfig.py); anything
#: else is refused at submit
CRAB_ALLOWED_CORES = (1, 2, 4, 8)


def crab_memory_ceiling(n_cores):
    """The most memory, in MB, CRAB grants a job of `n_cores` cores."""
    return max(CRAB_MB_SINGLE_CORE, n_cores * CRAB_MB_PER_CORE)


def crab_resources(task_family, n_cpus, memory_mb):
    """(numCores, maxMemoryMB) of a CRAB job whose payload runs `n_cpus` threads.

    maxMemoryMB is a kill threshold, not a reservation: a job above it is removed (exit
    50660) and CRAB never retries that, while law's retries repeat the same peak. So with no
    explicit request (`memory_mb` <= 0) a job asks for the most CRAB grants for its cores,
    max(3000, 2500 * cores). An explicit request is honoured exactly, buying cores when it
    needs more than the payload's threads come with, and refused rather than shrunk when no
    core count CRAB accepts can hold it. The cores are the smallest count CRAB accepts that
    is at least `n_cpus`; the payload still runs `n_cpus` threads.
    """
    n_cpus = max(1, int(n_cpus))
    if n_cpus > CRAB_ALLOWED_CORES[-1]:
        raise ValueError(
            f"{task_family}: CRAB accepts at most {CRAB_ALLOWED_CORES[-1]} cores, but the "
            f"task asks for n_cpus={n_cpus}"
        )
    memory_mb = int(memory_mb or 0)
    if 0 < memory_mb < 1000:
        raise ValueError(
            f"{task_family}: a CRAB memory request of {memory_mb} is read as MB, and "
            f"{memory_mb} MB per job cannot be meant -- pass the value in MB"
        )
    for n_cores in CRAB_ALLOWED_CORES:
        if n_cores < n_cpus:
            continue
        if memory_mb <= 0:
            return n_cores, crab_memory_ceiling(n_cores)
        if memory_mb <= crab_memory_ceiling(n_cores):
            return n_cores, memory_mb
    raise ValueError(
        f"{task_family}: {memory_mb} MB per job is more than CRAB grants at any core count "
        f"(at most {crab_memory_ceiling(CRAB_ALLOWED_CORES[-1])} MB at "
        f"{CRAB_ALLOWED_CORES[-1]} cores); ask for less"
    )


class CrabWorkflow(law.cms.CrabWorkflow):
    """CRAB (WLCG) remote workflow, built on law.contrib.cms.CrabWorkflow.

    CRAB is only the batch backend. **All analysis products and logs use FLAF remote
    I/O** (``fs_default`` / gfal via task targets and ``stageout_logs.sh``). CRAB
    ``transferOutputs`` / ``transferLogs`` / ``JobType.outputFiles`` are forced off so
    nothing is duplicated onto CRAB's stageout area.

    ``Site.storageSite`` / ``Data.outLFNDirBase`` are derived from ``fs_default``
    (submit-time write check only). Cores and memory follow ``crab_resources``: the
    cores CRAB accepts for ``n_cpus`` and, unless ``--crab-memory`` asks otherwise, the
    most memory CRAB grants for them.

    Law injects dummy ``userInputFiles`` when ``Data.inputDataset`` is empty,
    and the CRAB client then requires ``Site.whitelist``. If ``crab.whitelist``
    is unset, FLAF defaults to ``T1_*`` / ``T2_*`` / ``T3_*`` so jobs can run
    at every CMS processing site. CRAB gives the whitelist precedence over the
    blacklist, so excluded sites (configured ``crab.blacklist`` and the automatic
    quarantine alike) are removed from the whitelist itself, expanding globs from
    the CRIC Processing Site Name list where needed (see ``run_tools/crab_sites.py``).

    CRAB workers have no AFS, so code is always shipped via the existing BundleTask
    mechanism (same as ``--bundle`` on HTCondor). Tasks must declare ``bundle_flavours``.

    Config (``global.yaml`` / user_custom YAML), all optional::

        crab:
          # whitelist: [T2_CH_CERN]   # omit to use all T1/T2/T3 sites
          # blacklist: [T2_US_MIT]
          # parallel_jobs: 5000       # --parallel-jobs default; CLI wins
          # refill_fraction: 0.2      # min wave size as a fraction of parallel_jobs
          # retry_release_minutes: 45 # a parked retry goes out after this long anyway
          # poll_interval: 5          # minutes between crab status polls; CLI wins
          # min_runtime_min: 60       # floor for CRAB maxJobRuntimeMin
          # auto_blacklist:           # site quarantine; see crab_sites.DEFAULTS
          #   enabled: true
          # watchdog:                 # stall watchdog; see crab_watchdog.DEFAULTS
          #   enabled: true
          # ignore_global_blacklist: false  # waive CMS's own site blacklist (not recommended)
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
    crab_memory = luigi.IntParameter(
        default=0,
        significant=False,
        description="CRAB JobType.maxMemoryMB per job, in MB; 0 (default) = the most CRAB "
        "grants for the job's cores, max(3000, 2500 * cores). A larger request buys cores; "
        "one CRAB cannot grant at 8 cores is refused. CRAB only.",
    )

    # A per-task resource request, like max_runtime and n_cpus: never handed from a
    # requiring task to what it requires, and not needed on the worker command line.
    exclude_params_req = {"crab_memory"}
    exclude_params_branch = {"crab_memory"}

    # A job CRAB reports as `transferring`/`transferred` has finished its payload and only
    # waits for a stageout FLAF disables, so it counts as finished. law otherwise decides
    # that per poll by reading `disableAutomaticOutputCollection` out of the project's
    # crab.log: a log without that line reads False (such jobs are then polled as running
    # for ever), and a missing log makes the query raise before crab even runs.
    crab_job_kwargs_query = {"skip_transfers": True}

    #: lazily-built, throttled `kinit -R` used while polling (see crab_poll_callback)
    _crab_kinit_update = None

    #: the job manager of this workflow, so the poll callback can see what it found
    _flaf_crab_job_manager = None

    #: stall watchdog, shared between the poll callback and the job manager
    _watchdog_obj = None

    #: throttle for the watchdog's one directory listing per interval
    _watchdog_refresh = None

    def _crab_cfg(self):
        cfg = self.global_params.get("crab") or {}
        if "memory_mb_per_cpu" in cfg:
            raise RuntimeError(
                "`crab.memory_mb_per_cpu` is no longer used: CRAB memory is now the most CRAB "
                "grants for a job's cores, max(3000, 2500 * cores), and a task that needs a "
                "different amount asks for it itself -- `--<Task>-crab-memory <MB>` on the "
                "command line, `crab_memory` in a `[luigi_<Task>]` section of law.cfg, or "
                "`payload_producers.<producer>.crab_memory` for AnalysisCacheTask. Remove "
                "the key from the `crab:` block."
            )
        return cfg

    def site_stats(self):
        """Rolling per-site job record, kept in the analysis data area across runs.

        One instance per file and process: a law run chains several CRAB workflows, and a
        record of its own per workflow would let a later one overwrite an earlier one's
        outcomes and quarantines.
        """
        return SiteStats.shared(
            os.path.join(self.ana_data_path(), "crab_site_stats.json"),
            self._crab_cfg().get("auto_blacklist"),
        )

    def site_cache_path(self):
        """Where the CRIC site list is cached.

        Deliberately not the old `cms_sites.json`: that file holds a list built by a
        different rule (see `processing_sites`), and reusing it would keep a refused
        submission refused for the whole cache lifetime after the rule was corrected.
        """
        return os.path.join(self.ana_data_path(), "cms_psn_sites.json")

    def _heartbeat_dir_parts(self):
        parts = [self.version, HEARTBEAT_DIR, self.__class__.__name__, self.period]
        producer = getattr(self, "producer_to_run", None) or getattr(
            self, "producer_to_aggregate", None
        )
        if producer:
            parts.append(producer)
        return parts

    def heartbeat_dir_target(self):
        """The one flat directory holding a flag per running CRAB job of this workflow.

        Flat, so the driver lists it once per interval whatever the job count; named like
        the staged logs, so a workflow of another task, era or producer has its own.
        """
        return self.remote_dir_target(*self._heartbeat_dir_parts())

    def heartbeat_target(self, branch):
        """This branch's flag; the driver maps it back to a job through law's job data."""
        return self.remote_target(*self._heartbeat_dir_parts(), str(branch))

    def job_watchdog(self):
        """The stall watchdog, built once per workflow and shared with the job manager."""
        if self._watchdog_obj is None:
            self._watchdog_obj = StallWatchdog(
                lambda: self.heartbeat_dir_target().uri(),
                watchdog_config(self._crab_cfg()),
                voms_token=os.environ.get("X509_USER_PROXY") or None,
                publish=self.publish_message,
            )
        return self._watchdog_obj

    def crab_heartbeat(self):
        """Job side: a Heartbeat refreshing this branch's flag, or None.

        Only in a real CRAB job (`LAW_CRAB_JOB_NUMBER` is set by law's CRAB wrapper and by
        nothing else), only for the branch the job was submitted to run, and only while
        the watchdog is on, matching the driver side, which only watches CRAB.
        """
        if "LAW_CRAB_JOB_NUMBER" not in os.environ or not self.is_branch():
            return None
        family = submitted_task_family()
        if family is not None and family != self.get_task_family():
            return None
        cfg = watchdog_config(self._crab_cfg())
        if not cfg["enabled"]:
            return None
        return Heartbeat(
            self.heartbeat_target(self.branch).uri(),
            int(cfg["interval_minutes"]) * 60,
            voms_token=os.environ.get("X509_USER_PROXY") or None,
            label={"task": self.task_family, "branch": self.branch},
            log=self.publish_message,
        )

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
        ]
        # the unique suffix is what tells the CRAB tasks of one workflow apart (and names
        # their staged logs), so only the prefix is truncated
        prefix = re.sub(r"[^A-Za-z0-9_\-]", "_", "_".join(parts))[:91]
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

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
        """Believe CRAB's FINISHED only when the branch's outputs are on storage.

        CRAB parks a job in `transferring` between the payload exiting and the post-job
        classifying it, and it parks a payload that exited non-zero there too; with
        transfers skipped, law maps that state to FINISHED. Without this check a poll
        landing in that window writes a failed job off as finished and never queries it
        again (DSProd booked 113 failed jobs as `finished: 113` in one poll). With it, law
        checks the branch outputs before accepting FINISHED and demotes a job whose outputs
        are missing to a retry; each job is checked once for the whole run.

        law calls this once per poll iteration, right before it checks the jobs reported
        finished, which is the moment to require that every "absent" answer of this
        iteration rests on a listing taken after the status it judges. Cached answers do
        not: a CRAB worker cannot reach the path-cache server, so a listing published before
        the job wrote its file answers "absent" for it until it expires. Positive answers
        stay cache-served; a negative costs one listing per directory per poll, and the
        fresh listing republishes the directory for every other process.
        """
        require_fresh_negatives()
        return True

    def crab_poll_callback(self, poll_data):
        # The one hook the poll loop calls outside its own error handling, so the one place
        # a condition found while querying can end the run (see FLAFCrabJobManager).
        manager = self._flaf_crab_job_manager
        if manager is not None and manager.stop_reason:
            raise RuntimeError(manager.stop_reason)
        # A large CRAB production polls for days while law keeps writing its job-status
        # files to the AFS work area — renew the Kerberos ticket, hourly and verbosely: a
        # silent renewal leaves no way to tell, after a credential failure, whether it
        # had been running at all.
        if self._crab_kinit_update is None:
            self._crab_kinit_update = timed_call_wrapper(
                lambda: update_kinit(verbose=1), 3600
            )
        self._crab_kinit_update()
        # One listing of the heartbeat directory per interval, however many jobs are in
        # flight. The verdicts are applied in the job manager's query(), on the fresh
        # status of each CRAB task, so nothing here changes the number of jobs law polls.
        watchdog = self.job_watchdog()
        if watchdog.enabled and self._watchdog_refresh is None:
            self._watchdog_refresh = timed_call_wrapper(
                watchdog.refresh, watchdog.interval_seconds
            )
        if self._watchdog_refresh is not None:
            proxy = getattr(self, "workflow_proxy", None)
            job_data = getattr(proxy, "job_data", None)
            if job_data is not None:
                watchdog.set_jobs(getattr(job_data, "jobs", None))
            self._watchdog_refresh()
        return True

    def crab_job_manager_cls(self):
        return FLAFCrabJobManager

    def crab_create_job_manager(self, **kwargs):
        # The sandbox preflight lives in the proxy's setup_job_manager: this runs at
        # workflow-proxy construction, i.e. also for --print-status and completeness
        # checks, which must not build a CMSSW sandbox.
        manager = super().crab_create_job_manager(**kwargs)
        manager.site_stats = self.site_stats()
        manager.site_cache_path = self.site_cache_path()
        manager.watchdog = self.job_watchdog()
        self._flaf_crab_job_manager = manager
        return manager

    def crab_job_file_factory_cls(self):
        return FLAFCrabJobFileFactory

    def crab_job_file(self):
        # the same grouped run as HTCondor (see grouped_law_job_script)
        from law.job.base import JobInputFile

        return JobInputFile(
            path=grouped_law_job_script(), copy=True, share=True, render_job=True
        )

    def crab_job_config(self, config, job_nums, branches=None):
        # law 0.1.21 calls crab_job_config(config, job_nums, branches) once per submission,
        # with the job numbers and their branch lists as two parallel lists.
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
        # CRAB numbers the jobs of every CRAB task from 1, and a production is many CRAB
        # tasks (waves, retries) staging logs into one directory: the task's unique suffix
        # keeps one job's log from overwriting another's.
        config.render_variables["crab_log_tag"] = config.request_name.rsplit("_", 1)[-1]

        # Cores + memory. CRAB requires JobType.numCores == PSet numberOfThreads, and
        # accepts only 1, 2, 4 or 8 of them; see crab_resources for the memory.
        n_cores, mem = crab_resources(
            self.task_family, getattr(self, "n_cpus", 1) or 1, self.crab_memory
        )
        config.crab.JobType.psetName = self._ensure_crab_pset(n_cores)
        config.crab.JobType.numCores = n_cores
        config.crab.JobType.maxMemoryMB = mem

        # Runtime limit (hours → minutes). CRAB jobs must download/unpack bundles before
        # the payload starts, so enforce a floor (default 60 min) even when the task's
        # max_runtime is tiny (e.g. HelloWorld 0.1 h would otherwise be 6 min). Without a
        # value CRAB applies its own 1250 min, which would kill every longer job silently,
        # so a floor that does not parse is an error.
        max_runtime = getattr(self, "max_runtime", None)
        if max_runtime is not None and float(max_runtime) > 0:
            raw_floor = self._crab_cfg().get("min_runtime_min", 60)
            try:
                cfg_floor = int(raw_floor)
            except (TypeError, ValueError):
                raise ValueError(
                    "crab.min_runtime_min must be a whole number of minutes, got "
                    f"{raw_floor!r}"
                ) from None
            config.crab.JobType.maxJobRuntimeMin = max(
                int(math.floor(float(max_runtime) * 60)), cfg_floor
            )

        # Law always sets dummy userInputFiles (no inputDataset). The CRAB client
        # then requires Site.whitelist. Default to every CMS processing site so
        # analyses need not pin T2_CH_CERN. An explicit crab.whitelist still restricts.
        whitelist = list(self._crab_cfg().get("whitelist") or [])
        blacklist = list(self._crab_cfg().get("blacklist") or [])
        if not whitelist:
            whitelist = ["T1_*", "T2_*", "T3_*"]

        # Sites quarantined by their recent failure record; every wave is a new CRAB
        # task, so this takes effect for the next one — retries included.
        quarantined = [s for s in self.site_stats().blacklist() if s not in blacklist]

        # CRAB gives the whitelist precedence over the blacklist, so a blacklisted site
        # matched by a glob would silently be kept — remove it from the whitelist itself
        # (see resolve_whitelist). CRIC is only consulted when something is excluded.
        all_sites = []
        if blacklist or quarantined:
            try:
                all_sites = processing_sites(self.site_cache_path())
            except RuntimeError:
                # A configured exclusion must not be silently defeated — but the
                # quarantine is advisory, and aborting a running production because
                # CRIC is down would cost more than one unquarantined wave.
                if blacklist:
                    raise
                self.publish_message(
                    "cannot expand the site whitelist (CRIC unreachable, no usable cache); "
                    "skipping the site quarantine for this CRAB task"
                )
                quarantined = []
        if quarantined:
            self.publish_message(
                "keeping {} site(s) out of this CRAB task after recent failures: {}".format(
                    len(quarantined), ", ".join(quarantined)
                )
            )
            blacklist += quarantined
        sites = resolve_whitelist(whitelist, blacklist, all_sites)
        config.crab.Site.whitelist = [str(s) for s in sites]
        config.crab.Data.ignoreLocality = True
        if blacklist:
            config.crab.Site.blacklist = [str(s) for s in blacklist]
        # CMS's global blacklist of known-broken sites stays in force unless explicitly
        # waived: with an open site pool it is the main protection against burning jobs
        # at bad sites.
        if self._crab_cfg().get("ignore_global_blacklist", False):
            config.crab.Site.ignoreGlobalBlacklist = True

        return config


_crab_heartbeats = {}


@CrabWorkflow.event_handler(luigi.Event.START)
def _start_crab_heartbeat(task):
    """Start refreshing the branch's heartbeat flag when a CRAB job starts its payload.

    luigi events rather than a decorator on every run(): they cover every FLAF task without
    touching its body. A run() that yields new requirements fires START again when resumed,
    so a beating task is not started twice.
    """
    if task.task_id in _crab_heartbeats:
        return
    heartbeat = task.crab_heartbeat()
    if heartbeat is None:
        return
    _crab_heartbeats[task.task_id] = heartbeat.__enter__()


@CrabWorkflow.event_handler(luigi.Event.SUCCESS)
@CrabWorkflow.event_handler(luigi.Event.FAILURE)
def _stop_crab_heartbeat(task, *args):
    """Stop the heartbeat and remove the flag, however the payload ended."""
    heartbeat = _crab_heartbeats.pop(task.task_id, None)
    if heartbeat is not None:
        heartbeat.__exit__(None, None, None)
