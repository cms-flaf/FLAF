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
from FLAF.run_tools.crab_sites import SiteStats, processing_sites, resolve_whitelist
from FLAF.RunKit.law_wlcg import WLCGFileSystem, WLCGFileTarget, WLCGDirectoryTarget
from FLAF.Common.Setup import Setup
from FLAF.AnaProd.CostModel import pack_units

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

    def output(self):
        name = self.flavour
        if self.bundle_cfg().get("hashed", False):
            name = f"{self.flavour}_{self.source_hash()}"
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
        return os.getenv("FLAF_PATH") or os.path.join(
            os.getenv("ANALYSIS_PATH"), "FLAF"
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


class LawProxyState:
    """Access to workflow-proxy state whose names differ between law versions.

    law 0.1.20 keeps the skip-verdict cache and the retry counters as ``_skip_jobs`` /
    ``_job_retries``; older releases (such as the copy vendored in the inference
    submodule) expose them without the underscore.  Both spellings hold the same state,
    so read whichever the installed version provides instead of pinning to one.
    """

    def _law_state(self, *names):
        for name in names:
            if hasattr(self, name):
                return getattr(self, name)
        raise AttributeError(f"none of {names} found on {type(self).__name__}")

    @property
    def _cost_skip_jobs(self):
        return self._law_state("_skip_jobs", "skip_jobs")

    @property
    def _cost_job_retries(self):
        return self._law_state("_job_retries", "job_retries")


class _BundleAwareHTCondorWorkflowProxy(
    LawProxyState, BundleAwareHTCondorWorkflowProxyBase
):
    """HTCondor proxy with remote log paths and cost-aware job composition.

    law groups branches into jobs with ``iter_chunks(sorted(branch_map), tasks_per_job)``:
    fixed-size and contiguous.  When the per-branch cost is heavy-tailed -- as it is for
    AnaTuple production, where a handful of dilepton-skim files cost twenty times what the
    rest do, and where expensive branches are adjacent because they belong to the same
    dataset -- that packs the expensive work together into jobs that overrun the wall
    clock, get removed, and are retried with exactly the same grouping.

    A task opts in by providing ``branch_cost_map()``; then jobs are built to a target
    duration instead, a failed group is retried split rather than whole, and the estimates
    are refreshed from observed durations while the workflow runs.
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

    def _apply_cost_parallel_jobs(self):
        """Bound the queue footprint by default.

        Beyond queue hygiene this is what creates submission waves, and therefore the
        opportunity to re-pack the work that has not been submitted yet with the better
        estimates that the finished jobs provide.
        """
        if not self._cost_scheduling_enabled() or _cli_has_parallel_jobs():
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
        self.task.publish_message(
            f"cost-aware packing: {len(branches)} branch(es), "
            f"{law.util.human_duration(seconds=int(total))} of estimated work "
            f"-> {len(groups)} job(s) of at most "
            f"{law.util.human_duration(seconds=int(capacity))}"
        )

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


class FLAFCrabJobManager(law.cms.CrabJobManager):
    """CRAB job manager that rides out a status response it cannot read, keeps the CRAB
    client out of the AFS home, and feeds the per-site job record.

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
    ``max_unreadable_polls`` consecutive polls does raise: a production that quietly
    stalls is worse than one that stops.
    """

    #: attempts, and the pause between them, before a status response is given up on
    query_retries = 3
    query_retry_delay = 15.0

    #: consecutive unreadable polls of one task that are tolerated before raising
    max_unreadable_polls = 10

    #: in-flight site counts of a project not queried for this long stop counting
    in_flight_stale_seconds = 3600.0

    def __init__(self, *args, **kwargs):
        super(FLAFCrabJobManager, self).__init__(*args, **kwargs)
        #: proj_dir -> number of consecutive polls whose response could not be read
        self._unreadable = {}
        #: sandbox env with HOME moved off AFS, built once per manager
        self._flaf_env = None
        #: per-site job record, injected by CrabWorkflow.crab_create_job_manager;
        #: None disables harvesting
        self.site_stats = None
        self._stats_lock = threading.Lock()
        self._stats_seen = set()
        #: proj_dir -> (timestamp, Counter of jobs still pending/running per site)
        self._in_flight = {}

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
            home = os.path.join(
                tempfile.gettempdir(), f"flaf_crab_home_{os.getuid()}"
            )
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
    def parse_query_output(cls, out, proj_dir, job_ids, skip_transfers=False):
        """Parse a status response, and say what it looked like when that fails.

        law's error names the server status it ended up with ("but got 'None'") but never
        the output it read, so an unreadable response cannot be diagnosed after the fact.
        Attach the head of it — the status lines live in the first few lines, and the
        per-job JSON that follows is megabytes, so a slice is enough.
        """
        try:
            return super(FLAFCrabJobManager, cls).parse_query_output(
                out, proj_dir, job_ids, skip_transfers=skip_transfers
            )
        except Exception as exc:
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

    def query(self, proj_dir, job_ids=None, *args, **kwargs):
        proj_dir = str(proj_dir)
        last_error = None
        for attempt in range(self.query_retries + 1):
            try:
                result = super(FLAFCrabJobManager, self).query(
                    proj_dir, job_ids=job_ids, *args, **kwargs
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.query_retries:
                    time.sleep(self.query_retry_delay)
                continue
            self._unreadable.pop(proj_dir, None)
            self._harvest_site_stats(proj_dir, result)
            return result

        n = self._unreadable.get(proj_dir, 0) + 1
        self._unreadable[proj_dir] = n
        if n > self.max_unreadable_polls:
            raise Exception(
                f"the status of {os.path.basename(proj_dir)} has been unreadable for {n} "
                f"consecutive polls; last error: {last_error}"
            )
        print(
            f"could not read the status of {os.path.basename(proj_dir)} "
            f"({n}/{self.max_unreadable_polls} consecutive), keeping its jobs pending: "
            f"{last_error}"
        )
        if job_ids is None:
            job_ids = self._job_ids_from_proj_dir(proj_dir)
        if job_ids is None:
            # without a readable crab.log there is nothing to degrade to
            raise last_error
        return {
            job_id: self.job_status_dict(job_id=job_id, status=self.PENDING)
            for job_id in job_ids
        }

    def _harvest_site_stats(self, proj_dir, result):
        """Record the outcome of every job that reached a terminal state, per site.

        Keyed by the per-attempt job id straight from the parsed query result: law's poll
        syncs per-job ``extra`` (which carries ``site_history``) onto ``job_data``
        positionally, so with several live CRAB projects the site info there can be
        attached to the wrong job — the record here never goes through that path. A
        retried job lands in a new CRAB task and therefore has a new job id, so each
        attempt counts once.

        Jobs still in flight are counted too — not as outcomes, but as part of what was
        sent to a site, which is the denominator its failure rate is measured against.
        """
        if self.site_stats is None:
            return
        outcome = {self.FINISHED: True, self.FAILED: False}
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
                if status not in outcome:
                    in_flight[site] += 1
                    continue
                key = (str(job_id), status)
                if key in self._stats_seen:
                    continue
                self._stats_seen.add(key)
                self.site_stats.record(site, outcome[status])
            cutoff = now - self.in_flight_stale_seconds
            self._in_flight[proj_dir] = (now, in_flight)
            self._in_flight = {
                p: (t, c) for p, (t, c) in self._in_flight.items() if t >= cutoff
            }
            combined = Counter()
            for _, counts in self._in_flight.values():
                combined.update(counts)
            self.site_stats.set_in_flight(combined)
            self.site_stats.save()


# Require VOMS + MyProxy before submit. The CRAB server retrieves the user proxy
# from myproxy.cern.ch (>= ~5 days remaining). A local VOMS proxy alone is not
# enough: the client may accept the task, then the server returns SUBMITFAILED.
# Do not fall back to interactive delegation or a law.cfg password file.
_FLAFCrabWorkflowProxyBase = law.cms.CrabWorkflow.workflow_proxy_cls


_CRAB_DEFAULT_PARALLEL_JOBS = 5000
_CRAB_DEFAULT_REFILL_FRACTION = 0.2
_CRAB_DEFAULT_POLL_INTERVAL = 5  # minutes


def _cli_has_param(name):
    """True when the user passed ``--<name>`` (or a task-prefixed form) on the CLI."""
    parser = luigi.cmdline_parser.CmdlineParser.get_instance()
    tokens = list(getattr(parser, "cmdline_args", None) or [])
    for variant in (name.replace("_", "-"), name.replace("-", "_")):
        for tok in tokens:
            if tok == f"--{variant}" or tok.startswith(f"--{variant}="):
                return True
            if tok.endswith(f"-{variant}") or f"-{variant}=" in tok:
                return True
    return False


def _cli_has_parallel_jobs():
    """True when the user passed ``--parallel-jobs`` (or a task-prefixed form)."""
    return _cli_has_param("parallel-jobs")


def _cli_has_tasks_per_job(task_family):
    """True when the user pinned the group size for *task_family* explicitly.

    Cost-aware packing then steps aside: an operator who asks for a specific
    ``--tasks-per-job`` gets it, which keeps the previous behaviour available as an
    escape hatch if an estimate ever misbehaves.  The match is exact, so setting the
    option for one task does not silently change how another one is scheduled.
    """
    parser = luigi.cmdline_parser.CmdlineParser.get_instance()
    tokens = list(getattr(parser, "cmdline_args", None) or [])
    wanted = set()
    for name in ("tasks-per-job", "tasks_per_job"):
        wanted.add(f"--{name}")
        if task_family:
            wanted.add(f"--{task_family}-{name}")
    return any(tok.split("=", 1)[0] in wanted for tok in tokens)


class _FLAFCrabWorkflowProxy(_FLAFCrabWorkflowProxyBase):
    def __init__(self, *args, **kwargs):
        super(_FLAFCrabWorkflowProxy, self).__init__(*args, **kwargs)
        self._apply_crab_parallel_jobs()
        self._apply_crab_poll_interval()

    def _crab_refill_fraction(self):
        raw = self.task._crab_cfg().get(
            "refill_fraction", _CRAB_DEFAULT_REFILL_FRACTION
        )
        try:
            frac = float(raw)
        except (TypeError, ValueError):
            frac = _CRAB_DEFAULT_REFILL_FRACTION
        return min(max(frac, 0.0), 1.0)

    def _apply_crab_parallel_jobs(self):
        """CRAB default is 5000 jobs in flight; yaml then CLI override.

        Multi-workflow tasks inherit HTCondor's unlimited ``parallel_jobs``, so
        the CrabWorkflow class default never wins. Apply the CRAB default here.
        """
        if _cli_has_parallel_jobs():
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
        """
        if _cli_has_param("poll-interval"):
            return
        yaml_v = self.task._crab_cfg().get("poll_interval")
        if yaml_v is not None:
            self.task.poll_interval = float(yaml_v)
            return
        htcondor_default = float(HTCondorWorkflow.poll_interval._default)
        if float(self.task.poll_interval) == htcondor_default:
            self.task.poll_interval = _CRAB_DEFAULT_POLL_INTERVAL

    def _should_submit_crab_group(self, n_waiting):
        """Whether to submit now, or hold jobs back so they accumulate into one CRAB task.

        Creating a CRAB task is expensive and a task holds only a few thousand jobs, so a
        production is submitted in waves of at least ``refill_fraction * parallel_jobs``
        jobs. Jobs are held back only while such a wave is still **achievable**: once the
        work left in the whole production — running plus waiting — can no longer fill
        one, waiting can only delay it, so whatever is waiting goes out immediately,
        however little that is. That covers the tail of a large production and every
        small production (which can never fill a wave and so is never batched at all),
        while a trickle of retries early on still accumulates.

        ``n_waiting`` (unsubmitted + jobs offered for retry) is what makes this an
        aggregation threshold at all. Gating on free slots alone let a handful of retries
        out as their own CRAB task whenever the production did not fill
        ``parallel_jobs``: with 3270 of 5000 slots taken, 1730 were free, so the gate was
        open from the first poll onwards.
        """
        n_parallel = self.poll_data.n_parallel
        if n_parallel >= self.n_parallel_max:
            # unlimited parallelism: keep law's own behaviour
            return True
        if n_waiting <= 0:
            return True
        n_active = self.poll_data.n_active
        min_wave = self._crab_refill_fraction() * n_parallel
        # a full-sized wave, and the room to run it
        if min(n_waiting, n_parallel - n_active) >= min_wave:
            return True
        # even if every job still running were to fail, the next wave could not reach
        # the bar
        return n_active + n_waiting < min_wave

    def submit(self, retry_jobs=None):
        retry_jobs = retry_jobs or OrderedDict()
        n_waiting = len(self.job_data.unsubmitted_jobs) + len(retry_jobs)
        if self._should_submit_crab_group(n_waiting):
            return super(_FLAFCrabWorkflowProxy, self).submit(retry_jobs or None)

        # Park retries as unsubmitted so the next eligible wave picks them up
        # as one larger CRAB task instead of a 1-job task now.
        if retry_jobs:
            for job_num, branches in retry_jobs.items():
                if self._can_skip_job(job_num, branches):
                    continue
                self.job_data.jobs.pop(job_num, None)
                self.job_data.unsubmitted_jobs[job_num] = branches
            self.dump_job_data()
        return OrderedDict()

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
    (submit-time write check only). Memory is ``2000 MB * n_cpus`` (override
    with ``crab.memory_mb_per_cpu``), matching the CRAB / site-guaranteed default.

    Law injects dummy ``userInputFiles`` when ``Data.inputDataset`` is empty,
    and the CRAB client then requires ``Site.whitelist``. If ``crab.whitelist``
    is unset, FLAF defaults to ``T1_*`` / ``T2_*`` / ``T3_*`` so jobs can run
    at every CMS processing site. CRAB gives the whitelist precedence over the
    blacklist, so excluded sites (configured ``crab.blacklist`` and the automatic
    quarantine alike) are removed from the whitelist itself, expanding globs from
    the CRIC processing-site list where needed (see ``run_tools/crab_sites.py``).

    CRAB workers have no AFS, so code is always shipped via the existing BundleTask
    mechanism (same as ``--bundle`` on HTCondor). Tasks must declare ``bundle_flavours``.

    Config (``global.yaml`` / user_custom YAML), all optional::

        crab:
          # whitelist: [T2_CH_CERN]   # omit to use all T1/T2/T3 sites
          # blacklist: [T2_US_MIT]
          # parallel_jobs: 5000       # --parallel-jobs default; CLI wins
          # refill_fraction: 0.2      # min wave size as a fraction of parallel_jobs
          # poll_interval: 5          # minutes between crab status polls; CLI wins
          # memory_mb_per_cpu: 2000   # CRAB JobType.maxMemoryMB / n_cpus
          # auto_blacklist:           # site quarantine; see crab_sites.DEFAULTS
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

    #: lazily-built, throttled `kinit -R` used while polling (see crab_poll_callback)
    _crab_kinit_update = None

    #: rolling per-site job statistics, shared with the job manager (see site_stats)
    _site_stats_obj = None

    def _crab_cfg(self):
        return self.global_params.get("crab") or {}

    def site_stats(self):
        """Rolling per-site job record, kept in the analysis data area across runs."""
        if self._site_stats_obj is None:
            self._site_stats_obj = SiteStats(
                os.path.join(self.ana_data_path(), "crab_site_stats.json"),
                self._crab_cfg().get("auto_blacklist"),
            )
        return self._site_stats_obj

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
        # A large CRAB production polls for days while law keeps writing its job-status
        # files to the AFS work area — renew the Kerberos ticket, hourly and verbosely: a
        # silent renewal leaves no way to tell, after a credential failure, whether it
        # had been running at all.
        if self._crab_kinit_update is None:
            self._crab_kinit_update = timed_call_wrapper(
                lambda: update_kinit(verbose=1), 3600
            )
        self._crab_kinit_update()
        return True

    def crab_job_manager_cls(self):
        return FLAFCrabJobManager

    def crab_create_job_manager(self, **kwargs):
        """Create the job manager, and build its CMSSW sandbox, before anything is submitted.

        law builds that sandbox lazily, inside every submission attempt. A failure there
        is swallowed per job: each one is stored with ``dummy_job_id``, polled as
        "unknown job id", retried, and the workflow only dies when the retry tolerance is
        exceeded — half an hour later, with the real cause nowhere in the log. Building
        it here turns that into a single actionable error before the first submission.
        """
        manager = super().crab_create_job_manager(**kwargs)
        manager.site_stats = self.site_stats()
        try:
            manager.cmssw_env
        except Exception as exc:
            raise RuntimeError(
                "could not set up the CMSSW sandbox that law runs `crab` in "
                "(job.crab_sandbox_name in law.cfg): "
                f"{exc}\nThe sandbox dumps its environment with bare `python`, which "
                "modern CMSSW does not ship — check that `python` on PATH resolves to a "
                "python3 (flaf_env provides one; see docs/workflow/crab.md)."
            ) from exc
        return manager

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
        # Default 2000 MB/CPU (CRAB default; all sites guarantee this per core),
        # then clamp to the CRAB client max (5000 MB for 1 core, 2500 MB * n_cpus
        # otherwise).
        n_cpus = max(1, int(getattr(self, "n_cpus", 1) or 1))
        try:
            mb_per_cpu = int(self._crab_cfg().get("memory_mb_per_cpu", 2000))
        except (TypeError, ValueError):
            mb_per_cpu = 2000
        # CRAB client cap: 5000 MB (1 core) or 2500 MB * n_cpus (multi-core).
        crab_max = 5000 if n_cpus == 1 else 2500 * n_cpus
        mem = min(n_cpus * max(mb_per_cpu, 1), crab_max)
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
        if quarantined:
            self.publish_message(
                "keeping {} site(s) out of this CRAB task after recent failures: {}".format(
                    len(quarantined), ", ".join(quarantined)
                )
            )
            blacklist += quarantined

        # CRAB gives the whitelist precedence over the blacklist, so a blacklisted site
        # matched by a glob would silently be kept — remove it from the whitelist itself
        # (see resolve_whitelist). CRIC is only consulted when something is excluded.
        all_sites = []
        if blacklist:
            all_sites = processing_sites(
                os.path.join(self.ana_data_path(), "cms_sites.json")
            )
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
