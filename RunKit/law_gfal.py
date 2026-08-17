import json
import time
import os
import sys

from law.target.remote.interface import RemoteFileInterface
from .grid_tools import (
    get_voms_proxy_info,
    GfalError,
    gfal_copy_safe,
    gfal_ls_safe,
    gfal_rm,
    gfal_stat,
    gfal_exists,
)
from .run_tools import repeat_until_success
from .pathCacheClient import (
    set_status as set_remote_cache_status,
    get_status as get_remote_cache_status,
    get_status_many as get_remote_cache_status_many,
)


class PathCacheEntry:
    def __init__(self, path, exists, expiration_time):
        self.path = path
        self.exists = exists
        self.expiration_time = expiration_time

    def is_valid(self):
        return self.expiration_time >= time.time()


# Cache key marking that a directory has been listed and that the cached entries for it are
# therefore complete. It is a key like any other, so it is shared through the cache server and
# expires with the entries it covers. Only a successful listing writes it — unlike the plain
# "this directory exists" entry, which is also (re)written when a parent is listed or when a
# file is copied into the directory, and which therefore says nothing about the directory's
# content.
LISTING_MARKER = ".flaf_listed"


def listing_marker(base_dir):
    return os.path.join(base_dir, LISTING_MARKER)


class PathCache:
    def __init__(self, validity_period):
        self.validity_period = validity_period
        self.cache = {}
        # Directories listed by this process. A marker learned from the cache server means
        # "the server's knowledge of this directory is complete", which does not hold for the
        # subset of entries kept locally, so only markers backed by a listing taken here may
        # be handed on in a snapshot (see iter_valid).
        self.listed_dirs = set()

    @staticmethod
    def _iter_parents(path):
        while True:
            parent = os.path.dirname(path)
            if not parent or parent == path:
                break
            path = parent
            yield path

    def set(self, path, exists):
        self.cache[path] = PathCacheEntry(
            path, exists, time.time() + self.validity_period
        )
        # If a path exists, every ancestor directory exists too: drop any stale negative
        # ancestor entry that would otherwise (via directory-negative inference in get())
        # wrongly imply this path is absent.
        if exists:
            for parent in self._iter_parents(path):
                pentry = self.cache.get(parent)
                if pentry is not None and pentry.exists is False:
                    del self.cache[parent]

    def set_local(self, path, exists):
        # Local-only set; identical to set() for the in-memory cache (kept for parity
        # with RemotePathCache, where set_local avoids a network round-trip).
        self.set(path, exists)

    def set_exists(self, base_dir, items):
        # The marker is written first so that it can never outlive the entries it covers.
        self.set(listing_marker(base_dir), True)
        for item in items:
            path = os.path.join(base_dir, item)
            self.set(path, True)
        self.set(base_dir, True)
        self.listed_dirs.add(base_dir)

    def has_listing(self, base_dir):
        return self.get(listing_marker(base_dir))[0] is True

    def get(self, path):
        entry = self.cache.get(path)
        if entry is not None:
            if entry.is_valid():
                return entry.exists, True
            del self.cache[path]
        # Directory-negative inference: if the nearest cached ancestor directory does not
        # exist, then this path cannot exist either.
        for parent in self._iter_parents(path):
            pentry = self.cache.get(parent)
            if pentry is None:
                continue
            if not pentry.is_valid():
                del self.cache[parent]
                continue
            if pentry.exists is False:
                return False, True
            break
        return None, True

    def get_many(self, paths):
        return {path: self.get(path)[0] for path in paths}

    def iter_valid(self):
        for path, entry in list(self.cache.items()):
            if not entry.is_valid():
                continue
            if (
                os.path.basename(path) == LISTING_MARKER
                and os.path.dirname(path) not in self.listed_dirs
            ):
                # Shipping this marker would tell the receiver that the entries it got are a
                # complete listing, which is only true for a listing taken by this process.
                continue
            yield path, entry.exists

    def load_entries(self, entries):
        """Refresh entries with this cache's validity period (snapshot has no timestamps)."""
        negatives = []
        positives = []
        for item in entries:
            path = item.get("path")
            if not path:
                continue
            if item.get("exists"):
                positives.append(path)
            else:
                negatives.append(path)
        for path in negatives:
            self.set(path, False)
        for path in positives:
            self.set(path, True)

    def invalidate(self, path):
        to_remove = []
        for p in self.cache:
            if path.startswith(p):
                to_remove.append(p)
        for p in to_remove:
            del self.cache[p]


class RemotePathCache:
    def __init__(self, host, port, local_cache_validity_period, timeout=5, verbose=0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.verbose = verbose
        self.local_cache = PathCache(local_cache_validity_period)

    def set(self, path, exists):
        set_remote_cache_status(
            [
                (path, exists),
            ],
            self.host,
            self.port,
            self.timeout,
            verbose=self.verbose,
        )
        self.local_cache.set(path, exists)

    def set_local(self, path, exists):
        # Update only the in-process cache, without a round-trip to the cache server.
        self.local_cache.set(path, exists)

    def set_exists(self, base_dir, items):
        # The marker is published first so that it can never outlive the entries it covers.
        entries = [(listing_marker(base_dir), True)]
        for item in items:
            path = os.path.join(base_dir, item)
            entries.append((path, True))
        entries.append((base_dir, True))
        set_remote_cache_status(
            entries, self.host, self.port, self.timeout, verbose=self.verbose
        )
        self.local_cache.set_exists(base_dir, items)

    def has_listing(self, base_dir):
        return self.get(listing_marker(base_dir))[0] is True

    def get(self, path):
        local_result, _ = self.local_cache.get(path)
        if local_result is not None:
            return local_result, True
        remote_result = get_remote_cache_status(
            path, self.host, self.port, self.timeout, verbose=self.verbose
        )
        if remote_result is not None:
            self.local_cache.set(path, remote_result)
        return remote_result, False

    def get_many(self, paths):
        """Resolve many paths in one shot: serve what the local cache knows, then query the
        remaining paths from the server in a single pipelined request. Returns {path: bool|None}.
        """
        results = {}
        missing = []
        for path in paths:
            local_result, _ = self.local_cache.get(path)
            if local_result is not None:
                results[path] = local_result
            else:
                missing.append(path)
        if missing:
            remote = get_remote_cache_status_many(
                missing, self.host, self.port, self.timeout, verbose=self.verbose
            )
            for path in missing:
                remote_result = remote.get(path)
                if remote_result is not None:
                    self.local_cache.set(path, remote_result)
                results[path] = remote_result
        return results

    def invalidate(self, path):
        set_remote_cache_status(
            [
                (path, None),
            ],
            self.host,
            self.port,
            self.timeout,
            verbose=self.verbose,
        )
        self.local_cache.invalidate(path)


SHIPPED_PATH_CACHE_BASENAME = "path_cache.json"
SHIPPED_PATH_CACHE_ENV = "FLAF_SHIPPED_PATH_CACHE"

_shipped_path_cache_entries = None


def local_path_cache(fs):
    """Return the in-process PathCache for a WLCG/GFAL filesystem, if any."""
    fi = getattr(fs, "file_interface", None)
    pc = getattr(fi, "path_cache", None)
    if pc is None:
        return None
    return getattr(pc, "local_cache", pc)


def collect_setup_path_cache_entries(setup):
    """Union of valid path-cache entries from every FS the Setup has already created."""
    entries = {}
    for fs in getattr(setup, "fs_dict", {}).values():
        pc = local_path_cache(fs)
        if pc is None:
            continue
        for path, exists in pc.iter_valid():
            entries[path] = exists
    return [{"path": path, "exists": exists} for path, exists in entries.items()]


def write_path_cache_file(path, entries):
    with open(path, "w") as f:
        json.dump({"entries": entries}, f)


def _resolve_shipped_path_cache_file():
    env_path = os.environ.get(SHIPPED_PATH_CACHE_ENV, "")
    if env_path and os.path.isfile(env_path):
        return env_path
    stem, ext = os.path.splitext(SHIPPED_PATH_CACHE_BASENAME)
    search_dirs = [
        os.environ.get("LAW_JOB_INIT_DIR", ""),
        os.environ.get("LAW_JOB_HOME", ""),
        "/srv",
        os.getcwd(),
    ]
    for d in search_dirs:
        if not d:
            continue
        direct = os.path.join(d, SHIPPED_PATH_CACHE_BASENAME)
        if os.path.isfile(direct):
            return direct
        if not os.path.isdir(d):
            continue
        try:
            for name in os.listdir(d):
                if name.startswith(stem + "_") and name.endswith(ext):
                    cand = os.path.join(d, name)
                    if os.path.isfile(cand):
                        return cand
        except OSError:
            pass
    return None


def apply_shipped_path_cache(fs):
    """Load a submit-time path-cache snapshot into ``fs`` (once per process)."""
    global _shipped_path_cache_entries
    if _shipped_path_cache_entries is None:
        _shipped_path_cache_entries = []
        path = _resolve_shipped_path_cache_file()
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                _shipped_path_cache_entries = data.get("entries") or []
                os.environ[SHIPPED_PATH_CACHE_ENV] = path
            except (OSError, ValueError, TypeError):
                _shipped_path_cache_entries = []
    pc = local_path_cache(fs)
    if pc is None or not _shipped_path_cache_entries:
        return
    pc.load_entries(_shipped_path_cache_entries)


class GFALFileInterface(RemoteFileInterface):
    local_prefix = "file://"

    def __init__(
        self,
        base,
        local_path_cache_validity_period=60,
        path_cache_host=None,
        path_cache_port=None,
        verbose=0,
    ):
        self.voms_token = get_voms_proxy_info()["path"]
        if path_cache_host is None:
            self.path_cache = PathCache(local_path_cache_validity_period)
        else:
            self.path_cache = RemotePathCache(
                path_cache_host,
                path_cache_port,
                local_cache_validity_period=local_path_cache_validity_period,
                verbose=verbose,
            )
        self.verbose = verbose
        super(GFALFileInterface, self).__init__(base=base)

    def is_local(self, path):
        return path.startswith(GFALFileInterface.local_prefix)

    exists_counter = 0
    remove_counter = 0
    filecopy_counter = 0
    listdir_counter = 0

    def exists(self, path, base=None, **kwargs):
        GFALFileInterface.exists_counter += 1
        path_dir, path_name = os.path.split(path)
        path_uri = self.uri(path, base=base)
        dir_uri = self.uri(path_dir, base=base)
        result = False
        cached_result, from_local_cache = self.path_cache.get(path_uri)
        if cached_result is None and self.path_cache.has_listing(dir_uri):
            # The directory has been listed and its cached entries are therefore complete,
            # so this path is absent. A listing is what proves absence: that the directory
            # itself exists says nothing about its content. Memoize the negative result
            # locally so repeated checks of the same path do not query the cache server
            # again (a fresh TCP round-trip per call otherwise).
            cached_result = False
            from_local_cache = True
            self.path_cache.set_local(path_uri, False)
        use_cache = cached_result is not None

        if use_cache:
            result = cached_result
        else:
            path_dir, path_name = os.path.split(path)
            dir_entries = self.listdir(path_dir, base=base, silent=True)
            result = path_name in dir_entries
            if not result:
                # Local-only: the listing just taken covers every absent sibling for this
                # process, while a file-level negative published to the cache server would
                # outlive the file's creation by a job whose own cache update is lost.
                self.path_cache.set_local(path_uri, False)

        if self.verbose > 0:
            print(
                f"GFALFileInterface.exists: cnt={GFALFileInterface.exists_counter} path={path} taken_from_cache={use_cache} from_local_cache={from_local_cache} result={result}",
                file=sys.stderr,
            )

        return result

    def remove(self, path, base=None, silent=True, **kwargs):
        GFALFileInterface.remove_counter += 1
        path_uri = self.uri(path, base=base)
        if self.verbose > 0:
            print(
                f"GFALFileInterface.remove: cnt={GFALFileInterface.remove_counter} path={path}",
                file=sys.stderr,
            )
        try:
            if gfal_exists(path_uri, voms_token=self.voms_token):
                gfal_rm(path_uri, voms_token=self.voms_token, recursive=True)
            self.path_cache.set(path_uri, False)
            return True
        except GfalError as e:
            if not silent:
                raise e
        return False

    def filecopy(self, src, dst, base=None, **kwargs):
        GFALFileInterface.filecopy_counter += 1
        if self.verbose > 0:
            print(
                f"GFALFileInterface.filecopy: cnt={GFALFileInterface.filecopy_counter} src={src} dst={dst}",
                file=sys.stderr,
            )
        src_local = self.is_local(src)
        dst_local = self.is_local(dst)
        if src_local and not dst_local:
            dst_uris = self.uri(dst, base=base, return_all=True)
            src_uri = src
            for dst_uri in dst_uris:
                dst_dir_uri, _ = os.path.split(dst_uri)
                self.path_cache.set(dst_uri, False)
                gfal_copy_safe(src_uri, dst_uri, voms_token=self.voms_token, verbose=0)
                self.path_cache.set(dst_uri, True)
                cached_dst_dir, _ = self.path_cache.get(dst_dir_uri)
                if cached_dst_dir is not True:
                    # The directory now exists. Record it also when it was simply unknown:
                    # a listing of its parent taken before it was created would otherwise
                    # answer "absent" for it until that listing expires.
                    self.path_cache.set(dst_dir_uri, True)
            return src_uri, dst_uris
        elif dst_local and not src_local:
            dst_uri = dst
            src_uris = self.uri(src, base=base, return_all=True)
            opt_list = [
                [
                    uri,
                ]
                for uri in src_uris
            ]
            successful_src_uri = None

            def copy(src_uri):
                nonlocal successful_src_uri
                gfal_copy_safe(
                    src_uri, dst_uri, voms_token=self.voms_token, n_retries=1, verbose=0
                )
                successful_src_uri = src_uri

            repeat_until_success(
                copy,
                opt_list=opt_list,
                exception=GfalError(
                    f"GFALFileInterface: failed to copy {src} to {dst}"
                ),
            )
            return successful_src_uri, dst_uri
        raise RuntimeError(
            f"GFALFileInterface: unable to copy {src} -> {dst}. Either source or destination must be local"
        )

    def listdir(self, path, base=None, silent=False, **kwargs):
        GFALFileInterface.listdir_counter += 1
        if self.verbose > 0:
            print(
                f"GFALFileInterface.listdir: cnt={GFALFileInterface.listdir_counter} path={path}",
                file=sys.stderr,
            )
        path_uri = self.uri(path, base=base)
        entries = gfal_ls_safe(
            path_uri, voms_token=self.voms_token, catch_stderr=True, verbose=0
        )
        if entries is None:
            # A failed listing may be a transient error rather than an absent directory.
            # Confirm before acting on it: the negative is published to the cache server and
            # suppresses the whole subtree there for every client.
            entries = gfal_ls_safe(
                path_uri, voms_token=self.voms_token, catch_stderr=True, verbose=0
            )
        if entries is None:
            if not silent:
                gfal_ls_safe(
                    path_uri, voms_token=self.voms_token, catch_stderr=False, verbose=1
                )
                raise GfalError(f"GFALFileInterface: failed to list directory {path}")
            entry_names = []
            self.path_cache.set(path_uri, False)
            # Walk up to record the highest absent ancestor as well, so the server can
            # answer the whole missing subtree by inference and clients can skip the
            # per-subdirectory gfal-ls on subsequent lookups.
            self._mark_absent_ancestors(path_uri)
        else:
            entry_names = [entry.name for entry in entries]
            self.path_cache.set_exists(path_uri, entry_names)
        return entry_names

    def _mark_absent_ancestors(self, dir_uri, max_climb=32):
        # A directory was found absent. Walk upward to record the highest absent ancestor
        # too, so the cache server can answer the whole missing subtree by directory-negative
        # inference and clients can skip the per-subdirectory gfal-ls. A negative cached at a
        # high level suppresses a large subtree, so each absent ancestor is confirmed with a
        # second gfal-ls before caching it (a single failure may be transient).
        current = dir_uri
        for _ in range(max_climb):
            parent = os.path.dirname(current)
            if not parent or parent == current:
                break
            cached, _ = self.path_cache.get(parent)
            if cached is not None:
                # Already known: False => the subtree is already covered by inference;
                # True => we reached an existing ancestor, stop.
                break
            entries = gfal_ls_safe(
                parent, voms_token=self.voms_token, catch_stderr=True, verbose=0
            )
            if entries is None:
                # Confirm the absence before caching a wide-reaching negative.
                entries = gfal_ls_safe(
                    parent, voms_token=self.voms_token, catch_stderr=True, verbose=0
                )
            if entries is None:
                self.path_cache.set(parent, False)
                current = parent
            else:
                self.path_cache.set_exists(parent, [entry.name for entry in entries])
                break

    def prefetch(self, paths, base=None):
        """Warm the cache for many paths with a single pipelined request to the cache
        server. Existence results are stored in the local cache so subsequent exists()
        calls are served without further round-trips. Returns {path: bool|None}."""
        uri_map = {path: self.uri(path, base=base) for path in paths}
        uri_results = self.path_cache.get_many(list(uri_map.values()))
        return {path: uri_results.get(uri) for path, uri in uri_map.items()}

    @staticmethod
    def _raise_not_implemented(method_name):
        raise NotImplementedError(
            f"{method_name} is not supported by the GFAL interface"
        )

    def chmod(self, file, perm, **kwargs):
        return True

    def isdir(self, path, **kwargs):
        stat = gfal_stat(path, voms_token=self.voms_token)
        return stat["type"] == "directory"

    def isfile(self):
        self._raise_not_implemented("isfile")

    def mkdir(self, *args, **kwargs):
        return True

    def mkdir_rec(self, *args, **kwargs):
        return True

    def rmdir(self):
        self._raise_not_implemented("rmdir")

    def stat(self):
        self._raise_not_implemented("stat")

    def unlink(self):
        self._raise_not_implemented("unlink")
