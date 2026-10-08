import json
import math
import time
import os
import sys

from law.target.remote.interface import RemoteFileInterface
from .grid_tools import (
    get_voms_proxy_info,
    GfalError,
    gfal_copy_safe,
    gfal_ls_checked,
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
        # Paths whose entry was learned from the cache server: their record time here is the
        # fetch, not when the server learned it, so it says nothing about how fresh it is.
        self.from_server = set()

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
        self.from_server.discard(path)
        # If a path exists, every ancestor directory exists too: drop any stale negative
        # ancestor entry that would otherwise (via directory-negative inference in get())
        # wrongly imply this path is absent.
        if exists:
            for parent in self._iter_parents(path):
                pentry = self.cache.get(parent)
                if pentry is not None and pentry.exists is False:
                    del self.cache[parent]

    def set_from_server(self, path, exists):
        self.set(path, exists)
        self.from_server.add(path)

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

    def iter_valid(self, negatives_after=0.0):
        """Valid entries, for a snapshot. An "absent" -- a negative entry, or a listing marker,
        which implies one for every file not listed -- recorded before `negatives_after` is
        left out (see require_fresh_negatives)."""
        for path, entry in list(self.cache.items()):
            if not entry.is_valid():
                continue
            if negatives_after > 0 and (
                entry.exists is False or os.path.basename(path) == LISTING_MARKER
            ):
                if (
                    path in self.from_server
                    or entry.expiration_time - self.validity_period < negatives_after
                ):
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
            self.local_cache.set_from_server(path, remote_result)
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
                    self.local_cache.set_from_server(path, remote_result)
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
    """Union of valid path-cache entries from every FS the Setup has already created.

    In a process that requires fresh negatives (a CRAB driver), an "absent" recorded before
    that may predate a CRAB job's write, and a job that trusted it would find an input
    missing; it is not shipped.
    """
    entries = {}
    negatives_after = GFALFileInterface.negatives_valid_after
    for fs in getattr(setup, "fs_dict", {}).values():
        pc = local_path_cache(fs)
        if pc is None:
            continue
        for path, exists in pc.iter_valid(negatives_after=negatives_after):
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


def require_fresh_negatives():
    """From now on, answer "absent" only from a listing taken by this process after this call.

    `exists()` answers a missing file from cached knowledge (its own entry, a directory
    listing marker, an absent ancestor), shared between processes through the cache server,
    where a listing stays valid for 24 h by default. CRAB jobs cannot reach that server, so
    a file that a CRAB job wrote after its directory was listed reads as absent everywhere
    until the marker expires; clearing only the in-process caches would not help, because
    the next lookup falls through to the same marker on the server. After this call, a
    negative answer for a directory not listed by this process since the call costs one
    listing, which also republishes the directory's entries to the cache server. Positive
    answers stay cache-served.
    """
    GFALFileInterface.negatives_valid_after = time.time()


class GFALFileInterface(RemoteFileInterface):
    local_prefix = "file://"

    # Set by require_fresh_negatives(); 0 keeps every cached negative usable.
    negatives_valid_after = 0.0

    # A failed listing is reported once per directory within this many seconds. It is not
    # remembered otherwise: every path asks again, so that once the storage answers, the
    # rest of the directory is judged on a real listing -- a remembered failure would turn a
    # brief blip into "absent" for every file of the directory.
    failed_listing_report_seconds = 60.0

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
        # Sizes from the most recent listing of each directory, so that collecting input
        # file metadata does not cost a second gfal-ls.
        self.listing_sizes = {}
        # dir uri -> time of the last listing by this process that the storage answered
        self._listed_at = {}
        # dir uri -> time a failed listing of it was last reported
        self._failed_at = {}
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
        from_listing = cached_result is None and self.path_cache.has_listing(dir_uri)
        if from_listing:
            # The directory has been listed and its cached entries are therefore complete,
            # so this path is absent. A listing is what proves absence: that the directory
            # itself exists says nothing about its content.
            cached_result = False
            from_local_cache = True
        epoch = GFALFileInterface.negatives_valid_after
        if (
            cached_result is False
            and epoch > 0
            and self._listed_at.get(dir_uri, 0.0) < epoch
        ):
            # See require_fresh_negatives: a cached "absent" may predate a CRAB job's write.
            cached_result = None
        elif from_listing:
            # Memoize the negative result locally so repeated checks of the same path do
            # not query the cache server again (a fresh TCP round-trip per call otherwise).
            self.path_cache.set_local(path_uri, False)
        use_cache = cached_result is not None

        if use_cache:
            result = cached_result
        else:
            dir_entries, answered = self._list(dir_uri, silent=True)
            result = path_name in dir_entries
            if not result and answered:
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
                # Publish by renaming a checksum-verified upload onto the target, so that the
                # target never exists with partial content and an existing one is not removed
                # before its replacement is complete. No "absent" is published for the target
                # meanwhile: it stays in place until the rename, and a failed upload leaves it
                # intact.
                gfal_copy_safe(
                    src_uri,
                    dst_uri,
                    voms_token=self.voms_token,
                    copy_mode="copy_rename",
                    verbose=0,
                )
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
        entry_names, _ = self._list(self.uri(path, base=base), silent=silent)
        return entry_names

    def _list(self, path_uri, silent):
        """Entry names of the directory `path_uri`, and whether the storage answered.

        Only an answer is cached: a listing, or gfal reporting that the directory does not
        exist. A listing that fails raises GfalError, or with `silent` returns ([], False)
        and caches nothing: a negative taken from it would be published to the cache server
        and hide existing files from every client.
        """
        GFALFileInterface.listdir_counter += 1
        if self.verbose > 0:
            print(
                f"GFALFileInterface.listdir: cnt={GFALFileInterface.listdir_counter} path={path_uri}",
                file=sys.stderr,
            )
        listed_at = time.time()
        try:
            entries = gfal_ls_checked(path_uri, voms_token=self.voms_token)
        except GfalError as e:
            if not silent:
                raise
            last = self._failed_at.get(path_uri, -math.inf)
            if listed_at - last >= self.failed_listing_report_seconds:
                self._failed_at[path_uri] = listed_at
                reason = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
                print(
                    f"GFALFileInterface: could not list {path_uri} ({reason}); files in it "
                    "read as missing until it can be listed, and nothing is cached",
                    file=sys.stderr,
                )
            return [], False
        if entries is None:
            if not silent:
                raise GfalError(
                    f"GFALFileInterface: directory {path_uri} does not exist"
                )
            self._listed_at[path_uri] = listed_at
            self.path_cache.set(path_uri, False)
            # Walk up to record the highest absent ancestor as well, so the server can
            # answer the whole missing subtree by inference and clients can skip the
            # per-subdirectory gfal-ls on subsequent lookups.
            self._mark_absent_ancestors(path_uri)
            return [], True
        self._listed_at[path_uri] = listed_at
        entry_names = [entry.name for entry in entries]
        self.path_cache.set_exists(path_uri, entry_names)
        self._cache_listing_sizes(path_uri, entries)
        return entry_names, True

    def _cache_listing_sizes(self, path_uri, entries):
        self.listing_sizes[path_uri] = {
            entry.name: {"size": entry.size}
            for entry in entries
            if not entry.is_dir and entry.name not in (".", "..")
        }

    def listdir_info(self, path, base=None, silent=True, **kwargs):
        """``{name: {"size": bytes}}`` for a directory.

        ``gfal-ls --long`` (which :py:func:`gfal_ls` already runs) reports the size of
        every entry, but :py:meth:`listdir` returns only the names.  Sizes are what
        job-cost estimation needs, so keep them from whichever listing happened first
        rather than paying for a second one.  Returns an empty dict when the listing
        fails: callers treat that as "no metadata available".
        """
        path_uri = self.uri(path, base=base)
        if path_uri not in self.listing_sizes:
            entries = gfal_ls_safe(
                path_uri, voms_token=self.voms_token, catch_stderr=True, verbose=0
            )
            if entries is None:
                if not silent:
                    raise GfalError(
                        f"GFALFileInterface: failed to list directory {path}"
                    )
                return {}
            self._cache_listing_sizes(path_uri, entries)
        return self.listing_sizes[path_uri]

    def _mark_absent_ancestors(self, dir_uri, max_climb=32):
        # A directory was found absent. Walk upward to record the highest absent ancestor
        # too, so the cache server can answer the whole missing subtree by directory-negative
        # inference and clients can skip the per-subdirectory gfal-ls. A negative cached at a
        # high level suppresses a large subtree, so only gfal reporting the ancestor absent is
        # cached; a listing that fails stops the climb and records nothing.
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
            listed_at = time.time()
            try:
                entries = gfal_ls_checked(parent, voms_token=self.voms_token)
            except GfalError:
                break
            self._listed_at[parent] = listed_at
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
