import gzip
import json
import math
import threading
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

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
    ServerUnavailable,
    ensure_local_server,
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
# therefore complete. It is a key like any other, so it is shared through the machine's cache
# server and shipped in snapshots, and it expires no later than the entries it covers. Only a
# successful listing writes it — unlike the plain "this directory exists" entry, which is also
# (re)written when a parent is listed or when a file is copied into the directory, and which
# therefore says nothing about the directory's content.
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
                    self.cache.pop(parent, None)

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
            self.cache.pop(path, None)
        # Directory-negative inference: if the nearest cached ancestor directory does not
        # exist, then this path cannot exist either.
        for parent in self._iter_parents(path):
            pentry = self.cache.get(parent)
            if pentry is None:
                continue
            if not pentry.is_valid():
                self.cache.pop(parent, None)
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
    """This process's PathCache in front of the path-cache server of its user on this machine
    (pathCacheServer.py), through which what one process learns serves every other one --
    the forked workers of a law run, a later run, a --print-status -- also after a restart.

    A server that stops answering (e.g. the session that started it ended) is started again;
    while that fails the process works from its own cache, re-trying every `retry_seconds`.
    """

    retry_seconds = 60.0

    def __init__(self, address, local_cache_validity_period, timeout=5, verbose=0):
        self.address = address
        self.timeout = timeout
        self.verbose = verbose
        self.local_cache = PathCache(local_cache_validity_period)
        # time the server was last found unusable, None while it answers
        self._down_since = None

    def _server(self, call, *args, default=None):
        """`call(*args, address, timeout)` against the server, or `default` while it cannot
        be used."""
        now = time.time()
        if self._down_since is not None and now - self._down_since < self.retry_seconds:
            return default
        try:
            try:
                result = call(*args, self.address, self.timeout, verbose=self.verbose)
            except ServerUnavailable:
                ensure_local_server(os.path.dirname(self.address))
                result = call(*args, self.address, self.timeout, verbose=self.verbose)
        except ServerUnavailable as e:
            if self._down_since is None:
                print(
                    f"GFALFileInterface: the path-cache server {self.address} cannot be used "
                    f"({e}); this process works from its own cache until it can, so no other "
                    "process learns from it meanwhile",
                    file=sys.stderr,
                )
            self._down_since = now
            return default
        if self._down_since is not None:
            print(
                f"GFALFileInterface: the path-cache server {self.address} answers again",
                file=sys.stderr,
            )
            self._down_since = None
        return result

    def set(self, path, exists):
        self._server(set_remote_cache_status, [(path, exists)])
        self.local_cache.set(path, exists)

    def set_local(self, path, exists):
        # Update only the in-process cache, without a round-trip to the cache server.
        self.local_cache.set(path, exists)

    def set_exists(self, base_dir, items):
        # The marker is published last: a publication cut short then leaves no marker that
        # would answer "absent" for the entries that did not arrive. The server keeps it no
        # longer than the entries (a marker lives as long as an "absent", pathCacheServer).
        entries = [(os.path.join(base_dir, item), True) for item in items]
        entries.append((base_dir, True))
        entries.append((listing_marker(base_dir), True))
        self._server(set_remote_cache_status, entries)
        self.local_cache.set_exists(base_dir, items)

    def has_listing(self, base_dir):
        return self.get(listing_marker(base_dir))[0] is True

    def get(self, path):
        local_result, _ = self.local_cache.get(path)
        if local_result is not None:
            return local_result, True
        remote_result = self._server(get_remote_cache_status, path)
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
            remote = self._server(get_remote_cache_status_many, missing, default={})
            for path in missing:
                remote_result = remote.get(path)
                if remote_result is not None:
                    self.local_cache.set_from_server(path, remote_result)
                results[path] = remote_result
        return results

    def invalidate(self, path):
        self._server(set_remote_cache_status, [(path, None)])
        self.local_cache.invalidate(path)


# The snapshot a submission ships to its jobs: <prefix><anything><suffix>, gzipped JSON.
SHIPPED_PATH_CACHE_PREFIX = "path_cache_"
SHIPPED_PATH_CACHE_SUFFIX = ".json.gz"
SHIPPED_PATH_CACHE_ENV = "FLAF_SHIPPED_PATH_CACHE"

# How many directories are listed at once when many have to be looked at, e.g. the output
# directories of the jobs that finished since the last poll.
PARALLEL_LISTINGS = 8

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

    In a process that requires fresh negatives (a driver that submits or polls), an "absent"
    recorded before that may predate a job's write, and a job that trusted it would find an
    input missing; it is not shipped.
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


def _gfal_interface(target):
    fi = getattr(getattr(target, "fs", None), "file_interface", None)
    return fi if isinstance(fi, GFALFileInterface) else None


def list_target_directories(targets, n_threads=PARALLEL_LISTINGS):
    """Take at once, `n_threads` at a time, every listing that the exists() of the GFAL
    targets among `targets` would take one after the other, so that those calls are then
    answered from the cache. Returns the ids of the targets whose directory could not be
    listed: exists() reads them as absent, though nothing is known about them.

    The cache server is asked first, in one pipelined request per file interface. Every base
    of a file system is looked at, since exists() picks one at random.
    """
    per_interface = {}
    for target in targets:
        fi = _gfal_interface(target)
        if fi is None:
            continue
        uris = fi.uri(target.fs.abspath(target.path), return_all=True)
        per_interface.setdefault(id(fi), (fi, []))[1].append((target, uris))
    to_list = {}
    target_dirs = []
    for fi, items in per_interface.values():
        dirs = {os.path.dirname(uri) for _, uris in items for uri in uris}
        fi.path_cache.get_many(
            [uri for _, uris in items for uri in uris]
            + [listing_marker(d) for d in dirs]
        )
        for target, uris in items:
            # the listings this target's own answer depends on
            keys = []
            for uri in uris:
                dir_uri = os.path.dirname(uri)
                if fi._cached_answer(uri, dir_uri)[0] is None:
                    key = (id(fi), dir_uri)
                    keys.append(key)
                    to_list.setdefault(key, fi)
            target_dirs.append((target, keys))
    unlisted = set()
    if to_list:

        def answered(item):
            (_, dir_uri), fi = item
            return fi._list(dir_uri, silent=True)[1]

        with ThreadPoolExecutor(max_workers=min(n_threads, len(to_list))) as pool:
            for key, ok in zip(to_list, pool.map(answered, to_list.items())):
                if not ok:
                    unlisted.add(key)
    return {id(t) for t, keys in target_dirs if any(k in unlisted for k in keys)}


def collect_existing_target_entries(targets):
    """Snapshot entries for the files among `targets` that exist on GFAL storage, and how many
    of the targets exist.

    Each existence is resolved now, whatever the age of the local entries: the cache server is
    asked first, and the directories it cannot answer for are listed, several at once. Only
    "exists" is recorded; an absent file is left to the reader, which lists it. A file is
    recorded under every base of its filesystem, since a reader picks one at random.
    """
    list_target_directories(targets)
    entries = {}
    n_found = 0
    for target in targets:
        fi = _gfal_interface(target)
        if fi is None:
            continue
        if target.exists():
            n_found += 1
            uris = fi.uri(target.fs.abspath(target.path), return_all=True)
            entries.update(dict.fromkeys(uris, True))
    return [
        {"path": path, "exists": exists} for path, exists in entries.items()
    ], n_found


def write_path_cache_file(path, entries):
    with gzip.open(path, "wt", compresslevel=6) as f:
        json.dump({"entries": entries}, f)


def _resolve_shipped_path_cache_file():
    env_path = os.environ.get(SHIPPED_PATH_CACHE_ENV, "")
    if env_path and os.path.isfile(env_path):
        return env_path
    search_dirs = [
        os.environ.get("LAW_JOB_INIT_DIR", ""),
        os.environ.get("LAW_JOB_HOME", ""),
        "/srv",
        os.getcwd(),
    ]
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for name in os.listdir(d):
                if name.startswith(SHIPPED_PATH_CACHE_PREFIX) and name.endswith(
                    SHIPPED_PATH_CACHE_SUFFIX
                ):
                    cand = os.path.join(d, name)
                    if os.path.isfile(cand):
                        return cand
        except OSError:
            pass
    return None


def apply_shipped_path_cache(fs):
    """Load the snapshot shipped with this job into ``fs`` (read once per process)."""
    global _shipped_path_cache_entries
    if _shipped_path_cache_entries is None:
        _shipped_path_cache_entries = []
        path = _resolve_shipped_path_cache_file()
        if path is None:
            print(
                "GFALFileInterface: no path-cache snapshot was shipped with this job; every "
                "directory it looks at is listed",
                file=sys.stderr,
            )
        else:
            try:
                with gzip.open(path, "rt") as f:
                    data = json.load(f)
                _shipped_path_cache_entries = data.get("entries") or []
                os.environ[SHIPPED_PATH_CACHE_ENV] = path
            except (OSError, EOFError, ValueError, TypeError, AttributeError) as e:
                print(
                    f"GFALFileInterface: the path-cache snapshot {path} cannot be read "
                    f"({e}); every directory this job looks at is listed",
                    file=sys.stderr,
                )
    pc = local_path_cache(fs)
    if pc is None or not _shipped_path_cache_entries:
        return
    pc.load_entries(_shipped_path_cache_entries)


def require_fresh_negatives():
    """From now on, answer "absent" only from a listing taken by this process after this call.

    `exists()` answers a missing file from cached knowledge (its own entry, a directory
    listing marker, an absent ancestor), shared between the processes of a machine through
    its cache server. No job publishes what it writes -- a job reaches no cache server -- so
    a file that a job wrote after its directory was listed reads as absent until the marker
    expires; clearing only the in-process caches would not help, because the next lookup
    falls through to the same marker on the server. After this call, a negative answer for a
    directory not listed by this process since the call costs one listing, which also
    republishes the directory's entries to the cache server. Positive answers stay
    cache-served.
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
        path_cache_server=None,
        verbose=0,
    ):
        self.voms_token = get_voms_proxy_info()["path"]
        if path_cache_server is None:
            self.path_cache = PathCache(local_path_cache_validity_period)
        else:
            self.path_cache = RemotePathCache(
                path_cache_server,
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
    # directories are listed from several threads at once (list_target_directories)
    _counter_lock = threading.Lock()

    def exists(self, path, base=None, **kwargs):
        GFALFileInterface.exists_counter += 1
        path_dir, path_name = os.path.split(path)
        path_uri = self.uri(path, base=base)
        dir_uri = self.uri(path_dir, base=base)
        result = False
        cached_result, from_local_cache = self._cached_answer(path_uri, dir_uri)
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

    def _cached_answer(self, path_uri, dir_uri):
        """What the cache can say about `path_uri` (True, False, or None when only a listing
        of `dir_uri` can tell), and whether it came from the in-process level."""
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
            # See require_fresh_negatives: a cached "absent" may predate a job's write.
            cached_result = None
        elif from_listing:
            # Memoize the negative result locally so repeated checks of the same path do
            # not query the cache server again (a round-trip per call otherwise).
            self.path_cache.set_local(path_uri, False)
        return cached_result, from_local_cache

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
        with GFALFileInterface._counter_lock:
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
