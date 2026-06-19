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
)


class PathCacheEntry:
    def __init__(self, path, exists, expiration_time):
        self.path = path
        self.exists = exists
        self.expiration_time = expiration_time

    def is_valid(self):
        return self.expiration_time >= time.time()


class PathCache:
    def __init__(self, validity_period):
        self.validity_period = validity_period
        self.cache = {}

    def set(self, path, exists):
        self.cache[path] = PathCacheEntry(
            path, exists, time.time() + self.validity_period
        )

    def set_exists(self, base_dir, items):
        for item in items:
            path = os.path.join(base_dir, item)
            self.set(path, True)

    def get(self, path):
        if path in self.cache:
            entry = self.cache[path]
            if entry.is_valid():
                return entry.exists, True
            else:
                del self.cache[path]
        return None, True

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

    def set_exists(self, base_dir, items):
        entries = []
        for item in items:
            path = os.path.join(base_dir, item)
            entries.append((path, True))
        entries.append((base_dir, True))
        set_remote_cache_status(
            entries, self.host, self.port, self.timeout, verbose=self.verbose
        )
        self.local_cache.set_exists(base_dir, items)

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
        if cached_result is None:
            cached_dir_result, from_local_cache = self.path_cache.get(dir_uri)
            if cached_dir_result is not None:
                cached_result = False
        use_cache = cached_result is not None

        if use_cache:
            result = cached_result
        else:
            path_dir, path_name = os.path.split(path)
            dir_entries = self.listdir(path_dir, base=base, silent=True)
            result = path_name in dir_entries
            if not result:
                self.path_cache.set(path_uri, False)

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
                if cached_dst_dir is not None and not cached_dst_dir:
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
            if not silent:
                gfal_ls_safe(
                    path_uri, voms_token=self.voms_token, catch_stderr=False, verbose=1
                )
                raise GfalError(f"GFALFileInterface: failed to list directory {path}")
            entry_names = []
            self.path_cache.set(path_uri, False)
        else:
            entry_names = [entry.name for entry in entries]
            self.path_cache.set_exists(path_uri, entry_names)
        return entry_names

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
