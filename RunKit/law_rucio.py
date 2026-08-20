from law.target.remote.interface import RemoteFileInterface
from .grid_tools import (
    get_voms_proxy_info,
    copy_remote_file,
    rucio_list_files,
    rucio_list_replicas,
    das_dataset_file_info,
)


class RucioFileInterface(RemoteFileInterface):
    local_prefix = "file://"

    def __init__(self, *, ls_cache_validity_period=60):
        self.voms_token = get_voms_proxy_info()["path"]
        self.dataset_files = {}
        self.dataset_available_files = {}
        self.dataset_event_counts = {}
        super(RucioFileInterface, self).__init__(base=["/"])

    def is_local(self, path):
        return path.startswith(RucioFileInterface.local_prefix)

    def exists(self, path, base=None, **kwargs):
        self._raise_not_implemented("exists")

    def remove(self, path, base=None, silent=True, **kwargs):
        self._raise_not_implemented("remove")

    def filecopy(self, src, dst, base=None, **kwargs):
        src_local = self.is_local(src)
        dst_local = self.is_local(dst)
        if not (not src_local and dst_local):
            raise RuntimeError(
                "RucioFileInterface: only copy from remote to local is supported"
            )
        dst_path = dst[len(RucioFileInterface.local_prefix) :]
        copy_remote_file(src, dst_path, voms_token=self.voms_token)
        return src, dst

    def listdir(self, path, base=None, silent=False, **kwargs):
        return list(self._dataset_file_info(path).keys())

    def listdir_info(self, path, base=None, silent=True, **kwargs):
        """``{lfn: {"size": bytes, "n_events": int|None}}`` for a dataset.

        Rucio always knows the size but leaves the CMS ``events`` field empty, so the
        event counts come from a single DAS query per dataset.  Both are optional inputs
        to job-cost estimation, so a DAS failure leaves ``n_events`` as None rather than
        propagating.
        """
        info = dict(self._dataset_file_info(path))
        if path not in self.dataset_event_counts:
            self.dataset_event_counts[path] = das_dataset_file_info(path)
        for name, das_info in self.dataset_event_counts[path].items():
            if name in info and das_info.get("n_events"):
                info[name]["n_events"] = das_info["n_events"]
        return info

    def _dataset_file_info(self, path):
        if path not in self.dataset_files:
            self.dataset_files[path] = {
                f["name"]: {"size": f.get("bytes"), "n_events": None}
                for f in rucio_list_files(path)
            }
        return self.dataset_files[path]

    def is_available(self, dataset, file, verbose=0):
        if dataset not in self.dataset_available_files:
            if verbose > 0:
                print(f"{dataset}: querying Rucio for available replicas...")
            all_files = self.listdir(dataset)
            replicas = rucio_list_replicas(all_files)
            available_files = set(
                name for name, info in replicas.items() if info["available"]
            )
            if verbose > 0:
                print(
                    f"  {len(available_files)}/{len(all_files)} available files found"
                )
            self.dataset_available_files[dataset] = available_files
        return file in self.dataset_available_files[dataset]

    @staticmethod
    def _raise_not_implemented(method_name):
        raise NotImplementedError(
            f"{method_name} is not supported by the Rucio interface"
        )

    def chmod(self, file, perm, **kwargs):
        self._raise_not_implemented("chmod")

    def isdir(self, path, **kwargs):
        self._raise_not_implemented("isdir")

    def isfile(self):
        self._raise_not_implemented("isfile")

    def mkdir(self):
        self._raise_not_implemented("mkdir")

    def mkdir_rec(self):
        self._raise_not_implemented("mkdir_rec")

    def rmdir(self):
        self._raise_not_implemented("rmdir")

    def stat(self):
        self._raise_not_implemented("stat")

    def unlink(self):
        self._raise_not_implemented("unlink")
