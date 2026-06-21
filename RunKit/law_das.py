import datetime
import os
from law.target.remote.interface import RemoteFileInterface
from .grid_tools import get_voms_proxy_info, copy_remote_file, run_dasgoclient


class DASFileInterface(RemoteFileInterface):
    local_prefix = "file://"

    def __init__(self, *, ls_cache_validity_period=60):
        self.voms_token = get_voms_proxy_info()["path"]
        self.dataset_available_files = {}
        self.dataset_file_records = {}
        super(DASFileInterface, self).__init__(base=["/"])

    def is_local(self, path):
        return path.startswith(DASFileInterface.local_prefix)

    def exists(self, path, base=None, **kwargs):
        self._raise_not_implemented("exists")

    def remove(self, path, base=None, silent=True, **kwargs):
        self._raise_not_implemented("remove")

    def filecopy(self, src, dst, base=None, **kwargs):
        src_local = self.is_local(src)
        dst_local = self.is_local(dst)
        if not (not src_local and dst_local):
            raise RuntimeError(
                "DASFileInterface: only copy from remote to local is supported"
            )
        dst_path = dst[len(DASFileInterface.local_prefix) :]
        copy_remote_file(src, dst_path, voms_token=self.voms_token)
        return src, dst

    def get_dataset_sites(self, dataset, disk_only=True, verbose=0):
        output = run_dasgoclient(
            f"site dataset={dataset}",
            inputDBS="global",
            json_output=True,
            verbose=verbose,
        )
        sites = []
        for entry in output:
            if "site" not in entry:
                continue
            for site_entry in entry["site"]:
                if disk_only and site_entry["kind"] != "DISK":
                    continue
                sites.append((site_entry["name"], site_entry["total_files"]))

        sites = sorted(sites, key=lambda x: x[1], reverse=True)
        site_list = []
        for site in sites:
            if site[0] not in site_list:
                site_list.append(site[0])
        return site_list

    def is_available(self, dataset, file, verbose=0):
        if dataset not in self.dataset_available_files:
            if verbose > 0:
                print(f"{dataset}: searching for available files...")
            all_files = set(self.listdir(dataset, verbose=verbose))
            sites = self.get_dataset_sites(dataset, verbose=verbose)
            available_files = set()
            for site in sites:
                if all_files == available_files:
                    break
                output = run_dasgoclient(
                    f"file site={site} dataset={dataset}",
                    inputDBS="global",
                    json_output=True,
                    verbose=verbose,
                )
                for entry in output:
                    if "file" not in entry:
                        continue
                    for file_entry in entry["file"]:
                        file_name = file_entry["name"]
                        if file_name in all_files:
                            available_files.add(file_name)
                if verbose > 0:
                    print(
                        f"  {len(available_files)}/{len(all_files)} available files found"
                    )
            self.dataset_available_files[dataset] = available_files
        return file in self.dataset_available_files[dataset]

    def file_records(self, dataset, verbose=0):
        if dataset not in self.dataset_file_records:
            output = run_dasgoclient(
                f"file dataset={dataset}",
                inputDBS="global",
                json_output=True,
                verbose=verbose,
            )
            file_records = {}
            for entry in output:
                if "file" not in entry:
                    continue
                for file_entry in entry["file"]:
                    file_name = file_entry["name"]
                    file_records[file_name] = file_entry
            self.dataset_file_records[dataset] = file_records
        return self.dataset_file_records[dataset]

    def n_events(self, dataset, file, verbose=0):
        file_records = self.file_records(dataset, verbose=verbose)
        if file not in file_records:
            raise RuntimeError(f'File "{file}" record not found in dataset "{dataset}"')
        return file_records[file]["nevents"]

    def listdir(self, path, base=None, silent=False, **kwargs):
        file_records = self.file_records(path, verbose=0)
        return list(file_records.keys())

    @staticmethod
    def _raise_not_implemented(method_name):
        raise NotImplementedError(
            f"{method_name} is not supported by the DAS interface"
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
