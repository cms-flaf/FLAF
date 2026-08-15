import json
import os
import tempfile
import traceback

import law
import luigi

from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    CrabWorkflow,
    copy_param,
)


class HelloWorldTask(Task, HTCondorWorkflow, CrabWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 0.5)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    poll_interval = copy_param(HTCondorWorkflow.poll_interval, 1)
    bundle_flavours = ["core"]
    force_fail = luigi.BoolParameter(
        default=False,
        significant=False,
        description="raise an exception in run() to test log transfer on crash",
    )
    # Remote X509 / grid access probe (significant so different probes do not collide).
    download_url = luigi.Parameter(
        default="",
        significant=True,
        description="if set, probe VOMS/X509 access: gfal_stat/sum and download when small",
    )
    max_download_bytes = luigi.IntParameter(
        default=20 * 1024 * 1024,
        significant=False,
        description="full download only if remote size is at most this many bytes",
    )
    test_rucio = luigi.BoolParameter(
        default=False,
        significant=True,
        description="if true, attempt Rucio Client() authentication on the worker",
    )

    def create_branch_map(self):
        return {0: "hello"}

    def output(self):
        return self.remote_target(
            self.version, self.__class__.__name__, self.period, "hello_world_done.txt"
        )

    def _probe_grid_access(self):
        """Return a JSON-serializable report of X509/gfal/Rucio status on this host."""
        from FLAF.RunKit.grid_tools import (
            get_voms_proxy_info,
            gfal_stat,
            gfal_sum,
            gfal_copy,
            get_rucio_client,
            copy_remote_file,
        )

        report = {
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "cwd": os.getcwd(),
            "X509_USER_PROXY": os.environ.get("X509_USER_PROXY"),
            "proxy_file_exists": False,
            "voms_proxy_info": None,
            "voms_proxy_error": None,
            "download_url": self.download_url,
            "gfal_stat": None,
            "gfal_stat_error": None,
            "gfal_sum": None,
            "gfal_sum_error": None,
            "download": None,
            "download_error": None,
            "rucio": None,
            "rucio_error": None,
        }

        proxy = os.environ.get("X509_USER_PROXY", "")
        report["proxy_file_exists"] = bool(proxy) and os.path.isfile(proxy)

        try:
            info = get_voms_proxy_info()
            # keep only plain types
            report["voms_proxy_info"] = {
                k: (float(v) if k == "timeleft" else str(v))
                for k, v in info.items()
                if k
                in (
                    "path",
                    "timeleft",
                    "identity",
                    "issuer",
                    "type",
                    "strength",
                    "VO",
                )
            }
        except Exception as e:
            report["voms_proxy_error"] = f"{type(e).__name__}: {e}"

        if self.download_url:
            try:
                st = gfal_stat(self.download_url)
                report["gfal_stat"] = {
                    k: (int(v) if k == "size" else str(v)) for k, v in st.items()
                }
            except Exception as e:
                report["gfal_stat_error"] = (
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )

            try:
                asum = gfal_sum(self.download_url, sum_type="adler32")
                report["gfal_sum"] = {"adler32": str(asum)}
            except Exception as e:
                report["gfal_sum_error"] = (
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )

            size = None
            if report.get("gfal_stat") and "size" in report["gfal_stat"]:
                size = int(report["gfal_stat"]["size"])
            do_full = size is not None and size <= int(self.max_download_bytes)
            report["download"] = {
                "attempted": do_full,
                "reason": (
                    "size_ok"
                    if do_full
                    else (
                        "size_unknown"
                        if size is None
                        else f"size {size} > max_download_bytes {self.max_download_bytes}"
                    )
                ),
                "local_size": None,
            }
            if do_full:
                try:
                    tmpdir = tempfile.mkdtemp(prefix="hello_dl_")
                    local = os.path.join(tmpdir, "download.bin")
                    # Protocol URLs (davs/root/srm): gfal_copy. Pure /store/ LFNs: Rucio path.
                    if self.download_url.startswith("/store/"):
                        copy_remote_file(self.download_url, local, verbose=1)
                    else:
                        gfal_copy(self.download_url, local, verbose=1)
                    report["download"]["local_size"] = os.path.getsize(local)
                    report["download"]["ok"] = True
                except Exception as e:
                    report["download"]["ok"] = False
                    report["download_error"] = (
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )

        if self.test_rucio:
            try:
                client = get_rucio_client()
                # Lightweight authenticated call: whoami if available, else ping list.
                who = None
                if hasattr(client, "whoami"):
                    who = client.whoami()
                report["rucio"] = {
                    "ok": True,
                    "whoami": str(who) if who is not None else None,
                    "account": os.environ.get("RUCIO_ACCOUNT"),
                }
            except Exception as e:
                report["rucio"] = {"ok": False}
                report["rucio_error"] = (
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )

        return report

    def run(self):
        if self.force_fail:
            raise RuntimeError(
                f"Forced failure for testing log transfer on crash. version = {self.version}"
            )
        print(f"hello world from {self.version}")

        body = "done\n"
        if self.download_url or self.test_rucio:
            report = self._probe_grid_access()
            print("=== grid access probe report ===")
            print(json.dumps(report, indent=2, default=str))
            body = json.dumps(report, indent=2, default=str) + "\n"

        with self.output().localize("w") as tmp:
            with open(tmp.path, "w") as f:
                f.write(body)
