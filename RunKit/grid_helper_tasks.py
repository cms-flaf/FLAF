import law
import luigi
import os

from .run_tools import PsCallError, ps_call, on_batch_node
from .grid_tools import get_voms_proxy_info


class CreateVomsProxy(law.Task):
    time_limit = luigi.Parameter(default="24")

    def __init__(self, *args, **kwargs):
        super(CreateVomsProxy, self).__init__(*args, **kwargs)
        self.proxy_path = os.getenv("X509_USER_PROXY")
        if not self.proxy_path:
            raise RuntimeError("CreateVomsProxy requires X509_USER_PROXY to be set")

    @property
    def on_batch_node(self):
        return on_batch_node()

    def complete(self):
        if not os.path.exists(self.proxy_path):
            return False
        try:
            timeleft = get_voms_proxy_info().get("timeleft", 0.0)
        except PsCallError:
            # voms-proxy-info exits non-zero on an expired or unreadable proxy
            return False
        if self.on_batch_node:
            # Any valid proxy the batch system delegated will do: its remaining lifetime
            # is not ours to police (CRAB's lives for slightly under 24 h, i.e. below the
            # interactive renewal threshold), and voms-proxy-init cannot run unattended on
            # a worker. Enforcing the threshold here would delete the delegated proxy,
            # after which every remote-storage call in the job fails.
            return True
        return timeleft >= float(self.time_limit)

    def output(self):
        return law.LocalFileTarget(self.proxy_path)

    def create_proxy(self, proxy_file):
        self.publish_message("Creating voms proxy...")
        proxy_file.makedirs()
        ps_call(
            [
                "voms-proxy-init",
                "-voms",
                "cms",
                "-rfc",
                "-valid",
                "192:00",
                "--out",
                proxy_file.path,
            ]
        )

    def run(self):
        if self.on_batch_node:
            raise RuntimeError(
                f"No usable voms proxy at {self.proxy_path} on a batch node, and a new "
                "one cannot be created there. Check that the batch system delegated a "
                "proxy."
            )
        proxy_file = self.output()
        if proxy_file.exists():
            self.publish_message("Removing old proxy.")
            proxy_file.remove()
        self.create_proxy(proxy_file)
        if not proxy_file.exists():
            raise RuntimeError("Unable to create voms proxy")
