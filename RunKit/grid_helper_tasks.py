import law
import luigi
import os
import re

from .run_tools import ps_call
from .grid_tools import get_voms_proxy_info


class CreateVomsProxy(law.Task):
    time_limit = luigi.Parameter(default="24")

    def __init__(self, *args, **kwargs):
        super(CreateVomsProxy, self).__init__(*args, **kwargs)
        self.proxy_path = os.getenv("X509_USER_PROXY")
        if os.path.exists(self.proxy_path):
            proxy_info = get_voms_proxy_info()
            timeleft = proxy_info.get("timeleft", 0.0)
            if timeleft < float(self.time_limit):
                self.publish_message(
                    f"Removing old proxy which expires in a less than {timeleft:.1f} hours."
                )
                self.output().remove()

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
        proxy_file = self.output()
        self.create_proxy(proxy_file)
        if not proxy_file.exists():
            raise RuntimeError("Unable to create voms proxy")
