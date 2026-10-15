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
        # The old proxy is replaced only by one that exists: voms-proxy-init asks for the
        # certificate passphrase, so without a controlling terminal (setsid, CI) it fails,
        # and the old proxy, possibly still valid for hours, has to survive that.
        proxy_file = self.output()
        new_proxy = law.LocalFileTarget(f"{proxy_file.path}.new.{os.getpid()}")
        try:
            self.create_proxy(new_proxy)
        except PsCallError as e:
            if new_proxy.exists():
                new_proxy.remove()
            kept = (
                f"the existing one at {proxy_file.path} is kept"
                if proxy_file.exists()
                else f"there is none at {proxy_file.path}"
            )
            raise RuntimeError(
                f"Unable to create a voms proxy; {kept}. Re-run the command from a "
                "terminal, or accept an existing proxy with "
                "--CreateVomsProxy-time-limit <hours>."
            ) from e
        if not new_proxy.exists():
            raise RuntimeError("Unable to create voms proxy")
        self.publish_message("Moving the new proxy into place.")
        os.replace(new_proxy.path, proxy_file.path)
