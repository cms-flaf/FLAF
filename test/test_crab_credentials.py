#!/usr/bin/env python3
"""Which credentials let the CRAB backend submit at all.

`crab submit --proxy <file>`, which law always uses, makes CRABClient return from
`handleMyProxy` before it delegates or renews anything, so the credential the TaskWorker
retrieves is whatever is already on myproxy.cern.ch -- and it is looked up under `sha1(DN)`
and under no other name. A credential stored under the plain DN, which is what a bare
`myproxy-init -d` writes, is therefore not one CRAB can use: letting it open the gate sends a
whole production out to fail on the TaskWorker instead of failing at submission in a second
(ported from DSProd test/test_myproxy_gate.py).

law builds the CMSSW sandbox it runs `crab` in lazily, inside every submission attempt, where a
failure is swallowed per job and resurfaces half an hour later as a retry-limit failure. The
gate builds it first, so a broken sandbox is reported as such, before any credential check can
blame the proxy for it.
"""

import hashlib
import importlib.util
import inspect
import os
import shlex
import sys
import tempfile
import types
import unittest
from collections import OrderedDict
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

# law_customizations imports FLAF.Common.Setup, whose Utilities imports ROOT at module level.
# The unit-test environment has no ROOT and nothing exercised here touches it, so an empty
# stand-in is provided for the import only, and withdrawn again so that no other test sees it.
_root_stand_in = None
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    _root_stand_in = sys.modules["ROOT"] = types.ModuleType("ROOT")

import law  # noqa: E402
import law.contrib.cms.job  # noqa: E402
import law.contrib.wlcg.util  # noqa: E402
from law.util import no_value  # noqa: E402

from FLAF.run_tools import law_customizations as lc  # noqa: E402

if _root_stand_in is not None and sys.modules.get("ROOT") is _root_stand_in:
    del sys.modules["ROOT"]

DAY = 24 * 3600
PLAIN_DN = "/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=someone"
HASHED = hashlib.sha1(PLAIN_DN.encode("utf-8")).hexdigest()
MYPROXY_SERVER = "myproxy.cern.ch"

_LAW_GET_MYPROXY_INFO = law.contrib.wlcg.util.get_myproxy_info


class FakeMyProxyInfo:
    """`law.wlcg.get_myproxy_info` against a myproxy.cern.ch that holds credentials under the
    hashed and/or the plain DN. Every call is recorded with law's own defaults applied, so a
    caller that leaves an argument out is judged by what law then does."""

    def __init__(self, hashed_timeleft=None, plain_timeleft=None, error=None):
        self.store = {}
        if hashed_timeleft is not None:
            self.store[HASHED] = hashed_timeleft
        if plain_timeleft is not None:
            self.store[PLAIN_DN] = plain_timeleft
        self.error = error
        self.calls = []

    def __call__(self, *args, **kwargs):
        bound = inspect.signature(_LAW_GET_MYPROXY_INFO).bind(*args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        # law's own resolution: the proxy's identity, sha1-encoded unless told otherwise
        username = call["username"] or PLAIN_DN
        if call["encode_username"]:
            username = hashlib.sha1(username.encode("utf-8")).hexdigest()
        timeleft = None
        if call["endpoint"] == MYPROXY_SERVER:
            timeleft = self.store.get(username)
        if timeleft is None:
            if call["silent"]:
                return None
            raise Exception("myproxy-info failed with code 1")
        return {"username": username, "subject": PLAIN_DN, "timeleft": timeleft}


class Sandbox:
    """The CMSSW sandbox behind `CrabJobManager.cmssw_env`; `error` makes loading it fail the
    way law's CMSSW sandbox does."""

    def __init__(self, error=None):
        self.error = error
        self.env_requests = 0

    @property
    def env(self):
        self.env_requests += 1
        if self.error is not None:
            raise self.error
        return {"CMSSW_VERSION": "CMSSW_15_0_0"}


def make_workflow_proxy(sandbox):
    """A real `_FLAFCrabWorkflowProxy` around a real `FLAFCrabJobManager` whose CMSSW sandbox
    is `sandbox`. The proxy is built without its constructor, which needs a task; every
    attribute the gate reads is still looked up on the real class."""
    manager = lc.FLAFCrabJobManager(sandbox_name="CMSSW_15_0_0")
    manager.cmssw_sandbox = sandbox
    proxy = object.__new__(lc._FLAFCrabWorkflowProxy)
    proxy.job_manager = manager
    proxy._job_manager_setup_kwargs = no_value
    return proxy


class GateCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.proxy_file = os.path.join(self.tmp, "x509up_u12345")
        with open(self.proxy_file, "w") as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
        self.sandbox = Sandbox()
        self.workflow_proxy = make_workflow_proxy(self.sandbox)
        self.server = None
        self.voms_check = None
        self.delegation = None

    def crab_home(self):
        """The job manager's sandbox env writes a `crab` wrapper into a home under the
        system tmp; it is kept inside this test's directory."""
        return mock.patch("tempfile.gettempdir", return_value=self.tmp)

    def run_gate(self, server=None, proxy_file=no_value, voms_valid=True, call=None):
        """Run the gate with `server` as myproxy.cern.ch and `proxy_file` as X509_USER_PROXY
        (None: unset). law's own renewal and delegation helpers are replaced by recorders:
        the gate must refuse, never delegate on its own (that asks for a passphrase)."""
        self.server = server if server is not None else FakeMyProxyInfo()
        if proxy_file is no_value:
            proxy_file = self.proxy_file
        call = call or self.workflow_proxy.setup_job_manager
        with self.crab_home(), mock.patch.dict(os.environ), mock.patch(
            "law.wlcg.get_myproxy_info", side_effect=self.server
        ), mock.patch(
            "law.wlcg.check_vomsproxy_validity", return_value=voms_valid
        ) as self.voms_check, mock.patch(
            "law.contrib.cms.workflow.delegate_myproxy"
        ) as delegate, mock.patch(
            "law.contrib.cms.workflow.renew_vomsproxy"
        ) as renew:
            self.delegation = (delegate, renew)
            if proxy_file is None:
                os.environ.pop("X509_USER_PROXY", None)
            else:
                os.environ["X509_USER_PROXY"] = proxy_file
            return call()


class MyProxyGate(GateCase):
    """The DSProd gate, ported: only a >= 5-day credential under sha1(DN) opens it."""

    def test_hashed_credential_opens_the_gate(self):
        kwargs = self.run_gate(FakeMyProxyInfo(hashed_timeleft=30 * DAY))
        self.assertEqual(kwargs["myproxy_username"], HASHED)
        self.assertEqual(kwargs["proxy_file"], self.proxy_file)

    def test_fresh_plain_dn_credential_does_not(self):
        """The regression: a 30-day DN-keyed credential next to a 2-day hashed one."""
        with self.assertRaises(RuntimeError) as caught:
            self.run_gate(
                FakeMyProxyInfo(hashed_timeleft=2 * DAY, plain_timeleft=30 * DAY)
            )
        self.assertIn("crab createmyproxy", str(caught.exception))

    def test_plain_dn_credential_alone_does_not(self):
        with self.assertRaises(RuntimeError) as caught:
            self.run_gate(FakeMyProxyInfo(plain_timeleft=30 * DAY))
        self.assertIn("crab createmyproxy", str(caught.exception))

    def test_five_days_is_the_boundary(self):
        kwargs = self.run_gate(FakeMyProxyInfo(hashed_timeleft=5 * DAY))
        self.assertEqual(kwargs["myproxy_username"], HASHED)
        with self.assertRaises(RuntimeError):
            self.run_gate(FakeMyProxyInfo(hashed_timeleft=5 * DAY - 1))

    def test_myproxy_is_only_asked_for_the_hashed_name(self):
        servers = {
            "hashed only": FakeMyProxyInfo(hashed_timeleft=30 * DAY),
            "short hashed, fresh plain": FakeMyProxyInfo(
                hashed_timeleft=2 * DAY, plain_timeleft=30 * DAY
            ),
            "plain only": FakeMyProxyInfo(plain_timeleft=30 * DAY),
            "nothing": FakeMyProxyInfo(),
        }
        for name, server in servers.items():
            with self.subTest(name):
                try:
                    self.run_gate(server)
                except RuntimeError:
                    pass
                self.assertTrue(server.calls, "the myproxy server was never asked")
                for call in server.calls:
                    self.assertIs(call["encode_username"], True)
                    self.assertEqual(call["endpoint"], MYPROXY_SERVER)

    def test_no_answer_from_myproxy_closes_the_gate(self):
        servers = {
            "myproxy-info fails": FakeMyProxyInfo(
                error=Exception("myproxy-info failed with code 1")
            ),
            "no credential": FakeMyProxyInfo(),
        }
        for name, server in servers.items():
            with self.subTest(name):
                with self.assertRaises(RuntimeError) as caught:
                    self.run_gate(server)
                self.assertIn("crab createmyproxy", str(caught.exception))

    def test_incomplete_myproxy_answer_closes_the_gate(self):
        answers = {
            "no timeleft": {"username": HASHED},
            "no username": {"timeleft": 30 * DAY},
            "empty username": {"username": "", "timeleft": 30 * DAY},
        }
        for name, answer in answers.items():
            with self.subTest(name):
                with self.assertRaises(RuntimeError) as caught:
                    self.run_gate(mock.Mock(return_value=answer))
                self.assertIn("crab createmyproxy", str(caught.exception))

    def test_refusal_never_delegates_on_its_own(self):
        """law's stock CRAB gate renews the VOMS proxy and delegates to MyProxy itself, which
        asks for the certificate passphrase and writes the plain-DN credential CRAB ignores.
        """
        for server in (FakeMyProxyInfo(plain_timeleft=30 * DAY), FakeMyProxyInfo()):
            with self.assertRaises(RuntimeError):
                self.run_gate(server)
            for helper in self.delegation:
                self.assertFalse(helper.called)
        with self.assertRaises(RuntimeError):
            self.run_gate(FakeMyProxyInfo(hashed_timeleft=30 * DAY), voms_valid=False)
        for helper in self.delegation:
            self.assertFalse(helper.called)

    def test_returned_kwargs_are_accepted_by_the_crab_job_manager(self):
        """law forwards the gate's kwargs to every submit, query, cancel and cleanup call."""
        kwargs = self.run_gate(FakeMyProxyInfo(hashed_timeleft=30 * DAY))
        for method in ("submit", "query", "cancel", "cleanup"):
            with self.subTest(method):
                params = inspect.signature(
                    getattr(law.contrib.cms.job.CrabJobManager, method)
                ).parameters
                self.assertLessEqual(set(kwargs), set(params))

    def test_laws_real_status_query_runs_crab_with_the_gates_proxy(self):
        """The same, executed: law's own `CrabJobManager.query`, handed the gate's kwargs merged
        with the workflow's query kwargs as law's poll loop merges them, runs a `crab` found on
        PATH with `--proxy <file>`.
        A keyword law does not take never reaches crab: the query is ridden out as unreadable.
        """
        kwargs = self.run_gate(FakeMyProxyInfo(hashed_timeleft=30 * DAY))
        fake_bin = os.path.join(self.tmp, "bin")
        os.makedirs(fake_bin)
        crab = os.path.join(fake_bin, "crab")
        with open(crab, "w") as f:
            f.write(
                "#!/bin/bash\n"
                'printf "%s\\n" "$@" > "$CRAB_ARGV"\n'
                'printf "Status on the CRAB server:\\tSUBMITTED\\n"\n'
                'printf "Status on the scheduler:\\tSUBMITTED\\n"\n'
                'printf \'{"1": {"State": "running"}}\\n\'\n'
            )
        os.chmod(crab, 0o755)
        argv_file = os.path.join(self.tmp, "crab_argv")
        proj_dir = os.path.join(self.tmp, "crab_task")
        os.makedirs(proj_dir)
        manager = self.workflow_proxy.job_manager
        job_id = manager.JobId(1, "task", proj_dir)
        env = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "CRAB_ARGV": argv_file}
        with mock.patch.object(
            lc.FLAFCrabJobManager,
            "cmssw_env",
            new_callable=mock.PropertyMock,
            return_value=env,
        ), mock.patch.object(lc.time, "sleep"), mock.patch("builtins.print"):
            result = manager.query(
                proj_dir,
                job_ids=[job_id],
                **law.util.merge_dicts(kwargs, lc.CrabWorkflow.crab_job_kwargs_query),
            )
        self.assertTrue(os.path.exists(argv_file), "crab never ran")
        with open(argv_file) as f:
            argv = f.read().splitlines()
        self.assertEqual(argv[:2], ["status", "--dir"])
        self.assertEqual(argv[argv.index("--proxy") + 1], self.proxy_file)
        self.assertEqual(result[job_id]["status"], manager.RUNNING)


class VomsProxyGate(GateCase):
    def test_missing_proxy_file_is_reported_as_the_proxy(self):
        cases = {
            "nonexistent file": "/nonexistent/proxy",
            "unset": None,
            "empty": "",
            "a directory": self.tmp,
        }
        for name, proxy_file in cases.items():
            with self.subTest(name):
                with self.assertRaises(RuntimeError) as caught:
                    self.run_gate(
                        FakeMyProxyInfo(hashed_timeleft=30 * DAY), proxy_file=proxy_file
                    )
                self.assertIn("X509_USER_PROXY", str(caught.exception))
                self.assertFalse(self.server.calls)

    def test_expired_voms_proxy_is_refused_before_myproxy_is_asked(self):
        with self.assertRaises(RuntimeError) as caught:
            self.run_gate(FakeMyProxyInfo(hashed_timeleft=30 * DAY), voms_valid=False)
        self.assertIn(self.proxy_file, str(caught.exception))
        self.assertIn("voms-proxy-init", str(caught.exception))
        self.assertTrue(self.voms_check.called)
        self.assertFalse(self.server.calls)

    def test_valid_voms_proxy_is_checked_on_every_opening(self):
        self.run_gate(FakeMyProxyInfo(hashed_timeleft=30 * DAY))
        self.assertTrue(self.voms_check.called)


class SandboxPreflight(GateCase):
    def test_sandbox_failure_is_reported_before_any_credential_check(self):
        # with every credential missing as well: the sandbox is what must be named
        failure = Exception("cmssw sandbox env loading failed with exit code 127")
        self.workflow_proxy = make_workflow_proxy(Sandbox(error=failure))
        with mock.patch.object(
            law.contrib.wlcg.util, "interruptable_popen"
        ) as grid_command:
            with self.assertRaises(RuntimeError) as caught:
                self.run_gate(FakeMyProxyInfo(), proxy_file=None, voms_valid=False)
        message = str(caught.exception)
        self.assertIn("sandbox", message.lower())
        self.assertIn(str(failure), message)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertNotIn("X509_USER_PROXY", message)
        self.assertFalse(self.voms_check.called)
        self.assertFalse(self.server.calls)
        self.assertFalse(grid_command.called)

    def test_working_sandbox_is_built_before_the_gate_opens(self):
        self.run_gate(FakeMyProxyInfo(hashed_timeleft=30 * DAY))
        self.assertGreaterEqual(self.sandbox.env_requests, 1)


def myproxy_info_output(username, timeleft):
    hours, rest = divmod(timeleft, 3600)
    minutes, seconds = divmod(rest, 60)
    return (
        f"username: {username}\n"
        f"owner: {PLAIN_DN}\n"
        "  retrieval policy: *\n"
        f"  timeleft: {hours}:{minutes:02d}:{seconds:02d}  ({timeleft / DAY:.1f} days)\n"
    )


class FakeGridCommands:
    """`voms-proxy-info` and `myproxy-info` as law's wlcg helpers run them through
    `interruptable_popen`; `store` maps myproxy usernames to remaining seconds."""

    def __init__(self, proxy_file, store):
        self.proxy_file = proxy_file
        self.store = store
        self.argvs = []

    def __call__(self, cmd, *args, **kwargs):
        argv = shlex.split(cmd)
        self.argvs.append(argv)
        if argv[0] == "voms-proxy-info":
            if "--file" in argv and argv[argv.index("--file") + 1] != self.proxy_file:
                return 1, "", "Proxy not found: " + argv[argv.index("--file") + 1]
            if "--exists" in argv:
                return 0, "", ""
            if "--timeleft" in argv:
                return 0, f"{8 * DAY}\n", ""
            if "--identity" in argv:
                return 0, PLAIN_DN + "\n", ""
        if argv[0] == "myproxy-info":
            server = argv[argv.index("-s") + 1]
            username = argv[argv.index("-l") + 1]
            if server == MYPROXY_SERVER and username in self.store:
                return 0, myproxy_info_output(username, self.store[username]), ""
            return 1, "", f"ERROR from myproxy-server: no credentials for {username}"
        raise AssertionError(f"unexpected command run by the gate: {cmd}")

    def myproxy_lookups(self):
        return [a[a.index("-l") + 1] for a in self.argvs if a[0] == "myproxy-info"]


class GateWithLawsOwnHelpers(GateCase):
    """The same gate with law's real `check_vomsproxy_validity` and `get_myproxy_info`, only
    the two command-line tools faked: the sha1 encoding and the `timeleft` parsing the gate
    relies on are law's."""

    def run_real_gate(self, store):
        commands = FakeGridCommands(self.proxy_file, store)
        with self.crab_home(), mock.patch.dict(
            os.environ, {"X509_USER_PROXY": self.proxy_file}
        ), mock.patch.object(
            law.contrib.wlcg.util, "interruptable_popen", side_effect=commands
        ), mock.patch(
            "law.contrib.cms.workflow.delegate_myproxy",
            side_effect=AssertionError("the gate delegated"),
        ):
            try:
                return self.workflow_proxy.setup_job_manager()
            finally:
                self.commands = commands

    def test_hashed_credential_opens_the_gate(self):
        kwargs = self.run_real_gate({HASHED: 30 * DAY})
        self.assertEqual(kwargs["myproxy_username"], HASHED)
        self.assertEqual(kwargs["proxy_file"], self.proxy_file)
        self.assertEqual(self.commands.myproxy_lookups(), [HASHED])

    def test_fresh_plain_dn_credential_does_not(self):
        with self.assertRaises(RuntimeError) as caught:
            self.run_real_gate({HASHED: 2 * DAY, PLAIN_DN: 30 * DAY})
        self.assertIn("crab createmyproxy", str(caught.exception))
        self.assertNotIn(PLAIN_DN, self.commands.myproxy_lookups())

    def test_plain_dn_credential_alone_does_not(self):
        with self.assertRaises(RuntimeError):
            self.run_real_gate({PLAIN_DN: 30 * DAY})
        self.assertEqual(self.commands.myproxy_lookups(), [HASHED])

    def test_five_days_is_the_boundary(self):
        kwargs = self.run_real_gate({HASHED: 5 * DAY})
        self.assertEqual(kwargs["myproxy_username"], HASHED)
        with self.assertRaises(RuntimeError):
            self.run_real_gate({HASHED: 5 * DAY - 1})


class LawRunsTheGateBeforeSubmitting(GateCase):
    """law's own `_submit_group` calls the gate before the job manager sees any job, forwards
    its kwargs to the submission, and lets a gate failure end the submission."""

    def setUp(self):
        super().setUp()
        self.submitted = []

    def submit(self, server, n_jobs=2):
        """Run law's `_submit_group` for `n_jobs` jobs; the job manager records what it is
        asked to submit in `self.submitted`."""
        proxy = self.workflow_proxy

        def submit_group(job_files, retries=None, threads=None, **kwargs):
            self.submitted.append(kwargs)
            return [f"job-{i}" for i in range(len(job_files))]

        proxy.job_manager.submit_group = submit_group
        proxy.task = types.SimpleNamespace(
            submission_threads=1,
            crab_job_kwargs=lc.CrabWorkflow.crab_job_kwargs,
            crab_job_kwargs_submit=lc.CrabWorkflow.crab_job_kwargs_submit,
        )
        job_file = os.path.join(self.tmp, "job.py")
        proxy.create_job_file = lambda submit_jobs: {"job": job_file, "log": None}
        proxy.job_data = types.SimpleNamespace(jobs={n: {} for n in range(n_jobs)})
        submit_jobs = OrderedDict((n, [n]) for n in range(n_jobs))
        return self.run_gate(
            server,
            call=lambda: lc._FLAFCrabWorkflowProxy._submit_group(proxy, submit_jobs),
        )

    def test_gate_kwargs_reach_the_submission(self):
        server = FakeMyProxyInfo(hashed_timeleft=30 * DAY)
        job_ids, _ = self.submit(server)
        self.assertEqual(job_ids, ["job-0", "job-1"])
        self.assertEqual(len(self.submitted), 1)
        self.assertEqual(self.submitted[0]["myproxy_username"], HASHED)
        self.assertEqual(self.submitted[0]["proxy_file"], self.proxy_file)
        # law sets the job manager up once per workflow, not once per submission
        self.submit(server)
        self.assertEqual(len(self.submitted), 2)
        self.assertEqual(len(server.calls), 1)

    def test_refused_gate_submits_nothing(self):
        with self.assertRaises(RuntimeError) as caught:
            self.submit(FakeMyProxyInfo(plain_timeleft=30 * DAY))
        self.assertIn("crab createmyproxy", str(caught.exception))
        self.assertEqual(self.submitted, [])

    def test_sandbox_failure_ends_the_submission(self):
        failure = Exception("cmssw sandbox env loading failed with exit code 127")
        self.workflow_proxy = make_workflow_proxy(Sandbox(error=failure))
        server = FakeMyProxyInfo(hashed_timeleft=30 * DAY)
        with self.assertRaises(RuntimeError) as caught:
            self.submit(server)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(self.submitted, [])
        self.assertFalse(server.calls)


if __name__ == "__main__":
    unittest.main()
