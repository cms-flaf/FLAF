"""The path-cache server of one user on one machine (RunKit/pathCacheServer.py), started on
demand by pathCacheClient.ensure_local_server.

Real server processes on real Unix sockets: what is under test is process lifecycle -- one
server per user and machine however many processes ask at once, a server that died is
replaced, its state survives a restart, nobody else can reach it -- and that decides in the
kernel, not in Python.
"""

import multiprocessing
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import law_gfal, pathCacheClient, pathCacheServer  # noqa: E402
from FLAF.RunKit.law_gfal import RemotePathCache  # noqa: E402
from FLAF.RunKit.pathCacheClient import (  # noqa: E402
    ServerUnavailable,
    ensure_local_server,
    get_status,
    set_status,
)

FILE = "davs://storage.example:1094/store/user/flaf/data/file_0.root"
MARKER = "davs://storage.example:1094/store/user/flaf/data/.flaf_listed"
ABSENT_DIR = "davs://storage.example:1094/store/user/flaf/gone"


def _ensure_in_child(directory, queue):
    queue.put(ensure_local_server(directory))


class _ServerCase(unittest.TestCase):
    def setUp(self):
        # short: a Unix socket path holds at most 107 bytes
        parent = tempfile.mkdtemp(prefix="fpc", dir="/tmp")
        self.addCleanup(self._remove, parent)
        self.dir = os.path.join(parent, "cache")

    def _remove(self, parent):
        for pid in self.server_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # and their loggers, which notice only within a minute that the server is gone
        subprocess.run(
            ["pkill", "-KILL", "-f", "--", f"--socket {parent}/"], check=False
        )
        subprocess.run(["rm", "-rf", parent], check=False)

    def test_a_killed_server_takes_its_logger_with_it(self):
        if type(self) is not AServerThatEnds:
            self.skipTest("one class runs it")
        ensure_local_server(self.dir)
        (server,) = self.server_pids()
        with open(f"/proc/{server}/task/{server}/children") as f:
            (logger,) = [int(pid) for pid in f.read().split()]
        os.kill(server, signal.SIGKILL)
        self.wait_dead(server)
        self.wait_dead(logger, timeout=120)

    def log(self):
        try:
            with open(os.path.join(self.dir, "server.log")) as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def server_pids(self):
        """Every server that ever listened in this test's directory."""
        return [
            int(pid) for pid in re.findall(r"listening on .* \(pid (\d+)\)", self.log())
        ]

    def alive(self, pid):
        try:
            with open(f"/proc/{pid}/status") as f:
                return "\nState:\tZ" not in f.read()
        except FileNotFoundError:
            return False

    def wait_dead(self, pid, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive(pid):
                return
            time.sleep(0.1)
        self.fail(f"server {pid} still runs")


class OneServerPerUserAndMachine(_ServerCase):
    def test_it_is_started_once_and_then_reused(self):
        address = ensure_local_server(self.dir)
        self.assertEqual(address, os.path.join(self.dir, pathCacheClient.SOCKET_NAME))
        self.assertEqual(ensure_local_server(self.dir), address)
        self.assertEqual(len(self.server_pids()), 1)
        set_status([(FILE, True)], address, 5)
        self.assertIs(get_status(FILE, address, 5), True)

    def test_processes_asking_at_once_share_one_server(self):
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        children = [
            ctx.Process(target=_ensure_in_child, args=(self.dir, queue))
            for _ in range(6)
        ]
        for child in children:
            child.start()
        addresses = {queue.get(timeout=60) for _ in children}
        for child in children:
            child.join(timeout=60)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(len(addresses), 1)
        self.assertEqual(len(self.server_pids()), 1, self.log())

    def test_nobody_else_can_enter_its_directory_or_socket(self):
        address = ensure_local_server(self.dir)
        self.assertEqual(stat.S_IMODE(os.stat(self.dir).st_mode), 0o700)
        st = os.lstat(address)
        self.assertTrue(stat.S_ISSOCK(st.st_mode))
        self.assertEqual(stat.S_IMODE(st.st_mode) & 0o077, 0)

    def test_a_directory_others_can_enter_is_refused(self):
        os.mkdir(self.dir, 0o700)
        os.chmod(self.dir, 0o755)
        with self.assertRaises(ServerUnavailable):
            ensure_local_server(self.dir)
        self.assertEqual(self.server_pids(), [])

    def test_a_symlink_in_place_of_the_directory_is_refused(self):
        elsewhere = tempfile.mkdtemp(prefix="fpc", dir="/tmp")
        self.addCleanup(os.rmdir, elsewhere)
        os.chmod(elsewhere, 0o700)
        os.symlink(elsewhere, self.dir)
        with self.assertRaises(ServerUnavailable):
            ensure_local_server(self.dir)

    def test_a_socket_path_too_long_for_the_kernel_is_refused_with_a_reason(self):
        deep = os.path.join(self.dir, "d" * 120)
        os.makedirs(deep, 0o700)
        os.chmod(self.dir, 0o700)
        with self.assertRaises(ServerUnavailable) as caught:
            ensure_local_server(deep)
        self.assertIn(pathCacheClient.LOCAL_SERVER_DIR_ENV, str(caught.exception))


class AServerThatEnds(_ServerCase):
    def test_a_killed_server_is_replaced(self):
        address = ensure_local_server(self.dir)
        (first,) = self.server_pids()
        os.kill(first, signal.SIGKILL)
        self.wait_dead(first)
        # the socket file of the dead server is still there and refuses connections
        self.assertTrue(os.path.exists(address))
        self.assertFalse(pathCacheClient.server_answers(address))
        self.assertEqual(ensure_local_server(self.dir), address)
        pids = self.server_pids()
        self.assertEqual(len(pids), 2)
        self.assertTrue(self.alive(pids[1]))

    def test_its_state_survives_a_terminated_session(self):
        """logind ends a session's processes with SIGTERM: the server writes its state and
        removes its socket, and the next one starts from that state."""
        address = ensure_local_server(self.dir)
        set_status([(MARKER, True), (FILE, True), (ABSENT_DIR, False)], address, 5)
        (first,) = self.server_pids()
        os.kill(first, signal.SIGTERM)
        self.wait_dead(first)
        self.assertFalse(os.path.exists(address))
        ensure_local_server(self.dir)
        self.assertIs(get_status(FILE, address, 5), True)
        self.assertIs(get_status(MARKER, address, 5), True)
        self.assertIs(get_status(ABSENT_DIR + "/file.root", address, 5), False)

    def test_an_idle_server_exits(self):
        os.mkdir(self.dir, 0o700)
        cfg = os.path.join(self.dir, "cfg.yaml")
        with open(cfg, "w") as f:
            f.write(
                "max_connections: 4\ncache_write_interval: 60\nidle_timeout: 1\n"
                "validity_periods:\n  default: 3600\n"
            )
        address = os.path.join(self.dir, pathCacheClient.SOCKET_NAME)
        state = os.path.join(self.dir, "state.json")
        process = subprocess.Popen(
            [
                sys.executable,
                pathCacheServer.__file__,
                "--cfg",
                cfg,
                "--socket",
                address,
                "--cache-file",
                state,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 20
        while not pathCacheClient.server_answers(address, timeout=1):
            self.assertIsNone(process.poll())
            self.assertLess(time.time(), deadline)
            time.sleep(0.1)
        set_status([(FILE, True)], address, 5)
        self.assertEqual(process.wait(timeout=180), 0)
        self.assertFalse(os.path.exists(address))
        self.assertTrue(os.path.exists(state))

    def test_a_second_server_on_the_same_socket_leaves_the_first_alone(self):
        address = ensure_local_server(self.dir)
        result = subprocess.run(
            [sys.executable, pathCacheServer.__file__, "--socket", address],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(pathCacheClient.server_answers(address))
        self.assertEqual(len(self.server_pids()), 1)


class AProcessWhoseServerGoes(_ServerCase):
    """RemotePathCache, the client every law process on the machine uses."""

    def client(self):
        return RemotePathCache(
            ensure_local_server(self.dir), local_cache_validity_period=600
        )

    def test_the_server_is_started_again_and_nothing_is_lost(self):
        cache = self.client()
        (first,) = self.server_pids()
        os.kill(first, signal.SIGKILL)
        self.wait_dead(first)
        with mock.patch("sys.stderr") as stderr:
            cache.set(FILE, True)
            # another process of the machine finds it
            self.assertIs(self.client().get(FILE)[0], True)
        self.assertFalse(stderr.write.called, "a restart that worked was reported")
        self.assertEqual(len(self.server_pids()), 2)

    def test_it_works_alone_while_no_server_can_run_and_says_so_once(self):
        cache = self.client()
        (first,) = self.server_pids()
        os.kill(first, signal.SIGKILL)
        self.wait_dead(first)
        os.chmod(self.dir, 0o755)
        with mock.patch("sys.stderr") as stderr:
            cache.set(FILE, True)
            self.assertIs(cache.get(FILE)[0], True)  # from its own cache
            self.assertIsNone(cache.get(MARKER)[0])
            self.assertEqual(cache.get_many([MARKER]), {MARKER: None})
        reports = [c for c in stderr.write.call_args_list if "cannot be used" in str(c)]
        self.assertEqual(len(reports), 1)
        # and asks again once the retry interval is over
        os.chmod(self.dir, 0o700)
        cache._down_since -= RemotePathCache.retry_seconds + 1
        with mock.patch("sys.stderr"):
            cache.set(MARKER, True)
        self.assertIsNone(cache._down_since)
        self.assertIs(self.client().get(MARKER)[0], True)


class AnAbsentLivesShorter(unittest.TestCase):
    """No job publishes what it writes, so an "absent" on the machine's server -- a negative,
    or a listing marker, which implies one for every file it does not list -- is kept for
    `negative_validity` only; an "exists" for its prefix's validity."""

    def cache(self):
        db = pathCacheServer.ValidityPeriodDB({"default": 86400}, negative_validity=600)
        logger = types.SimpleNamespace(log=lambda *args, **kwargs: None, verbose=False)
        return pathCacheServer.PathCache(db, logger)

    def test_validities(self):
        cache = self.cache()
        now = time.time()
        for path, exists in ((FILE, True), (MARKER, True), (ABSENT_DIR, False)):
            cache.set(path, exists)
        expiry = {
            path: entry["expiration_time"] - now for path, entry in cache.cache.items()
        }
        std = pathCacheServer.standardize_path_name
        self.assertAlmostEqual(expiry[std(FILE)], 86400, delta=2)
        self.assertAlmostEqual(expiry[std(MARKER)], 600, delta=2)
        self.assertAlmostEqual(expiry[std(ABSENT_DIR)], 600, delta=2)

    def test_the_shipped_configuration_sets_it(self):
        import yaml

        cfg_path = os.path.join(
            os.path.dirname(pathCacheServer.__file__), "pathCacheServerCfg.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        db = pathCacheServer.ValidityPeriodDB(
            cfg["validity_periods"], cfg.get("negative_validity")
        )
        self.assertEqual(db.get_validity(FILE, exists=True), 24 * 3600)
        self.assertEqual(db.get_validity(FILE, exists=False), 600)
        self.assertEqual(db.get_validity(MARKER, exists=True), 600)


class TheFileSystemsOfAProcess(unittest.TestCase):
    """Setup._create_fs_instance: a job works from its snapshot, never a server; any other
    process uses its machine's server."""

    def create(self, env):
        from FLAF.Common import Setup

        setup = types.SimpleNamespace(
            global_params={"WLCGFileSystem": {"localPathCacheValidity": 600}}
        )
        with mock.patch.dict(os.environ, env), mock.patch.object(
            law_gfal, "get_voms_proxy_info", lambda: {"path": None}
        ), mock.patch.object(Setup, "_local_path_cache_server", []), mock.patch.object(
            Setup, "ensure_local_server", return_value="/tmp/x/server_v1.sock"
        ) as ensure, mock.patch.object(
            law_gfal, "apply_shipped_path_cache"
        ) as apply:
            if not env.get("LAW_JOB_HOME"):
                os.environ.pop("LAW_JOB_HOME", None)
            fs = Setup.Setup._create_fs_instance(
                setup, "davs://storage.example:1094/store"
            )
        return fs.file_interface.path_cache, ensure, apply

    def test_a_job_on_any_backend_loads_its_snapshot(self):
        cache, ensure, apply = self.create({"LAW_JOB_HOME": "/srv/job"})
        self.assertIsInstance(cache, law_gfal.PathCache)
        self.assertEqual(cache.validity_period, 86400)
        self.assertFalse(ensure.called)
        self.assertTrue(apply.called)

    def test_a_driver_uses_the_server_of_its_machine(self):
        cache, ensure, apply = self.create({})
        self.assertIsInstance(cache, RemotePathCache)
        self.assertEqual(cache.address, "/tmp/x/server_v1.sock")
        self.assertEqual(cache.local_cache.validity_period, 600)
        self.assertTrue(ensure.called)
        self.assertFalse(apply.called)


if __name__ == "__main__":
    unittest.main()
