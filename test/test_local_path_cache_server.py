"""The path-cache server of one user on one machine (RunKit/pathCacheServer.py), started on
demand by pathCacheClient.ensure_local_server.

Real server processes on real Unix sockets: what is under test is process lifecycle -- one
server per user and machine however many processes ask at once, a server that died is
replaced, its state survives a restart, nobody else can reach it -- and that decides in the
kernel, not in Python.
"""

import importlib.util
import json
import multiprocessing
import os
import re
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import law_gfal, pathCacheClient, pathCacheServer  # noqa: E402

# Setup imports ROOT and Corrections at module level, which the unit-test runner does not have
# and nothing tested here uses: stand-ins for these imports alone (as in test_era_jobs).
_HEAVY_MODULES = ["ROOT", "numpy", "awkward", "uproot", "Corrections"]


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_replaced = {
    name: sys.modules.get(name) for name in _HEAVY_MODULES if not _importable(name)
}
sys.modules.update({name: mock.MagicMock() for name in _replaced})
try:
    from FLAF.Common import Setup  # noqa: E402
finally:
    for _name, _module in _replaced.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module
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


class NobodyElsesServerIsBelieved(_ServerCase):
    """A server that another user runs -- e.g. one listening in /tmp/flaf-path-cache-<uid>
    created by someone else before this user's first run on the node -- would answer every
    question of this user's processes: "exists" for files never written, which jobs then
    skip and snapshots ship."""

    def poisoner(self, directory):
        """A listener answering "exists" to everything, as a foreign server could."""
        import threading

        os.makedirs(directory, exist_ok=True)
        os.chmod(directory, 0o777)
        address = os.path.join(directory, pathCacheClient.SOCKET_NAME)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(address)
        listener.listen(8)

        def answer():
            while True:
                try:
                    conn, _ = listener.accept()
                except OSError:
                    return
                with conn:
                    for _ in conn.makefile("rb"):
                        conn.sendall(b"1\n")

        threading.Thread(target=answer, daemon=True).start()
        return address

    def test_a_process_does_not_use_a_directory_others_can_enter(self):
        address = self.poisoner(self.dir)
        with mock.patch.dict(
            os.environ, {pathCacheClient.LOCAL_SERVER_DIR_ENV: self.dir}
        ), mock.patch.object(Setup, "_local_path_cache_server", []), mock.patch(
            "sys.stderr"
        ):
            self.assertIsNone(Setup.local_path_cache_server())
        with self.assertRaises(ServerUnavailable):
            ensure_local_server(self.dir)
        self.assertTrue(os.path.exists(address))

    def test_a_directory_that_cannot_be_made_leaves_the_process_alone(self):
        """No private directory (a parent missing or not writable, a full /tmp): the process
        works from its own cache and says so once, as when the server cannot run."""
        missing = os.path.join(self.dir, "no", "such", "parent", "cache")
        with mock.patch.dict(
            os.environ, {pathCacheClient.LOCAL_SERVER_DIR_ENV: missing}
        ), mock.patch.object(Setup, "_local_path_cache_server", []), mock.patch(
            "sys.stderr"
        ) as stderr:
            self.assertIsNone(Setup.local_path_cache_server())
            self.assertIsNone(Setup.local_path_cache_server())
        warnings = [c for c in stderr.write.call_args_list if "not used" in str(c)]
        self.assertEqual(len(warnings), 1)

    def test_a_uid_above_two_to_the_31_is_read_unsigned(self):
        """struct ucred holds unsigned ids: read signed, a user whose uid has the top bit set
        would never be believed by their own processes."""
        uid = 2**31 + 5
        creds = struct.pack("iII", 1234, uid, uid)
        sock = types.SimpleNamespace(getsockopt=lambda *args: creds)
        with mock.patch.object(pathCacheClient.os, "getuid", return_value=uid):
            pathCacheClient._check_peer(sock, "server.sock")
        with mock.patch.object(
            pathCacheClient.os, "getuid", return_value=uid + 1
        ), self.assertRaises(ServerUnavailable):
            pathCacheClient._check_peer(sock, "server.sock")

    def test_a_server_of_another_user_is_not_believed(self):
        address = self.poisoner(self.dir)
        real_uid = os.getuid()
        with mock.patch.object(
            pathCacheClient.os, "getuid", return_value=real_uid + 1
        ), self.assertRaises(ServerUnavailable) as caught:
            get_status(FILE, address, 5)
        self.assertIn("another user", str(caught.exception))
        # and the cache of a process falls back to its own answers
        cache = RemotePathCache(address, local_cache_validity_period=600)
        with mock.patch.object(
            pathCacheClient.os, "getuid", return_value=real_uid + 1
        ), mock.patch.object(
            law_gfal,
            "ensure_local_server",
            side_effect=ServerUnavailable("not private"),
        ), mock.patch(
            "sys.stderr"
        ):
            self.assertIsNone(cache.get(FILE)[0])


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

    def test_a_process_stopped_while_starting_the_server_does_not_stop_the_others(self):
        """One process starts the server under a lock; one stopped then (Ctrl-Z) must not
        keep every other process of the user waiting."""
        import fcntl

        os.mkdir(self.dir, 0o700)
        with open(os.path.join(self.dir, "server.lock"), "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            start = time.time()
            with self.assertRaises(ServerUnavailable) as caught:
                ensure_local_server(self.dir, startup_timeout=1.0)
            self.assertLess(time.time() - start, 10)
        self.assertIn("starting", str(caught.exception))

    def test_a_server_started_meanwhile_ends_the_wait_for_the_lock(self):
        """The process holding the start lock may have started the server and been stopped
        since: the others take the server as soon as it answers, not after the timeout.
        """
        import fcntl

        os.mkdir(self.dir, 0o700)
        address = os.path.join(self.dir, pathCacheClient.SOCKET_NAME)
        with open(os.path.join(self.dir, "server.lock"), "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            log = open(os.path.join(self.dir, "server.log"), "a")
            self.addCleanup(log.close)
            starter = threading.Timer(
                1.0,
                subprocess.Popen,
                args=([sys.executable, pathCacheServer.__file__, "--socket", address],),
                kwargs={"stdout": log, "stderr": subprocess.STDOUT},
            )
            starter.start()
            start = time.time()
            self.assertEqual(
                ensure_local_server(self.dir, startup_timeout=20.0), address
            )
            self.assertLess(time.time() - start, 10)

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


class ManyClientsAtOnce(_ServerCase):
    def test_forty_clients_pipelining_a_thousand_requests_each(self):
        """Every law process of a machine publishes up to 8 listings at once: the server
        must keep up with dozens of pipelined clients (it once read a request one byte per
        system call, and clients timed out from about 24 of them on)."""
        import threading

        address = ensure_local_server(self.dir)
        errors = []

        def client(n):
            paths = [f"{FILE}.{n}.{i}" for i in range(1000)]
            try:
                set_status([(p, True) for p in paths], address, 5)
                answers = pathCacheClient.get_status_many(paths, address, 5)
                if set(answers.values()) != {True}:
                    errors.append(f"client {n}: wrong answers")
            except ServerUnavailable as e:
                errors.append(f"client {n}: {e}")

        threads = [threading.Thread(target=client, args=(n,)) for n in range(40)]
        start = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertLess(time.time() - start, 60)


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

    def test_the_end_of_a_session_writes_the_state(self):
        """logind sends SIGTERM and then SIGHUP to a session's processes, the server's
        logger included: the second signal must not cut the shutdown short, nor a signal
        taken by another thread leave the server waiting. Several times: which thread the
        kernel hands a signal to varies."""
        for n in range(1, 9):
            address = ensure_local_server(self.dir)
            set_status([(f"{FILE}.{n}.{i}", True) for i in range(2000)], address, 5)
            server = self.server_pids()[-1]
            with open(f"/proc/{server}/task/{server}/children") as f:
                processes = [server] + [int(pid) for pid in f.read().split()]
            for signum in (signal.SIGTERM, signal.SIGHUP):
                for pid in processes:
                    os.kill(pid, signum)
            self.wait_dead(server, timeout=15)
            self.assertFalse(os.path.exists(address))
            with open(os.path.join(self.dir, pathCacheClient.STATE_NAME)) as f:
                self.assertEqual(len(json.load(f)["cache"]), 2000 * n)

    def test_a_process_starting_while_the_server_shuts_down_gets_the_next_one(self):
        """A server shutting down has removed its socket but still writes its state: a law
        process starting then must get the next server, not none for its whole life -- nor
        wait for a client that keeps its connection open."""
        address = ensure_local_server(self.dir)
        (first,) = self.server_pids()
        for n in range(200):
            set_status([(f"{FILE}.{n}.{i}", True) for i in range(1000)], address, 5)
        idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(idle.close)
        idle.connect(address)
        # served once, then kept open: its handler waits for the next request
        idle.sendall(b"GET /\n")
        self.assertEqual(idle.makefile("rb").readline().strip(), b"u")
        os.kill(first, signal.SIGTERM)
        deadline = time.time() + 10
        while os.path.exists(address):
            self.assertLess(time.time(), deadline)
            time.sleep(0.01)
        self.assertTrue(self.alive(first), "the state write was over too soon to test")
        start = time.time()
        self.assertEqual(ensure_local_server(self.dir), address)
        self.assertLess(time.time() - start, 25)
        self.wait_dead(first)
        self.assertEqual(len(self.server_pids()), 2)
        self.assertIs(get_status(f"{FILE}.199.999", address, 5), True)

    def test_only_the_main_thread_takes_the_exit_signals(self):
        """The main thread waits in accept() and handles a signal at once; another thread
        that took one would leave it asleep there until accept() times out."""
        address = ensure_local_server(self.dir)
        (server,) = self.server_pids()
        set_status([(f"{FILE}.{i}", True) for i in range(2000)], address, 5)
        pathCacheClient.get_status_many(
            [f"{FILE}.{i}" for i in range(2000)], address, 5
        )
        blocked = {}
        for tid in os.listdir(f"/proc/{server}/task"):
            with open(f"/proc/{server}/task/{tid}/status") as f:
                fields = dict(line.split(":", 1) for line in f if ":" in line)
            blocked[int(tid)] = int(fields["SigBlk"].strip(), 16)
        exit_mask = (1 << (signal.SIGTERM - 1)) | (1 << (signal.SIGHUP - 1))
        others = {tid: mask for tid, mask in blocked.items() if tid != server}
        self.assertTrue(others, "the server runs no other thread to check")
        for tid, mask in others.items():
            self.assertEqual(mask & exit_mask, exit_mask, f"thread {tid}")

    def test_a_server_whose_logger_died_still_exits(self):
        address = ensure_local_server(self.dir)
        (server,) = self.server_pids()
        with open(f"/proc/{server}/task/{server}/children") as f:
            (logger,) = [int(pid) for pid in f.read().split()]
        os.kill(logger, signal.SIGKILL)
        # far more log lines than the pipe to the dead logger holds
        try:
            pathCacheClient._pipelined(address, 5, ["BOGUS " + "x" * 200 + "\n"] * 1500)
        except ServerUnavailable:
            pass
        os.kill(server, signal.SIGTERM)
        self.wait_dead(server, timeout=20)

    def test_a_relative_directory_works(self):
        parent = os.path.dirname(self.dir)
        cwd = os.getcwd()
        os.chdir(parent)
        try:
            with mock.patch.dict(
                os.environ, {pathCacheClient.LOCAL_SERVER_DIR_ENV: "cache"}
            ):
                address = ensure_local_server()
        finally:
            os.chdir(cwd)
        self.assertEqual(address, os.path.join(self.dir, pathCacheClient.SOCKET_NAME))
        self.assertTrue(pathCacheClient.server_answers(address))

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
        """Also one that missed the first while it was starting: a server holds a lock for
        its whole life, so none ever replaces the socket of a live one."""
        address = ensure_local_server(self.dir)
        inode = os.lstat(address).st_ino
        result = subprocess.run(
            [sys.executable, pathCacheServer.__file__, "--socket", address],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(pathCacheClient.server_answers(address))
        self.assertEqual(os.lstat(address).st_ino, inode)
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


class TheStateFile(unittest.TestCase):
    def test_a_later_copy_is_never_replaced_by_an_earlier_one(self):
        """The periodic write runs in a thread: one that took longer must not replace the
        copy written after it (the last one, at shutdown)."""
        import threading

        directory = tempfile.mkdtemp(prefix="fpc", dir="/tmp")
        self.addCleanup(subprocess.run, ["rm", "-rf", directory])
        state = os.path.join(directory, "cache.json")
        db = pathCacheServer.ValidityPeriodDB({"default": 3600})
        logger = types.SimpleNamespace(log=lambda *args, **kwargs: None, verbose=False)
        cache = pathCacheServer.PathCache(db, logger, cache_file=state)
        cache.set(FILE, True)
        real_dump = json.dump
        slow = threading.Event()

        def dump(data, f, *args, **kwargs):
            if len(data["cache"]) == 1:
                slow.wait(10)
            return real_dump(data, f, *args, **kwargs)

        with mock.patch.object(pathCacheServer.json, "dump", side_effect=dump):
            early = threading.Thread(target=cache._write, args=(True,))
            early.start()
            time.sleep(0.2)
            cache.set(MARKER, True)
            cache._write(need_lock=True)
            slow.set()
            early.join()
        with open(state) as f:
            self.assertEqual(len(json.load(f)["cache"]), 2)


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
        setup = types.SimpleNamespace(
            global_params={"WLCGFileSystem": {"localPathCacheValidity": 600}}
        )
        with mock.patch.dict(os.environ, env), mock.patch.object(
            law_gfal, "get_voms_proxy_info", lambda: {"path": None}
        ), mock.patch.object(Setup, "_local_path_cache_server", []), mock.patch.object(
            Setup, "checked_server_address", return_value="/tmp/x/server_v1.sock"
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

    def test_the_renamed_job_validity_key_is_refused(self):
        setup = types.SimpleNamespace(
            global_params={"WLCGFileSystem": {"crabLocalPathCacheValidity": 1000}}
        )
        with self.assertRaises(RuntimeError) as caught:
            Setup.Setup._create_fs_instance(setup, "davs://storage.example:1094/store")
        self.assertIn("jobPathCacheValidity", str(caught.exception))

    def test_a_driver_uses_the_server_of_its_machine(self):
        cache, ensure, apply = self.create({})
        self.assertIsInstance(cache, RemotePathCache)
        self.assertEqual(cache.address, "/tmp/x/server_v1.sock")
        self.assertEqual(cache.local_cache.validity_period, 600)
        self.assertTrue(ensure.called)
        self.assertFalse(apply.called)


if __name__ == "__main__":
    unittest.main()
