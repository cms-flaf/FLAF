#!/usr/bin/env python3
""" "Could not list it" is not "it is not there".

`GFALFileInterface.exists()` answers by listing the parent directory, and the listing used
to return no entries for *any* failure -- a timeout, an SSL error, a missing credential --
which `exists()` reported as "the file does not exist", cached as a negative, published to
the cache server and propagated up the tree by marking the ancestors absent. In DSProd one
such blink per job turned into 1400 failed CRAB jobs. A listing that fails is now retried
and then raised, or with `silent` answered without caching anything; only gfal reporting
ENOENT means absent.
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import grid_tools, law_gfal, run_tools
from FLAF.RunKit.grid_tools import GfalError, gfal_ls_checked, is_absent_error
from FLAF.RunKit.law_gfal import GFALFileInterface, RemotePathCache

# stderr and exit code of `gfal-ls --long --all --time-style long-iso <path>` (gfal2 2.23.5,
# lxplus, 2026-10-08), unless marked otherwise.
ABSENT_XROOTD = (
    "gfal-ls error: 2 (No such file or directory) - Failed to stat file "
    "(No such file or directory)\n",
    2,
)
# captured by DSProd against CERNBox; no valid proxy was available to repeat it
ABSENT_DAVS = (
    "gfal-ls error: 2 (No such file or directory) - Result HTTP 404 : File not found "
    "after 1 attempts\n",
    2,
)
# X509_USER_PROXY pointing to a file that does not exist: davix prints the errno text of the
# failed open() before failing on the request itself, for any path.
NO_PROXY_FILE_DAVS = (
    "(Davix::OpenSSL) Error: impossible to open /nonexistent_proxy:  : "
    "error:80000002:system library::No such file or directory\n"
    "(Davix::OpenSSL) Error: impossible to open /nonexistent_proxy:  : "
    "error:80000002:system library::No such file or directory\n"
    "gfal-ls error: 1 (Operation not permitted) - HTTP 403 : Permission refused \n",
    1,
)
NO_CREDENTIAL_XROOTD = (
    "TLS: Unable to use cert+key file /nonexistent_proxy; does not exist.\n"
    "gfal-ls error: 13 (Permission denied) - Failed to stat file (Permission denied)\n",
    13,
)
EXPIRED_PROXY_DAVS = (
    "gfal-ls error: 112 (Host is down) - Result (Neon): Could not read status line: "
    "SSL error: ssl/tls alert certificate expired after 1 attempts\n",
    112,
)
REFUSED_DAVS = (
    "gfal-ls error: 112 (Host is down) - Result Could not connect to server after 1 "
    "attempts\n",
    112,
)
UNRESOLVED_DAVS = (
    "gfal-ls error: 113 (No route to host) - Result Domain name resolution failed after "
    "1 attempts\n",
    113,
)
UNRESOLVED_XROOTD = (
    "gfal-ls error: 113 (No route to host) - Failed to stat file (No route to host)\n",
    113,
)
# killed by a timeout before writing anything (xrootd to a closed port keeps reconnecting)
KILLED = ("", -9)
# `gfal-ls --timeout 10` against an xrootd endpoint that never answers (lxplus, 2026-10-08)
TIMED_OUT_XROOTD = ("Command timed out after 10 seconds!\n", 110)

NOT_ABSENT = {
    "no proxy file, davs": NO_PROXY_FILE_DAVS,
    "no credential, xrootd": NO_CREDENTIAL_XROOTD,
    "expired proxy, davs": EXPIRED_PROXY_DAVS,
    "connection refused, davs": REFUSED_DAVS,
    "unresolved host, davs": UNRESOLVED_DAVS,
    "unresolved host, xrootd": UNRESOLVED_XROOTD,
    "killed without output": KILLED,
    "gfal's own timeout, xrootd": TIMED_OUT_XROOTD,
}


def replayed_ls_error(case, path="davs://host:443/store/x"):
    """The GfalError that gfal_ls raises for `case`, produced by the real ps_call: a process
    writes the captured stderr and exits with the captured code. The stderr is read from a
    file: as an argument it would also show up in the command line that the error quotes.
    """
    stderr, return_code = case

    def call(cmd, **kwargs):
        script = '/bin/cat "$1" >&2; exit "$2"'
        if return_code < 0:
            script = f'/bin/cat "$1" >&2; kill {-return_code} $$'
        return run_tools.ps_call(
            ["/bin/sh", "-c", script, "sh", stderr_file, str(return_code)], **kwargs
        )

    with tempfile.TemporaryDirectory() as tmp:
        stderr_file = os.path.join(tmp, "stderr")
        with open(stderr_file, "w") as f:
            f.write(stderr)
        with mock.patch.object(grid_tools, "ps_call", call):
            try:
                grid_tools.gfal_ls(path, voms_token="tok", catch_stderr=True, verbose=0)
            except GfalError as e:
                return e
    raise AssertionError("the replayed listing did not fail")


class Entry:
    def __init__(self, name, size=0, is_dir=False):
        self.name = name
        self.size = size
        self.is_dir = is_dir


class TheErrorText(unittest.TestCase):
    """Without stderr in the exception the two cases cannot be told apart at all."""

    def test_a_failed_call_carries_its_stderr(self):
        with self.assertRaises(run_tools.PsCallError) as caught:
            run_tools.ps_call(
                ["/bin/sh", "-c", 'printf "%s-%s" the reason >&2; exit 3'],
                catch_stderr=True,
            )
        self.assertIn("the-reason", str(caught.exception))
        self.assertEqual(caught.exception.return_code, 3)

    def test_the_stderr_is_bounded_to_its_tail(self):
        script = "head -c 50000 /dev/zero | tr '\\0' x >&2; echo THE-END >&2; exit 1"
        with self.assertRaises(run_tools.PsCallError) as caught:
            run_tools.ps_call(["/bin/sh", "-c", script], catch_stderr=True)
        message = caught.exception.message
        self.assertTrue(message.endswith("THE-END"), message[-40:])
        self.assertLessEqual(len(message), 2003)

    def test_a_stderr_merged_into_the_output_is_used(self):
        with self.assertRaises(run_tools.PsCallError) as caught:
            run_tools.ps_call(
                ["/bin/sh", "-c", 'printf "%s-%s" merged reason >&2; exit 1'],
                catch_stdout=True,
                catch_stderr=True,
                print_output=True,
            )
        self.assertIn("merged-reason", str(caught.exception))

    def test_a_listing_error_carries_the_gfal_reason(self):
        self.assertIn("Failed to stat file", str(replayed_ls_error(ABSENT_XROOTD)))


class Classification(unittest.TestCase):
    def test_gfal_reporting_enoent_is_absent(self):
        self.assertTrue(is_absent_error(replayed_ls_error(ABSENT_XROOTD)))
        self.assertTrue(is_absent_error(replayed_ls_error(ABSENT_DAVS)))

    def test_every_other_failure_is_not_absent(self):
        # The first case is why the errno decides and not the text: with the proxy file
        # missing, davix writes "No such file or directory" for every path, and a text
        # match would condemn every product of a production as gone.
        for name, case in NOT_ABSENT.items():
            with self.subTest(name):
                self.assertFalse(is_absent_error(replayed_ls_error(case)))

    def test_the_reason_gfal_gives_last_decides(self):
        # a directory name is not evidence, whatever it looks like
        path = "davs://host:443/store/gfal-ls error: 2 (No such file or directory)"
        self.assertFalse(is_absent_error(replayed_ls_error(NO_PROXY_FILE_DAVS, path)))

    def test_a_listing_that_could_not_be_parsed_is_not_absent(self):
        self.assertFalse(is_absent_error(GfalError('gfal_ls: unable to parse "?"')))

    def test_listing_dates_are_utc(self):
        # gfal-ls prints the time of an entry in the client's timezone
        self.assertEqual(grid_tools.gfal_env("tok")["TZ"], "UTC")


class CheckedListing(unittest.TestCase):
    def test_an_absent_path_is_reported_at_once(self):
        absent = replayed_ls_error(ABSENT_DAVS)
        with mock.patch.object(grid_tools, "gfal_ls", side_effect=absent) as ls:
            self.assertIsNone(gfal_ls_checked("davs://host/x", voms_token="t"))
        self.assertEqual(ls.call_count, 1, "an absent path is not worth retrying")

    def test_a_blink_is_retried_and_the_listing_returned(self):
        down = replayed_ls_error(REFUSED_DAVS)
        ls = mock.Mock(side_effect=[down, [Entry("a.root")]])
        with mock.patch.object(grid_tools, "gfal_ls", ls):
            entries = gfal_ls_checked("davs://host/x", voms_token="t", delay=0)
        self.assertEqual([e.name for e in entries], ["a.root"])
        self.assertEqual(ls.call_count, 2)

    def test_an_endpoint_that_stays_down_raises(self):
        ls = mock.Mock(side_effect=replayed_ls_error(NO_PROXY_FILE_DAVS))
        with mock.patch.object(grid_tools, "gfal_ls", ls):
            with self.assertRaises(GfalError):
                gfal_ls_checked("davs://host/x", voms_token="t", attempts=3, delay=0)
        self.assertEqual(ls.call_count, 3, "every attempt is spent before giving up")

    def test_no_attempt_is_no_answer(self):
        with mock.patch.object(grid_tools, "gfal_ls") as ls:
            with self.assertRaises(GfalError):
                gfal_ls_checked("davs://host/x", voms_token="t", attempts=0)
        ls.assert_not_called()


class ListingTimeout(unittest.TestCase):
    """gfal's own limit is 1800 s per call, and a checked listing makes up to three: an
    endpoint that hangs would hold one exists() for an hour and a half."""

    @staticmethod
    def command(fn, *args, **kwargs):
        with mock.patch.object(
            grid_tools, "ps_call", return_value=(0, [], None)
        ) as call:
            fn(*args, **kwargs)
        (cmd,), _ = call.call_args
        return cmd

    def test_a_checked_listing_is_bounded_by_default(self):
        cmd = self.command(gfal_ls_checked, "davs://host/x", voms_token="t")
        self.assertEqual(cmd[0], "gfal-ls")
        self.assertEqual(cmd[cmd.index("--timeout") + 1], "300")
        self.assertEqual(cmd[-1], "davs://host/x")

    def test_the_bound_can_be_chosen(self):
        cmd = self.command(gfal_ls_checked, "davs://host/x", voms_token="t", timeout=30)
        self.assertEqual(cmd[cmd.index("--timeout") + 1], "30")

    def test_a_plain_listing_keeps_the_gfal_default(self):
        cmd = self.command(grid_tools.gfal_ls, "davs://host/x", voms_token="t")
        self.assertNotIn("--timeout", cmd)
        self.assertNotIn("-t", cmd)

    @unittest.skipUnless(shutil.which("gfal-ls"), "gfal-ls is not installed")
    def test_the_real_gfal_ls_accepts_the_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.root"), "w").close()
            entries = gfal_ls_checked("file://" + tmp, voms_token="/none", attempts=1)
        self.assertEqual([e.name for e in entries], ["a.root"])


BASE = "davs://server:1234/store/test"
UNREACHABLE = GfalError("gfal-ls error: 112 (Host is down) - Could not connect")


class FakeCacheServer:
    """In-memory stand-in for pathCacheServer, shared by RemotePathCache clients."""

    def __init__(self):
        self.entries = {}

    def set_status(self, entries, *args, **kwargs):
        for path, exists in entries:
            self.entries[path] = exists

    def get_status(self, path, *args, **kwargs):
        return self.entries.get(path)

    def get_status_many(self, paths, *args, **kwargs):
        return {path: self.entries.get(path) for path in paths}

    def patch(self):
        return mock.patch.multiple(
            law_gfal,
            set_remote_cache_status=self.set_status,
            get_remote_cache_status=self.get_status,
            get_remote_cache_status_many=self.get_status_many,
        )


def make_interface(path_cache=None):
    with mock.patch.object(law_gfal, "get_voms_proxy_info", lambda: {"path": None}):
        fs = GFALFileInterface(base=[BASE])
    if path_cache is not None:
        fs.path_cache = path_cache
    return fs


def listing(result):
    """Patch the one listing call of law_gfal; `result` is a return value, an exception,
    or a function of the uri."""
    if callable(result):
        return mock.patch.object(law_gfal, "gfal_ls_checked", side_effect=result)
    key = "side_effect" if isinstance(result, Exception) else "return_value"
    return mock.patch.object(law_gfal, "gfal_ls_checked", **{key: result})


class TheInterface(unittest.TestCase):
    def setUp(self):
        self.fs = make_interface()

    def cached(self, path):
        return self.fs.path_cache.get(self.fs.uri(path))[0]

    def test_an_absent_directory_lists_as_empty_when_silent(self):
        with listing(None):
            self.assertEqual(self.fs.listdir("some/dir", silent=True), [])

    def test_an_absent_directory_raises_when_not_silent(self):
        with listing(None):
            with self.assertRaises(GfalError):
                self.fs.listdir("some/dir", silent=False)

    def test_a_failed_listing_raises_when_not_silent(self):
        with listing(UNREACHABLE):
            with self.assertRaises(GfalError):
                self.fs.listdir("some/dir")

    def test_a_failed_silent_listing_is_empty_and_caches_nothing(self):
        with listing(UNREACHABLE):
            self.assertEqual(self.fs.listdir("some/dir", silent=True), [])
        for path in ("some/dir", "some"):
            self.assertIsNone(self.cached(path), path)

    def test_exists_caches_nothing_from_a_failed_listing(self):
        # the incident: exists() said False for a file that was there all along, and the
        # cache kept saying it after the storage was back
        with listing(UNREACHABLE):
            self.assertFalse(self.fs.exists("some/dir/file.txt"))
        for path in ("some/dir/file.txt", "some/dir", "some"):
            self.assertIsNone(self.cached(path), path)
        with listing([Entry("file.txt")]):
            self.assertTrue(self.fs.exists("some/dir/file.txt"))

    def test_an_absence_is_still_cached(self):
        with listing(None):
            self.assertFalse(self.fs.exists("some/dir/file.txt"))
        self.assertIs(self.cached("some/dir/file.txt"), False)
        self.assertIs(self.cached("some/dir"), False)

    def test_a_failed_ancestor_listing_stops_the_climb_without_a_negative(self):
        def ls(uri, **kwargs):
            if uri.endswith("/some/dir"):
                return None
            raise UNREACHABLE

        with listing(ls):
            self.assertFalse(self.fs.exists("some/dir/file.txt"))
        self.assertIs(self.cached("some/dir"), False, "the confirmed absence")
        self.assertIsNone(self.cached("some"), "an ancestor nobody could list")

    def test_an_absent_ancestor_is_still_recorded(self):
        def ls(uri, **kwargs):
            return [Entry("store")] if uri == "davs://server:1234/store" else None

        with listing(ls):
            self.assertFalse(self.fs.exists("some/dir/file.txt"))
        self.assertIs(self.cached("some"), False)


class NothingReachesTheServer(unittest.TestCase):
    """A negative on the cache server hides the path from every client for 24 h."""

    def setUp(self):
        self.server = FakeCacheServer()

    def client(self):
        return make_interface(
            RemotePathCache("host", 1, local_cache_validity_period=600)
        )

    def test_an_outage_publishes_no_negative(self):
        with self.server.patch():
            fs = self.client()
            with listing(UNREACHABLE):
                self.assertFalse(fs.exists("data/file_0.root"))
                self.assertFalse(fs.exists("data/file_0.root"))
                self.assertEqual(fs.listdir("data", silent=True), [])
        self.assertEqual(
            [p for p, exists in self.server.entries.items() if exists is False], []
        )

    def test_after_the_outage_the_file_is_found_by_everyone(self):
        with self.server.patch():
            fs = self.client()
            with listing(UNREACHABLE):
                self.assertFalse(fs.exists("data/file_0.root"))
            with listing([Entry("file_0.root")]):
                self.assertTrue(fs.exists("data/file_0.root"))
            other = self.client()
            with listing(UNREACHABLE):
                self.assertTrue(other.exists("data/file_0.root"))

    def test_a_confirmed_absence_is_still_shared(self):
        with self.server.patch():
            with listing(None):
                self.assertFalse(self.client().exists("data/file_0.root"))
        self.assertIs(self.server.entries.get(os.path.join(BASE, "data")), False)


class AFailedListingIsAskedAgain(unittest.TestCase):
    """A failed listing answers "missing" for the one lookup that took it. Remembered for the
    directory, a brief blip would answer "missing" for every file in it; what is limited per
    directory is the warning, which a persistent outage would otherwise print per file.
    """

    def setUp(self):
        self.fs = make_interface()
        self.stderr = io.StringIO()
        redirect = contextlib.redirect_stderr(self.stderr)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def warnings_for(self, path):
        uri = self.fs.uri(path)
        return [
            line
            for line in self.stderr.getvalue().splitlines()
            if "could not list" in line and f"{uri} " in line
        ]

    def test_a_sibling_after_a_brief_outage_is_answered_from_a_real_listing(self):
        recovered = [Entry("file_0.root"), Entry("file_1.root")]
        with listing(mock.Mock(side_effect=[UNREACHABLE, recovered])) as ls:
            self.assertFalse(self.fs.exists("data/file_0.root"))
            self.assertTrue(self.fs.exists("data/file_1.root"))
            self.assertEqual(ls.call_count, 2, "the sibling was listed again")
            self.assertTrue(self.fs.exists("data/file_0.root"))
            self.assertFalse(self.fs.exists("data/file_2.root"))
            self.assertEqual(ls.call_count, 2, "the listing answers the rest")

    def test_a_persistent_outage_is_reported_once_per_directory_per_interval(self):
        now = [1000.0]
        clock = types.SimpleNamespace(time=lambda: now[0])
        interval = GFALFileInterface.failed_listing_report_seconds
        with mock.patch.object(law_gfal, "time", clock), listing(UNREACHABLE) as ls:
            for i in range(5):
                self.assertFalse(self.fs.exists(f"data/file_{i}.root"))
            for i in range(3):
                self.assertFalse(self.fs.exists(f"other/file_{i}.root"))
            self.assertEqual(ls.call_count, 8, "every lookup asked the storage")
            self.assertEqual(len(self.warnings_for("data")), 1)
            self.assertEqual(len(self.warnings_for("other")), 1)
            self.assertIn("Host is down", self.warnings_for("data")[0])

            now[0] += interval - 1
            self.assertFalse(self.fs.exists("data/file_5.root"))
            self.assertEqual(len(self.warnings_for("data")), 1)

            now[0] += 2
            self.assertFalse(self.fs.exists("data/file_6.root"))
            self.assertEqual(len(self.warnings_for("data")), 2)
            self.assertEqual(len(self.warnings_for("other")), 1)


if __name__ == "__main__":
    unittest.main()
