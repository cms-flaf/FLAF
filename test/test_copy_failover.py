"""Input copies fail over between sources (RunKit/grid_tools.copy_remote_file).

The copy commands are replaced by a stub process that delivers, hangs, stalls, trickles,
fails or corrupts the file, and the time limits are scaled down, so every path runs in
seconds.
"""

import contextlib
import io
import os
import sys
import tempfile
import time
import unittest
import zlib
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import grid_tools

REAL_COPY_COMMAND = grid_tools._copy_command

PAYLOAD = b"nanoAOD payload " * 64
ADLER32 = zlib.adler32(PAYLOAD)

# stub://<mode>/<name>
#   good, fail (exit 3), failonce (exit 4 on its first start, then good), nofile (exit 0, no
#   file), writefail (the file, then exit 5), hang (no data), stall (some data, then nothing),
#   trickle-<n>[-after-<s>] (the file in n parts, 0.3 s apart, after s seconds),
#   mostly-<s> (90% of the file, then the rest in 10 parts over s seconds), mostlyhang (90% of
#   the file, then nothing), corrupt, short, empty, slow-<s> (nothing, s seconds later the file)
STUB = """
import os, sys, time
mode, out, log = sys.argv[1], sys.argv[2], sys.argv[3]
open(log, "a").write(f"{mode} {os.getpid()} {time.time()}\\n")
payload = b"nanoAOD payload " * 64
if mode == "failonce" and not os.path.exists(log + ".failed"):
    open(log + ".failed", "w").close()
    sys.exit(4)
if mode == "fail":
    print("stub: connection refused")
    sys.exit(3)
if mode == "nofile":
    sys.exit(0)
if mode == "writefail":
    open(out, "wb").write(payload)
    sys.exit(5)
if mode == "hang":
    time.sleep(600)
if mode == "stall":
    with open(out, "wb") as f:
        f.write(payload[:100])
        f.flush()
        time.sleep(600)
if mode.startswith("trickle-"):
    parts = mode.split("-")
    n = int(parts[1])
    time.sleep(float(parts[3]) if len(parts) > 3 else 0)
    step = len(payload) // n + 1
    with open(out, "wb") as f:
        for i in range(n):
            f.write(payload[i * step : (i + 1) * step])
            f.flush()
            time.sleep(0.3)
    sys.exit(0)
if mode.startswith("mostly"):
    with open(out, "wb") as f:
        f.write(payload[:921])
        f.flush()
        if mode == "mostlyhang":
            time.sleep(600)
        for i in range(10):
            time.sleep(float(mode[7:]) / 10)
            f.write(payload[921 + 11 * i : 921 + 11 * (i + 1)])
            f.flush()
    sys.exit(0)
if mode.startswith("slow-"):
    time.sleep(float(mode[5:]))
data = {"corrupt": payload[::-1], "short": payload[:10], "empty": b""}.get(mode, payload)
open(out, "wb").write(data)
"""

# Under these limits an attempt is stopped after 10 s and joined after 5 s; tests that are
# about joining lower the expected time with _join_after.
FAST_LIMITS = {
    "COPY_FIRST_TIMEOUT": 10,
    "COPY_MIN_RATE": 1e9,
    "COPY_UNKNOWN_SIZE_TIMEOUT": 10,
    "COPY_TIMEOUT_GROWTH": 2,
    "COPY_MAX_TIMEOUT": 100,
    "COPY_EXPECTED_RATE": 1e9,
    "COPY_EXPECTED_OVERHEAD": 5,
    "COPY_UNKNOWN_SIZE_EXPECTED": 5,
    "COPY_STALL_TIMEOUT": 100,
    "COPY_MAX_TOTAL_TIME": 100,
    "COPY_FEDERATION_PREFIX": "stub://good/federation",
}
REAL_LIMITS = {name: getattr(grid_tools, name) for name in FAST_LIMITS}


def stub_sources(modes):
    return [
        (f"stub://{mode}/{idx}", f"T2_XX_Site{idx}", idx)
        for idx, mode in enumerate(modes)
    ]


def _running(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in "ZX"
    except (FileNotFoundError, ProcessLookupError):  # reaped before or while reading
        return False


class TestCopyFailover(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "nano.root")
        self.started = os.path.join(self.tmp.name, "started.log")
        grid_tools._copy_failed_rses.clear()
        for key, value in FAST_LIMITS.items():
            self._patch(key, value)
        self._patch("_copy_command", self._stub_command)

    def tearDown(self):
        self.tmp.cleanup()

    def _patch(self, name, value, target=grid_tools):
        patch = mock.patch.object(target, name, value)
        patch.start()
        self.addCleanup(patch.stop)

    def _join_after(self, seconds):
        self._patch("COPY_EXPECTED_OVERHEAD", seconds)
        self._patch("COPY_UNKNOWN_SIZE_EXPECTED", seconds)

    def _time_limit(self, seconds):
        self._patch("COPY_FIRST_TIMEOUT", seconds)
        self._patch("COPY_UNKNOWN_SIZE_TIMEOUT", seconds)

    def _stub_command(self, source, output_file, timeout, voms_token):
        if not source.startswith("stub://"):
            return REAL_COPY_COMMAND(source, output_file, timeout, voms_token)
        mode = source[len("stub://") :].split("/")[0]
        if mode == "nocmd":
            return ["/nonexistent/copy-tool", source, output_file], dict(os.environ)
        # A shell that runs the copy as its child, like a wrapper script: stopping an attempt
        # has to stop the child too.
        cmd = ["sh", "-c", '"$@"; exit $?', "sh"]
        cmd += [sys.executable, "-c", STUB, mode, output_file, self.started]
        return cmd, dict(os.environ)

    def _copy(
        self,
        sources,
        size=len(PAYLOAD),
        adler32=ADLER32,
        n_rounds=2,
        round_sleep_interval=0,
        verbose=0,
    ):
        if isinstance(sources[0], str):
            sources = stub_sources(sources)
        t0 = time.monotonic()
        try:
            grid_tools.copy_with_failover(
                sources,
                self.out,
                size=size,
                expected_adler32sum=adler32,
                voms_token="proxy",
                n_rounds=n_rounds,
                round_sleep_interval=round_sleep_interval,
                poll_interval=0.05,
                verbose=verbose,
            )
        finally:
            self.elapsed = time.monotonic() - t0
        self._assert_payload()

    def _assert_payload(self):
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)

    def _started(self):
        """[(mode, pid, start time)] of the stubs in the order they started."""
        if not os.path.exists(self.started):
            return []
        with open(self.started) as f:
            return [line.split() for line in f if line.strip()]

    def _started_modes(self):
        return [mode for mode, _, _ in self._started()]

    def _start_gaps(self, mode):
        times = [float(t) for m, _, t in self._started() if m == mode]
        return [b - a for a, b in zip(times, times[1:])]

    def _assert_no_leftovers(self):
        expected = {"nano.root", "started.log", "started.log.failed"}
        self.assertEqual(sorted(set(os.listdir(self.tmp.name)) - expected), [])
        # Every stub has finished or was stopped. A stopped stub is a grandchild (of the
        # shell) and stays a zombie until its new parent reaps it, which is not waited for.
        pids = [pid for _, pid, _ in self._started()]
        deadline = time.monotonic() + 1
        while any(map(_running, pids)) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual([pid for pid in pids if _running(pid)], [])

    # -- limits and source order ---------------------------------------------------------

    def test_default_limits(self):
        with mock.patch.multiple(grid_tools, **REAL_LIMITS):
            limits = grid_tools.copy_attempt_limits
            # (timeout, expected, stall)
            self.assertEqual(limits(None, 0), (1800, 300, 300))
            self.assertEqual(limits(5_000_000, 0), (300, 60.5, 300))
            self.assertEqual(limits(4_000_000_000, 0), (4000, 460, 300))
            self.assertEqual(limits(4_000_000_000, 1), (12000, 1380, 300))
            self.assertEqual(limits(4_000_000_000, 2), (4 * 3600, 4140, 300))
            # Capped timeout: a copy can still be joined.
            self.assertEqual(limits(4_000_000_000, 3), (4 * 3600, 2 * 3600, 300))
        self.assertEqual(
            (
                REAL_LIMITS["COPY_MAX_TOTAL_TIME"],
                REAL_LIMITS["COPY_FEDERATION_PREFIX"],
                grid_tools.COPY_MAX_PARALLEL,
            ),
            (6 * 3600, "root://cms-xrd-global.cern.ch/", 2),
        )
        # The stall limit never outlasts the timeout (10 s here).
        self.assertEqual(grid_tools.copy_attempt_limits(len(PAYLOAD), 0)[2], 10)

    def test_source_order_spreads_sites_and_demotes_failed_ones(self):
        sources = [
            ("davs://budapest/f", "T2_HU_Budapest", 10),
            ("root://budapest/f", "T2_HU_Budapest", 10),
            ("davs://bari/f", "T2_IT_Bari", 10),
            ("davs://cnaf/f", "T1_IT_CNAF_Disk", 11),
            ("root://cnaf/f", "T1_IT_CNAF_Disk", 11),
            ("root://federation/f", "xrootd federation", float("inf")),
        ]

        def order():
            with mock.patch.object(grid_tools.random, "random", return_value=0.5):
                return [s[0] for s in grid_tools.copy_source_order(sources)]

        self.assertEqual(
            order(),
            [
                "root://budapest/f",
                "davs://bari/f",
                "root://cnaf/f",
                "root://federation/f",
                "davs://budapest/f",
                "davs://cnaf/f",
            ],
        )
        grid_tools._copy_failed_rses["T2_HU_Budapest"] = 1
        self.assertEqual(
            order(),
            [
                "davs://bari/f",
                "root://cnaf/f",
                "root://federation/f",
                "root://budapest/f",
                "davs://cnaf/f",
                "davs://budapest/f",
            ],
        )

    def test_sites_at_the_same_distance_are_drawn_at_random(self):
        sources = [
            ("davs://budapest/f", "T2_HU_Budapest", 10),
            ("davs://bari/f", "T2_IT_Bari", 10),
            ("davs://cern/f", "T2_CH_CERN", 0),
        ]
        orders = {
            tuple(s[1] for s in grid_tools.copy_source_order(sources))
            for _ in range(200)
        }
        self.assertEqual(
            orders,
            {
                ("T2_CH_CERN", "T2_HU_Budapest", "T2_IT_Bari"),
                ("T2_CH_CERN", "T2_IT_Bari", "T2_HU_Budapest"),
            },
        )

    # -- how an attempt ends ---------------------------------------------------------------

    def test_failed_source_falls_over_at_once(self):
        self._copy(["fail", "good"])
        self.assertLess(self.elapsed, 4)
        self.assertEqual(grid_tools._copy_failed_rses, {"T2_XX_Site0": 1})
        self._assert_no_leftovers()

    def test_corrupt_or_truncated_copy_is_rejected(self):
        self._copy(["corrupt", "short", "good"])
        self.assertEqual(self._started_modes(), ["corrupt", "short", "good"])
        self._assert_no_leftovers()

    def test_size_is_checked_without_a_checksum(self):
        self._copy(["short", "good"], adler32=None)
        self.assertEqual(self._started_modes(), ["short", "good"])

    def test_copy_that_succeeds_without_a_file_is_rejected(self):
        self._copy(["nofile", "good"])
        self.assertEqual(self._started_modes(), ["nofile", "good"])
        self._assert_no_leftovers()

    def test_copy_that_writes_the_file_but_fails_is_rejected(self):
        self._copy(["writefail", "good"])
        self.assertEqual(self._started_modes(), ["writefail", "good"])
        self._assert_no_leftovers()

    def test_an_empty_file_is_copied(self):
        grid_tools.copy_with_failover(
            stub_sources(["empty"]),
            self.out,
            size=0,
            expected_adler32sum=zlib.adler32(b""),
            voms_token="proxy",
            poll_interval=0.05,
            verbose=0,
        )
        self.assertEqual(os.path.getsize(self.out), 0)
        self._assert_no_leftovers()

    def test_unsupported_source_counts_as_a_failure(self):
        sources = [
            ("gopher://site/f", "T2_XX_Site0", 0),
            ("stub://good/1", "T2_XX_Site1", 1),
        ]
        self._copy(sources)
        self.assertEqual(grid_tools._copy_failed_rses, {"T2_XX_Site0": 1})
        self.assertEqual(self._started_modes(), ["good"])
        self._assert_no_leftovers()

    def test_copy_program_that_cannot_start_counts_as_a_failure(self):
        self._copy(["nocmd", "good"])
        self.assertEqual(grid_tools._copy_failed_rses, {"T2_XX_Site0": 1})
        self.assertEqual(self._started_modes(), ["good"])
        self._assert_no_leftovers()

    def test_stalled_copy_is_stopped_before_its_timeout(self):
        self._patch("COPY_STALL_TIMEOUT", 1)
        self._copy(["stall", "good"])
        self.assertLess(self.elapsed, 4)
        self.assertEqual(self._started_modes(), ["stall", "good"])
        self._assert_no_leftovers()

    def test_copy_that_never_writes_is_stopped_as_stalled(self):
        self._patch("COPY_STALL_TIMEOUT", 1)
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy(["hang", "stall"], n_rounds=1)
        message = str(ctx.exception)
        self.assertIn("stub://hang/0: no data for 1 s", message)
        self.assertIn("stub://stall/1: no data for 1 s", message)
        self.assertLess(self.elapsed, 5)
        self._assert_no_leftovers()

    def test_trickling_copy_is_not_taken_for_a_stall(self):
        self._patch("COPY_STALL_TIMEOUT", 2)
        self._copy(["trickle-8", "good"])  # 2.4 s in total, a new part every 0.3 s
        self.assertGreater(self.elapsed, 2)
        self.assertEqual(self._started_modes(), ["trickle-8"])
        self._assert_no_leftovers()

    def test_trickling_copy_is_stopped_at_its_timeout(self):
        self._patch("COPY_STALL_TIMEOUT", 2)
        self._time_limit(3)
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy(["trickle-20"], n_rounds=1)  # 6 s in total
        self.assertIn("stub://trickle-20/0: no result after 3 s", str(ctx.exception))
        self.assertLess(self.elapsed, 5)
        self._assert_no_leftovers()

    def test_limits_of_a_file_of_unknown_size(self):
        self._patch("COPY_UNKNOWN_SIZE_TIMEOUT", 1)
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy(["hang"], size=None, n_rounds=1)
        self.assertIn("stub://hang/0: no result after 1 s", str(ctx.exception))
        self._assert_no_leftovers()

    def test_leftovers_of_a_killed_job_are_overwritten(self):
        for path in (self.out + ".part0", self.out + ".part0.log"):
            with open(path, "wb") as f:
                f.write(b"left by a killed job" * 100)
        self._copy(["good"])
        self._assert_no_leftovers()

    def test_a_shrinking_file_is_not_taken_for_a_stall(self):
        # The copy truncates a larger file left at its path, then writes less than it held.
        self._patch("COPY_STALL_TIMEOUT", 2)
        with open(self.out + ".part0", "wb") as f:
            f.write(b"left by a killed job" * 250)
        self._copy(["trickle-8-after-0.5"])
        self.assertEqual(self._started_modes(), ["trickle-8-after-0.5"])
        self._assert_no_leftovers()

    def test_a_file_removed_while_its_size_is_read(self):
        part_file = self.out + ".part0"
        real_getsize = os.path.getsize
        raised = []

        def getsize(path):
            if path == part_file and not raised:
                raised.append(path)
                raise FileNotFoundError(path)
            return real_getsize(path)

        self._patch("getsize", getsize, target=os.path)
        self._copy(["trickle-4"])
        self.assertEqual(raised, [part_file])

    def test_an_interrupted_copy_stops_its_attempts(self):
        # e.g. the job is killed while one attempt has finished and another is running
        self._join_after(0.5)
        self._patch("check_download", mock.Mock(side_effect=KeyboardInterrupt))
        with self.assertRaises(KeyboardInterrupt):
            self._copy(["hang", "slow-1"])
        self.assertEqual(self._started_modes(), ["hang", "slow-1"])
        self.assertFalse(os.path.exists(self.out))
        self._assert_no_leftovers()

    def test_later_rounds_wait_longer(self):
        # slow-4 exceeds the 3 s limit of round 1 but fits the 6 s of round 2.
        self._time_limit(3)
        self._copy(["slow-4"], n_rounds=2)
        self.assertEqual(self._started_modes(), ["slow-4", "slow-4"])
        self.assertGreater(self.elapsed, 6.5)
        self._assert_no_leftovers()

    def test_a_source_is_retried_after_growing_pauses(self):
        with self.assertRaises(grid_tools.GfalError):
            self._copy(["fail"], n_rounds=3, round_sleep_interval=1)
        gaps = self._start_gaps("fail")
        self.assertEqual(len(gaps), 2)
        self.assertGreater(gaps[0], 0.95)  # 1 s after round 1
        self.assertGreater(gaps[1], 1.95)  # 2 s after round 2
        self.assertLess(self.elapsed, 8)

    def test_a_failing_source_keeps_its_pauses_next_to_a_running_one(self):
        # The source that fails fast is retried next to the hanging one, but only after its
        # pause, so it does not use up its rounds at once; the hanging one pauses as well.
        self._patch("COPY_STALL_TIMEOUT", 1)
        self._join_after(0.5)
        with self.assertRaises(grid_tools.GfalError):
            self._copy(["hang", "fail"], n_rounds=4, round_sleep_interval=1)
        for mode in ("hang", "fail"):
            gaps = self._start_gaps(mode)
            self.assertEqual(len(gaps), 3, mode)
            for pause, gap in zip((1, 2, 3), gaps):
                self.assertGreater(gap, pause - 0.05, mode)
        self._assert_no_leftovers()

    # -- several attempts at once ------------------------------------------------------------

    def test_hanging_source_is_joined_by_the_next_one(self):
        self._join_after(0.5)
        self._copy(["hang", "good"])
        self.assertLess(self.elapsed, 4)
        self.assertEqual(self._started_modes(), ["hang", "good"])
        self._assert_no_leftovers()

    def test_a_copy_that_is_about_to_finish_is_not_joined(self):
        # At 2 s, 90% of the file is there and data keeps arriving: at that rate the rest
        # takes 0.2 s.
        self._join_after(2)
        self._copy(["mostly-3", "good"])
        self.assertEqual(self._started_modes(), ["mostly-3"])
        self._assert_no_leftovers()

    def test_a_copy_of_unknown_size_is_joined_at_its_expected_time(self):
        # Without the size there is no telling how much is left, even while data arrives.
        self._join_after(0.5)
        self._copy(["trickle-8", "good"], size=None)
        self.assertEqual(self._started_modes(), ["trickle-8", "good"])
        self._assert_no_leftovers()

    def test_a_copy_that_stops_after_most_of_the_file_is_joined(self):
        # Joined once no data has arrived for the expected time, long before it stalls.
        self._join_after(1)
        self._copy(["mostlyhang", "good"])
        self.assertEqual(self._started_modes(), ["mostlyhang", "good"])
        self.assertLess(self.elapsed, 5)
        self._assert_no_leftovers()

    def test_at_most_two_attempts_run_at_once(self):
        self._join_after(0.5)
        self._time_limit(3)
        self._copy(["hang", "hang", "good"], n_rounds=1)
        self.assertEqual(self._started_modes(), ["hang", "hang", "good"])
        # good starts when the first hanging attempt times out.
        self.assertGreater(self.elapsed, 2.9)
        self.assertLess(self.elapsed, 8)
        self._assert_no_leftovers()

    def test_a_failing_attempt_leaves_the_one_it_joined_alone(self):
        self._join_after(0.5)
        self._copy(["trickle-8", "fail"], n_rounds=1)
        self.assertEqual(self._started_modes(), ["trickle-8", "fail"])
        self._assert_no_leftovers()

    def test_first_verified_copy_wins(self):
        # The first source is slow but finishes before the one that joined it.
        self._join_after(0.5)
        self._copy(["slow-1.5", "slow-5"])
        self.assertLess(self.elapsed, 4.5)
        self.assertEqual(self._started_modes(), ["slow-1.5", "slow-5"])
        self._assert_no_leftovers()

    def test_a_source_is_not_joined_by_itself(self):
        self._join_after(0.5)
        self._copy(["slow-2"], n_rounds=3)
        self.assertEqual(self._started_modes(), ["slow-2"])

    def test_a_busy_source_does_not_block_a_free_one(self):
        # failonce fails in round 1; its round-2 entry starts next to the hanging source
        # instead of waiting for the hanging one's round-2 entry ahead of it.
        self._join_after(0.5)
        self._copy(["hang", "failonce"], n_rounds=2)
        self.assertLess(self.elapsed, 4)
        self.assertEqual(self._started_modes(), ["hang", "failonce", "failonce"])
        self._assert_no_leftovers()

    def test_overtaken_site_is_tried_last_by_the_next_copy(self):
        self._join_after(0.5)
        self._copy(["hang", "good"])
        self.assertEqual(grid_tools._copy_failed_rses, {"T2_XX_Site0": 1})
        os.remove(self.out)
        self._join_after(5)
        self._copy(["hang", "good"])
        self.assertLess(self.elapsed, 4)
        self.assertEqual(self._started_modes(), ["hang", "good", "good"])
        self._assert_no_leftovers()

    def test_a_joining_site_overtaken_within_its_expected_time_is_not_demoted(self):
        # Joined at 2 s; the first source finishes at 2.5 s, well before the joining one is
        # late (at 4 s).
        self._join_after(2)
        self._copy(["slow-2.5", "hang"])
        self.assertEqual(self._started_modes(), ["slow-2.5", "hang"])
        self.assertEqual(grid_tools._copy_failed_rses, {})
        self._assert_no_leftovers()

    def test_a_winning_site_is_not_demoted_by_its_own_slow_endpoint(self):
        self._join_after(0.5)
        sources = [
            ("stub://hang/0", "T2_XX_Site", 0),
            ("stub://good/1", "T2_XX_Site", 0),
        ]
        self._copy(sources, n_rounds=1)
        self.assertEqual(grid_tools._copy_failed_rses, {})
        self._assert_no_leftovers()

    # -- giving up and reporting -----------------------------------------------------------------

    def test_all_sources_failing_raises_within_the_limits(self):
        self._join_after(0.5)
        self._time_limit(3)
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy(["hang", "fail"], n_rounds=2)
        message = str(ctx.exception)
        self.assertIn("every source failed", message)
        self.assertIn("stub://hang/0: no result after 3 s", message)
        self.assertIn("stub://fail/1: exit code 3: stub: connection refused", message)
        # round 1: hang (3 s) with fail next to it; round 2: hang (6 s).
        self.assertLess(self.elapsed, 14)
        self._assert_no_leftovers()

    def test_copy_gives_up_after_the_total_time(self):
        self._join_after(0.5)
        self._patch("COPY_MAX_TOTAL_TIME", 1.5)
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy(["hang", "hang"], n_rounds=4)
        self.assertIn("no copy within 2 s", str(ctx.exception))
        self.assertLess(self.elapsed, 4)
        self._assert_no_leftovers()

    def test_log_messages(self):
        self._join_after(0.5)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self._copy(["hang", "fail", "good"], n_rounds=1, verbose=1)
        lines = stdout.getvalue().splitlines()
        patterns = [
            r"Trying stub://hang/0 \(round 1, timeout 10 s, expected \d+ s, stall 10 s\)",
            r"Joining with stub://fail/1 \(round 1, ",
            r"Copy attempt from stub://fail/1 failed: exit code 3: stub: connection refused",
            r"Joining with stub://good/2 \(round 1, ",
            r"Copied from stub://good/2 in \d+ s",
        ]
        self.assertEqual(len(lines), len(patterns), lines)
        for pattern, line in zip(patterns, lines):
            self.assertRegex(line, "^" + pattern)

    # -- copy_remote_file ------------------------------------------------------------------------

    LFN = "/store/mc/X/NANOAODSIM/f.root"

    def _replicas(self, disk):
        return {
            self.LFN: {
                "pfns": {
                    "DISK": disk,
                    "TAPE": [("stub://good/tape", "T1_DE_KIT_Tape")],
                },
                "adler32": f"{ADLER32:08x}",
                "bytes": len(PAYLOAD),
            }
        }

    def _copy_remote(self, rucio, distances=None):
        distances = distances or {"T2_HU_Budapest": 10, "T2_IT_Bari": 11}
        rucio_mock = mock.Mock(side_effect=rucio)
        t0 = time.monotonic()
        try:
            with mock.patch.multiple(
                grid_tools,
                rucio_list_replicas=rucio_mock,
                get_local_site=mock.Mock(return_value="T2_CH_CERN"),
                get_distances=mock.Mock(return_value=distances),
            ):
                grid_tools.copy_remote_file(
                    self.LFN,
                    self.out,
                    voms_token="proxy",
                    retry_sleep_interval=0,
                    verbose=0,
                )
        finally:
            self.elapsed = time.monotonic() - t0
        return rucio_mock

    def test_copy_remote_file_reads_the_disk_replicas_with_size_and_checksum(self):
        self._join_after(0.5)
        replicas = self._replicas(
            [("stub://hang/0", "T2_HU_Budapest"), ("stub://good/1", "T2_IT_Bari")]
        )
        self._copy_remote([replicas])
        self._assert_payload()
        self.assertEqual(self._started_modes(), ["hang", "good"])
        # An intact local copy is kept without copying again.
        self._copy_remote([replicas])
        self.assertEqual(self._started_modes(), ["hang", "good"])
        self._assert_no_leftovers()

    def test_copy_remote_file_copies_again_without_a_checksum_in_rucio(self):
        replicas = self._replicas([("stub://good/0", "T2_IT_Bari")])
        replicas[self.LFN].update(adler32=None, bytes=None)
        with open(self.out, "wb") as f:
            f.write(PAYLOAD)
        # A local copy cannot be verified, so it is not trusted.
        self._copy_remote([replicas])
        self._assert_payload()
        self.assertEqual(self._started_modes(), ["good"])

    def test_copy_remote_file_replaces_a_damaged_local_copy(self):
        with open(self.out, "wb") as f:
            f.write(PAYLOAD[::-1])
        self._copy_remote([self._replicas([("stub://good/0", "T2_IT_Bari")])])
        self._assert_payload()
        self.assertEqual(self._started_modes(), ["good"])

    def test_copy_remote_file_rejects_a_corrupt_replica_by_the_rucio_checksum(self):
        replicas = self._replicas(
            [("stub://corrupt/0", "T2_HU_Budapest"), ("stub://good/1", "T2_IT_Bari")]
        )
        self._copy_remote([replicas])
        self._assert_payload()
        self.assertEqual(self._started_modes(), ["corrupt", "good"])

    def test_copy_remote_file_falls_back_to_the_xrootd_federation(self):
        replicas = self._replicas(
            [("stub://fail/0", "T2_HU_Budapest"), ("stub://fail/1", "T2_IT_Bari")]
        )
        self._copy_remote([replicas])
        self._assert_payload()
        self.assertEqual(self._started_modes(), ["fail", "fail", "good"])

    def test_copy_remote_file_tries_the_federation_after_sites_of_unknown_distance(
        self,
    ):
        self._patch("COPY_FEDERATION_PREFIX", "stub://fail/federation")
        replicas = self._replicas(
            [("stub://fail/0", "T2_IT_Bari"), ("stub://good/1", "T2_HU_Budapest")]
        )
        distances = {"T2_IT_Bari": 11, "T2_HU_Budapest": float("inf")}
        # The federation and the site would tie on distance; repeat against the random draw.
        for _ in range(10):
            grid_tools._copy_failed_rses.clear()
            for path in (self.started, self.out):
                if os.path.exists(path):
                    os.remove(path)
            self._copy_remote([replicas], distances)
            self.assertEqual(self._started_modes(), ["fail", "good"])

    def test_copy_remote_file_retries_a_failed_rucio_query(self):
        replicas = self._replicas([("stub://good/0", "T2_IT_Bari")])
        rucio = self._copy_remote([RuntimeError("HTTP 503"), replicas])
        self.assertEqual(rucio.call_count, 2)
        self._assert_payload()

    def test_copy_remote_file_gives_up_when_rucio_keeps_failing(self):
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy_remote([RuntimeError("HTTP 503")] * 3)
        self.assertIn("Unable to query Rucio", str(ctx.exception))
        self.assertLess(self.elapsed, 2)  # the retry pauses follow retry_sleep_interval
        self.assertEqual(self._started_modes(), [])

    def test_rucio_query_backs_off(self):
        with mock.patch.object(
            grid_tools, "rucio_list_replicas", side_effect=RuntimeError("HTTP 503")
        ), mock.patch.object(grid_tools.time, "sleep") as sleep:
            with self.assertRaises(grid_tools.GfalError):
                grid_tools.rucio_replica_info(self.LFN, verbose=0)
        self.assertEqual(sleep.call_args_list, [mock.call(10), mock.call(30)])

    def test_copy_remote_file_of_a_file_unknown_to_rucio(self):
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy_remote([{}] * 3)
        self.assertIn("is not known to Rucio", str(ctx.exception))
        self.assertEqual(self._started_modes(), [])

    def test_copy_remote_file_needs_a_disk_replica(self):
        with self.assertRaises(grid_tools.GfalError) as ctx:
            self._copy_remote([self._replicas([])])
        self.assertIn("No disk replica", str(ctx.exception))
        self.assertEqual(self._started_modes(), [])

    def test_copy_commands(self):
        out = os.path.join(self.tmp.name, "nano.root.part0")
        cmd, env = REAL_COPY_COMMAND("root://site//store/f", out, 300.7, "/proxy")
        self.assertEqual(
            cmd,
            [
                "xrdcp",
                "--force",
                "--nopbar",
                "--streams",
                "1",
                "root://site//store/f",
                out,
            ],
        )
        self.assertEqual(env["X509_USER_PROXY"], "/proxy")
        self.assertEqual(env["PATH"], os.environ["PATH"])
        for url in (
            "davs://site/f",
            "https://site/f",
            "gsiftp://site/f",
            "srm://site/f",
        ):
            cmd, env = REAL_COPY_COMMAND(url, "nano.root.part0", 300.7, "/proxy")
            self.assertEqual(
                cmd,
                [
                    "gfal-copy",
                    "--force",
                    "--timeout",
                    "300",
                    url,
                    "file://" + os.path.abspath("nano.root.part0"),
                ],
            )
            self.assertEqual(env, grid_tools.gfal_env("/proxy"))
        with self.assertRaises(grid_tools.GfalError):
            REAL_COPY_COMMAND("file:///store/f", out, 300, "/proxy")

    def test_copy_remote_file_from_a_url(self):
        with mock.patch.object(
            grid_tools, "gfal_sum", return_value=ADLER32
        ) as gfal_sum:
            grid_tools.copy_remote_file(
                "stub://good/url", self.out, voms_token="proxy", verbose=0
            )
        self._assert_payload()
        self.assertEqual(gfal_sum.call_args.kwargs["timeout"], 10)


if __name__ == "__main__":
    unittest.main()
