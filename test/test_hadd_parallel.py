#!/usr/bin/env python3
"""Two `hadd -j` merges must not share hadd's partial files.

hadd writes the partial files of `-j` to `<working dir>/partial<i>_<TUUID>.root`, the working
directory being the system temporary directory unless `-d` names another. In a GitHub-Actions
run the AnalysisCacheTask branches of two producers merged in the same second, drew the same
TUUID, and one cache came out with the other's content (FLAF#361). `hadd_parallel` gives every
merge a fresh directory of its own and removes it afterwards, also when the merge fails.
"""

import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import run_tools
from FLAF.RunKit.run_tools import PsCallError


def work_dir_of(cmd):
    """The directory a hadd command line passes with `-d`."""
    return cmd[cmd.index("-d") + 1]


class HaddParallel(unittest.TestCase):
    def test_the_command_line(self):
        seen = []

        def fake_ps_call(cmd, verbose=0):
            seen.append((list(cmd), os.path.isdir(work_dir_of(cmd))))

        with mock.patch.object(run_tools, "ps_call", side_effect=fake_ps_call):
            run_tools.hadd_parallel("out.root", ["a.root", "b.root"])
        self.assertEqual(len(seen), 1)
        cmd, existed = seen[0]
        work_dir = work_dir_of(cmd)
        self.assertEqual(
            cmd,
            ["hadd", "-j", str(os.cpu_count()), "-d", work_dir, "-ff", "out.root"]
            + ["a.root", "b.root"],
        )
        # hadd refuses a -d directory it cannot access, and does not create it
        self.assertTrue(existed)
        self.assertEqual(os.path.dirname(work_dir), tempfile.gettempdir())
        self.assertFalse(os.path.exists(work_dir))

    def test_merges_at_the_same_time_get_directories_of_their_own(self):
        n_merges = 4
        barrier = threading.Barrier(n_merges, timeout=10)
        work_dirs = []
        lock = threading.Lock()

        def fake_ps_call(cmd, verbose=0):
            work_dir = work_dir_of(cmd)
            with lock:
                work_dirs.append(work_dir)
            # the same partial file name in every merge, as two equal TUUIDs give
            partial = os.path.join(work_dir, "partial0_same-uuid.root")
            with open(partial, "x") as f:
                f.write(cmd[-1])
            barrier.wait()
            with open(partial) as f:
                self.assertEqual(f.read(), cmd[-1])

        errors = []

        def merge(i):
            try:
                run_tools.hadd_parallel(f"out{i}.root", [f"in{i}.root"])
            except Exception as e:
                errors.append(e)

        with mock.patch.object(run_tools, "ps_call", side_effect=fake_ps_call):
            threads = [
                threading.Thread(target=merge, args=(i,)) for i in range(n_merges)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(set(work_dirs)), n_merges)
        for work_dir in work_dirs:
            self.assertFalse(os.path.exists(work_dir))

    def test_a_failed_merge_leaves_no_directory(self):
        work_dirs = []

        def failing_ps_call(cmd, verbose=0):
            work_dir = work_dir_of(cmd)
            work_dirs.append(work_dir)
            open(os.path.join(work_dir, "partial0_uuid.root"), "w").close()
            raise PsCallError(cmd, 1)

        with mock.patch.object(run_tools, "ps_call", side_effect=failing_ps_call):
            with self.assertRaises(PsCallError):
                run_tools.hadd_parallel("out.root", ["a.root", "b.root"])
        self.assertEqual(len(work_dirs), 1)
        self.assertFalse(os.path.exists(work_dirs[0]))


if __name__ == "__main__":
    unittest.main()
