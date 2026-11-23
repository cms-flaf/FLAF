#!/usr/bin/env python3
"""stream_merge (Common/HistMergeStream.py), the HistMergerTask side of HistMergerFromHists.py
--stream, against a stand-in merger: one process's inputs are fetched at a time (the next one
while a process is merged), deleted only once the merger has acknowledged writing them, and the
post step comes last. A merger that fails, exits early or answers out of turn stops the merge,
and nothing fetched is left behind. Standard library only.
"""

import json
import os
import sys
import tempfile
import threading
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common.HistMergeStream import MERGER_ACK, stream_merge  # noqa: E402

# A stand-in merger: logs each message to a file, checks the inputs exist, prints some output
# and acknowledges. FAIL_AT=<process> makes it exit with 3 instead; WRONG_ACK answers wrongly.
FAKE_MERGER = r"""
import json, os, sys
log = open(sys.argv[1], "a")
for line in sys.stdin:
    msg = json.loads(line)
    log.write(line); log.flush()
    if "process" in msg:
        if os.environ.get("FAIL_AT") == msg["process"]:
            sys.exit(3)
        assert all(os.path.exists(p) for _, p in msg["inputs"]), msg
        print("merging", msg["process"], flush=True)
        ack = "merged " + (msg["process"] + "x" if os.environ.get("WRONG_ACK") else msg["process"])
        print("@@HistMerger " + ack, flush=True)
    else:
        print("@@HistMerger postprocessed", flush=True)
print("done", flush=True)
"""


class TestStreamMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.script = os.path.join(self.tmp.name, "merger.py")
        with open(self.script, "w") as f:
            f.write(FAKE_MERGER)
        self.msg_log = os.path.join(self.tmp.name, "messages.jsonl")
        self.events = []
        self.lock = threading.Lock()
        self.processes = [
            ("TT", [("TT_a", 2), ("TT_b", 1)]),
            ("DY", [("DY_a", 1)]),
            ("data", [("data", 3)]),
        ]

    def event(self, *what):
        with self.lock:
            self.events.append(what)

    def fetch(self, items):
        """Writes one local file per chunk; the stack's callbacks delete them."""
        import contextlib

        stack = contextlib.ExitStack()
        local = []
        for dataset, chunks in items:
            for chunk in range(chunks):
                path = os.path.join(self.tmp.name, f"{dataset}_{chunk}.root")
                open(path, "w").close()
                self.event("fetched", dataset, chunk)
                stack.callback(os.remove, path)
                stack.callback(self.event, "deleted", dataset, chunk)
                local.append((dataset, path))
        return stack, local

    def run_merge(self, env=None):
        lines = []
        old = {k: os.environ.get(k) for k in ("FAIL_AT", "WRONG_ACK")}
        os.environ.update(env or {})
        try:
            stream_merge(
                [sys.executable, self.script, self.msg_log],
                self.processes,
                self.fetch,
                ["TT_a", "TT_a", "TT_b", "DY_a", "data"],
                log=lambda line, end="": lines.append(line),
            )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return lines

    def messages(self):
        with open(self.msg_log) as f:
            return [json.loads(line) for line in f]

    def leftovers(self):
        return sorted(f for f in os.listdir(self.tmp.name) if f.endswith(".root"))

    def test_one_process_at_a_time_then_the_post_step(self):
        lines = self.run_merge()
        messages = self.messages()
        self.assertEqual(
            [m.get("process", "post") for m in messages], ["TT", "DY", "data", "post"]
        )
        self.assertEqual(
            [d for d, _ in messages[0]["inputs"]], ["TT_a", "TT_a", "TT_b"]
        )
        self.assertEqual(
            messages[-1]["postprocess"], ["TT_a", "TT_a", "TT_b", "DY_a", "data"]
        )
        # the merger's own output reaches the log, the acknowledgements do not
        self.assertEqual(
            [l.strip() for l in lines],
            ["merging TT", "merging DY", "merging data", "done"],
        )
        self.assertFalse(any(MERGER_ACK in l for l in lines))
        self.assertEqual(self.leftovers(), [])

    def test_inputs_deleted_after_their_merge_and_at_most_two_processes_on_disk(self):
        self.run_merge()
        on_disk, peak = set(), 0
        for kind, dataset, chunk in self.events:
            if kind == "fetched":
                on_disk.add((dataset, chunk))
            else:
                on_disk.discard((dataset, chunk))
            processes = {d.split("_")[0] for d, _ in on_disk}
            peak = max(peak, len(processes))
        self.assertLessEqual(peak, 2)
        # TT is deleted only after the merger had it: DY was fetched meanwhile, at most
        deleted_tt = self.events.index(("deleted", "TT_a", 0))
        fetched_data = self.events.index(("fetched", "data", 0))
        self.assertLess(deleted_tt, fetched_data)

    def test_a_failing_merger_stops_and_leaves_nothing(self):
        with self.assertRaisesRegex(
            RuntimeError, "merger exited with 3 before 'merged DY'"
        ):
            self.run_merge({"FAIL_AT": "DY"})
        self.assertEqual(self.leftovers(), [])

    def test_an_answer_out_of_turn_stops_the_merge(self):
        with self.assertRaisesRegex(
            RuntimeError, "expected 'merged TT', got 'merged TTx'"
        ):
            self.run_merge({"WRONG_ACK": "1"})
        self.assertEqual(self.leftovers(), [])


if __name__ == "__main__":
    unittest.main()
