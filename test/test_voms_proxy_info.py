#!/usr/bin/env python3
"""A stale CRL must not abort a job that still has a usable proxy.

In the DSProd CRAB production, 309 of 1400 jobs died during scheduling: `voms-proxy-info`
exited 1 with `CRL has expired` for `cms-auth.cern.ch`, and `get_voms_proxy_info()` raised
that as `PsCallError`. `GFALFileInterface` calls it while law builds the task's targets, so
the job never reached its payload. The read passes `-dont-verify-ac`: the callers need the
proxy path and its remaining lifetime, not the attribute-certificate check.

A missing proxy still exits non-zero. That failure has to keep raising, with the command's
stderr in the message, or `Couldn't find a valid proxy.` is lost.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import grid_tools, run_tools
from FLAF.RunKit.run_tools import PsCallError

STDOUT = [
    "subject   : /DC=ch/DC=cern/OU=Organic Units/OU=Users/CN=kandroso",
    "path      : /tmp/x509up_u34016",
    "timeleft  : 23:59:00",
    "",
]


def replay(stdout, stderr, return_code):
    """A ps_call that runs a real process printing `stdout`/`stderr` and exiting with
    `return_code`, so that the error text is produced by the real run_tools.ps_call. The
    texts are read from files: as arguments they would also show up in the command line
    that the exception quotes."""

    def call(cmd, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            files = [os.path.join(tmp, "stdout"), os.path.join(tmp, "stderr")]
            for path, text in zip(files, (stdout, stderr)):
                with open(path, "w") as f:
                    f.write(text)
            script = '/bin/cat "$1"; /bin/cat "$2" >&2; exit "$3"'
            return run_tools.ps_call(
                ["/bin/sh", "-c", script, "sh", *files, str(return_code)], **kwargs
            )

    return call


class VomsProxyInfoTests(unittest.TestCase):
    def test_ac_verification_is_skipped_and_the_proxy_is_parsed(self):
        with mock.patch.object(
            grid_tools, "ps_call", return_value=(0, STDOUT, [])
        ) as call:
            info = grid_tools.get_voms_proxy_info()
        (cmd,), _ = call.call_args
        self.assertEqual(cmd, ["voms-proxy-info", "-dont-verify-ac"])
        self.assertEqual(info["path"], "/tmp/x509up_u34016")
        self.assertAlmostEqual(info["timeleft"], 23 + 59 / 60.0)

    def test_a_missing_proxy_is_still_an_error(self):
        err = PsCallError(
            "voms-proxy-info -dont-verify-ac", 1, "Couldn't find a valid proxy."
        )
        with mock.patch.object(grid_tools, "ps_call", side_effect=err):
            with self.assertRaises(PsCallError) as caught:
                grid_tools.get_voms_proxy_info()
        self.assertIn("Couldn't find a valid proxy.", str(caught.exception))
        self.assertEqual(caught.exception.return_code, 1)

    def test_the_reason_a_missing_proxy_fails_reaches_the_exception(self):
        # What voms-proxy-info really does without a proxy (checked 2026-10-08): nothing on
        # stdout, the reason on stderr, exit code 1.
        with mock.patch.object(
            grid_tools, "ps_call", replay("", "\nCouldn't find a valid proxy.\n", 1)
        ):
            with self.assertRaises(PsCallError) as caught:
                grid_tools.get_voms_proxy_info()
        self.assertIn("Couldn't find a valid proxy.", str(caught.exception))
        self.assertEqual(caught.exception.return_code, 1)

    def test_a_usable_proxy_is_parsed_through_the_real_call(self):
        with mock.patch.object(grid_tools, "ps_call", replay("\n".join(STDOUT), "", 0)):
            info = grid_tools.get_voms_proxy_info()
        self.assertEqual(info["path"], "/tmp/x509up_u34016")


if __name__ == "__main__":
    unittest.main()
