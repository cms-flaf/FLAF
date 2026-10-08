#!/usr/bin/env python3
"""How a finished artefact becomes visible at its final path.

Every remote write that law makes through GFALFileInterface builds a complete file locally
and then publishes it; nothing appends, and no reader watches a product grow. `copy_flag`
copies straight onto the final path, so the final name exists while its content is partial,
and it removes the previous file first, so a copy that dies leaves nothing behind.
`copy_rename` uploads to a tmp name, verifies the checksum and renames onto the target. That
is an improvement only if three things hold together, each tested below: the target is not
removed up front, the publish happens through the rename, and the tmp name is unique per
writer -- a shared tmp path only moves the collision between two jobs writing one target.
"""

import contextlib
import os
import sys
import unittest
import zlib
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import grid_tools, law_gfal
from FLAF.RunKit.grid_tools import GfalError

SOURCE = "file:///local/nano.root"
TARGET = "davs://example.cern.ch:443/store/x/nano_1.root"


class Recorder:
    """Stands in for the gfal CLI, recording what would have been done to the storage.

    `contents` maps a path to its content. With `refuse_existing`, a rename onto an existing
    name fails, as on a storage that does not replace the destination of a rename.
    """

    def __init__(self, existing=None, refuse_existing=False):
        self.contents = {SOURCE: "new"}
        self.contents.update({TARGET: "old"} if existing is None else existing)
        self.refuse_existing = refuse_existing
        self.removed = []
        self.copied = []
        self.renamed = []

    def install(self, stack):
        def patch(name, fn):
            stack.enter_context(mock.patch.object(grid_tools, name, fn))

        patch("gfal_exists", lambda path, **kw: path in self.contents)
        patch("gfal_rm", self._rm)
        patch("gfal_copy", self._copy)
        patch("gfal_rename", self._rename)
        patch("gfal_stat", lambda path, **kw: {"type": "regular file"})
        patch("gfal_sum", self._sum)
        return self

    def _sum(self, path, **kw):
        if path not in self.contents:
            raise GfalError(f"gfal-sum error: 2 (No such file or directory) {path}")
        return zlib.adler32(self.contents[path].encode())

    def _rm(self, path, **kw):
        self.removed.append(path)
        self.contents.pop(path, None)

    def _copy(self, src, dst, **kw):
        self.copied.append((src, dst))
        # A source that is not tracked is a local file, e.g. the copy_flag marker.
        self.contents[dst] = self.contents.get(src, src)

    def _rename(self, src, dst, **kw):
        if self.refuse_existing and dst in self.contents:
            raise GfalError("gfal-rename error: 17 (File exists)")
        self.renamed.append((src, dst))
        self.contents[dst] = self.contents.pop(src)


def publish(mode, **recorder_kwargs):
    with contextlib.ExitStack() as stack:
        rec = Recorder(**recorder_kwargs).install(stack)
        grid_tools.gfal_copy_safe(
            SOURCE, TARGET, copy_mode=mode, voms_token="tok", n_retries=1, verbose=0
        )
    return rec


class PublishingByRename(unittest.TestCase):
    def test_the_published_file_is_never_taken_away_first(self):
        """`copy_rename` used to remove the target before uploading, so the product was
        absent for the whole duration of the copy."""
        rec = publish("copy_rename")
        self.assertNotIn(TARGET, rec.removed)
        self.assertEqual(rec.contents[TARGET], "new")

    def test_the_target_is_only_ever_created_by_the_rename(self):
        rec = publish("copy_rename")
        self.assertEqual([dst for _, dst in rec.renamed], [TARGET])
        self.assertNotIn(TARGET, [dst for _, dst in rec.copied])

    def test_the_upload_goes_to_a_marked_tmp_name_after_the_extension(self):
        """A marker before `.root` would be picked up by anything globbing `*.root`."""
        rec = publish("copy_rename")
        ((_, tmp),) = rec.copied
        self.assertTrue(tmp.startswith(TARGET), tmp)
        self.assertTrue(grid_tools.is_copy_rename_tmp(tmp), tmp)
        self.assertFalse(grid_tools.is_copy_rename_tmp(TARGET))

    def test_two_writers_of_the_same_target_do_not_share_a_tmp_path(self):
        """`download()` removes the tmp at the top of every attempt, so with a shared tmp
        path one writer would delete the other's upload."""
        first, second = publish("copy_rename"), publish("copy_rename")
        self.assertNotEqual(first.copied[0][1], second.copied[0][1])

    def test_the_checksum_is_verified_before_the_rename(self):
        with contextlib.ExitStack() as stack:
            rec = Recorder().install(stack)
            stack.enter_context(
                mock.patch.object(grid_tools, "gfal_sum", side_effect=[1, 2])
            )
            with self.assertRaises(GfalError):
                grid_tools.gfal_copy_safe(
                    SOURCE,
                    TARGET,
                    copy_mode="copy_rename",
                    voms_token="tok",
                    n_retries=1,
                    verbose=0,
                )
        self.assertEqual(rec.renamed, [], "a corrupt upload must never be published")
        self.assertEqual(rec.contents[TARGET], "old")

    def test_copy_flag_still_behaves_as_it_did(self):
        """The other mode is unchanged: it copies onto the target and clears it first."""
        rec = publish("copy_flag")
        self.assertIn(TARGET, rec.removed)
        self.assertIn(TARGET, [dst for _, dst in rec.copied])
        self.assertEqual(rec.renamed, [])

    def test_a_first_publish_needs_no_removal_either(self):
        rec = publish("copy_rename", existing={})
        self.assertEqual(rec.removed, [])
        self.assertEqual([dst for _, dst in rec.renamed], [TARGET])


class StorageThatDoesNotReplaceOnRename(unittest.TestCase):
    """EOS replaces the destination of a rename (xrootd, checked 2026-10-08); a storage
    that refuses must still end with the new content published."""

    def test_a_different_target_is_replaced(self):
        rec = publish("copy_rename", refuse_existing=True)
        self.assertEqual(rec.contents[TARGET], "new")
        self.assertEqual(rec.removed, [TARGET])
        self.assertEqual(
            [p for p in rec.contents if grid_tools.is_copy_rename_tmp(p)], []
        )

    def test_an_identical_target_is_kept_and_the_upload_dropped(self):
        rec = publish("copy_rename", existing={TARGET: "new"}, refuse_existing=True)
        self.assertNotIn(TARGET, rec.removed)
        self.assertEqual(rec.contents[TARGET], "new")
        self.assertEqual(
            [p for p in rec.contents if grid_tools.is_copy_rename_tmp(p)], []
        )

    def test_a_failed_rename_without_a_target_is_not_papered_over(self):
        with contextlib.ExitStack() as stack:
            rec = Recorder(existing={}).install(stack)
            stack.enter_context(
                mock.patch.object(
                    grid_tools,
                    "gfal_rename",
                    side_effect=GfalError(
                        "gfal-rename error: 110 (Connection timed out)"
                    ),
                )
            )
            with self.assertRaises(GfalError):
                grid_tools.gfal_copy_safe(
                    SOURCE,
                    TARGET,
                    copy_mode="copy_rename",
                    voms_token="tok",
                    n_retries=1,
                    verbose=0,
                )
        self.assertNotIn(TARGET, rec.contents)
        self.assertEqual(rec.removed, [])


class GfalCopyForce(unittest.TestCase):
    def command(self, **kwargs):
        with mock.patch.object(grid_tools, "ps_call") as call:
            grid_tools.gfal_copy("a", "b", voms_token="tok", verbose=0, **kwargs)
        (cmd,), _ = call.call_args
        return cmd

    def test_force_overwrites(self):
        self.assertIn("--force", self.command(force=True))

    def test_no_overwrite_by_default(self):
        self.assertNotIn("--force", self.command())


class WhatLawUses(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(
            law_gfal, "get_voms_proxy_info", return_value={"path": "/tmp/token"}
        ):
            self.fs = law_gfal.GFALFileInterface(base="davs://host:443/store")

    def copy_modes(self, src, dst):
        modes = []

        def copy_safe(*args, **kwargs):
            modes.append(kwargs.get("copy_mode", "copy_flag"))

        with mock.patch.object(law_gfal, "gfal_copy_safe", copy_safe):
            self.fs.filecopy(src, dst)
        return modes

    def test_every_local_to_remote_write_publishes_by_rename(self):
        """The chokepoint: every remote write of a law target goes through this call."""
        self.assertEqual(
            self.copy_modes("file:///tmp/x.root", "data/x.root"), ["copy_rename"]
        )

    def test_a_download_keeps_its_mode(self):
        self.assertEqual(
            self.copy_modes("data/x.root", "file:///tmp/x.root"), ["copy_flag"]
        )


if __name__ == "__main__":
    unittest.main()
