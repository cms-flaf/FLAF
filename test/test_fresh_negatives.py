#!/usr/bin/env python3
"""A CRAB job's output must not stay "absent" because of a listing taken before it was written.

`exists()` answers a missing file from cached knowledge that the cache server shares between
processes for 24 h: a directory listing marker, an entry, an absent ancestor. HTCondor jobs
publish what they write to the server; CRAB jobs cannot reach it. So a listing taken before
a CRAB job wrote its output answers "absent" for that output, in every process, until the
marker expires -- and law, checking the outputs of the jobs CRAB reports finished, demotes
a job that did finish. Clearing the caches of the checking process is not enough: its next
lookup falls through to the same marker on the server. `require_fresh_negatives()` makes
every negative rest on a listing taken by this process after the call.
"""

import os
import sys
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import law_gfal
from FLAF.RunKit.grid_tools import GfalError
from FLAF.RunKit.law_gfal import (
    GFALFileInterface,
    PathCache,
    RemotePathCache,
    require_fresh_negatives,
)

BASE = "davs://server:1234/store/test"
DATA = os.path.join(BASE, "data")


class Entry:
    def __init__(self, name):
        self.name = name
        self.size = 0
        self.is_dir = False


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


class Storage:
    """The directory tree on the storage, listed through gfal_ls_checked; counts listings."""

    def __init__(self, tree):
        self.tree = tree
        self.listed = []
        self.down = False

    def ls(self, uri, **kwargs):
        self.listed.append(uri)
        if self.down:
            raise GfalError("gfal-ls error: 112 (Host is down) - Could not connect")
        path = uri[len(BASE) :].strip("/")
        if path not in self.tree:
            return None
        return [Entry(name) for name in self.tree[path]]

    def patch(self):
        return mock.patch.object(law_gfal, "gfal_ls_checked", self.ls)


class FreshNegatives(unittest.TestCase):
    def setUp(self):
        self.server = FakeCacheServer()
        self.storage = Storage({"": ["data"], "data": ["file_0.root"]})
        self.patches = [self.server.patch(), self.storage.patch()]
        for patch in self.patches:
            patch.start()
        # A status check at submit time lists the directory and publishes the listing.
        self.assertFalse(self.client().exists("data/file_1.root"))
        # A CRAB job writes file_1 and cannot tell the cache server.
        self.storage.tree["data"].append("file_1.root")
        self.storage.listed.clear()

    def tearDown(self):
        GFALFileInterface.negatives_valid_after = 0.0
        for patch in reversed(self.patches):
            patch.stop()

    def client(self):
        with mock.patch.object(law_gfal, "get_voms_proxy_info", lambda: {"path": None}):
            fs = GFALFileInterface(base=[BASE])
        fs.path_cache = RemotePathCache("host", 1, local_cache_validity_period=600)
        return fs

    def test_the_server_marker_answers_absent_while_the_epoch_is_off(self):
        self.assertFalse(self.client().exists("data/file_1.root"))
        self.assertEqual(self.storage.listed, [], "answered from the server's marker")

    def test_a_fresh_negative_lists_once_and_republishes(self):
        fs = self.client()
        require_fresh_negatives()
        self.assertTrue(fs.exists("data/file_1.root"))
        self.assertEqual(self.storage.listed, [DATA])
        self.assertIs(self.server.entries.get(os.path.join(DATA, "file_1.root")), True)
        # every other process now finds it without listing
        self.assertTrue(self.client().exists("data/file_1.root"))
        self.assertEqual(self.storage.listed, [DATA])

    def test_one_listing_per_directory_per_epoch(self):
        fs = self.client()
        require_fresh_negatives()
        self.assertTrue(fs.exists("data/file_1.root"))
        for i in range(2, 10):
            self.assertFalse(fs.exists(f"data/file_{i}.root"))
        self.assertEqual(self.storage.listed, [DATA])

    def test_a_confirmed_absence_counts_as_a_fresh_listing(self):
        fs = self.client()
        require_fresh_negatives()
        self.assertFalse(fs.exists("gone/file_0.root"))
        n_listed = len(self.storage.listed)
        self.assertFalse(fs.exists("gone/file_1.root"))
        self.assertEqual(self.storage.listed[n_listed:], [])

    def test_a_positive_is_never_relisted(self):
        fs = self.client()
        require_fresh_negatives()
        self.assertTrue(fs.exists("data/file_0.root"))
        self.assertEqual(self.storage.listed, [])

    def test_clearing_the_local_cache_alone_does_not_help(self):
        # The trap that makes an in-process cache flush insufficient: the next lookup falls
        # through to the marker that the server still holds.
        fs = self.client()
        self.assertFalse(fs.exists("data/file_1.root"))
        fs.path_cache.local_cache = PathCache(600)
        self.assertFalse(fs.exists("data/file_1.root"))
        self.assertEqual(self.storage.listed, [])

    def test_a_listing_taken_before_the_epoch_does_not_count(self):
        fs = self.client()
        fs.listdir("data")
        self.storage.tree["data"].append("file_2.root")
        require_fresh_negatives()
        self.assertTrue(fs.exists("data/file_2.root"))
        self.assertEqual(self.storage.listed, [DATA, DATA])

    def test_a_new_epoch_needs_a_new_listing(self):
        fs = self.client()
        require_fresh_negatives()
        self.assertFalse(fs.exists("data/file_2.root"))
        self.storage.tree["data"].append("file_2.root")
        require_fresh_negatives()
        self.assertTrue(fs.exists("data/file_2.root"))
        self.assertEqual(self.storage.listed, [DATA, DATA])

    def test_a_directory_known_absent_is_listed_again(self):
        # The negative comes from directory inference: `new` was absent when it was looked
        # up, and a CRAB job has created it since.
        fs = self.client()
        self.assertFalse(fs.exists("new/file_0.root"))
        self.storage.tree["new"] = ["file_1.root"]
        self.assertFalse(fs.exists("new/file_1.root"))
        n_listed = len(self.storage.listed)
        require_fresh_negatives()
        self.assertTrue(fs.exists("new/file_1.root"))
        self.assertEqual(self.storage.listed[n_listed:], [os.path.join(BASE, "new")])

    def test_a_failed_fresh_listing_publishes_nothing(self):
        fs = self.client()
        require_fresh_negatives()
        self.storage.down = True
        self.assertFalse(fs.exists("data/file_1.root"))
        self.assertIsNone(self.server.entries.get(os.path.join(DATA, "file_1.root")))
        self.assertNotIn(
            os.path.join(DATA, "file_1.root"), fs.path_cache.local_cache.cache
        )
        self.storage.down = False
        self.assertTrue(fs.exists("data/file_1.root"))


if __name__ == "__main__":
    unittest.main()
