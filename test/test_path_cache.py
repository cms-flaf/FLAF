import os
import sys
import unittest
from collections import namedtuple
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.RunKit import law_gfal
from FLAF.RunKit.law_gfal import GFALFileInterface, PathCache, RemotePathCache

Entry = namedtuple("Entry", ["name"])
FakeFS = namedtuple("FakeFS", ["file_interface"])


def as_fs(file_interface):
    """Wrap an interface the way WLCGFileSystem does, for the snapshot helpers."""
    return FakeFS(file_interface)


BASE = "davs://server:1234/store/test"


class FakeGFALFileInterface(GFALFileInterface):
    """GFALFileInterface without a grid proxy, backed by an in-memory directory tree."""

    def __init__(self, tree, path_cache):
        self.voms_token = None
        self.path_cache = path_cache
        self.verbose = 0
        self.tree = tree
        super(GFALFileInterface, self).__init__(base=[BASE])


def make_interface(tree, path_cache=None, ls_failures=0):
    """Interface over *tree* ({dir_path: [entry names]}); the first *ls_failures* listings
    of any directory fail, mimicking a transient gfal error."""
    fs = FakeGFALFileInterface(tree, path_cache or PathCache(600))
    state = {"failures": ls_failures}

    def fake_ls(uri, **kwargs):
        if state["failures"] > 0:
            state["failures"] -= 1
            return None
        path = uri[len(BASE) :].strip("/")
        if path not in tree:
            return None
        return [Entry(name) for name in tree[path]]

    fs.fake_ls = fake_ls
    return fs


class FakeCacheServer:
    """In-memory stand-in for pathCacheServer, shared by several RemotePathCache clients."""

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


class TestPathCacheListing(unittest.TestCase):
    def test_absence_requires_a_listing_not_a_known_directory(self):
        # Regression test: an entry created after the cache learned that its directory
        # exists must not be reported as absent (production v2608 reported 2260 of 11066
        # existing AnaTuple outputs as missing because of this inference).
        tree = {"data": ["file_0.root", "file_1.root"]}
        cache = PathCache(600)
        cache.set(os.path.join(BASE, "data"), True)  # directory known, content unknown
        fs = make_interface(tree, path_cache=cache)
        with mock.patch.object(law_gfal, "gfal_ls_safe", fs.fake_ls):
            self.assertTrue(fs.exists("data/file_0.root"))
            self.assertTrue(fs.exists("data/file_1.root"))
            self.assertFalse(fs.exists("data/file_2.root"))

    def test_listing_answers_absent_siblings_without_further_listings(self):
        tree = {"data": ["file_0.root"]}
        fs = make_interface(tree)
        with mock.patch.object(law_gfal, "gfal_ls_safe", fs.fake_ls):
            n0 = GFALFileInterface.listdir_counter
            self.assertTrue(fs.exists("data/file_0.root"))
            for i in range(1, 20):
                self.assertFalse(fs.exists(f"data/file_{i}.root"))
            self.assertEqual(GFALFileInterface.listdir_counter - n0, 1)

    def test_expired_listing_is_not_used(self):
        tree = {"data": ["file_0.root"]}
        fs = make_interface(tree, path_cache=PathCache(-1))
        with mock.patch.object(law_gfal, "gfal_ls_safe", fs.fake_ls):
            self.assertFalse(fs.exists("data/file_1.root"))
            n0 = GFALFileInterface.listdir_counter
            tree["data"].append("file_1.root")
            self.assertTrue(fs.exists("data/file_1.root"))
            self.assertEqual(GFALFileInterface.listdir_counter - n0, 1)

    def test_absent_directory_is_still_reported_absent(self):
        fs = make_interface({"data": []})
        with mock.patch.object(law_gfal, "gfal_ls_safe", fs.fake_ls):
            self.assertFalse(fs.exists("no_such_dir/file_0.root"))
            self.assertFalse(fs.exists("data/file_0.root"))

    def test_transient_listing_failure_does_not_cache_absence(self):
        tree = {"data": ["file_0.root"]}
        fs = make_interface(tree, ls_failures=1)
        with mock.patch.object(law_gfal, "gfal_ls_safe", fs.fake_ls):
            self.assertTrue(fs.exists("data/file_0.root"))


class TestSharedListing(unittest.TestCase):
    """One listing must serve every other process and job through the cache server, so that
    a status check or a job never has to list the directory again while it is cached."""

    def setUp(self):
        self.server = FakeCacheServer()
        self.tree = {"data": ["file_0.root"]}

    def _client(self):
        return make_interface(
            self.tree,
            path_cache=RemotePathCache("host", 1, local_cache_validity_period=600),
        )

    def test_second_process_answers_from_the_shared_listing(self):
        with self.server.patch():
            first = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", first.fake_ls):
                self.assertTrue(first.exists("data/file_0.root"))

            # A second process starts with an empty local cache; it must answer both the
            # existing and the absent path from the server, without listing anything.
            second = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", second.fake_ls):
                n0 = GFALFileInterface.listdir_counter
                self.assertTrue(second.exists("data/file_0.root"))
                self.assertFalse(second.exists("data/file_1.root"))
                self.assertEqual(GFALFileInterface.listdir_counter - n0, 0)

    def test_file_published_after_the_listing_is_found_without_listing(self):
        with self.server.patch():
            first = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", first.fake_ls):
                self.assertFalse(first.exists("data/file_1.root"))

            # A job creates the file and publishes it, as filecopy() does.
            self.tree["data"].append("file_1.root")
            producer = self._client()
            producer.path_cache.set(os.path.join(BASE, "data/file_1.root"), True)

            consumer = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", consumer.fake_ls):
                n0 = GFALFileInterface.listdir_counter
                self.assertTrue(consumer.exists("data/file_1.root"))
                self.assertEqual(GFALFileInterface.listdir_counter - n0, 0)

    def test_directory_created_after_a_parent_listing_becomes_known(self):
        with self.server.patch():
            self.tree[""] = []
            first = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", first.fake_ls):
                first.listdir("")  # the new directory does not exist yet
                self.assertFalse(first.exists("data"))

            producer = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", producer.fake_ls):
                with mock.patch.object(
                    law_gfal, "gfal_copy_safe", lambda *a, **k: None
                ):
                    self.tree[""] = ["data"]
                    producer.filecopy("file:///tmp/x.root", "data/file_0.root")

            consumer = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", consumer.fake_ls):
                self.assertTrue(consumer.exists("data"))
                self.assertTrue(consumer.exists("data/file_0.root"))

    def test_snapshot_ships_a_marker_only_for_a_listing_taken_here(self):
        # The submit-time snapshot shipped to jobs must not claim that a partial set of
        # entries is a complete listing.
        with self.server.patch():
            lister = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", lister.fake_ls):
                self.assertTrue(lister.exists("data/file_0.root"))
            shipped = dict(law_gfal.local_path_cache(as_fs(lister)).iter_valid())
            self.assertIn(os.path.join(BASE, "data", law_gfal.LISTING_MARKER), shipped)

            # This one only learned the marker from the server, so it may not pass it on.
            reader = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", reader.fake_ls):
                self.assertFalse(reader.exists("data/file_1.root"))
            shipped = dict(law_gfal.local_path_cache(as_fs(reader)).iter_valid())
            self.assertNotIn(
                os.path.join(BASE, "data", law_gfal.LISTING_MARKER), shipped
            )

    def test_directory_existence_alone_never_implies_absence(self):
        # Regression test for the production failure: the era directory being listed marks
        # every dataset directory as existing, which must not suppress the listing that is
        # needed to learn the dataset directory's content.
        with self.server.patch():
            self.tree["data"] = ["file_0.root", "file_1.root"]
            self.tree[""] = ["data"]
            first = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", first.fake_ls):
                first.listdir("")  # marks BASE/data as existing, content unknown
            self.assertIs(self.server.entries.get(os.path.join(BASE, "data")), True)

            second = self._client()
            with mock.patch.object(law_gfal, "gfal_ls_safe", second.fake_ls):
                self.assertTrue(second.exists("data/file_0.root"))
                self.assertTrue(second.exists("data/file_1.root"))
                self.assertFalse(second.exists("data/file_2.root"))


if __name__ == "__main__":
    unittest.main()
