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
from FLAF.RunKit.grid_tools import GfalError
from FLAF.RunKit.law_gfal import GFALFileInterface, PathCache, RemotePathCache

# The fields of RunKit.grid_tools.FileInfo that law_gfal reads off a listing. Defaults keep
# the common case readable: a plain file whose size does not matter to the test at hand.
Entry = namedtuple("Entry", ["name", "size", "is_dir"], defaults=(0, False))
FakeFS = namedtuple("FakeFS", ["file_interface"])


def as_fs(file_interface):
    """Wrap an interface the way WLCGFileSystem does, for the snapshot helpers."""
    return FakeFS(file_interface)


BASE = "davs://server:1234/store/test"


class FakeGFALFileInterface(GFALFileInterface):
    """GFALFileInterface without a grid proxy, backed by an in-memory directory tree."""

    def __init__(self, tree, path_cache):
        # Run the real __init__, with only the grid-proxy lookup stubbed out, instead of
        # mirroring the attributes it sets. Mirroring them by hand is what broke this file
        # when `listing_sizes` was added: the fake kept working until something read it.
        with mock.patch.object(law_gfal, "get_voms_proxy_info", lambda: {"path": None}):
            super().__init__(base=[BASE])
        self.path_cache = path_cache
        self.tree = tree


def make_interface(tree, path_cache=None, ls_failures=0):
    """Interface over *tree*, mapping a directory path to its contents: either a list of
    entry names, or a {name: size} mapping when a test cares about sizes. An entry that is
    itself a key of *tree* is reported as a directory, as a real listing would. The first
    *ls_failures* listings of any directory fail, mimicking a gfal error that outlasts the
    retries of gfal_ls_checked. Patch both listing functions with fs.patch_ls()."""
    fs = FakeGFALFileInterface(tree, path_cache or PathCache(600))
    state = {"failures": ls_failures}

    def fake_ls(uri, **kwargs):
        if state["failures"] > 0:
            state["failures"] -= 1
            return None
        path = uri[len(BASE) :].strip("/")
        if path not in tree:
            return None
        contents = tree[path]
        sizes = contents if isinstance(contents, dict) else dict.fromkeys(contents, 0)
        return [
            Entry(
                name=name,
                size=size,
                is_dir="/".join(filter(None, (path, name))) in tree,
            )
            for name, size in sizes.items()
        ]

    def fake_ls_checked(uri, **kwargs):
        # gfal_ls_checked: None only for an absent path, a failure raises.
        if state["failures"] > 0:
            state["failures"] -= 1
            raise GfalError("gfal-ls error: 110 (Connection timed out)")
        return fake_ls(uri)

    fs.fake_ls = fake_ls
    fs.patch_ls = lambda: mock.patch.multiple(
        law_gfal, gfal_ls_safe=fake_ls, gfal_ls_checked=fake_ls_checked
    )
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
        with fs.patch_ls():
            self.assertTrue(fs.exists("data/file_0.root"))
            self.assertTrue(fs.exists("data/file_1.root"))
            self.assertFalse(fs.exists("data/file_2.root"))

    def test_listing_answers_absent_siblings_without_further_listings(self):
        tree = {"data": ["file_0.root"]}
        fs = make_interface(tree)
        with fs.patch_ls():
            n0 = GFALFileInterface.listdir_counter
            self.assertTrue(fs.exists("data/file_0.root"))
            for i in range(1, 20):
                self.assertFalse(fs.exists(f"data/file_{i}.root"))
            self.assertEqual(GFALFileInterface.listdir_counter - n0, 1)

    def test_expired_listing_is_not_used(self):
        tree = {"data": ["file_0.root"]}
        fs = make_interface(tree, path_cache=PathCache(-1))
        with fs.patch_ls():
            self.assertFalse(fs.exists("data/file_1.root"))
            n0 = GFALFileInterface.listdir_counter
            tree["data"].append("file_1.root")
            self.assertTrue(fs.exists("data/file_1.root"))
            self.assertEqual(GFALFileInterface.listdir_counter - n0, 1)

    def test_absent_directory_is_still_reported_absent(self):
        fs = make_interface({"data": []})
        with fs.patch_ls():
            self.assertFalse(fs.exists("no_such_dir/file_0.root"))
            self.assertFalse(fs.exists("data/file_0.root"))

    def test_transient_listing_failure_does_not_cache_absence(self):
        # A listing that fails answers nothing: exists() reports False for it, but caches
        # nothing, so the next lookup lists again and finds the file.
        tree = {"data": ["file_0.root"]}
        fs = make_interface(tree, ls_failures=1)
        with fs.patch_ls():
            self.assertFalse(fs.exists("data/file_0.root"))
            self.assertTrue(fs.exists("data/file_0.root"))


class TestListingSizes(unittest.TestCase):
    """`listdir_info` exists so that collecting input-file metadata costs no extra listing;
    it was added without a test, and adding it is what broke this file's fakes."""

    def test_sizes_come_from_the_listing_and_exclude_directories(self):
        tree = {
            "data": {"file_0.root": 11, "file_1.root": 22, "sub": 0},
            "data/sub": [],
        }
        fs = make_interface(tree)
        with fs.patch_ls():
            self.assertEqual(
                fs.listdir_info("data"),
                {"file_0.root": {"size": 11}, "file_1.root": {"size": 22}},
            )

    def test_sizes_reuse_the_listing_already_taken(self):
        # The whole point: an exists() or listdir() that already listed the directory must
        # leave the sizes behind, so asking for them costs no second gfal-ls.
        tree = {"data": {"file_0.root": 11}}
        fs = make_interface(tree)
        with fs.patch_ls():
            self.assertTrue(fs.exists("data/file_0.root"))
            n0 = GFALFileInterface.listdir_counter
            self.assertEqual(fs.listdir_info("data"), {"file_0.root": {"size": 11}})
            self.assertEqual(GFALFileInterface.listdir_counter - n0, 0)

    def test_a_failed_listing_yields_no_metadata(self):
        fs = make_interface({"data": {}})
        with fs.patch_ls():
            self.assertEqual(fs.listdir_info("no_such_dir"), {})


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
            with first.patch_ls():
                self.assertTrue(first.exists("data/file_0.root"))

            # A second process starts with an empty local cache; it must answer both the
            # existing and the absent path from the server, without listing anything.
            second = self._client()
            with second.patch_ls():
                n0 = GFALFileInterface.listdir_counter
                self.assertTrue(second.exists("data/file_0.root"))
                self.assertFalse(second.exists("data/file_1.root"))
                self.assertEqual(GFALFileInterface.listdir_counter - n0, 0)

    def test_file_published_after_the_listing_is_found_without_listing(self):
        with self.server.patch():
            first = self._client()
            with first.patch_ls():
                self.assertFalse(first.exists("data/file_1.root"))

            # A job creates the file and publishes it, as filecopy() does.
            self.tree["data"].append("file_1.root")
            producer = self._client()
            producer.path_cache.set(os.path.join(BASE, "data/file_1.root"), True)

            consumer = self._client()
            with consumer.patch_ls():
                n0 = GFALFileInterface.listdir_counter
                self.assertTrue(consumer.exists("data/file_1.root"))
                self.assertEqual(GFALFileInterface.listdir_counter - n0, 0)

    def test_directory_created_after_a_parent_listing_becomes_known(self):
        with self.server.patch():
            self.tree[""] = []
            first = self._client()
            with first.patch_ls():
                first.listdir("")  # the new directory does not exist yet
                self.assertFalse(first.exists("data"))

            producer = self._client()
            with producer.patch_ls():
                with mock.patch.object(
                    law_gfal, "gfal_copy_safe", lambda *a, **k: None
                ):
                    self.tree[""] = ["data"]
                    producer.filecopy("file:///tmp/x.root", "data/file_0.root")

            consumer = self._client()
            with consumer.patch_ls():
                self.assertTrue(consumer.exists("data"))
                self.assertTrue(consumer.exists("data/file_0.root"))

    def _failed_upload(self, path):
        producer = self._client()
        with producer.patch_ls():
            with mock.patch.object(
                law_gfal,
                "gfal_copy_safe",
                side_effect=GfalError(f'Unable to copy "x.root" to "{path}".'),
            ):
                with self.assertRaises(GfalError):
                    producer.filecopy("file:///tmp/x.root", path)
        return producer

    def test_a_failed_upload_leaves_a_known_target_visible(self):
        # copy_rename keeps the target in place until the rename, so a failed upload leaves
        # it intact; an "absent" published before the upload would hide it from everyone.
        target = os.path.join(BASE, "data/file_0.root")
        with self.server.patch():
            first = self._client()
            with first.patch_ls():
                self.assertTrue(first.exists("data/file_0.root"))

            producer = self._failed_upload("data/file_0.root")
            self.assertIs(self.server.entries.get(target), True)
            with producer.patch_ls():
                self.assertTrue(producer.exists("data/file_0.root"))

            other = self._client()
            with other.patch_ls():
                n0 = GFALFileInterface.listdir_counter
                self.assertTrue(other.exists("data/file_0.root"))
                self.assertEqual(GFALFileInterface.listdir_counter - n0, 0)

    def test_a_failed_upload_publishes_nothing_for_an_unknown_target(self):
        target = os.path.join(BASE, "data/file_0.root")
        with self.server.patch():
            producer = self._failed_upload("data/file_0.root")
            self.assertNotIn(target, self.server.entries)

            other = self._client()
            with other.patch_ls():
                self.assertTrue(other.exists("data/file_0.root"))
            with producer.patch_ls():
                self.assertTrue(producer.exists("data/file_0.root"))

    def test_snapshot_ships_a_marker_only_for_a_listing_taken_here(self):
        # The submit-time snapshot shipped to jobs must not claim that a partial set of
        # entries is a complete listing.
        with self.server.patch():
            lister = self._client()
            with lister.patch_ls():
                self.assertTrue(lister.exists("data/file_0.root"))
            shipped = dict(law_gfal.local_path_cache(as_fs(lister)).iter_valid())
            self.assertIn(os.path.join(BASE, "data", law_gfal.LISTING_MARKER), shipped)

            # This one only learned the marker from the server, so it may not pass it on.
            reader = self._client()
            with reader.patch_ls():
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
            with first.patch_ls():
                first.listdir("")  # marks BASE/data as existing, content unknown
            self.assertIs(self.server.entries.get(os.path.join(BASE, "data")), True)

            second = self._client()
            with second.patch_ls():
                self.assertTrue(second.exists("data/file_0.root"))
                self.assertTrue(second.exists("data/file_1.root"))
                self.assertFalse(second.exists("data/file_2.root"))


if __name__ == "__main__":
    unittest.main()
