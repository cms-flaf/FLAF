#!/usr/bin/env python3
"""A bundle whose content changed must get a different name.

A bundle's output is otherwise just a path, and law treats an existing one as complete
forever: jobs keep unpacking the code and configs of whenever it was first built and rebuild
their branch map from those, so a newly declared dataset shifts every branch index and the
jobs silently work on the wrong one.
"""

import os
import shutil
import sys
import tempfile
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.run_tools.law_customizations import BundleTask

BUNDLES = {
    "core": {"hashed": True, "patterns": ["config", "AnaProd"]},
    "soft": {"patterns": ["soft/flaf_env"]},
}


class FakeBundleTask(BundleTask):
    """BundleTask without the law machinery: only the naming and hashing are exercised."""

    @property
    def global_params(self):
        return {"bundles": BUNDLES}

    def remote_target(self, *parts):
        return "/".join(parts)


def make_task(flavour, period="Era"):
    # luigi's metaclass owns __call__, so the instance is built without it.
    task = object.__new__(FakeBundleTask)
    task.flavour = flavour
    task.version = "v1"
    task.period = period
    return task


class TestBundleHash(unittest.TestCase):
    def setUp(self):
        self.ana = tempfile.mkdtemp()
        os.environ["ANALYSIS_PATH"] = self.ana
        self.write("config/global.yaml", "datasets: [a, b]\n")
        self.write("AnaProd/tasks.py", "print('hello')\n")
        self.write("soft/flaf_env/lib/big.so", "x" * 4096)
        self.addCleanup(shutil.rmtree, self.ana)

    def write(self, rel, content):
        path = os.path.join(self.ana, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def hash_of(self, flavour="core", period="Era"):
        BundleTask._source_hash_cache.clear()
        return make_task(flavour, period).source_hash()

    def test_editing_a_packed_file_changes_the_hash(self):
        before = self.hash_of()
        self.write("config/global.yaml", "datasets: [a, b, c]\n")
        self.assertNotEqual(before, self.hash_of())

    def test_same_size_edit_is_still_noticed_through_the_timestamp(self):
        before = self.hash_of()
        path = os.path.join(self.ana, "config", "global.yaml")
        info = os.lstat(path)
        self.write("config/global.yaml", "datasets: [a, c]\n")  # same length
        self.assertEqual(os.lstat(path).st_size, info.st_size)
        self.assertNotEqual(before, self.hash_of())

    def test_hash_is_stable_when_nothing_changed(self):
        self.assertEqual(self.hash_of(), self.hash_of())

    def test_adding_and_removing_a_file_changes_the_hash(self):
        before = self.hash_of()
        added = self.write("config/Era/datasets.yaml", "x: 1\n")
        self.assertNotEqual(before, self.hash_of())
        os.remove(added)
        self.assertEqual(before, self.hash_of())

    def test_a_symlink_is_hashed_by_its_target(self):
        link = os.path.join(self.ana, "config", "link.yaml")
        os.symlink("/somewhere/a.yaml", link)
        before = self.hash_of()
        os.remove(link)
        os.symlink("/somewhere/b.yaml", link)
        self.assertNotEqual(before, self.hash_of())

    def test_the_unhashed_payload_does_not_affect_the_hashed_bundle(self):
        # The reason the environment lives in its own flavour: hashing it would mean walking
        # a large tree on every submission for something that only a reinstall changes.
        before = self.hash_of()
        self.write("soft/flaf_env/lib/big.so", "y" * 4096)
        self.assertEqual(before, self.hash_of())

    def test_only_a_hashed_flavour_carries_the_hash_in_its_name(self):
        BundleTask._source_hash_cache.clear()
        core = make_task("core").output()
        soft = make_task("soft").output()
        self.assertEqual(soft, "v1/bundles/Era/soft.tar.bz2")
        self.assertTrue(core.startswith("v1/bundles/Era/core_"), core)
        self.assertTrue(core.endswith(".tar.bz2"), core)
        self.assertEqual(len(core.split("core_")[1].split(".tar")[0]), 12)

    def test_pycache_is_ignored(self):
        before = self.hash_of()
        self.write("AnaProd/__pycache__/tasks.cpython-312.pyc", "compiled")
        self.write("AnaProd/tasks.pyc", "compiled")
        self.assertEqual(before, self.hash_of())


if __name__ == "__main__":
    unittest.main()
