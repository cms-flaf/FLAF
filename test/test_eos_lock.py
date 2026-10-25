#!/usr/bin/env python3
"""run_tools/eos_lock.py against an in-memory model of the EOS namespace.

The model enforces the EOS rules the lock depends on, as measured on cmseos (EOS 5.3.38) in
lpcflaf/debug on 2026-10-17: an immutable (`z:i`) directory refuses every change except an ACL
change; chmod is decided by the parent's ACL; only the owner or an `A` holder may set a
directory's `sys.acl`; a directory owner is exempt from `!d`; a new directory inherits its
parent's attributes. The central test interrupts a lock after every single operation and
checks that a rerun converges without losing or changing a file.
"""

import os
import posixpath
import signal
import sys
import threading
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.run_tools import eos_lock
from FLAF.run_tools.eos_lock import (
    Entry,
    LockError,
    Refused,
    acl_rights,
    is_immutable,
    lock_acl,
    parse_find,
    run_lock,
    run_status,
    run_unlock,
    split_target,
)

ME, OTHER, GID = 44535, 17345, 5063
TEMPLATE = f"g:5063:rx!d!u,u:{ME}:rwx+d+u,u:55339:rwx+d+u,u:{OTHER}:rwx+d+u"
WEAK_LOCK = "g:5063:rx!d!u"
LOCK = lock_acl(ME, [])
# the model needs no pause between retries
eos_lock.RETRY_DELAYS = (0, 0, 0, 0)
eos_lock.READ_RETRY_DELAY = 0
NOW = 2_000_000_000.0
OLD_MTIME = NOW - 86400


class Crash(Exception):
    """A lock killed in the middle (nothing after it runs, not even `finally`)."""


class Node:
    def __init__(
        self, type, uid, mode=0o755, attrs=None, size=0, checksum="", mtime=OLD_MTIME
    ):
        self.type = type
        self.uid = uid
        self.mode = mode
        self.attrs = dict(attrs or {})
        self.size = size
        self.checksum = checksum
        self.mtime = mtime


class FakeEos:
    def __init__(self):
        self.nodes = {"/eos": Node("directory", 0, attrs={"sys.acl": TEMPLATE})}
        self.ops = []
        self.crash_after = None
        self.dead = False
        self.hooks = {}
        # per rename: None, "before" (not carried out) or "after" (reply lost)
        self.mv_failures = []
        # read requests that fail: a predicate on (method, path)
        self.read_failure = None
        self.corrupt_copies = False
        self.partial_copies = False
        # copies that are carried out but report "File exists" (a client retry after a lost reply)
        self.copy_lost_replies = 0
        # the lock runs copies and renames from threads
        self.lock = threading.RLock()
        for name in (
            "find",
            "size",
            "exists",
            "checksum",
            "mkdir",
            "rmdir",
            "rm",
            "mv",
            "copy",
            "attr_set",
            "attr_rm",
            "chmod",
        ):
            setattr(self, name, self._locked(getattr(self, name)))

    def _locked(self, method):
        def call(*args, **kw):
            with self.lock:
                return method(*args, **kw)

        return call

    def _read(self, method, path):
        if self.dead:
            raise Crash("dead")
        if self.read_failure and self.read_failure(method, path):
            raise LockError(f"xrdfs {method} {path}: [FATAL] Socket timeout")

    # -- test helpers
    def add_dir(self, path, uid, acl=TEMPLATE, mode=0o755):
        self.nodes[path] = Node("directory", uid, mode, {"sys.acl": acl})

    def add_file(self, path, uid, content, mtime=OLD_MTIME, attrs=None):
        self.nodes[path] = Node(
            "file",
            uid,
            0o644,
            attrs,
            len(content),
            f"{hash(content) & 0xFFFFFFFF:08x}",
            mtime,
        )

    def content(self, root):
        return {
            posixpath.relpath(p, root): (n.size, n.checksum)
            for p, n in self.nodes.items()
            if n.type == "file" and p.startswith(root + "/")
        }

    def snapshot(self):
        return {
            p: (n.type, n.uid, n.mode, dict(n.attrs), n.size, n.checksum)
            for p, n in self.nodes.items()
        }

    # -- permission model
    def _acl(self, path):
        return self.nodes[path].attrs.get("sys.acl", "")

    def _can_write_in(self, d):
        n = self.nodes.get(d)
        if (
            n is None
            or n.type != "directory"
            or is_immutable(n.attrs.get("sys.acl", ""))
        ):
            return False
        write = acl_rights(n.attrs.get("sys.acl", ""), ME, GID)[0]
        return write or (n.uid == ME and n.mode & 0o200)

    def _can_delete_in(self, d):
        n = self.nodes[d]
        write, delete, _ = acl_rights(n.attrs.get("sys.acl", ""), ME, GID)
        return self._can_write_in(d) and (n.uid == ME or delete)

    def _op(self, name, *args):
        if self.dead:
            raise Crash("dead")
        if self.crash_after is not None and len(self.ops) >= self.crash_after:
            self.dead = True
            raise Crash(f"killed before {name} {args}")
        hook = self.hooks.get(name)
        if hook:
            hook(*args)
        self.ops.append((name,) + args)

    def _refuse(self, what):
        raise LockError(f"Operation not permitted: {what}")

    # -- the EosClient interface
    def whoami(self):
        return ME, GID

    def find(self, path, maxdepth=None):
        self._read("find", path)
        if path not in self.nodes:
            raise LockError(f"no such path {path}")
        out = []
        base = path.count("/")
        for p in sorted(self.nodes):
            if p != path and not p.startswith(path + "/"):
                continue
            if maxdepth is not None and p.count("/") - base > maxdepth:
                continue
            n = self.nodes[p]
            out.append(
                Entry(
                    p,
                    n.type,
                    n.uid,
                    n.size,
                    n.checksum,
                    n.mode | 0o40000 if n.type == "directory" else 0,
                    n.mtime,
                    dict(n.attrs),
                )
            )
        return out

    def size(self, path):
        self._read("size", path)
        n = self.nodes.get(path)
        return None if n is None else n.size

    def exists(self, path):
        self._read("exists", path)
        return path in self.nodes

    def checksum(self, path):
        self._read("checksum", path)
        return self.nodes[path].checksum

    def mkdir(self, path):
        self._op("mkdir", path)
        parent = posixpath.dirname(path)
        if path in self.nodes or not self._can_write_in(parent):
            self._refuse(f"mkdir {path}")
        p = self.nodes[parent]
        self.nodes[path] = Node("directory", ME, p.mode, p.attrs)

    def rmdir(self, path):
        self._op("rmdir", path)
        if any(p.startswith(path + "/") for p in self.nodes):
            raise LockError(f"Directory not empty: {path}")
        parent = posixpath.dirname(path)
        if not self._can_delete_in(parent):
            self._refuse(f"rmdir {path}")
        # measured 2026-10-25: owning the parent does not lift its `!d` for a directory of
        # another user ("Unable to rmdir by ACL"), as it does for files
        if self.nodes[path].uid != ME and not acl_rights(self._acl(parent), ME, GID)[1]:
            self._refuse(f"rmdir {path}: by ACL")
        del self.nodes[path]

    def rm(self, path):
        self._op("rm", path)
        if self.nodes[path].type != "file" or not self._can_delete_in(
            posixpath.dirname(path)
        ):
            self._refuse(f"rm {path}")
        del self.nodes[path]

    def mv(self, src, dst):
        self._op("mv", src, dst)
        failure = self.mv_failures.pop(0) if self.mv_failures else None
        if failure == "before":
            raise LockError(f"xrdfs mv {src} {dst}: [FATAL] Socket error")
        self._mv(src, dst)
        if failure == "after":
            raise LockError(f"xrdfs mv {src} {dst}: [FATAL] Socket timeout")

    def _mv(self, src, dst):
        n = self.nodes[src]
        sp, dp = posixpath.dirname(src), posixpath.dirname(dst)
        if not self._can_delete_in(sp) or not self._can_write_in(dp):
            self._refuse(f"mv {src} {dst}")
        if n.type == "directory":
            acl = n.attrs.get("sys.acl", "")
            if is_immutable(acl):
                self._refuse(f"mv immutable {src}")
            if n.uid != ME and not acl_rights(acl, ME, GID)[1]:
                self._refuse(f"mv {src}: no delete right on it")
            if dst in self.nodes and self.nodes[dst].type == "directory":
                dst = posixpath.join(dst, posixpath.basename(src))
            for p in sorted(
                p for p in self.nodes if p == src or p.startswith(src + "/")
            ):
                self.nodes[dst + p[len(src) :]] = self.nodes.pop(p)
        else:
            if dst in self.nodes and self.nodes[dst].type != "file":
                self._refuse(f"mv {src} onto directory {dst}")
            self.nodes[dst] = self.nodes.pop(src)

    def copy(self, src, dst):
        self._op("copy", src, dst)
        if dst in self.nodes or not self._can_write_in(posixpath.dirname(dst)):
            self._refuse(f"copy to {dst}")
        s = self.nodes[src]
        if self.partial_copies:
            self.nodes[dst] = Node(
                "file", ME, 0o644, None, s.size // 2, "0badc0de", NOW
            )
            raise LockError(f"xrdcp {src} {dst}: connection lost")
        checksum = "deadbeef" if self.corrupt_copies else s.checksum
        self.nodes[dst] = Node("file", ME, 0o644, None, s.size, checksum, NOW)
        if self.copy_lost_replies:
            self.copy_lost_replies -= 1
            raise LockError(
                f"xrdcp {src} {dst}: [3018] Unable to create file - (O_EXCL); File exists"
            )

    def _targets(self, path, recursive):
        if not recursive:
            return [path]
        return sorted(
            (
                p
                for p, n in self.nodes.items()
                if n.type == "directory" and (p == path or p.startswith(path + "/"))
            ),
            key=lambda p: p.count("/"),
        )

    def attr_set(self, path, key, value, recursive=False):
        self._op("attr_set", path, key, value, recursive)
        failed = []
        for p in self._targets(path, recursive):
            n = self.nodes[p]
            if n.uid == ME or (
                key == "sys.acl" and acl_rights(n.attrs.get("sys.acl", ""), ME, GID)[2]
            ):
                n.attrs[key] = value
            else:
                failed.append(p)
        if failed:
            self._refuse(f"attr set on {failed}")

    def attr_rm(self, path, key):
        self._op("attr_rm", path, key)
        n = self.nodes[path]
        if n.uid != ME or (n.type == "directory" and not n.mode & 0o200):
            self._refuse(f"attr rm {path}")
        n.attrs.pop(key, None)

    def chmod(self, path, mode, recursive=False):
        self._op("chmod", path, mode, recursive)
        failed = []
        for p in self._targets(path, recursive):
            parent_acl = self.nodes[posixpath.dirname(p)].attrs.get("sys.acl", "")
            if is_immutable(parent_acl) or self.nodes[p].uid != ME:
                failed.append(p)
            else:
                self.nodes[p].mode = mode
        if failed:
            self._refuse(f"chmod {failed}")


def mixed_tree():
    """/eos/t (caller's) with U: a folder rebuilt and copied at every level."""
    eos = FakeEos()
    eos.add_dir("/eos/t", ME)
    eos.add_dir("/eos/t/U", OTHER)
    eos.add_file("/eos/t/U/a.root", OTHER, "a")
    eos.add_file("/eos/t/U/b.root", ME, "b")
    eos.add_dir("/eos/t/U/mine", ME)
    eos.add_file("/eos/t/U/mine/c.root", OTHER, "c")
    eos.add_file("/eos/t/U/mine/c2.root", ME, "c2")
    eos.add_dir("/eos/t/U/theirs", OTHER)
    eos.add_file("/eos/t/U/theirs/d.root", OTHER, "d")
    eos.add_dir("/eos/t/U/theirs/deep", OTHER)
    eos.add_file("/eos/t/U/theirs/deep/e.root", ME, "e")
    eos.add_dir("/eos/t/U/theirs/empty", OTHER)
    return eos


def lock(eos, path="/eos/t/U", **kw):
    kw.setdefault("admins", [])
    kw.setdefault("apply", True)
    kw.setdefault("min_age", 60)
    kw.setdefault("jobs", 1)
    run_lock(eos, path, now=NOW, **kw)


class LockedState:
    def assert_locked(self, eos, root, acl=LOCK):
        self.assertEqual(eos_lock.lock_problems(eos_lock.scan(eos, root), ME, acl), [])
        for p, n in eos.nodes.items():
            self.assertFalse(posixpath.basename(p).endswith(eos_lock.LOCK_SUFFIXES), p)


class TestAcl(unittest.TestCase):
    def test_immutable_needs_z_i_and_no_negation(self):
        self.assertTrue(is_immutable(LOCK))
        self.assertTrue(is_immutable("g:5063:rx,z:i"))
        self.assertTrue(is_immutable("z:i"))
        for acl in (
            TEMPLATE,
            WEAK_LOCK,
            eos_lock.staging_acl(ME),
            "",
            "u:1:i",
            "z:!i",
            "g:5063:rx,z:i,u:7:!i",
        ):
            self.assertFalse(is_immutable(acl), acl)

    def test_rights(self):
        self.assertEqual(acl_rights(TEMPLATE, ME, GID), (True, True, False))
        self.assertEqual(acl_rights(WEAK_LOCK, ME, GID), (False, False, False))
        self.assertEqual(acl_rights(LOCK, ME, GID), (False, False, True))
        self.assertEqual(acl_rights(f"u:{ME}:rwx!d", ME, GID), (True, False, False))
        self.assertEqual(acl_rights(f"u:{ME}:rwo", ME, GID), (False, False, False))
        self.assertEqual(acl_rights(TEMPLATE, 999, 999), (False, False, False))

    def test_lock_acl(self):
        self.assertEqual(lock_acl(ME, []), f"g:5063:rx,u:{ME}:A,z:i")
        self.assertEqual(lock_acl(ME, [7121, ME]), f"g:5063:rx,u:{ME}:A,u:7121:A,z:i")
        self.assertNotIn("!d", lock_acl(ME, [7121]))


class TestParsing(unittest.TestCase):
    # the line layout of mgm/proc/user/NewfindCmd.cc (EOS 5.3.38) for --format FIND_FORMAT
    SAMPLE = (
        'path="/eos/x/U/" type=directory  uid=44535 mode=42755 mtime=1760000000.000000001 '
        'attr.sys.acl="g:5063:rx!d!u,u:44535:rwx+d+u" attr.sys.forced.layout="replica" attr.user.acl=""\n'
        'path="/eos/x/U/f 1.root" type=file  size=12 uid=17345 checksum=0a1b2c3d mtime=1760000001.5 '
        'attr.sys.acl="u:17345:rwx,z:!i"\n'
        'path="/eos/x/U/l" type=symlink  uid=17345 mtime=1760000002.0\n'
        "warning: ignored line\n"
    )

    def test_parse_find(self):
        d, f, l = parse_find(self.SAMPLE)
        self.assertEqual(
            (d.path, d.type, d.uid, d.mode & 0o777),
            ("/eos/x/U", "directory", ME, 0o755),
        )
        self.assertEqual(d.acl, "g:5063:rx!d!u,u:44535:rwx+d+u")
        self.assertEqual(d.attrs["user.acl"], "")
        self.assertEqual(
            (f.path, f.size, f.uid, f.checksum),
            ("/eos/x/U/f 1.root", 12, OTHER, "0a1b2c3d"),
        )
        self.assertAlmostEqual(f.mtime, 1760000001.5)
        self.assertEqual(f.attrs["sys.acl"], "u:17345:rwx,z:!i")
        self.assertEqual(l.type, "symlink")

    def test_parse_find_refuses_missing_fields(self):
        # a missing mtime would silently switch off the --min-age guard
        for line in (
            'path="/eos/x/f" type=file  size=1 uid=1 checksum=00000001\n',
            'path="/eos/x/f" type=file  size=1 checksum=00000001 mtime=1.0\n',
            'path="/eos/x/f" type=file  uid=1 checksum=00000001 mtime=1.0\n',
            'path="/eos/x/f" type=file  size=1 uid=1 mtime=1.0\n',
            'path="/eos/x/d/" type=directory  uid=1 mtime=1.0\n',
        ):
            with self.assertRaises(LockError, msg=line):
                parse_find(line)

    def test_split_target(self):
        self.assertEqual(
            split_target(
                "root://cmseos.fnal.gov//eos/uscms/store/user/x/", "root://other"
            ),
            ("root://cmseos.fnal.gov", "/eos/uscms/store/user/x"),
        )
        self.assertEqual(
            split_target("/eos/a/b/", "root://m"), ("root://m", "/eos/a/b")
        )
        with self.assertRaises(Refused):
            split_target("/store/user/x", "root://m")

    def test_logical_paths_of_leftovers(self):
        root = "/eos/t/U"
        cases = {
            "/eos/t/U.flaf-lock-new/a": "a",
            "/eos/t/U.flaf-lock-new/.flaf-lock-old": "",
            "/eos/t/U.flaf-lock-new/.flaf-lock-old/s/x.root": "s/x.root",
            "/eos/t/U/s/.flaf-lock-old/x.root": "s/x.root",
            "/eos/t/U/s.flaf-lock-new/x.root": "s/x.root",
            "/eos/t/U/s/x.root.flaf-lock-tmp": "s/x.root",
        }
        for path, logical in cases.items():
            self.assertEqual(eos_lock.logical_path(path, root), logical, path)


class TestLock(unittest.TestCase, LockedState):
    def test_mixed_ownership(self):
        eos = mixed_tree()
        before = eos.content("/eos/t/U")
        lock(eos)
        self.assert_locked(eos, "/eos/t/U")
        self.assertEqual(eos.content("/eos/t/U"), before)
        for p, n in eos.nodes.items():
            if p.startswith("/eos/t/U"):
                self.assertEqual(n.uid, ME, p)
        self.assertEqual(eos.nodes["/eos/t"].attrs["sys.acl"], TEMPLATE)
        self.assertNotIn("/eos/t/U" + eos_lock.RUN_MARKER_SUFFIX, eos.nodes)

    def test_lock_refuses_everything_afterwards(self):
        eos = mixed_tree()
        lock(eos)
        with self.assertRaises(LockError):
            eos.rm("/eos/t/U/a.root")
        with self.assertRaises(LockError):
            eos.copy("/eos/t/U/a.root", "/eos/t/U/new.root")
        with self.assertRaises(LockError):
            eos.mv("/eos/t/U/theirs/d.root", "/eos/t/d.root")
        with self.assertRaises(LockError):
            eos.mv("/eos/t/U/theirs", "/eos/t/theirs")
        with self.assertRaises(LockError):
            eos.mkdir("/eos/t/U/new")

    def test_dry_run_changes_nothing(self):
        eos = mixed_tree()
        snap = eos.snapshot()
        lock(eos, apply=False)
        self.assertEqual(eos.snapshot(), snap)
        self.assertEqual(eos.ops, [])

    def test_second_run_is_a_noop(self):
        eos = mixed_tree()
        lock(eos)
        eos.ops.clear()
        lock(eos)
        self.assertEqual(eos.ops, [])

    def test_admin_change_relocks(self):
        eos = mixed_tree()
        lock(eos)
        lock(eos, admins=[7121])
        self.assert_locked(eos, "/eos/t/U", lock_acl(ME, [7121]))

    def test_folder_owned_by_other_under_weak_lock_of_caller(self):
        # the PProd shape: the caller's directory carries g:5063:rx!d!u, a child is somebody else's
        eos = mixed_tree()
        eos.nodes["/eos/t"].attrs["sys.acl"] = WEAK_LOCK
        before = eos.content("/eos/t/U")
        lock(eos)
        self.assert_locked(eos, "/eos/t/U")
        self.assertEqual(eos.content("/eos/t/U"), before)

    def test_file_level_acl_is_removed(self):
        eos = mixed_tree()
        eos.nodes["/eos/t/U/b.root"].attrs["sys.acl"] = f"u:{ME}:rwx+u,z:!i"
        eos.nodes["/eos/t/U/a.root"].attrs["sys.acl"] = f"u:{OTHER}:rwx+u,z:!i"
        lock(eos)
        self.assert_locked(eos, "/eos/t/U")

    def test_subfolder_of_a_locked_folder_is_opened_and_relocked(self):
        eos = mixed_tree()
        lock(eos, path="/eos/t/U/theirs")
        eos.add_file("/eos/t/U/late.root", OTHER, "late")
        before = eos.content("/eos/t/U")
        lock(eos)
        self.assert_locked(eos, "/eos/t/U")
        self.assertEqual(eos.content("/eos/t/U"), before)

    def test_corrupt_copy_keeps_the_original(self):
        eos = mixed_tree()
        eos.corrupt_copies = True
        before = eos.content("/eos/t/U")
        with self.assertRaises(LockError):
            lock(eos)
        self.assertEqual(eos.content("/eos/t/U"), before)
        self.assertFalse(any(n.checksum == "deadbeef" for n in eos.nodes.values()))

    def test_file_written_during_rebuild_is_kept(self):
        eos = mixed_tree()

        def late_write(src, dst):
            if src == "/eos/t/U" and dst.endswith(eos_lock.OLD_DIR_NAME):
                eos.add_file("/eos/t/U/late.root", OTHER, "late")

        eos.hooks["mv"] = late_write
        before = eos.content("/eos/t/U")
        lock(eos)
        self.assert_locked(eos, "/eos/t/U")
        after = eos.content("/eos/t/U")
        self.assertEqual(
            after.pop("late.root"), (4, f"{hash('late') & 0xFFFFFFFF:08x}")
        )
        self.assertEqual(after, before)

    def test_file_changed_during_rebuild_is_not_deleted(self):
        eos = mixed_tree()

        def change(src, dst):
            if src == "/eos/t/U" and dst.endswith(eos_lock.OLD_DIR_NAME):
                # same size: only the checksum tells the two versions apart
                eos.add_file("/eos/t/U/a.root", OTHER, "A")

        eos.hooks["mv"] = change
        with self.assertRaises(LockError):
            lock(eos)
        old = "/eos/t/U/" + eos_lock.OLD_DIR_NAME
        self.assertEqual(
            eos.nodes[old + "/a.root"].checksum, f"{hash('A') & 0xFFFFFFFF:08x}"
        )
        self.assertEqual(
            eos.nodes["/eos/t/U/a.root"].checksum, f"{hash('a') & 0xFFFFFFFF:08x}"
        )


def files_in_place(eos, root, expected):
    """The paths of `expected` that do not hold their original content."""
    bad = []
    for rel, (size, checksum) in expected.items():
        n = eos.nodes.get(posixpath.join(root, rel))
        if n is None or n.type != "file" or (n.size, n.checksum) != (size, checksum):
            bad.append(rel)
    return bad


class StopAfter:
    """A stop request (Ctrl-C, SIGTERM, SIGHUP) that arrives after n operations."""

    def __init__(self, eos, n):
        self.eos, self.n = eos, n

    @property
    def requested(self):
        return len(self.eos.ops) >= self.n

    def check(self):
        if self.requested:
            raise eos_lock.Interrupted("stop requested")


class TestResume(unittest.TestCase, LockedState):
    def test_stopped_before_every_operation_files_stay_in_place(self):
        for jobs in (1, 3):
            reference = mixed_tree()
            expected = reference.content("/eos/t/U")
            lock(reference, jobs=jobs)
            for n in range(len(reference.ops) + 1):
                with self.subTest(jobs=jobs, stop_requested_after=n):
                    eos = mixed_tree()
                    try:
                        lock(eos, jobs=jobs, stop=StopAfter(eos, n))
                    except eos_lock.Interrupted:
                        pass
                    # every original file at its path with its content, the marker gone
                    self.assertEqual(files_in_place(eos, "/eos/t/U", expected), [])
                    self.assertNotIn("/eos/t/U" + eos_lock.RUN_MARKER_SUFFIX, eos.nodes)
                    # its own copies are not taken for a production still writing
                    lock(eos, jobs=jobs)
                    self.assert_locked(eos, "/eos/t/U")
                    self.assertEqual(eos.content("/eos/t/U"), expected)

    def test_killed_after_every_operation_then_rerun(self):
        for jobs in (1, 3):
            reference = mixed_tree()
            expected = reference.content("/eos/t/U")
            lock(reference, jobs=jobs)
            total = len(reference.ops)
            self.assertGreater(total, 20)
            for n in range(total):
                with self.subTest(jobs=jobs, killed_after=n):
                    eos = mixed_tree()
                    eos.crash_after = n
                    with self.assertRaises(Crash):
                        lock(eos, jobs=jobs)
                    missing = files_in_place(eos, "/eos/t/U", expected)
                    if missing:
                        # only a kill inside a swap, after one of its renames, leaves a path missing
                        self.assertEqual(eos.ops[-1][0], "mv", missing)
                    eos.dead = False
                    eos.crash_after = None
                    # the user removes the marker of the killed run, as the refusal says
                    eos.nodes.pop("/eos/t/U" + eos_lock.RUN_MARKER_SUFFIX, None)
                    # a rerun stopped at its first safe point has put every path back
                    try:
                        lock(eos, jobs=jobs, stop=StopAfter(eos, 0))
                    except eos_lock.Interrupted:
                        pass
                    self.assertEqual(files_in_place(eos, "/eos/t/U", expected), [])
                    lock(eos, jobs=jobs)
                    self.assert_locked(eos, "/eos/t/U")
                    self.assertEqual(eos.content("/eos/t/U"), expected)

    def test_renames_are_retried_and_lost_replies_accepted(self):
        for failures in (["before"], ["after"], ["before", "after", "before"]):
            with self.subTest(failures=failures):
                eos = mixed_tree()
                expected = eos.content("/eos/t/U")
                eos.mv_failures = list(failures)
                lock(eos)
                self.assert_locked(eos, "/eos/t/U")
                self.assertEqual(eos.content("/eos/t/U"), expected)

    def test_storage_failing_inside_a_swap(self):
        swapped = "/eos/t/U.flaf-lock-new/.flaf-lock-old"
        for failures in (1, len(eos_lock.RETRY_DELAYS) + 1):
            with self.subTest(failing_requests=failures):
                eos = mixed_tree()
                expected = eos.content("/eos/t/U")
                left = [failures]

                def after_first_rename(method, path):
                    # every read fails for a while once the original went into the new directory
                    if swapped in eos.nodes and left[0] > 0:
                        left[0] -= 1
                        return True
                    return False

                eos.read_failure = after_first_rename
                if failures == 1:
                    lock(eos)
                    self.assert_locked(eos, "/eos/t/U")
                    continue
                with self.assertRaises(LockError) as caught:
                    lock(eos)
                self.assertIn(eos_lock.OLD_DIR_NAME, str(caught.exception))
                self.assertNotIn("/eos/t/U" + eos_lock.RUN_MARKER_SUFFIX, eos.nodes)
                eos.read_failure = None
                lock(eos)
                self.assertEqual(eos.content("/eos/t/U"), expected)

    def test_marker_removal_does_not_hide_the_error(self):
        eos = mixed_tree()
        eos.partial_copies = True
        real_rmdir = eos.rmdir

        def rmdir(path):
            if path.endswith(eos_lock.RUN_MARKER_SUFFIX):
                raise LockError("[FATAL] Auth failed")
            return real_rmdir(path)

        eos.rmdir = rmdir
        with self.assertRaises(LockError) as caught:
            lock(eos)
        self.assertIn("connection lost", str(caught.exception))

    def test_sigint_inside_a_swap_stops_after_it(self):
        eos = mixed_tree()
        expected = eos.content("/eos/t/U")
        real_mv = eos.mv

        def mv(src, dst):
            real_mv(src, dst)
            if dst.endswith(eos_lock.OLD_DIR_NAME):
                os.kill(os.getpid(), signal.SIGINT)

        eos.mv = mv
        with eos_lock.StopSignals() as stop:
            with self.assertRaises(eos_lock.Interrupted):
                lock(eos, stop=stop)
        self.assertEqual(files_in_place(eos, "/eos/t/U", expected), [])
        self.assertEqual(signal.getsignal(signal.SIGINT), signal.default_int_handler)
        # stopped right after the swap: nothing deleted after its last rename
        last_mv = max(i for i, op in enumerate(eos.ops) if op[0] == "mv")
        self.assertFalse(
            [op for op in eos.ops[last_mv:] if op[0] in ("rm", "rmdir")][1:]
        )
        self.assertIn("/eos/t/U/" + eos_lock.OLD_DIR_NAME, eos.nodes)
        eos.mv = real_mv
        lock(eos)
        self.assertEqual(eos.content("/eos/t/U"), expected)

    def test_stop_during_copies_does_not_start_the_swap(self):
        eos = mixed_tree()
        expected = eos.content("/eos/t/U")

        class StopAfterFirstCopy(eos_lock.NoStop):
            requested = False

            def check(self):
                if self.requested:
                    raise eos_lock.Interrupted("stop requested")

        stop = StopAfterFirstCopy()
        real_copy = eos.copy

        new = "/eos/t/U" + eos_lock.NEW_DIR_SUFFIX

        def copy(src, dst):
            real_copy(src, dst)
            # after the last copy of the folder's own files: only the check before the swap is left
            if {new + "/a.root", new + "/b.root"} <= set(eos.nodes):
                stop.requested = True

        eos.copy = copy
        with self.assertRaises(eos_lock.Interrupted):
            lock(eos, stop=stop)
        self.assertEqual([op for op in eos.ops if op[0] == "mv"], [])
        self.assertEqual(files_in_place(eos, "/eos/t/U", expected), [])
        eos.copy = real_copy
        lock(eos)
        self.assertEqual(eos.content("/eos/t/U"), expected)

    def test_nohup_is_kept(self):
        previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            with eos_lock.StopSignals():
                self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
        finally:
            signal.signal(signal.SIGHUP, previous)

    def test_inherited_ignored_sigint_is_handled(self):
        # how a non-interactive shell starts a background job (`lock ... &` in a script)
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with eos_lock.StopSignals() as stop:
                os.kill(os.getpid(), signal.SIGINT)
                self.assertTrue(stop.requested)
        finally:
            signal.signal(signal.SIGINT, previous)

    def test_rename_failing_for_good_stops_the_lock(self):
        eos = mixed_tree()
        eos.mv_failures = ["before"] * (len(eos_lock.RETRY_DELAYS) + 1)
        with self.assertRaises(LockError):
            lock(eos)

    def test_marker_of_a_killed_run_refuses(self):
        eos = mixed_tree()
        eos.crash_after = 5
        with self.assertRaises(Crash):
            lock(eos)
        eos.dead, eos.crash_after = False, None
        with self.assertRaises(Refused):
            lock(eos, min_age=0)


class TestRefusals(unittest.TestCase):
    def refused(self, eos, path="/eos/t/U", **kw):
        snap = eos.snapshot()
        with self.assertRaises(Refused):
            lock(eos, path, **kw)
        self.assertEqual(eos.snapshot(), snap)

    def test_upload_in_flight(self):
        for name in ("x.root.flaf-tmp-12-ab", ".dsprod-tmp-x", ".sys.a#.x"):
            eos = mixed_tree()
            eos.add_file(f"/eos/t/U/{name}", OTHER, "partial")
            self.refused(eos)

    def test_recent_file(self):
        eos = mixed_tree()
        eos.add_file("/eos/t/U/fresh.root", OTHER, "fresh", mtime=NOW - 60)
        self.refused(eos)
        lock(eos, min_age=0)

    def test_symlink(self):
        eos = mixed_tree()
        eos.nodes["/eos/t/U/link"] = Node("symlink", OTHER)
        self.refused(eos)

    def test_locked_parent(self):
        eos = mixed_tree()
        eos.nodes["/eos/t"].attrs["sys.acl"] = LOCK
        self.refused(eos)

    def test_foreign_directory_without_member_rights(self):
        eos = mixed_tree()
        eos.nodes["/eos/t/U/theirs"].attrs["sys.acl"] = WEAK_LOCK
        self.refused(eos)

    def test_foreign_directory_locked_by_its_owner(self):
        eos = mixed_tree()
        eos.nodes["/eos/t/U/theirs"].attrs["sys.acl"] = f"g:5063:rx,u:{OTHER}:A,z:i"
        self.refused(eos)

    def test_foreign_locked_directory_with_admin_rule_is_taken(self):
        eos = mixed_tree()
        eos.nodes["/eos/t/U/theirs"].attrs[
            "sys.acl"
        ] = f"g:5063:rx,u:{OTHER}:A,u:{ME}:A,z:i"
        eos.nodes["/eos/t/U/theirs"].mode = 0o555
        before = eos.content("/eos/t/U")
        lock(eos)
        self.assertEqual(eos.content("/eos/t/U"), before)

    def test_foreign_folder_in_foreign_parent_without_rights(self):
        eos = mixed_tree()
        eos.nodes["/eos/t"].uid = OTHER
        eos.nodes["/eos/t"].attrs["sys.acl"] = "g:5063:rx"
        self.refused(eos)

    def test_inconsistent_leftovers(self):
        eos = mixed_tree()
        eos.add_dir("/eos/t/U/mine" + eos_lock.NEW_DIR_SUFFIX, ME)
        self.refused(eos)

    def test_orphan_tmp_copy(self):
        eos = mixed_tree()
        eos.add_file("/eos/t/U/gone.root" + eos_lock.TMP_FILE_SUFFIX, ME, "x")
        self.refused(eos)

    def test_symlinked_path(self):
        eos = mixed_tree()
        eos.find = lambda path, maxdepth=None: [Entry("/eos/real/U", "directory", ME)]
        with self.assertRaises(Refused):
            eos_lock.scan(eos, "/eos/t/U")


class TestUnlock(unittest.TestCase):
    def test_unlock_restores_the_parent_acl(self):
        eos = mixed_tree()
        lock(eos)
        run_unlock(eos, "/eos/t/U", None, True)
        for p, n in eos.nodes.items():
            if p.startswith("/eos/t/U") and n.type == "directory":
                self.assertEqual((n.attrs["sys.acl"], n.mode), (TEMPLATE, 0o755), p)
        eos.rm("/eos/t/U/a.root")

    def test_unlock_refuses_directories_of_others(self):
        eos = mixed_tree()
        with self.assertRaises(Refused):
            run_unlock(eos, "/eos/t/U", None, True)

    def test_unlock_refuses_below_a_locked_parent(self):
        eos = mixed_tree()
        lock(eos)
        with self.assertRaises(Refused):
            run_unlock(eos, "/eos/t/U/theirs", None, True)

    def test_status_runs(self):
        eos = mixed_tree()
        run_status(eos, "/eos/t/U", [])
        lock(eos)
        run_status(eos, "/eos/t/U", [])


class TestCopiesAndRaces(unittest.TestCase, LockedState):
    def test_copy_with_a_lost_reply_is_kept(self):
        eos = mixed_tree()
        expected = eos.content("/eos/t/U")
        eos.copy_lost_replies = 3
        lock(eos)
        self.assert_locked(eos, "/eos/t/U")
        self.assertEqual(eos.content("/eos/t/U"), expected)

    def test_failed_copy_leaves_nothing_behind(self):
        eos = mixed_tree()
        eos.partial_copies = True
        before = eos.content("/eos/t/U")
        with self.assertRaises(LockError):
            lock(eos)
        self.assertEqual(eos.content("/eos/t/U"), before)
        self.assertFalse(any(n.checksum == "0badc0de" for n in eos.nodes.values()))

    def test_partial_copy_of_a_deleted_original_is_not_promoted(self):
        reference = mixed_tree()
        lock(reference)
        first_copy = next(i for i, op in enumerate(reference.ops) if op[0] == "copy")
        eos = mixed_tree()
        eos.crash_after = (
            first_copy + 1
        )  # killed right after the first copy into the new directory
        with self.assertRaises(Crash):
            lock(eos)
        eos.dead, eos.crash_after = False, None
        eos.nodes.pop("/eos/t/U" + eos_lock.RUN_MARKER_SUFFIX, None)
        new = "/eos/t/U" + eos_lock.NEW_DIR_SUFFIX
        copies = [p for p in eos.nodes if p.startswith(new + "/")]
        self.assertTrue(copies)
        # the copy is truncated and its original then deleted by somebody
        partial = copies[0]
        eos.nodes[partial].size //= 2
        eos.nodes[partial].checksum = "0badc0de"
        del eos.nodes["/eos/t/U/" + posixpath.basename(partial)]
        with self.assertRaises(LockError):
            lock(eos, min_age=0)
        self.assertEqual(eos.nodes[partial].checksum, "0badc0de")
        self.assertNotIn("0badc0de", {c for _, c in eos.content("/eos/t/U").values()})

    def test_folder_recreated_between_the_renames(self):
        eos = mixed_tree()
        before = eos.content("/eos/t/U")
        real_mv = eos.mv

        def mv(src, dst):
            real_mv(src, dst)
            if src == "/eos/t/U" and dst.endswith(eos_lock.OLD_DIR_NAME):
                eos.add_dir(
                    "/eos/t/U", OTHER
                )  # somebody recreates the folder right now

        eos.mv = mv
        with self.assertRaises(LockError):
            lock(eos)
        # the rebuilt directory was not moved into the recreated one, nothing was lost
        new = "/eos/t/U" + eos_lock.NEW_DIR_SUFFIX
        self.assertEqual(eos.content(new + "/" + eos_lock.OLD_DIR_NAME), before)
        self.assertFalse(any(p.startswith("/eos/t/U/U") for p in eos.nodes))

    def test_attributes_of_a_rebuilt_directory_are_kept(self):
        eos = mixed_tree()
        eos.nodes["/eos/t/U/theirs"].attrs["sys.forced.layout"] = "raid6"
        lock(eos)
        self.assert_locked(eos, "/eos/t/U")
        self.assertEqual(
            eos.nodes["/eos/t/U/theirs"].attrs["sys.forced.layout"], "raid6"
        )


class TestPlan(unittest.TestCase):
    def test_volume_peak_and_swap(self):
        eos = mixed_tree()
        folder = eos_lock.scan(eos, "/eos/t/U")
        n, volume, peak, swap = eos_lock.plan(folder, ME, jobs=4)
        # rebuilt U (a, b), theirs (d), deep (e); c of another owner in the caller's mine
        self.assertEqual(n, 5)
        self.assertEqual(volume, 1 + 1 + 1 + 1 + 1)
        # U's two files are copied before either original is removed
        self.assertEqual(peak, 2)
        # U has two subdirectories, theirs two
        self.assertEqual(swap, 2)


class TestOriginalChangedBeforeReplacement(unittest.TestCase):
    def test_original_rewritten_after_its_copy_is_kept(self):
        eos = mixed_tree()
        real_copy = eos.copy
        original = "/eos/t/U/mine/c.root"

        def copy(src, dst):
            real_copy(src, dst)
            if src == original and dst.endswith(eos_lock.TMP_FILE_SUFFIX):
                # same size, new content, between the copy and its rename onto the original
                eos.nodes[original].checksum = "c0ffee00"

        eos.copy = copy
        with self.assertRaises(LockError):
            lock(eos)
        self.assertEqual(eos.nodes[original].checksum, "c0ffee00")
        self.assertNotIn(original + eos_lock.TMP_FILE_SUFFIX, eos.nodes)


class TestNestedRuns(unittest.TestCase):
    def test_marker_of_a_subfolder_lock_refuses(self):
        eos = mixed_tree()
        eos.add_dir("/eos/t/U/theirs" + eos_lock.RUN_MARKER_SUFFIX, ME)
        with self.assertRaises(Refused):
            lock(eos)

    def test_marker_of_an_enclosing_lock_refuses(self):
        eos = mixed_tree()
        eos.add_dir("/eos/t/U" + eos_lock.RUN_MARKER_SUFFIX, ME)
        with self.assertRaises(Refused):
            lock(eos, path="/eos/t/U/theirs")

    def test_sticky_ownership_of_another_user_refuses(self):
        eos = mixed_tree()
        eos.nodes["/eos/t/U/theirs"].attrs["sys.owner.auth"] = "*"
        with self.assertRaises(Refused):
            lock(eos)


class TestClient(unittest.TestCase):
    """EosClient on outputs recorded from the eos, xrdfs and xrdcp clients on lxplus."""

    def client(self, *answers):
        client = eos_lock.EosClient("root://cmseos.fnal.gov")
        calls = []

        def run(cmd):
            calls.append(cmd)
            return answers[len(calls) - 1]

        client._run = run
        client.calls = calls
        return client

    def test_whoami(self):
        c = self.client(
            (
                0,
                "Virtual Identity: uid=44535 (44535,65534) gid=5063 (5063,65534) [authz:gsi]\n",
                "",
            )
        )
        self.assertEqual(c.whoami(), (44535, 5063))

    def test_stat(self):
        absent = (
            54,
            "",
            "[ERROR] Server responded with an error: [3011] Unable to stat /eos/x; No such file or directory\n",
        )
        failed = (51, "", "[FATAL] Auth failed: No protocols left to try\n")
        present = (0, "Path:   /eos/x\nId:     1\nSize:   12\nMTime:  2026-10-25\n", "")
        c = self.client(absent, present, *[failed] * eos_lock.READ_ATTEMPTS)
        self.assertIsNone(c.size("/eos/x"))
        self.assertEqual(c.size("/eos/x"), 12)
        with self.assertRaises(LockError):
            c.exists("/eos/x")

    def test_find_limit_refuses(self):
        warning = "warning(7): find results are limited for you to 50000 directories and 100000 files.\n"
        c = self.client(
            (0, 'path="/eos/x/" type=directory  uid=1 mode=40755 mtime=1.0\n', warning)
        )
        with self.assertRaises(Refused):
            c.find("/eos/x")

    def test_partial_recursive_change_is_an_error(self):
        out = "success: mode of file/directory /eos/x/ is now '2555'\n"
        err = "error: unable to chmod of directory /eos/x/y (errc=1) (Operation not permitted)\n"
        for answer in (
            (0, out, err),
            (1, out, err),
            (0, out, "warning(7): find results are limited ...\n"),
        ):
            with self.assertRaises(LockError):
                self.client(answer).chmod("/eos/x", 0o555, recursive=True)
        self.client((0, out, "")).chmod("/eos/x", 0o555, recursive=True)

    def test_reads_are_retried_without_an_answer(self):
        expired = (52, "", "[ERROR] Operation expired\n")
        present = (0, "Path:   /eos/x\nSize:   12\n", "")
        absent = (
            54,
            "",
            "[ERROR] Server responded with an error: [3011] Unable to stat /eos/x; No such file or directory\n",
        )
        self.assertEqual(self.client(expired, present).size("/eos/x"), 12)
        # a definitive answer is not retried
        c = self.client(absent, present)
        self.assertIsNone(c.size("/eos/x"))
        self.assertEqual(len(c.calls), 1)
        with self.assertRaises(LockError):
            self.client(*[expired] * eos_lock.READ_ATTEMPTS).size("/eos/x")

    def test_checksum(self):
        self.assertEqual(
            self.client((0, "adler32 0A1B2C3D\n", "")).checksum("/eos/x"), "0a1b2c3d"
        )

    def test_commands(self):
        c = self.client(*[(0, "", "")] * 4)
        c.copy("/eos/a", "/eos/b")
        c.mv("/eos/a", "/eos/b")
        c.attr_set("/eos/a", "sys.acl", "z:i", recursive=True)
        c.chmod("/eos/a", 0o555)
        self.assertEqual(
            c.calls,
            [
                [
                    "xrdcp",
                    "--nopbar",
                    "--tpc",
                    "first",
                    "root://cmseos.fnal.gov//eos/a",
                    "root://cmseos.fnal.gov//eos/b",
                ],
                ["xrdfs", "cmseos.fnal.gov", "mv", "/eos/a", "/eos/b"],
                [
                    "eos",
                    "root://cmseos.fnal.gov",
                    "attr",
                    "-r",
                    "set",
                    "sys.acl=z:i",
                    "/eos/a",
                ],
                ["eos", "root://cmseos.fnal.gov", "chmod", "555", "/eos/a"],
            ],
        )

    def test_jobs_must_be_positive(self):
        with self.assertRaises(SystemExit):
            eos_lock.main(["lock", "/eos/x", "--jobs", "0"])


if __name__ == "__main__":
    unittest.main()
