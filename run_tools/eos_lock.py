#!/usr/bin/env python3
"""Lock a finished production folder on EOS against accidental deletion and overwrite.

A locked folder belongs, directories and files, to the user who locked it; every directory
has mode 0555 and the ACL `g:5063:rx,u:<locker>:A[,u:<admin>:A...],z:i`. The `z:i` rule makes
each directory immutable for everybody, its owner included, over xrootd, davs and the eos CLI:
no file can be created, overwritten, updated, deleted or renamed, no directory created or
removed, and reading is unaffected. The owner, or a user named in an `A` rule, lifts the lock
again with `unlock`.

The lock takes ownership first, because ownership is what lets somebody undo it: the owner of
a directory may change its ACL, and the owner of a file may give that file an ACL of its own
that bypasses the directory's. Regular users cannot chown on EOS, but the lpcflaf member ACL
(`rwx+d+u`) lets a member create, rename and delete in directories owned by others, so:
- a file owned by somebody else is copied next to itself (third-party copy, adler32 checked)
  and the copy renamed onto the original;
- a directory owned by somebody else is rebuilt: a new directory created next to it receives
  copies of all its files, the original is moved into the new one and the new one renamed into
  its place, then the subdirectories are moved out of the original, which is removed. The
  original never has to be removed from the parent, where a `!d` rule would forbid it even to
  the parent's owner. An original file is deleted only once a copy with the same size and
  checksum is in place.
Copies cost I/O and, while one directory is rebuilt, quota for its files.

Interruptions. Every file stays at its path with its content: a copy replaces its original by
one rename, and a reader that has the original open keeps reading it. The exception is the
swap of a rebuilt directory: between its two renames the directory itself, files included, is
missing from its path for about two requests (longer while the storage does not answer), then
its subdirectories until they are moved back, MOVE_JOBS at a time. Ctrl-C, Ctrl-Z, SIGTERM
and SIGHUP never stop the lock there, but at the next point where every file is in place (exit
code 3). A swap is retried as a whole; if the storage does not answer within the retries, the
error names where the missing entries are, and the next run puts them back first. Only a hard
kill (SIGKILL, a crashed node) inside a swap leaves paths missing without that message. Ending
an lxplus session sends SIGTERM to the lock and its clients and SIGKILL 90 s later. A rerun
resumes any interrupted lock.

Not stopped by the lock: renaming the folder's (open) parent, which moves the locked folder
as a whole; a deliberate unlock by the owner or an `A` holder; root. Never edit a locked
folder's ACL with `eos acl` (it silently drops `z:i`) or remove it with `eos attr rm sys.acl`.

Usage:
  eos_lock.py status <folder>
  eos_lock.py lock   <folder> [--admin UID ...] [--apply]
  eos_lock.py unlock <folder> [--acl ACL] [--apply]

<folder> is an EOS path (/eos/...) on --mgm (default root://cmseos.fnal.gov) or a
root://<host>//eos/... URL. Without --apply only the plan is printed. An interrupted lock
leaves `*.flaf-lock-*` entries behind, and a hard kill also the `<folder>.flaf-lock-running`
marker; remove the marker if no lock runs, and rerun the same command. Needs the eos, xrdfs and xrdcp clients and a valid grid
proxy.
"""

import argparse
import os
import posixpath
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MGM = "root://cmseos.fnal.gov"

#: read and browse for every us_cms user at FNAL, the read rule of the lpcflaf member template
READ_RULE = "g:5063:rx"
LOCKED_MODE = 0o555
OPEN_MODE = 0o755

#: temporaries of an interrupted lock, from which a rerun resumes
NEW_DIR_SUFFIX = ".flaf-lock-new"
#: name of the original directory inside the one rebuilt to replace it
OLD_DIR_NAME = ".flaf-lock-old"
TMP_FILE_SUFFIX = ".flaf-lock-tmp"
RUN_MARKER_SUFFIX = ".flaf-lock-running"
LOCK_SUFFIXES = (NEW_DIR_SUFFIX, OLD_DIR_NAME, TMP_FILE_SUFFIX, RUN_MARKER_SUFFIX)

#: name fragments of uploads still in flight: the FLAF and DSProd copy-then-rename
#: temporaries and EOS atomic-upload leftovers
IN_FLIGHT_MARKERS = (".flaf-tmp-", ".dsprod-tmp-", ".sys.a#")

#: file attributes that change who may modify that file
FILE_PERMISSION_ATTRS = ("sys.acl", "sys.eval.useracl", "user.acl")

#: directory attributes a rebuilt directory does not take over from the original: the lock
#: sets the ACL, and the rest would hand permissions or ownership to somebody else
NOT_COPIED_DIR_ATTRS = (
    "sys.acl",
    "sys.eos.btime",
    "sys.eval.useracl",
    "user.acl",
    "sys.owner.auth",
)

#: pauses before retrying a step during which paths of the folder are missing (a rebuilt
#: directory replacing the original); no pause once a stop was requested, since the session
#: may end (lxplus kills what is left of it 90 s after logout)
RETRY_DELAYS = (5, 15, 45, 120)

#: renames in flight at once while a rebuilt directory takes over the subdirectories of the
#: original, which are missing from their paths until then
MOVE_JOBS = 16

#: the eos-CLI calls of one directory rebuild, without its copies, as measured from lxplus
#: (live test, 2026-10-25)
REBUILD_SECONDS = 20

#: attempts of a read (stat, listing, checksum) that ends without an answer from the server,
#: such as a timeout or a dropped connection
READ_ATTEMPTS = 3
READ_RETRY_DELAY = 5

#: request timeouts in seconds (the XrdCl default is 1800): a hung rename or listing inside a
#: swap is given up and retried instead of holding the swap open; xrdcp keeps the default, as a
#: third-party copy waits in one request for the whole transfer
REQUEST_TIMEOUT = {"xrdfs": "120", "eos": "300"}

FIND_FORMAT = "type,uid,size,checksum,mode,mtime,attr.*"
FIND_LIMIT_MESSAGE = "find results are limited"


class LockError(Exception):
    pass


class Refused(LockError):
    """The folder is not in a state the operation can handle; nothing was changed."""


class Interrupted(LockError):
    """The lock was asked to stop and stopped where every file is in place."""


class SwapConflict(LockError):
    """The path of a directory being rebuilt was taken by something else meanwhile."""


def log(msg):
    try:
        print(time.strftime("%H:%M:%S ") + msg, flush=True)
    except OSError:
        # the terminal went away (SIGHUP): losing the log must not stop the lock mid-step
        pass


class StopSignals:
    """SIGINT, SIGTERM, SIGHUP and SIGTSTP ask the lock to stop at its next safe point.

    A safe point is one where every file of the folder is at its path, with its content:
    between directories and between files, never while a rebuilt directory takes the place
    of the original. Ctrl-Z stops the lock the same way instead of suspending it there.
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP)

    def __init__(self):
        self.received = None

    @property
    def requested(self):
        return self.received is not None

    def __enter__(self):
        self.previous = {}
        for sig in self.SIGNALS:
            # an ignored SIGHUP (nohup) stays ignored; SIGINT is handled even when inherited
            # ignored, as a non-interactive shell starts its background jobs that way
            if sig == signal.SIGHUP and signal.getsignal(sig) is signal.SIG_IGN:
                continue
            self.previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def _handle(self, signum, frame):
        if self.received is not None:
            return
        self.received = signal.Signals(signum).name
        # not through sys.stdout: the handler may interrupt a write to it
        msg = f"{self.received}: stopping at the next point where every file is in place\n"
        try:
            os.write(2, msg.encode())
        except OSError:
            pass

    def check(self):
        if self.received:
            raise Interrupted(
                f"stopped on {self.received}; every file is at its path, rerun the same "
                "command to resume"
            )


class NoStop:
    requested = False

    def check(self):
        pass


# ---------------------------------------------------------------- ACLs


def parse_acl(acl):
    """`[(tag, flags)]` of an EOS ACL, e.g. `[("u:44535", "rwx+d+u"), ("z", "i")]`."""
    rules = []
    for rule in filter(None, acl.split(",")):
        if rule.startswith("z:"):
            rules.append(("z", rule[2:]))
        else:
            tag, _, flags = rule.rpartition(":")
            rules.append((tag, flags))
    return rules


def acl_flags(flags):
    """`[(prefix, letter)]` with prefix "", "!" or "+"; `wo` (write-once) is the letter "W"."""
    out = []
    prefix = ""
    i = 0
    while i < len(flags):
        c = flags[i]
        if c in "!+":
            prefix = c
        else:
            if c == "w" and flags[i + 1 : i + 2] == "o":
                c = "W"
                i += 1
            out.append((prefix, c))
            prefix = ""
        i += 1
    return out


def is_immutable(acl):
    """True if the ACL makes the directory immutable for everybody.

    EOS sets the immutable flag from any matching rule with `i` and lifts it when any matching
    rule has `!i` (mgm/Acl.cc, EOS 5.3.38: `case 'i'` and the post-processing of denials), so
    only a `z:` rule with `i` binds everybody, and any `!i` undoes it.
    """
    rules = parse_acl(acl)
    zi = any(tag == "z" and ("", "i") in acl_flags(f) for tag, f in rules)
    not_i = any(("!", "i") in acl_flags(f) for _, f in rules)
    return zi and not not_i


def acl_rights(acl, uid, gid):
    """`(write, delete, sys_acl)` granted to uid/gid by the ACL rules that match it.

    A simplification of mgm/Acl.cc, used only to refuse early: denials beat grants, `+d` beats
    `!d`, and write-once grants no write. The server takes the real decision.
    """
    matching = [f for tag, f in parse_acl(acl) if tag in ("z", f"u:{uid}", f"g:{gid}")]
    flags = [fl for f in matching for fl in acl_flags(f)]
    write = ("", "w") in flags and ("!", "w") not in flags and ("", "W") not in flags
    delete = write and (("!", "d") not in flags or ("+", "d") in flags)
    sys_acl = ("", "A") in flags and ("!", "A") not in flags
    return write, delete, sys_acl


def lock_acl(uid, admins):
    rules = [READ_RULE] + [f"u:{u}:A" for u in dict.fromkeys([uid] + list(admins))]
    return ",".join(rules + ["z:i"])


def staging_acl(uid):
    """ACL while a lock is in progress: members lose write, the caller keeps it."""
    return f"{READ_RULE},u:{uid}:rwx+d+u"


# ---------------------------------------------------------------- EOS access


class Entry:
    __slots__ = ("path", "type", "uid", "size", "checksum", "mode", "mtime", "attrs")

    def __init__(
        self, path, type, uid, size=0, checksum="", mode=0, mtime=0.0, attrs=None
    ):
        self.path = path
        self.type = type
        self.uid = uid
        self.size = size
        self.checksum = checksum
        self.mode = mode
        self.mtime = mtime
        self.attrs = attrs or {}

    @property
    def acl(self):
        return self.attrs.get("sys.acl", "")


_FIND_LINE = re.compile(r'^path="(.*)" type=(\S+)\s*(.*)$')
_FIND_FIELD = re.compile(r'(\S+?)=(?:"([^"]*)"|(\S*))')


def parse_find(text):
    """Entries of `eos find --format FIND_FORMAT` output (mgm/proc/user/NewfindCmd.cc)."""
    entries = []
    for line in text.splitlines():
        m = _FIND_LINE.match(line.strip())
        if not m:
            continue
        path, type_, rest = m.groups()
        fields = {}
        for k, quoted, plain in _FIND_FIELD.findall(rest):
            fields[k] = quoted if quoted or plain == "" else plain
        attrs = {k[5:]: v for k, v in fields.items() if k.startswith("attr.")}
        # what NewfindCmd.cc prints per type: files have no mode, directories no checksum
        needed = {"file": ("size", "checksum"), "directory": ("mode",)}.get(type_, ())
        missing = [k for k in ("uid", "mtime") + needed if not fields.get(k)]
        if missing:
            raise LockError(f"eos find printed no {', '.join(missing)} for {path}")
        entries.append(
            Entry(
                path=path.rstrip("/") or "/",
                type=type_,
                uid=int(fields["uid"]),
                size=int(fields.get("size") or 0),
                checksum=fields.get("checksum", "").lower(),
                mode=int(fields.get("mode") or "0", 8),
                mtime=float(fields["mtime"]),
                attrs=attrs,
            )
        )
    return entries


def same_checksum(a, b):
    try:
        return int(a, 16) == int(b, 16)
    except ValueError:
        return False


class EosClient:
    """The eos, xrdfs and xrdcp command-line clients against one MGM."""

    def __init__(self, mgm):
        self.mgm = mgm.rstrip("/")
        self.host = self.mgm[len("root://") :]

    def _run(self, cmd):
        start = time.time()
        # their own session: a Ctrl-C at the terminal must reach the lock, which stops at a
        # safe point, and not kill a rename half-way
        env = None
        if cmd[0] in REQUEST_TIMEOUT:
            env = dict(os.environ, XRD_REQUESTTIMEOUT=REQUEST_TIMEOUT[cmd[0]])
        p = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            start_new_session=True,
            env=env,
        )
        if time.time() - start > 60:
            log(f"slow: {' '.join(cmd)} took {time.time() - start:.0f} s")
        return p.returncode, p.stdout, p.stderr

    def _read(self, cmd):
        """Run a read-only command, again if it ended without an answer from the server."""
        for attempt in range(READ_ATTEMPTS):
            rc, out, err = self._run(cmd)
            answered = rc == 0 or any(
                m in out + err for m in ("[3011]", "[3027]", FIND_LIMIT_MESSAGE)
            )
            if answered or attempt == READ_ATTEMPTS - 1:
                return rc, out, err
            log(f"retrying {' '.join(cmd[:3])} {cmd[-1]}: {(err or out).strip()[:200]}")
            time.sleep(READ_RETRY_DELAY)

    def _check(self, cmd, read=False):
        rc, out, err = self._read(cmd) if read else self._run(cmd)
        # eos reports a failed recursive change per directory on stderr; the find limit of a
        # recursive command is only a warning with rc 0
        failed = any(
            line.strip().startswith("error") for line in (out + err).splitlines()
        )
        if rc != 0 or failed or FIND_LIMIT_MESSAGE in out + err:
            raise LockError(f"{' '.join(cmd)} failed (rc={rc}): {(err or out).strip()}")
        return out

    def _eos(self, *args):
        return self._check(["eos", self.mgm] + list(args))

    def url(self, path):
        return f"{self.mgm}/{path}"

    def whoami(self):
        out = self._check(["eos", self.mgm, "whoami"], read=True)
        m = re.search(r"uid=(\d+).*gid=(\d+)", out)
        if not m:
            raise LockError(f"cannot read the EOS identity from: {out.strip()}")
        return int(m.group(1)), int(m.group(2))

    def find(self, path, maxdepth=None):
        cmd = ["eos", self.mgm, "find", "--format", FIND_FORMAT]
        if maxdepth is not None:
            cmd += ["--maxdepth", str(maxdepth)]
        rc, out, err = self._read(cmd + [path])
        if FIND_LIMIT_MESSAGE in out + err:
            # the limit is 50000 directories or 100000 files, but a listing counts each
            # directory's files once more (NewfindCmd.cc), so about 50000 files in practice
            raise Refused(
                f"{path} has more entries than one eos find may list (about 50000 "
                "directories or files): lock its subfolders one by one"
            )
        if rc != 0:
            raise LockError(f"eos find {path} failed (rc={rc}): {(err or out).strip()}")
        return parse_find(out)

    def size(self, path):
        """Size of an existing path, None if it does not exist."""
        rc, out, err = self._read(["xrdfs", self.host, "stat", path])
        if rc == 0:
            m = re.search(r"Size:\s+(\d+)", out)
            return int(m.group(1)) if m else 0
        if "[3011]" in out + err:
            return None
        raise LockError(f"xrdfs stat {path} failed (rc={rc}): {(err or out).strip()}")

    def exists(self, path):
        return self.size(path) is not None

    def checksum(self, path):
        out = self._check(["xrdfs", self.host, "query", "checksum", path], read=True)
        return out.split()[-1].lower()

    def mkdir(self, path):
        self._eos("mkdir", path)

    def mv(self, src, dst):
        self._check(["xrdfs", self.host, "mv", src, dst])

    def _remove(self, cmd, path):
        try:
            self._check(cmd)
        except LockError as e:
            # a request repeated by the client after a lost reply finds the path gone
            if "[3011]" in str(e) and not self.exists(path):
                log(f"{path} is already removed")
                return
            raise

    def rm(self, path):
        self._remove(["xrdfs", self.host, "rm", path], path)

    def rmdir(self, path):
        self._remove(["xrdfs", self.host, "rmdir", path], path)

    def copy(self, src, dst):
        self._check(
            ["xrdcp", "--nopbar", "--tpc", "first", self.url(src), self.url(dst)]
        )

    def attr_set(self, path, key, value, recursive=False):
        self._eos("attr", *(["-r"] if recursive else []), "set", f"{key}={value}", path)

    def attr_rm(self, path, key):
        self._eos("attr", "rm", key, path)

    def chmod(self, path, mode, recursive=False):
        self._eos("chmod", *(["-r"] if recursive else []), f"{mode:o}", path)


# ---------------------------------------------------------------- the folder as found


def base_name(name):
    for suffix in (NEW_DIR_SUFFIX, TMP_FILE_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def logical_path(path, root):
    """Path relative to root with the lock's temporaries mapped to the entries they stand for."""
    if path == root + NEW_DIR_SUFFIX or path.startswith(root + NEW_DIR_SUFFIX + "/"):
        path = root + path[len(root + NEW_DIR_SUFFIX) :]
    rel = posixpath.relpath(path, root)
    if rel == ".":
        return ""
    return "/".join(base_name(p) for p in rel.split("/") if p != OLD_DIR_NAME)


class Folder:
    """A folder as listed by one eos find, plus the leftovers of an interrupted lock of it."""

    def __init__(self, root, entries):
        self.root = root
        self.entries = entries
        self.dirs = {e.path: e for e in entries if e.type == "directory"}
        self.files = {e.path: e for e in entries if e.type == "file"}
        self.others = [e for e in entries if e.type not in ("directory", "file")]
        # every directory entry under the logical directory it stands for, by its own name:
        # an interrupted lock may have left it inside an ancestor's rebuilt or original copy,
        # from where that ancestor's completion moves it to its own path, name unchanged
        self.states = {}
        for p, e in self.dirs.items():
            name = posixpath.basename(p)
            kind = (
                "new"
                if name.endswith(NEW_DIR_SUFFIX)
                else "old" if name == OLD_DIR_NAME else "dir"
            )
            self.states.setdefault(logical_path(p, root), {}).setdefault(
                kind, []
            ).append(e)

    def logical_dirs(self):
        """Logical directory paths, parents before children."""
        names = {logical_path(p, self.root) for p in self.dirs}
        return sorted(names, key=lambda p: (p.count("/") if p else -1, p))

    def manifest(self):
        """`{logical path: (size, checksum)}` of every file as it was before the lock.

        Copies made by the lock are left out: `*.flaf-lock-tmp` files and the files of a
        `*.flaf-lock-new` directory (a copy there may be partial). Where the original of a
        rebuilt directory still holds a file, that file counts, not the rebuilt directory's copy.
        """
        out = {}
        originals_first = sorted(
            self.files.items(), key=lambda pe: f"/{OLD_DIR_NAME}/" not in pe[0]
        )
        for p, e in originals_first:
            if p.endswith(TMP_FILE_SUFFIX) or posixpath.dirname(p).endswith(
                NEW_DIR_SUFFIX
            ):
                continue
            out.setdefault(logical_path(p, self.root), (e.size, e.checksum))
        return out


def children(eos, path):
    return [e for e in eos.find(path, maxdepth=1) if e.path != path]


def scan(eos, folder):
    """The folder and, if an interrupted lock left it, the directory rebuilt to replace it."""
    entries = []
    for path in (folder, folder + NEW_DIR_SUFFIX):
        if not eos.exists(path):
            continue
        found = eos.find(path)
        if not any(e.path == path and e.type == "directory" for e in found) or any(
            e.path != path and not e.path.startswith(path + "/") for e in found
        ):
            raise Refused(
                f"eos find does not list {path} under its own name: pass the folder's "
                "canonical path (no symlink on the way) of a directory"
            )
        entries += found
    if not entries:
        raise Refused(f"{folder} does not exist")
    return Folder(folder, entries)


def dir_state(folder, logical):
    """The entries standing for one logical directory: (dir, new, old), each Entry or None.

    Raises Refused if one of them appears more than once.
    """
    state = folder.states.get(logical, {})
    for kind, found in state.items():
        if len(found) > 1:
            raise Refused(
                f"{', '.join(e.path for e in found)}: one directory found more than once, "
                "leftovers of an interrupted lock to resolve by hand"
            )
    return tuple(state.get(kind, [None])[0] for kind in ("dir", "new", "old"))


# ---------------------------------------------------------------- checks


def check_lock(folder, parent, uid, gid, now, min_age_minutes):
    """Reasons the folder cannot be locked as it is (empty list: it can)."""
    problems = []
    for e in folder.others:
        problems.append(f"{e.type} {e.path}: only files and directories can be locked")
    for p, e in list(folder.files.items()) + list(folder.dirs.items()):
        name = posixpath.basename(p)
        if any(m in name for m in IN_FLIGHT_MARKERS):
            problems.append(f"{p}: an upload is still in flight (or was abandoned)")
        if name.endswith(RUN_MARKER_SUFFIX):
            problems.append(
                f"{p}: a lock of a subfolder is running or was killed; let it finish, or "
                "remove the marker if none runs"
            )
    for e in list(folder.dirs.values()) + ([parent] if parent is not None else []):
        if "sys.owner.auth" in e.attrs and e.uid != uid:
            problems.append(
                f"{e.path} has sys.owner.auth: what the lock creates there would belong to "
                f"uid {e.uid}"
            )
    # not in flight: files in an immutable directory, and the copies of an interrupted lock
    # (its temporaries, its rebuilt directories, directories it has frozen)
    frozen = staging_acl(uid)
    recent = []
    for e in folder.files.values():
        d = folder.dirs[posixpath.dirname(e.path)]
        if (
            now - e.mtime < min_age_minutes * 60
            and not e.path.endswith(TMP_FILE_SUFFIX)
            and not d.path.endswith(NEW_DIR_SUFFIX)
            and not is_immutable(d.acl)
            and d.acl != frozen
        ):
            recent.append(e.path)
    if recent:
        problems.append(
            f"{len(recent)} files were modified in the last {min_age_minutes} min, e.g. {recent[0]}: "
            "the production may still be writing"
        )
    if parent is None:
        problems.append(f"{folder.root}: its parent cannot be listed")
    elif is_immutable(parent.acl):
        problems.append(
            f"{parent.path} is locked: lock or unlock the parent folder instead"
        )

    top, top_new, top_old = dir_state(folder, "")
    top_foreign = (
        (top is not None and top.uid != uid)
        or top_new is not None
        or top_old is not None
    )
    if parent is not None and top_foreign and parent.uid != uid:
        write, delete, _ = acl_rights(parent.acl, uid, gid)
        if not (write and delete):
            problems.append(
                f"{parent.path}: its ACL grants uid {uid} no write and delete, which rebuilding "
                f"{folder.root} needs"
            )

    for logical in folder.logical_dirs():
        d, new, old = dir_state(folder, logical)
        where = posixpath.join(folder.root, logical) if logical else folder.root
        # the states a rebuild passes through: (foreign dir) -> (foreign dir, new) -> (new
        # holding the original as old) -> (new renamed onto dir, old inside) -> (dir)
        state = (d is not None, new is not None, old is not None)
        consistent = (
            state == (True, False, False)
            or (state == (True, True, False) and d.uid != uid and new.uid == uid)
            or (
                state == (False, True, True)
                and new.uid == uid
                and posixpath.dirname(old.path) == new.path
            )
            or (
                state == (True, False, True)
                and d.uid == uid
                and posixpath.dirname(old.path) == d.path
            )
        )
        if not consistent:
            problems.append(
                f"{where}: inconsistent leftovers of an interrupted lock "
                f"(dir, {NEW_DIR_SUFFIX}, {OLD_DIR_NAME}) = {state}, resolve by hand"
            )
            continue
        for e in filter(None, (d, new, old)):
            if e.uid == uid:
                continue
            write, delete, sys_acl = acl_rights(e.acl, uid, gid)
            if is_immutable(e.acl) and not sys_acl:
                problems.append(
                    f"{e.path} is locked by uid {e.uid}: only its owner can unlock it"
                )
            elif not is_immutable(e.acl) and not (write and delete):
                problems.append(
                    f"{e.path} (uid {e.uid}): its ACL grants uid {uid} no write and delete, "
                    "so it cannot be rebuilt"
                )
    for p in folder.files:
        if (
            p.endswith(TMP_FILE_SUFFIX)
            and p[: -len(TMP_FILE_SUFFIX)] not in folder.files
        ):
            problems.append(
                f"{p}: copy of a file that no longer exists, resolve by hand"
            )
    return problems


# ---------------------------------------------------------------- lock


class Locker:
    def __init__(self, eos, folder_path, uid, gid, admins, jobs, stop):
        self.eos = eos
        self.root = folder_path
        self.uid = uid
        self.gid = gid
        self.lock_acl = lock_acl(uid, admins)
        self.jobs = jobs
        self.stop = stop

    def _move(self, src, dst):
        """Rename, retried: used where a failure would leave a path of the folder missing."""
        for attempt, delay in enumerate(RETRY_DELAYS + (None,)):
            try:
                self.eos.mv(src, dst)
                return
            except Exception as e:
                try:
                    # carried out, but the reply was lost
                    if not self.eos.exists(src) and self.eos.exists(dst):
                        return
                except Exception:
                    pass
                if delay is None:
                    raise
                delay = 1 if self.stop.requested else delay
                log(
                    f"rename {src} -> {dst} failed ({e}), retry {attempt + 1} in {delay} s"
                )
                time.sleep(delay)

    def _parallel(self, fn, items, workers=None):
        pool = ThreadPoolExecutor(max_workers=workers or self.jobs)
        try:
            for future in [pool.submit(fn, item) for item in items]:
                future.result()
        except BaseException:
            # copies already running finish, queued ones are dropped
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown()

    def _copy_verified(self, src, dst, size, checksum):
        """Copy src to dst and check the copy's checksum against the source's."""
        self.stop.check()
        if self.eos.exists(dst):
            self.eos.rm(dst)
        try:
            self.eos.copy(src, dst)
        except LockError as failure:
            try:
                # a copy whose reply was lost is repeated by the client, which then finds the
                # file it made ("File exists"): a destination that verifies is that copy
                if self.eos.size(dst) == size and same_checksum(
                    self.eos.checksum(dst), checksum
                ):
                    log(f"{dst} is complete although its copy reported: {failure}")
                    return
                if self.eos.exists(dst):
                    self.eos.rm(dst)
            except LockError as e:
                # a partial copy left here is refused before anything is promoted
                log(f"could not check or remove the failed copy {dst}: {e}")
            raise
        got_size, got = self.eos.size(dst), self.eos.checksum(dst)
        if got_size != size or not same_checksum(got, checksum):
            self.eos.rm(dst)
            raise LockError(
                f"copy of {src} has {got_size} bytes / adler32 {got}, expected {size} / {checksum}; "
                "original kept"
            )
        log(f"copied {src}")

    # phase 0: directories already locked (an earlier lock of this folder or of a subfolder)
    def open_locked_dirs(self, folder):
        for p in sorted(folder.dirs, key=lambda x: x.count("/")):
            e = folder.dirs[p]
            if is_immutable(e.acl) or (e.uid == self.uid and not e.mode & 0o200):
                log(f"open {p} for the duration of the lock")
                self.eos.attr_set(p, "sys.acl", staging_acl(self.uid))
                e.attrs["sys.acl"] = staging_acl(self.uid)

    # phase 1: every directory to the caller
    def take_dirs(self, folder):
        states = [
            (logical,) + dir_state(folder, logical) for logical in folder.logical_dirs()
        ]
        # first the swaps a hard kill interrupted, so that every path is back before anything
        # is copied; no safe point until they are all done
        unfinished = [st for st in states if st[1] is None or st[3] is not None]
        for logical, d, new, old in unfinished:
            self._complete_swap(self._path(logical))
        for logical, d, new, old in states:
            self.stop.check()
            if (logical, d, new, old) in unfinished:
                self._clear_old(self._path(logical))
            elif d is None or d.uid != self.uid:
                self.take_dir(self._path(logical), d, new, old)

    def _path(self, logical):
        return posixpath.join(self.root, logical) if logical else self.root

    def take_dir(self, path, d, new, old):
        new_path = path + NEW_DIR_SUFFIX
        if d is not None and d.uid != self.uid:
            log(f"rebuild {path} (uid {d.uid})")
            if new is None:
                self.eos.mkdir(new_path)
                done = {}
            # not the parent's ACL it inherited: under a `!d` rule the original could not be
            # removed from it, by the caller as its owner either ("Unable to rmdir by ACL")
            self.eos.attr_set(new_path, "sys.acl", staging_acl(self.uid))
            inherited = self.eos.find(new_path, maxdepth=0)[0].attrs
            for key, value in d.attrs.items():
                if key not in NOT_COPIED_DIR_ATTRS and inherited.get(key) != value:
                    self.eos.attr_set(new_path, key, value)
            if new is not None:
                done = {
                    posixpath.basename(e.path): e for e in children(self.eos, new_path)
                }
            # the caller's own files are copied too: moved, they would be missing from their
            # paths between the swap and their move
            originals = {
                posixpath.basename(e.path): e
                for e in children(self.eos, path)
                if e.type == "file"
            }
            todo = []
            for name, e in originals.items():
                have = done.get(name)
                if (
                    have is not None
                    and have.size == e.size
                    and same_checksum(have.checksum, e.checksum)
                ):
                    continue
                todo.append(e)
            self._parallel(
                lambda e: self._copy_verified(
                    e.path,
                    posixpath.join(new_path, posixpath.basename(e.path)),
                    e.size,
                    e.checksum,
                ),
                todo,
            )
            # only verified copies are promoted: a copy whose original is gone (an interrupted
            # lock, then a deletion) may be partial
            stray = [
                c.path
                for c in children(self.eos, new_path)
                if c.type != "file"
                or posixpath.basename(c.path) not in originals
                or c.size != originals[posixpath.basename(c.path)].size
                or not same_checksum(
                    c.checksum, originals[posixpath.basename(c.path)].checksum
                )
            ]
            if stray:
                raise LockError(
                    f"{new_path} holds {len(stray)} entries that are not verified copies of "
                    f"files in {path} (an original deleted after an interrupted lock?), e.g. "
                    f"{stray[0]}; resolve by hand"
                )
            # the last safe point before the swap: from the next rename until _complete_swap
            # returns, the directory (between the two renames) and then its subdirectories
            # are missing from their paths; every step is retried
            self.stop.check()
            try:
                self._move(path, posixpath.join(new_path, OLD_DIR_NAME))
            except Exception as e:
                raise LockError(
                    f"{path} could not be moved into {new_path} to be replaced ({e}); if it is "
                    "missing from its path, rerun the same command to put it back"
                )
        self._complete_swap(path)
        self.stop.check()
        self._clear_old(path)

    def _complete_swap(self, path):
        """Put every path of a swapped directory back, retried as a whole from a fresh look."""
        new_path = path + NEW_DIR_SUFFIX
        old_path = posixpath.join(path, OLD_DIR_NAME)
        for attempt, delay in enumerate(RETRY_DELAYS + (None,)):
            try:
                if not self.eos.exists(path):
                    self.eos.mv(new_path, path)
                elif self.eos.exists(new_path):
                    # a rename onto an existing directory would move the source into it
                    raise SwapConflict(
                        f"{path} reappeared while it was rebuilt; the original is in "
                        f"{new_path}/{OLD_DIR_NAME}, resolve by hand"
                    )
                if self.eos.exists(old_path):
                    present = {
                        posixpath.basename(e.path) for e in children(self.eos, path)
                    }
                    missing = [
                        e.path
                        for e in children(self.eos, old_path)
                        if posixpath.basename(e.path) not in present
                    ]
                    start = time.time()
                    self._parallel(
                        lambda src: self.eos.mv(
                            src, posixpath.join(path, posixpath.basename(src))
                        ),
                        missing,
                        workers=MOVE_JOBS,
                    )
                    if missing:
                        log(
                            f"moved {len(missing)} entries back into {path} in "
                            f"{time.time() - start:.0f} s"
                        )
                return
            except SwapConflict:
                raise
            except Exception as e:
                if delay is None:
                    raise LockError(
                        f"{path} could not be put back ({e}): what is missing from it is in "
                        f"{old_path} (or, with the whole directory, {new_path}/{OLD_DIR_NAME}); "
                        "rerun the same command as soon as the storage answers again"
                    )
                delay = 1 if self.stop.requested else delay
                log(
                    f"putting {path} back failed ({e}), retry {attempt + 1} in {delay} s"
                )
                time.sleep(delay)

    def _clear_old(self, path):
        """Delete the originals whose verified copies are in place, then the original itself."""
        old_path = posixpath.join(path, OLD_DIR_NAME)
        if not self.eos.exists(old_path):
            return
        conflicts = []
        current = {posixpath.basename(e.path): e for e in children(self.eos, path)}
        for e in children(self.eos, old_path):
            have = current.get(posixpath.basename(e.path))
            if have is None:
                conflicts.append(e.path)
            elif (
                e.type == "file"
                and have.type == "file"
                and have.size == e.size
                and same_checksum(have.checksum, e.checksum)
            ):
                self.eos.rm(e.path)
            else:
                conflicts.append(e.path)
        if conflicts:
            raise LockError(
                f"{old_path} keeps {len(conflicts)} entries that differ from the rebuilt "
                f"{path} (changed while the lock ran?), e.g. {conflicts[0]}; nothing of them "
                "was deleted, resolve by hand and rerun"
            )
        self.eos.rmdir(old_path)

    # phase 2: keep members out while files are taken
    def freeze(self):
        self.eos.attr_set(self.root, "sys.acl", staging_acl(self.uid), recursive=True)
        self.eos.chmod(self.root, OPEN_MODE, recursive=True)

    # phase 3: every file to the caller
    def take_files(self, folder):
        for p, e in folder.dirs.items():
            if "sys.eval.useracl" in e.attrs:
                self.eos.attr_rm(p, "sys.eval.useracl")
        for p in folder.files:
            if p.endswith(TMP_FILE_SUFFIX):
                self.eos.rm(p)
        foreign = [
            e
            for p, e in folder.files.items()
            if e.uid != self.uid and not p.endswith(TMP_FILE_SUFFIX)
        ]
        if foreign:
            log(
                f"copy {len(foreign)} files of other owners ({sum(e.size for e in foreign) / 1e9:.2f} GB)"
            )

        def take(e):
            tmp = e.path + TMP_FILE_SUFFIX
            self._copy_verified(e.path, tmp, e.size, e.checksum)
            if self.eos.size(e.path) != e.size or not same_checksum(
                self.eos.checksum(e.path), e.checksum
            ):
                self.eos.rm(tmp)
                raise LockError(
                    f"{e.path} changed while the lock ran; it was not replaced, resolve by "
                    "hand and rerun"
                )
            # replaces the original in one rename: the path never misses its file
            self._move(tmp, e.path)

        self._parallel(take, foreign)
        for p, e in folder.files.items():
            for key in FILE_PERMISSION_ATTRS:
                if key in e.attrs and e.uid == self.uid:
                    self.eos.attr_rm(p, key)

    # phase 4
    def seal(self):
        self.eos.chmod(self.root, LOCKED_MODE, recursive=True)
        self.eos.attr_set(self.root, "sys.acl", self.lock_acl, recursive=True)


def lock_problems(folder, uid, lock_acl_value):
    """Reasons a locked folder is not fully locked (empty list: it is)."""
    problems = []
    for p, e in folder.dirs.items():
        if e.uid != uid:
            problems.append(f"directory {p} belongs to uid {e.uid}")
        if e.acl != lock_acl_value:
            problems.append(f"directory {p} has sys.acl={e.acl!r}")
        if e.mode & 0o777 != LOCKED_MODE:
            problems.append(f"directory {p} has mode {e.mode & 0o777:o}")
        if posixpath.basename(p).endswith(LOCK_SUFFIXES):
            problems.append(f"leftover {p}")
    for p, e in folder.files.items():
        if e.uid != uid:
            problems.append(f"file {p} belongs to uid {e.uid}")
        for key in FILE_PERMISSION_ATTRS:
            if key in e.attrs:
                problems.append(f"file {p} has {key}")
        if p.endswith(TMP_FILE_SUFFIX):
            problems.append(f"leftover {p}")
    return problems


def manifest_problems(before, after):
    """Files lost or changed between two manifests (files added are not a problem)."""
    problems = []
    for rel, (size, checksum) in sorted(before.items()):
        got = after.get(rel)
        if got is None:
            problems.append(f"missing after the lock: {rel}")
        elif got[0] != size or not same_checksum(got[1], checksum):
            problems.append(
                f"changed by the lock: {rel} {size}/{checksum} -> {got[0]}/{got[1]}"
            )
    return problems


def plan(folder, uid, jobs):
    """What the lock copies and moves: `(files, bytes, peak extra bytes, largest swap)`.

    A directory of another owner is rebuilt with copies of all its files, its subdirectories
    then moved one rename each; a file of another owner in the caller's directory is copied
    next to itself, `jobs` at a time.
    """
    foreign_dirs = {p for p, e in folder.dirs.items() if e.uid != uid}
    rebuilt = {}
    taken = []
    for p, e in folder.files.items():
        parent = posixpath.dirname(p)
        if parent in foreign_dirs:
            rebuilt.setdefault(parent, []).append(e.size)
        elif e.uid != uid:
            taken.append(e.size)
    subdirs = {}
    for p in folder.dirs:
        if posixpath.dirname(p) in foreign_dirs:
            subdirs[posixpath.dirname(p)] = subdirs.get(posixpath.dirname(p), 0) + 1
    n = len(taken) + sum(len(v) for v in rebuilt.values())
    volume = sum(taken) + sum(sum(v) for v in rebuilt.values())
    peak = max(
        max((sum(v) for v in rebuilt.values()), default=0),
        sum(sorted(taken, reverse=True)[:jobs]),
    )
    return n, volume, peak, max(subdirs.values(), default=0)


def describe(folder, uid, jobs=1):
    dirs = list(folder.dirs.values())
    files = list(folder.files.values())
    foreign_dirs = [e for e in dirs if e.uid != uid]
    locked = [e for e in dirs if is_immutable(e.acl)]
    n, volume, peak, swap = plan(folder, uid, jobs)
    lines = [
        f"{len(dirs)} directories ({len(foreign_dirs)} of other owners, {len(locked)} locked), "
        f"{len(files)} files ({sum(e.size for e in files) / 1e9:.2f} GB)",
        f"to copy: {n} files ({volume / 1e9:.2f} GB); at most {peak / 1e9:.2f} GB of extra "
        "quota at a time",
    ]
    if foreign_dirs:
        lines.append(
            f"directories to rebuild: {len(foreign_dirs)}, about {REBUILD_SECONDS} s each from "
            "lxplus plus their copies (keep the session open while the lock runs)"
        )
    if swap:
        lines.append(
            f"largest swap: the directory is missing for about two requests, then up to {swap} "
            "subdirectories until they are moved back"
        )
    owners = sorted({e.uid for e in dirs + files})
    lines.append(f"owners (uid): {', '.join(map(str, owners))}; caller uid {uid}")
    parents = {posixpath.dirname(p) for p in list(folder.dirs) + list(folder.files)}
    empty = sorted(p for p in folder.dirs if p not in parents)
    if empty:
        lines.append(
            f"note: {len(empty)} empty directories, e.g. {empty[0]}; an empty locked directory "
            "can still be removed together with its open parent"
        )
    return lines


def ancestors(path):
    """`/eos/a/b/c` -> `/eos/a/b`, `/eos/a`."""
    out = []
    while path.count("/") > 2:
        path = posixpath.dirname(path)
        out.append(path)
    return out


def parent_entry(eos, path):
    found = eos.find(posixpath.dirname(path), maxdepth=0)
    return found[0] if found else None


def lock_preconditions(eos, path, folder, uid, gid, now, min_age):
    parent = parent_entry(eos, path)
    problems = check_lock(folder, parent, uid, gid, now, min_age)
    for ancestor in ancestors(path):
        if eos.exists(ancestor + RUN_MARKER_SUFFIX):
            problems.append(
                f"{ancestor}{RUN_MARKER_SUFFIX}: a lock of {ancestor} is running or was "
                "killed; let it finish, or remove the marker if none runs"
            )
    if problems:
        raise Refused("cannot lock:\n  " + "\n  ".join(problems))


def run_lock(eos, path, admins, apply, min_age, jobs, now=None, stop=None):
    stop = stop or NoStop()
    now = time.time() if now is None else now
    uid, gid = eos.whoami()
    folder = scan(eos, path)
    acl = lock_acl(uid, admins)
    log(f"lock {path} as uid {uid} with sys.acl={acl}, mode {LOCKED_MODE:o}")
    for line in describe(folder, uid, jobs):
        log(line)
    if not lock_problems(folder, uid, acl):
        log("already locked")
        return
    lock_preconditions(eos, path, folder, uid, gid, now, min_age)
    if not apply:
        log("dry run: nothing changed (add --apply)")
        return

    # a stop before the start changes nothing, unless a killed lock left paths missing:
    # those are put back first
    if not any(
        p.endswith(NEW_DIR_SUFFIX) or posixpath.basename(p) == OLD_DIR_NAME
        for p in folder.dirs
    ):
        stop.check()
    marker = path + RUN_MARKER_SUFFIX
    try:
        eos.mkdir(marker)
    except LockError as e:
        raise Refused(
            f"cannot create {marker} ({e}): a lock of this folder is running or was killed "
            "(remove the marker if none runs), or the parent is not writable"
        )

    def remove_marker():
        try:
            eos.rmdir(marker)
        except LockError as e:
            log(f"could not remove {marker} ({e}); remove it before the next run")

    try:
        # again under the marker: a lock of a subfolder or an enclosing folder may have
        # started since the first look
        folder = scan(eos, path)
        lock_preconditions(eos, path, folder, uid, gid, now, min_age)
    except BaseException:
        remove_marker()
        raise
    before = folder.manifest()
    try:
        locker = Locker(eos, path, uid, gid, admins, jobs, stop)
        locker.open_locked_dirs(folder)
        locker.take_dirs(folder)
        stop.check()
        locker.freeze()
        locker.take_files(scan(eos, path))
        stop.check()
        locker.seal()
    except Refused as e:
        raise LockError(f"stopped after changes were made: {e}")
    finally:
        remove_marker()

    after = scan(eos, path)
    problems = lock_problems(after, uid, acl) + manifest_problems(
        before, after.manifest()
    )
    if problems:
        raise LockError("the lock did not complete:\n  " + "\n  ".join(problems))
    added = set(after.manifest()) - set(before)
    if added:
        log(
            f"{len(added)} files appeared while the lock ran and are locked too, e.g. {sorted(added)[0]}"
        )
    log(
        f"locked: {len(after.dirs)} directories, {len(after.files)} files, content unchanged"
    )


# ---------------------------------------------------------------- unlock and status


def run_unlock(eos, path, acl, apply):
    uid, _ = eos.whoami()
    folder = scan(eos, path)
    parent = parent_entry(eos, path)
    if parent is None or is_immutable(parent.acl):
        raise Refused(
            f"the parent of {path} is locked: unlock the parent folder instead"
        )
    if acl is None:
        if not parent.acl:
            raise Refused(
                f"the parent of {path} has no ACL to copy: give one with --acl"
            )
        acl = parent.acl
    not_mine = [p for p, e in folder.dirs.items() if e.uid != uid]
    if not_mine:
        raise Refused(
            f"{len(not_mine)} directories are not yours, e.g. {not_mine[0]}: only the owner can "
            "unlock (a user with an A rule can only change the ACL)"
        )
    log(
        f"unlock {path}: sys.acl={acl}, mode {OPEN_MODE:o}, {len(folder.dirs)} directories"
    )
    if not apply:
        log("dry run: nothing changed (add --apply)")
        return
    # the ACL first: chmod is refused below an immutable parent
    eos.attr_set(path, "sys.acl", acl, recursive=True)
    eos.chmod(path, OPEN_MODE, recursive=True)
    after = scan(eos, path)
    bad = [
        p for p, e in after.dirs.items() if e.acl != acl or e.mode & 0o777 != OPEN_MODE
    ]
    if bad:
        raise LockError(f"{len(bad)} directories were not unlocked, e.g. {bad[0]}")
    log("unlocked")


def run_status(eos, path, admins):
    uid, _ = eos.whoami()
    folder = scan(eos, path)
    for line in describe(folder, uid):
        log(line)
    swaps = sorted(
        p
        for p in folder.dirs
        if p.endswith(NEW_DIR_SUFFIX) or posixpath.basename(p) == OLD_DIR_NAME
    )
    if swaps:
        log(
            f"INTERRUPTED LOCK: {len(swaps)} directories of an interrupted rebuild, e.g. "
            f"{swaps[0]}; paths may be missing until the lock is run again"
        )
    if eos.exists(path + RUN_MARKER_SUFFIX):
        log(
            f"{path}{RUN_MARKER_SUFFIX} exists: a lock of this folder is running, or one was "
            "killed (remove the marker before running it again)"
        )
    problems = lock_problems(folder, uid, lock_acl(uid, admins))
    if not problems:
        log("LOCKED by the caller")
        return
    locked = [p for p, e in folder.dirs.items() if is_immutable(e.acl)]
    owners = {e.uid for e in folder.dirs.values()}
    if len(locked) == len(folder.dirs) and len(owners) == 1:
        log(
            f"LOCKED by uid {owners.pop()} (not by the caller, or with another admin list)"
        )
        return
    log(f"NOT LOCKED: {len(problems)} findings, e.g.")
    for p in problems[:10]:
        log("  " + p)


# ---------------------------------------------------------------- command line


def split_target(folder, mgm):
    m = re.match(r"^(root://[^/]+)/(/.*)$", folder)
    if m:
        mgm, folder = m.groups()
    if not folder.startswith("/eos/"):
        raise Refused(
            f"{folder}: expected an EOS path /eos/... or a root://host//eos/... URL"
        )
    return mgm, posixpath.normpath(folder)


def positive_int(value):
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"{value}: expected a positive integer")
    return int(value)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mgm", default=DEFAULT_MGM, help="EOS MGM URL (default %(default)s)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_status = sub.add_parser("status", help="report whether a folder is locked")
    p_lock = sub.add_parser("lock", help="lock a folder")
    p_unlock = sub.add_parser("unlock", help="unlock a folder")
    for p in (p_status, p_lock, p_unlock):
        p.add_argument("folder")
    for p in (p_status, p_lock):
        p.add_argument(
            "--admin",
            type=int,
            action="append",
            default=[],
            metavar="UID",
            help="EOS uid that may unlock besides the caller (repeatable); check it with "
            "'getent passwd <uid>' on cmslpc",
        )
    for p in (p_lock, p_unlock):
        p.add_argument(
            "--apply", action="store_true", help="change the folder (default: dry run)"
        )
    p_lock.add_argument(
        "--min-age",
        type=float,
        default=60,
        metavar="MINUTES",
        help="refuse if a file was modified more recently (default %(default)s)",
    )
    p_lock.add_argument(
        "--jobs",
        type=positive_int,
        default=4,
        help="parallel copies (default %(default)s)",
    )
    p_unlock.add_argument("--acl", help="ACL to restore (default: the parent's)")
    args = parser.parse_args(argv)

    try:
        mgm, folder = split_target(args.folder, args.mgm)
        eos = EosClient(mgm)
        if args.command == "status":
            run_status(eos, folder, args.admin)
        elif args.command == "lock":
            with StopSignals() as stop:
                run_lock(
                    eos,
                    folder,
                    args.admin,
                    args.apply,
                    args.min_age,
                    args.jobs,
                    stop=stop,
                )
        else:
            run_unlock(eos, folder, args.acl, args.apply)
    except Refused as e:
        log(f"REFUSED: {e}")
        return 2
    except Interrupted as e:
        log(f"INTERRUPTED: {e}")
        return 3
    except LockError as e:
        log(f"ERROR: {e}")
        return 1
    finally:
        try:
            sys.stdout.flush()
        except OSError:
            sys.stdout = open(os.devnull, "w")
    return 0


if __name__ == "__main__":
    sys.exit(main())
