"""Finding a CRAB job that still holds its slot after its payload has stopped.

CRAB reports a job as ``running`` for as long as the batch system says the slot is held, which is
not the same as the payload still doing anything. A production of 600 branches ended with two jobs
whose entire status record -- wall duration, memory, CPU time -- was byte-identical across eight
consecutive polls: they had started promptly, run for ~2.7 h and then stopped reporting, and
nothing would have reclaimed them until CRAB's 24 h wall-clock rule fired 21 hours later. The
production had 598 of 600 branches done and sat there.

The signal here is a flag file per running job in ONE flat directory on the same storage the job
must write its products to, refreshed by the job itself. Its modification time is the evidence, so
the driver needs exactly one directory listing per interval however many jobs are in flight (about
1.6 s at 3000 entries) and never reads a flag's content. A job whose flag has not moved for
``missed_checks`` intervals is declared failed, which puts it straight through law's ordinary
retry path -- there is no separate resubmission mechanism, and none is wanted.

Two things this deliberately does not do. It does not kill the job: ``crab kill`` has no per-job
form, and law's ``cancel`` ignores the ids it is given and kills the whole task, so condemning one
job would take every healthy sibling with it. The slot is abandoned instead, and CRAB's wall-clock
rule reclaims it. And it never condemns on the flag alone in bulk: if writing to the storage breaks
while reading it still works, every flag goes stale at once while the jobs are fine, so a listing
in which most running jobs look stale is read as an infrastructure fault and produces no verdicts.

Ported from the DSProd CRAB production tooling.
"""

import datetime
import json
import os
import socket
import tempfile
import threading

from FLAF.RunKit.grid_tools import gfal_copy, gfal_ls_safe, gfal_rm

#: the flat directory, under the production's own storage prefix, that holds one flag per live job
HEARTBEAT_DIR = "heartbeat"

DEFAULTS = {
    # on for CRAB and nothing else: the failure mode is a batch system holding a slot it cannot
    # account for, and a local run has no slot to hold
    "enabled": True,
    #: how often a job refreshes its flag, and how often the driver lists the directory
    "interval_minutes": 30,
    #: consecutive refreshes a job may miss before it is declared failed
    "missed_checks": 2,
    #: never condemn more than this many jobs in one interval, whatever the evidence says
    "max_per_interval": 5,
    #: how often one branch may be rescued this way before it is left to the wall-clock rule; a
    #: branch that stalls wherever it runs is the branch's problem, not the site's, and each
    #: verdict spends one of its attempts
    "max_per_branch": 1,
    #: a listing in which this fraction of running jobs looks stale is an infrastructure fault
    "max_stale_fraction": 0.5,
    #: log the verdicts that would have been issued, issue none
    "dry_run": False,
}


def watchdog_config(crab_cfg):
    """The `watchdog` block of the CRAB config, merged over the defaults."""
    raw = (crab_cfg or {}).get("watchdog", {})
    if raw is False or raw is None:
        return dict(DEFAULTS, enabled=False)
    if raw is True:
        return dict(DEFAULTS)
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise RuntimeError(
            f"unknown watchdog setting(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}"
        )
    cfg = dict(DEFAULTS, **raw)
    if int(cfg["interval_minutes"]) < 1 or int(cfg["missed_checks"]) < 1:
        raise RuntimeError(
            "watchdog.interval_minutes and watchdog.missed_checks must both be >= 1"
        )
    return cfg


class Heartbeat:
    """Job side: refresh one flag every `interval_seconds` for as long as the payload runs.

    Used as a context manager so the flag is dropped on the way out however the payload ends. A
    failure to write is never allowed to disturb the job: the whole point is to observe it, and a
    storage hiccup that killed the payload would be far worse than a missed beat.
    """

    def __init__(self, uri, interval_seconds, voms_token=None, label=None, log=None):
        self.uri = uri
        self.interval = max(1.0, float(interval_seconds))
        self.voms_token = voms_token
        self.label = label or {}
        self.log = log or (lambda msg: None)
        self._stop = threading.Event()
        self._thread = None
        self._beats = 0

    def _write(self):
        # the driver only ever reads the modification time; the content is for a human looking at
        # a verdict after the fact
        payload = dict(
            self.label,
            beat=self._beats,
            utc=datetime.datetime.utcnow().replace(microsecond=0).isoformat(),
            host=socket.gethostname(),
            pid=os.getpid(),
        )
        fd, path = tempfile.mkstemp(prefix="flaf-beat-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            # force: gfal-copy does not overwrite, and an in-place overwrite is the signal -- a
            # remove-then-copy would leave a window in which the flag is simply absent, which the
            # driver cannot tell from a stalled job
            gfal_copy(path, self.uri, voms_token=self.voms_token, force=True, verbose=0)
            self._beats += 1
        finally:
            os.unlink(path)

    def _loop(self):
        while True:
            try:
                self._write()
            except Exception as exc:  # never let the heartbeat break the payload
                self.log(f"heartbeat: could not refresh {self.uri}: {exc}")
            if self._stop.wait(self.interval):
                return

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        try:
            gfal_rm(self.uri, voms_token=self.voms_token, verbose=0)
        except Exception as err:
            # a flag left behind must not fail a payload that has already finished
            self.log(f"heartbeat: could not remove {self.uri}: {err}")
        return False


class StallWatchdog:
    """Driver side: one listing per interval, and a verdict for jobs whose flag has stopped moving.

    `refresh()` runs from the poll callback (throttled there) and `verdicts()` from the job
    manager's `query()`, which law calls once per CRAB project **concurrently**, so every piece of
    state here is taken under one lock.
    """

    def __init__(self, flag_dir_uri, cfg=None, voms_token=None, publish=None):
        # may be a callable: resolving the uri builds the remote file system, which shells out to
        # `voms-proxy-info`, and this object is constructed alongside the job manager -- long
        # before anything is listed, and in tests that have no grid environment at all
        self._flag_dir = flag_dir_uri
        self.cfg = dict(DEFAULTS) if cfg is None else dict(cfg)
        self.voms_token = voms_token
        self.publish = publish or (lambda msg: None)
        self._lock = threading.Lock()
        self._ages = None  # branch name -> mtime, from the last listing that worked
        self._listed = False
        self._first_running = {}  # (crab_num, task_name) -> when first seen running
        self._per_branch = {}  # branch -> verdicts issued so far
        self._by_id = {}  # (crab_num, task_name) -> (job_num, branches)
        self._seen_flag = set()  # job ids a flag has ever been observed for
        # messages already published, so a poll loop cannot repeat them
        self._reported = set()
        self._issued_this_interval = 0
        self._listing_failures = 0

    @property
    def flag_dir(self):
        if callable(self._flag_dir):
            self._flag_dir = self._flag_dir()
        return self._flag_dir

    @property
    def enabled(self):
        return bool(self.cfg.get("enabled"))

    @property
    def interval_seconds(self):
        return int(self.cfg["interval_minutes"]) * 60

    @property
    def stale_seconds(self):
        return self.interval_seconds * int(self.cfg["missed_checks"])

    def refresh(self):
        """List the flag directory once. Returns False if the listing could not be read."""
        if not self.enabled:
            return False
        # catch_stderr: until a job writes its first flag the directory does not exist, so the
        # CLI's own "404 File not found" would be printed on every interval of every wave. The
        # outcome is reported here instead, once per transition, so a listing that starts failing
        # after it had been working -- the case worth noticing -- is not lost in that noise.
        entries = gfal_ls_safe(
            self.flag_dir, voms_token=self.voms_token, catch_stderr=True, verbose=0
        )
        with self._lock:
            self._issued_this_interval = 0
            if entries is None:
                # Could be an outage, could be a directory no job has written to yet. Either way
                # there is no evidence, and stale evidence must not accumulate across it.
                self._listing_failures += 1
                if self._listing_failures == 1:
                    self.publish(
                        f"watchdog: cannot list {self.flag_dir} -- no verdicts until it can be "
                        "read (expected until the first job writes a flag)"
                    )
                self._ages = None
                self._listed = False
                return False
            if self._listing_failures:
                self.publish(
                    f"watchdog: {self.flag_dir} readable again after "
                    f"{self._listing_failures} failed listing(s)"
                )
                self._listing_failures = 0
            self._ages = {
                e.name: e.date for e in entries if e.date is not None and not e.is_dir
            }
            self._listed = True
            return True

    #: slack for the minute resolution of listed modification times
    _stamp_resolution = datetime.timedelta(seconds=60)

    def _age(self, branches, now, since=None):
        """Seconds since the freshest flag of `branches`, or None if none of them has one.

        Flags older than `since` are ignored: flags are named by branch, so a flag left by an
        earlier attempt of the same branch -- a worker that died without removing it -- is
        not evidence about the attempt running now, and would condemn it on the first poll
        that sees it running, before its own first beat. Ignoring a stamp only ever delays
        evidence: a live job's next beat is newer than `since`.
        """
        stamps = [self._ages.get(str(b)) for b in branches]
        stamps = [s for s in stamps if s is not None and (since is None or s >= since)]
        if not stamps:
            return None
        # a timestamp in the future (skew between driver and storage) counts as fresh
        return max(0.0, (now - max(stamps)).total_seconds())

    @staticmethod
    def _key(job_id):
        """A key that matches law's `JobId` namedtuple and its json form in job_data.

        Both are (crab_num, task_name, proj_dir) in that order, but one arrives as a namedtuple
        from the job manager and the other as a list from the dumped job data, and the project
        directory of the same job can differ between them (law rewrites it on resubmission).
        """
        parts = tuple(job_id)[:2]
        return (int(parts[0]), str(parts[1]))

    def set_jobs(self, jobs):
        """Publish law's job_data mapping so verdicts can name the branches behind a job id."""
        by_id = {}
        for job_num, entry in (jobs or {}).items():
            job_id = (entry or {}).get("job_id")
            branches = (entry or {}).get("branches") or []
            if not job_id or not branches:
                continue
            try:
                by_id[self._key(job_id)] = (job_num, list(branches))
            except (TypeError, ValueError, IndexError):
                continue
        with self._lock:
            self._by_id = by_id

    def verdicts(self, result, now=None):
        """Which of the jobs `result` reports running have stopped refreshing their flag.

        `result` is the status dict the job manager just fetched, keyed by law's `JobId`. Returns
        {that same key: reason} for the jobs to be failed.
        """
        if not self.enabled:
            return {}
        now = now or datetime.datetime.utcnow()
        with self._lock:
            if not self._listed or self._ages is None:
                return {}
            running, stale = [], []
            for job_id, data in result.items():
                if not isinstance(data, dict) or data.get("status") != "running":
                    continue
                try:
                    known = self._by_id.get(self._key(job_id))
                except (TypeError, ValueError, IndexError):
                    continue
                if not known:
                    continue
                job_num, branches = known
                key = self._key(job_id)
                self._first_running.setdefault(key, now)
                running.append(job_id)
                age = self._age(
                    branches,
                    now,
                    since=self._first_running[key] - self._stamp_resolution,
                )
                # a job that has not had time to write its first flag is not evidence of anything
                since_seen = (now - self._first_running[key]).total_seconds()
                grace = self.stale_seconds + self.interval_seconds
                if age is not None:
                    self._seen_flag.add(key)
                    if age >= self.stale_seconds:
                        stale.append((job_id, job_num, branches, age))
                elif key in self._seen_flag:
                    # The flag was there and is gone, which is exactly what a job does on its way
                    # out: the heartbeat context removes it, and CRAB goes on reporting the job as
                    # running for minutes afterwards. Failing it here would resubmit a branch that
                    # had just been produced. A worker that dies without exiting cleanly leaves
                    # its flag behind instead, and that is caught above as a stale one, which is
                    # the shape the real incident had.
                    self._note(
                        ("gone", key),
                        f"watchdog: job {job_num} had a heartbeat and no longer does -- reading "
                        "that as exiting or restarting, not stalled",
                    )
                elif since_seen >= grace:
                    stale.append((job_id, job_num, branches, None))
            if not stale:
                return {}
            # writing to the storage can break while reading it still works, and then every flag
            # goes stale at once while every job is healthy
            fraction = float(self.cfg["max_stale_fraction"])
            if running and len(stale) > max(1, int(fraction * len(running))):
                self.publish(
                    f"watchdog: {len(stale)} of {len(running)} running jobs have a stale "
                    f"heartbeat -- reading that as a storage fault, not {len(stale)} dead jobs, "
                    "and issuing no verdicts"
                )
                return {}
            out = {}
            for job_id, job_num, branches, age in sorted(
                stale, key=lambda s: str(s[1])
            ):
                if self._issued_this_interval >= int(self.cfg["max_per_interval"]):
                    self.publish(
                        "watchdog: reached max_per_interval "
                        f"({self.cfg['max_per_interval']}); leaving the rest for the next interval"
                    )
                    break
                repeat = max(self._per_branch.get(b, 0) for b in branches)
                if repeat >= int(self.cfg["max_per_branch"]):
                    self._note(
                        ("cap", tuple(branches)),
                        f"watchdog: branch(es) {list(branches)} have stalled {repeat + 1} times "
                        "now -- that is the branch, not the slot, so it is left to CRAB's "
                        "wall-clock limit instead of spending another attempt",
                    )
                    continue
                seen = (
                    "no heartbeat"
                    if age is None
                    else f"heartbeat {age / 60:.0f} min old"
                )
                reason = (
                    f"stalled: {seen}, threshold {self.stale_seconds / 60:.0f} min "
                    f"({self.cfg['missed_checks']} x {self.cfg['interval_minutes']} min)"
                )
                if self.cfg.get("dry_run"):
                    # once per job, not once per poll: a dry run exists to be read
                    self._note(
                        ("dry", self._key(job_id)),
                        f"watchdog (dry run): would fail job {job_num} -- {reason}",
                    )
                    continue
                for b in branches:
                    self._per_branch[b] = self._per_branch.get(b, 0) + 1
                self._issued_this_interval += 1
                out[job_id] = reason
            return out

    def _note(self, key, message):
        """Publish `message` the first time `key` produces it, and never again."""
        if key in self._reported:
            return
        self._reported.add(key)
        self.publish(message)

    def forget(self, job_id):
        """Drop the grace clock of a job id law has replaced, so a resubmission starts clean."""
        with self._lock:
            key = self._key(job_id)
            self._first_running.pop(key, None)
            self._seen_flag.discard(key)
            self._reported -= {("gone", key), ("dry", key)}
