"""CRAB site selection: the CMS processing-site list, whitelist resolution, and the
rolling per-site job record used to quarantine misbehaving sites.

Ported from the DSProd production tooling, where all three were hardened in real
CRAB productions (cms-flaf/DSProd #7, #16, #40, #42, #44). The site list is the set
of CMS Processing Site Names (PSNs) the CRAB server validates a whitelist against.
"""

import fnmatch
import json
import os
import re
import threading
import time
import urllib.request
from collections import Counter

#: CRIC's site table, asked the way CRAB asks it: the `psn` rows of `preset=site-names`
#: are what `WMCore.Services.CRIC.CRIC.getAllPSNs` returns, and that is the list the CRAB
#: TaskWorker validates a whitelist against
CRIC_URL = "https://cms-cric.cern.ch/api/cms/site/query/?json&preset=site-names"

#: how long a cached site list is reused before CRIC is asked again
CRIC_CACHE_SECONDS = 24 * 3600

#: a site list shorter than this is treated as a failed fetch. CRIC lists ~120 PSNs, so
#: sites going offline cannot get there — only a payload that changed shape can, and that
#: would otherwise shrink the whitelist silently instead of raising.
CRIC_MIN_SITES = 50


def _parse_cric_sites(payload):
    """Processing Site Names out of a `preset=site-names` payload.

    The preset answers `{"desc": {"columns": [...]}, "result": [[...], ...]}` — rows are
    lists, not objects — so the columns are read by name: their order is CRIC's to change.
    Any other shape yields an empty list, which `_checked` then refuses.
    """
    if not isinstance(payload, dict):
        return []
    desc = payload.get("desc")
    columns = desc.get("columns") if isinstance(desc, dict) else None
    rows = payload.get("result")
    if not columns or not isinstance(rows, list):
        return []
    entries = [
        dict(zip(columns, row)) for row in rows if isinstance(row, (list, tuple))
    ]
    return sorted(
        {e["alias"] for e in entries if e.get("type") == "psn" and e.get("alias")}
    )


def _checked(sites, source):
    """`sites`, if it is long enough to be the real site list.

    Applied to every path out of `processing_sites`, cache included: a short list is not a
    small grid but a payload that changed shape, and it shrinks the whitelist without any
    error — `resolve_whitelist` only objects when the blacklist empties it completely.
    """
    n = len(sites) if isinstance(sites, list) else 0
    if n < CRIC_MIN_SITES:
        raise RuntimeError(
            f"only {n} processing sites from {source}; expected at least "
            f"{CRIC_MIN_SITES}, so the site pool would be silently shrunk"
        )
    return sites


def processing_sites(cache_path=None, url=CRIC_URL, timeout=60):
    """CMS Processing Site Names, from CRIC, cached on disk.

    These are the only names a `Site.whitelist` may contain. Any other name makes the CRAB
    server refuse the whole task — "A site name T1_US_FNAL_Disk that user specified is not
    in the list of known CMS Processing Site Names" — so neither of the obvious sources
    will do. `/cvmfs/cms.cern.ch/SITECONF` also lists storage endpoints such as
    `T1_US_FNAL_Disk` and `T3_CH_CERNBOX`. CRIC's sites carrying `computeunits` are not the
    PSNs either: measured on 2026-09-13 and again on 2026-10-07, that rule admitted three
    names that are not PSNs (`T3_CH_CERN_HelixNebula_REHA` got a DSProd production refused)
    and missed 39 that are.

    A fresh cache is reused without asking CRIC. When CRIC cannot be read, a stale cache
    is used instead, with a message saying so; otherwise RuntimeError is raised — also
    when the cached list itself is unusable — since that is what callers degrade on.
    """
    try:
        fresh = bool(cache_path) and (
            time.time() - os.path.getmtime(cache_path) < CRIC_CACHE_SECONDS
        )
    except OSError:  # missing, or removed underneath
        fresh = False
    if fresh:
        try:
            with open(cache_path) as f:
                return _checked(json.load(f), cache_path)
        except (OSError, ValueError, RuntimeError):
            pass
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            sites = _checked(_parse_cric_sites(json.load(response)), url)
    except Exception as exc:
        if cache_path and os.path.exists(cache_path):
            try:
                age_h = (time.time() - os.path.getmtime(cache_path)) / 3600.0
                # announced, never silent: this path resubmits an old list, and a wrong
                # list is exactly what gets a submission refused
                print(
                    f"could not read the CMS site list from {url} ({exc}); falling back "
                    f"to {cache_path}, written {age_h:.1f} h ago"
                )
                with open(cache_path) as f:
                    return _checked(json.load(f), cache_path)
            except (OSError, ValueError, RuntimeError) as cache_exc:
                raise RuntimeError(
                    f"could not read the CMS site list from {url}: {exc}; "
                    f"the cached {cache_path} is unusable too: {cache_exc}"
                ) from cache_exc
        raise RuntimeError(f"could not read the CMS site list from {url}: {exc}")
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            tmp = f"{cache_path}.tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(sites, f)
            os.replace(tmp, cache_path)
        except OSError:
            pass
    return sites


def resolve_whitelist(whitelist, blacklist, sites):
    """A `Site.whitelist` from which `blacklist` is actually absent.

    CRAB gives the whitelist precedence: a site matched by both lists is *kept*, and it says
    so only in a warning ("Since the whitelist has precedence, these sites are not considered
    in the blacklist"). With the default all-tier globs that silently defeats every
    exclusion — the configured `crab.blacklist` and the automatic site quarantine alike.

    So a whitelist entry covering an excluded site is expanded, from `sites`, into the sites
    it actually matches minus the excluded ones. Entries covering nothing excluded are left
    alone, which keeps the pool wide and the expansion small: excluding one T2 lists the T2s
    and leaves `T1_*` and `T3_*` as they are. Blacklist entries may be globs too — a site
    is excluded when any blacklist pattern matches it, and a whitelist entry disappears
    when a pattern matches the entry itself.

    An entry is expanded also when it matches a blacklist entry as a name, whether or not
    that name is in `sites`: a site missing from a stale or short list would otherwise
    leave the glob in place, and the glob would let CRAB send jobs there regardless.
    """
    if not blacklist:
        return list(whitelist)

    def excluded(name):
        return any(fnmatch.fnmatch(name, b) for b in blacklist)

    out = []
    for entry in whitelist:
        # an entry that is itself excluded simply disappears
        if excluded(entry):
            continue
        matched = [site for site in sites if fnmatch.fnmatch(site, entry)]
        overlaps = any(fnmatch.fnmatch(b, entry) for b in blacklist)
        if not overlaps and not any(excluded(site) for site in matched):
            out.append(entry)
            continue
        out += [site for site in matched if not excluded(site)]
    if not out:
        raise RuntimeError(
            f"the blacklist {', '.join(blacklist)} excludes every site the whitelist "
            f"{', '.join(whitelist)} allows"
        )
    return out


# CMS site names, e.g. T1_DE_KIT, T2_UK_London_IC. CRAB reports "Unknown" when it does not
# know where a job ran; feeding that back as a blacklist entry would be meaningless at best
# and could have the client reject the whole submission.
_SITE_RE = re.compile(r"^T\d_[A-Za-z0-9_]+$")


def is_site(name):
    """Whether `name` is a CMS site name rather than a placeholder such as "Unknown"."""
    return bool(name and _SITE_RE.match(str(name)))


#: `crab.auto_blacklist` settings and their defaults
DEFAULTS = {
    # set false to keep only the statically configured `crab.blacklist`
    "enabled": True,
    # a site needs at least this many failures before it can be quarantined at all
    "min_failures": 5,
    # ... and at least this fraction of the jobs sent there (ended + in flight) must have failed
    "min_failure_rate": 0.5,
    # ... and it must be failing this many times more often than the other sites, so a bug of
    # our own -- which fails everywhere -- cannot blacklist every site that runs it
    "relative_factor": 2.0,
    # ... judged against at least this many jobs elsewhere. Without a baseline the first site
    # to collect `min_failures` would be quarantined on its own record alone, before there is
    # anything to compare it with; with a single site there is also nowhere else to send the
    # work.
    "min_baseline_jobs": 20,
    # how long a site's FIRST quarantine lasts. Each further one doubles the last, up to
    # `max_quarantine_hours`: the count is kept when a quarantine is lifted, so a site that
    # is still broken is held out for longer and longer instead of returning on a fixed
    # timer. A 6-hour fixed ban let three sites that failed 93-98 % of everything sent to
    # them cycle back into the whitelist three times each (DSProd, 2026-09-08..10), eating a
    # wave every time.
    "quarantine_hours": 24.0,
    # the ceiling the doubling stops at -- 32 days
    "max_quarantine_hours": 768.0,
    # A site that fails this many jobs inside `burst_minutes` is quarantined at once,
    # without waiting for the 24 h rate to clear `min_failure_rate`. The rate test is slow
    # against exactly the site that hurts most: a black hole fails in seconds, cycles
    # through slots faster than any healthy site, and its own earlier successes keep the
    # ratio down until they age out (measured 2026-09-13: ~2 h and several hundred jobs).
    # The burst test still has to pass the relative checks above, so a fault in the payload
    # itself -- which fails everywhere at once -- cannot ban every site in a quarter of an
    # hour. It reads `min_failure_rate` against the jobs that ENDED inside the window only,
    # since a job still in flight carries no timestamp that could place it there. A very
    # large value leaves only the rate test.
    "burst_failures": 20,
    "burst_minutes": 15.0,
    # outcomes older than this stop counting
    "window_hours": 24.0,
    # never quarantine more than this many sites at once
    "max_sites": 10,
}


def resolve_config(cfg):
    """Merge a user `crab.auto_blacklist` mapping onto `DEFAULTS`."""
    out = dict(DEFAULTS)
    if isinstance(cfg, bool):
        out["enabled"] = cfg
    elif cfg:
        out.update({k: v for k, v in cfg.items() if k in DEFAULTS})
    return out


class SiteStats:
    """Job outcomes per site, persisted as JSON, with a rolling window and quarantines.

    A single broken worker node fails jobs in seconds, frees its slot and picks up the next
    one, so one bad host can eat a large share of a production before anything else notices.
    CRAB accepts a blacklist only per *site* and only at submission time, so the record is
    kept here and a site whose recent jobs mostly fail is quarantined; since every wave is a
    new CRAB task, the next wave — retries included — is submitted without it.

    A site's failure rate is measured against every job *sent* there — the ones that already
    ended plus the ones still in flight. Counting only finished jobs does not work: a job
    fails in seconds and succeeds in hours, so early in a production every site's finished
    set is ~100% failures, no site looks worse than the others, and nothing is ever
    quarantined.

    One law process can run several CRAB workflows against the same file; they must share
    one instance (`shared`), or each one's `save()` overwrites the outcomes and quarantines
    the others recorded. Job managers harvest from law's query thread pools, so every public
    method takes the instance's lock.
    """

    _shared = {}
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls, path, cfg=None):
        """The one record for `path` in this process, created on first use.

        The first caller's `cfg` is the one the record keeps; a later caller's `cfg` is
        ignored, since a record judged by two configurations at once has no meaning.
        """
        key = os.path.abspath(path)
        with cls._shared_lock:
            stats = cls._shared.get(key)
            if stats is None:
                stats = cls._shared[key] = cls(path, cfg)
            return stats

    def __init__(self, path, cfg=None):
        self.path = path
        self.cfg = resolve_config(cfg)
        self.sites = {}
        #: jobs currently pending or running per site, summed over every source reporting
        #: them; part of the denominator, never persisted
        self.in_flight = {}
        self._in_flight_by_source = {}
        self._dirty = False
        self._lock = threading.RLock()
        self.load()

    # -- persistence ------------------------------------------------------------------------

    def load(self):
        """Read the persisted record, dropping every entry that does not read back.

        Nothing in here may raise: the file is advisory — it rebuilds within a poll or two —
        and `load` runs while a CRAB workflow is being submitted, so a file this version
        cannot read would otherwise stop a production before its first job.
        """
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        sites = data.get("sites") if isinstance(data, dict) else None
        if not isinstance(sites, dict):
            return
        loaded = {}
        for name, rec in sites.items():
            if not isinstance(rec, dict) or not is_site(name):
                continue
            try:
                loaded[name] = {
                    "events": [
                        (float(t), int(ok)) for t, ok in (rec.get("events") or [])
                    ],
                    "quarantined_until": float(rec.get("quarantined_until") or 0.0),
                    # absent from a record written before quarantines escalated: such a
                    # site starts at the base duration, which is what it would have had
                    "quarantines": int(rec.get("quarantines") or 0),
                    "cleared_at": float(rec.get("cleared_at") or 0.0),
                }
            except (TypeError, ValueError):
                # a record written by a different version is dropped, like corrupt JSON
                continue
        with self._lock:
            self.sites = loaded

    def save(self):
        with self._lock:
            if not self._dirty:
                return
            tmp = f"{self.path}.tmp{os.getpid()}"
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w") as f:
                json.dump({"version": 1, "sites": self.sites}, f)
            os.replace(tmp, self.path)
            self._dirty = False

    # -- recording --------------------------------------------------------------------------

    #: in-flight counts of a source that has not reported for this long stop counting: a
    #: workflow that stopped polling (it ended, or its run was stopped) would otherwise keep
    #: its last snapshot in every other workflow's denominator for the rest of the process
    in_flight_stale_seconds = 3600.0

    def set_in_flight(self, counts, source=None, now=None):
        """Jobs still pending or running per site, as of the latest poll of `source`.

        Each source's counts replace that source's previous ones only, and `in_flight` is
        their sum over the sources heard from recently, so job managers sharing one record
        do not overwrite each other's.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._in_flight_by_source[source] = (
                now,
                {s: n for s, n in (counts or {}).items() if is_site(s)},
            )
            self._sum_in_flight(now)

    def _sum_in_flight(self, now):
        """`in_flight` as the sum over the sources heard from within the stale window."""
        cutoff = now - self.in_flight_stale_seconds
        self._in_flight_by_source = {
            src: (t, per_site)
            for src, (t, per_site) in self._in_flight_by_source.items()
            if t >= cutoff
        }
        total = Counter()
        for _, per_site in self._in_flight_by_source.values():
            total.update(per_site)
        self.in_flight = dict(total)

    def record(self, site, ok, now=None):
        """Note one finished (`ok=True`) or failed job at `site`."""
        if not is_site(site):
            return
        now = time.time() if now is None else now
        with self._lock:
            rec = self.sites.setdefault(site, self._new_record())
            rec["events"].append((float(now), int(bool(ok))))
            self._dirty = True
            self._prune(now)
        # judging happens in `blacklist()`, once the caller has also reported what is still
        # in flight -- doing it here would use a stale, usually empty, denominator

    # -- blacklisting -----------------------------------------------------------------------

    def blacklist(self, now=None):
        """The sites to keep out of the next submission, worst first."""
        if not self.cfg["enabled"]:
            return []
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            self._expire(now)
            # a source that stopped reporting must not stay in the denominator: a whitelist
            # may be built before the first poll of the workflow that is submitting
            self._sum_in_flight(now)
            # re-judge here as well: the in-flight counts move between polls even when
            # nothing new fails
            self._quarantine(now)
            active = [
                (name, rec)
                for name, rec in self.sites.items()
                if rec["quarantined_until"] > now
            ]
            # most failures first, so the cap keeps the worst offenders
            active.sort(key=lambda item: -self._counts(item[0], item[1])[1])
            return [name for name, _ in active[: int(self.cfg["max_sites"])]]

    # -- internals --------------------------------------------------------------------------

    @staticmethod
    def _new_record():
        return {
            "events": [],
            "quarantined_until": 0.0,
            #: quarantines served, which is what the next one's length doubles on
            "quarantines": 0,
            #: when the last quarantine was lifted; outcomes before it no longer judge the site
            "cleared_at": 0.0,
        }

    def _counts(self, site, rec, since=0.0):
        """(jobs sent to `site`, failures among them): ended jobs plus those still in flight.

        `since` drops the outcomes recorded before a moment — the end of the last quarantine.
        The record itself is kept (a quarantine doubles on how many came before it), so
        without this a site would be re-quarantined the instant its ban lifted, on the very
        evidence that ban was served for. Only ended outcomes are filtered: a job dispatched
        before a ban can still be running when it lifts, which biases the fresh rate towards
        leaving the site in until those jobs end.
        """
        events = [e for e in rec["events"] if e[0] >= since] if since else rec["events"]
        n_fail = sum(1 for _, ok in events if not ok)
        return len(events) + self.in_flight.get(site, 0), n_fail

    @staticmethod
    def _ended_since(rec, since):
        """(outcomes recorded since `since`, failures among them).

        Without `in_flight`, unlike `_counts`: a burst is measured over what ended in a short
        window, and a job still running was not necessarily sent inside it.
        """
        events = [e for e in rec["events"] if e[0] >= since]
        return len(events), sum(1 for _, ok in events if not ok)

    def _burst_baseline(self, site, since):
        """(outcomes, failure rate) of every *other* site over the same window."""
        n = n_fail = 0
        for name, rec in self.sites.items():
            if name == site:
                continue
            a, b = self._ended_since(rec, since)
            n += a
            n_fail += b
        return n, ((n_fail / n) if n else 0.0)

    def _is_burst(self, site, rec, now):
        """Whether `site` has just failed a lot of jobs in a short time, and only it has."""
        window = float(self.cfg["burst_minutes"]) * 60.0
        since = max(now - window, float(rec["cleared_at"]))
        n, n_fail = self._ended_since(rec, since)
        if not n or n_fail < int(self.cfg["burst_failures"]):
            return False
        rate = n_fail / n
        if rate < float(self.cfg["min_failure_rate"]):
            return False
        n_other, rate_other = self._burst_baseline(site, since)
        if n_other < int(self.cfg["min_baseline_jobs"]):
            return False
        return rate >= float(self.cfg["relative_factor"]) * rate_other

    def _is_failing(self, site, rec):
        """The standing test: most of what the site was sent, over the whole window, failed."""
        n, n_fail = self._counts(site, rec, since=rec["cleared_at"])
        if not n or n_fail < int(self.cfg["min_failures"]):
            return False
        rate = n_fail / n
        if rate < float(self.cfg["min_failure_rate"]):
            return False
        n_other, rate_other = self._baseline(site)
        if n_other < int(self.cfg["min_baseline_jobs"]):
            return False
        return rate >= float(self.cfg["relative_factor"]) * rate_other

    def _prune(self, now):
        cutoff = now - float(self.cfg["window_hours"]) * 3600.0
        for rec in self.sites.values():
            kept = [(t, ok) for t, ok in rec["events"] if t >= cutoff]
            if len(kept) != len(rec["events"]):
                rec["events"] = kept
                self._dirty = True

    def _expire(self, now):
        """Lift quarantines that have run out, without forgetting that they happened.

        Wiping the record here let a site that stays broken return with a clean sheet, earn
        `min_failures` all over again and buy itself another wave each time. The count is
        therefore kept and drives the next ban's length; only the *evidence* stops judging
        the site, through `cleared_at`.
        """
        for rec in self.sites.values():
            if 0.0 < rec["quarantined_until"] <= now:
                # the moment the ban ended, not the moment it was noticed: whatever the site
                # failed between the two is evidence about it after its ban
                rec["cleared_at"] = rec["quarantined_until"]
                rec["quarantined_until"] = 0.0
                self._dirty = True

    def _baseline(self, site):
        """(jobs, failure rate) of every *other* site.

        The baseline has to exclude the site under test: a black hole that has eaten most of
        the production would otherwise dominate the baseline and excuse itself.
        """
        n = n_fail = 0
        for name, rec in self.sites.items():
            if name == site:
                continue
            a, b = self._counts(name, rec)
            n += a
            n_fail += b
        return n, ((n_fail / n) if n else 0.0)

    def _quarantine_seconds(self, rec):
        """How long this site's next quarantine lasts: the base, doubled per previous one."""
        # clamped only so that an absurd count cannot overflow the multiplication; 2**20 base
        # durations is already many times the ceiling
        doublings = min(int(rec["quarantines"]), 20)
        hours = float(self.cfg["quarantine_hours"]) * 2.0**doublings
        return min(hours, float(self.cfg["max_quarantine_hours"])) * 3600.0

    def _quarantine(self, now):
        for site, rec in self.sites.items():
            if rec["quarantined_until"] > now:
                continue
            if not (self._is_burst(site, rec, now) or self._is_failing(site, rec)):
                continue
            rec["quarantined_until"] = now + self._quarantine_seconds(rec)
            rec["quarantines"] += 1
            self._dirty = True
