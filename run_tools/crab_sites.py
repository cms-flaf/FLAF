"""CRAB site selection: the CMS processing-site list, whitelist resolution, and the
rolling per-site job record used to quarantine misbehaving sites.

Ported from the DSProd production tooling, where all three were hardened in a real
115k-job CRAB campaign (cms-flaf/DSProd #7, #16).
"""

import fnmatch
import json
import os
import re
import time
import urllib.request

#: CRIC's site table — the same source CRAB validates a whitelist against
CRIC_URL = "https://cms-cric.cern.ch/api/cms/site/query/?json"

#: how long a cached site list is reused before CRIC is asked again
CRIC_CACHE_SECONDS = 24 * 3600


def processing_sites(cache_path=None, url=CRIC_URL, timeout=60):
    """CMS site names that actually run jobs, from CRIC, cached on disk.

    `/cvmfs/cms.cern.ch/SITECONF` cannot be used for this: it also lists storage endpoints
    such as `T1_US_FNAL_Disk` and `T3_CH_CERNBOX`, and a whitelist naming one gets the task
    refused server-side — "A site name T1_US_FNAL_Disk that user specified is not in the
    list of known CMS Processing Site Names". CRIC marks the difference: a site that runs
    jobs has `computeunits`.
    """
    if cache_path and os.path.exists(cache_path):
        if time.time() - os.path.getmtime(cache_path) < CRIC_CACHE_SECONDS:
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except (OSError, ValueError):
                pass
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:  # stale is better than nothing
                return json.load(f)
        raise RuntimeError(f"could not read the CMS site list from {url}: {exc}")
    entries = payload.values() if isinstance(payload, dict) else payload
    sites = sorted(
        e["name"] for e in entries if e.get("name") and e.get("computeunits")
    )
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(sites, f)
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
        if not any(excluded(site) for site in matched):
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
    # how long a quarantine lasts; afterwards the site starts from a clean record
    "quarantine_hours": 6.0,
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
    """

    def __init__(self, path, cfg=None):
        self.path = path
        self.cfg = resolve_config(cfg)
        self.sites = {}
        #: jobs currently pending or running per site; part of the denominator, never persisted
        self.in_flight = {}
        self._dirty = False
        self.load()

    # -- persistence ------------------------------------------------------------------------

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        sites = data.get("sites")
        if isinstance(sites, dict):
            self.sites = {
                name: {
                    "events": [
                        (float(t), int(ok)) for t, ok in (rec.get("events") or [])
                    ],
                    "quarantined_until": float(rec.get("quarantined_until") or 0.0),
                }
                for name, rec in sites.items()
                if isinstance(rec, dict) and is_site(name)
            }

    def save(self):
        if not self._dirty:
            return
        tmp = f"{self.path}.tmp"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump({"version": 1, "sites": self.sites}, f)
        os.replace(tmp, self.path)
        self._dirty = False

    # -- recording --------------------------------------------------------------------------

    def set_in_flight(self, counts):
        """Jobs still pending or running per site, as of the latest poll."""
        self.in_flight = {s: n for s, n in (counts or {}).items() if is_site(s)}

    def record(self, site, ok, now=None):
        """Note one finished (`ok=True`) or failed job at `site`."""
        if not is_site(site):
            return
        now = time.time() if now is None else now
        rec = self.sites.setdefault(site, {"events": [], "quarantined_until": 0.0})
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
        self._prune(now)
        self._expire(now)
        # re-judge here as well: the in-flight counts move between polls even when nothing
        # new fails
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

    def _counts(self, site, rec):
        """(jobs sent to `site`, failures among them): ended jobs plus those still in flight."""
        n_fail = sum(1 for _, ok in rec["events"] if not ok)
        return len(rec["events"]) + self.in_flight.get(site, 0), n_fail

    def _prune(self, now):
        cutoff = now - float(self.cfg["window_hours"]) * 3600.0
        for rec in self.sites.values():
            kept = [(t, ok) for t, ok in rec["events"] if t >= cutoff]
            if len(kept) != len(rec["events"]):
                rec["events"] = kept
                self._dirty = True

    def _expire(self, now):
        """Lift quarantines that have run out, and let the site start over."""
        for rec in self.sites.values():
            if 0.0 < rec["quarantined_until"] <= now:
                rec["quarantined_until"] = 0.0
                rec["events"] = []
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

    def _quarantine(self, now):
        for site, rec in self.sites.items():
            if rec["quarantined_until"] > now:
                continue
            n, n_fail = self._counts(site, rec)
            if not n or n_fail < int(self.cfg["min_failures"]):
                continue
            rate = n_fail / n
            if rate < float(self.cfg["min_failure_rate"]):
                continue
            n_other, rate_other = self._baseline(site)
            if n_other < int(self.cfg["min_baseline_jobs"]):
                continue
            if rate < float(self.cfg["relative_factor"]) * rate_other:
                continue
            rec["quarantined_until"] = (
                now + float(self.cfg["quarantine_hours"]) * 3600.0
            )
            self._dirty = True
