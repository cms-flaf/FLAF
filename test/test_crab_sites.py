#!/usr/bin/env python3
"""The CRAB site list, the whitelist built from it, and the per-site quarantine record.

Ported from the DSProd tests that pin the same code there, each one tied to an incident
of a real CRAB production:

- a task refused by the CRAB server for a whitelist naming `T3_CH_CERN_HelixNebula_REHA`,
  which the old `computeunits` rule took for a processing site (2026-09-12);
- three broken sites cycling back into the whitelist every six hours, because a lifted
  quarantine also wiped the record that earned it (2026-09-08..10);
- a black hole quarantined two hours after its failures became visible, because the
  24 h failure rate is slow against a site that fails in seconds (2026-09-13).

Plus what FLAF adds on top: several CRAB workflows of one law process share one record.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.run_tools import crab_sites
from FLAF.run_tools.crab_sites import (
    CRIC_MIN_SITES,
    CRIC_URL,
    DEFAULTS,
    SiteStats,
    _parse_cric_sites,
    processing_sites,
    resolve_whitelist,
)

MINUTE = 60.0
HOUR = 3600.0

#: a list long enough to pass the shape guard, as a real one is
MANY_SITES = [f"T2_XX_Site{i:03d}" for i in range(60)]

UNREACHABLE = "http://127.0.0.1:9/nope"


def stats(tmpdir, **cfg):
    return SiteStats(os.path.join(tmpdir, "crab_site_stats.json"), cfg or None)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)


# -- the site list ---------------------------------------------------------------------------


class TheSiteListCrabValidatesAgainst(unittest.TestCase):
    PAYLOAD = {
        "desc": {"columns": ["type", "site_name", "alias"]},
        "result": [
            ["psn", "T2_DE_DESY", "T2_DE_DESY"],
            ["psn", "T1_US_FNAL", "T1_US_FNAL"],
            ["phedex", "T1_US_FNAL_Disk", "T1_US_FNAL_Disk"],
            ["psn", "T2_CH_CERN", "T2_CH_CERN"],
            ["phedex", "T3_CH_CERN_HelixNebula_REHA", "T3_CH_CERN_HelixNebula_REHA"],
        ],
    }

    #: a full-size payload, so that the shape guard passes and the parse itself is tested
    BIG_PAYLOAD = {
        "desc": {"columns": ["type", "site_name", "alias"]},
        "result": [["psn", s, s] for s in MANY_SITES]
        + [
            ["phedex", "T1_US_FNAL_Disk", "T1_US_FNAL_Disk"],
            ["lcg", "CERN-HNREHA", "T3_CH_CERN_HelixNebula_REHA"],
            ["phedex", "T3_CH_CERN_HelixNebula_REHA", "T3_CH_CERN_HelixNebula_REHA"],
            ["psn", MANY_SITES[0], MANY_SITES[0]],
        ],
    }

    def test_the_url_is_the_preset_crab_asks(self):
        self.assertIn("preset=site-names", CRIC_URL)
        self.assertEqual(CRIC_MIN_SITES, 50)

    def test_only_processing_site_names_survive(self):
        sites = _parse_cric_sites(self.PAYLOAD)
        self.assertEqual(sites, ["T1_US_FNAL", "T2_CH_CERN", "T2_DE_DESY"])
        self.assertNotIn("T3_CH_CERN_HelixNebula_REHA", sites)
        self.assertNotIn("T1_US_FNAL_Disk", sites)

    def test_the_columns_are_read_by_name_not_by_position(self):
        reordered = {
            "desc": {"columns": ["alias", "type", "site_name"]},
            "result": [[row[2], row[0], row[1]] for row in self.PAYLOAD["result"]],
        }
        self.assertEqual(_parse_cric_sites(reordered), _parse_cric_sites(self.PAYLOAD))

    def test_a_payload_of_another_shape_yields_nothing_rather_than_guessing(self):
        for payload in (
            {},
            {"result": None},
            {"desc": {}, "result": []},
            None,
            [],
            [{"name": "T2_CH_CERN", "computeunits": [1]}],
            {"desc": "columns", "result": [["psn", "T2_CH_CERN", "T2_CH_CERN"]]},
            # the old, plain `?json` shape: one object per site, keyed by name
            {"T2_CH_CERN": {"name": "T2_CH_CERN", "computeunits": [1]}},
        ):
            self.assertEqual(_parse_cric_sites(payload), [], payload)

    def test_a_fetch_never_yields_a_site_crab_would_refuse(self):
        """The incident itself: the name that got a production refused never comes back,
        from any row type it appears under, and a duplicated PSN row is listed once."""
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            with mock.patch.object(
                crab_sites.urllib.request, "urlopen"
            ), mock.patch.object(
                crab_sites.json, "load", return_value=self.BIG_PAYLOAD
            ):
                sites = processing_sites(cache_path=cache)
            with open(cache) as f:
                cached = json.load(f)
        self.assertEqual(sites, sorted(MANY_SITES))
        self.assertNotIn("T3_CH_CERN_HelixNebula_REHA", sites)
        self.assertNotIn("T1_US_FNAL_Disk", sites)
        self.assertEqual(cached, sites)

    def test_a_short_parse_is_a_failure_not_a_small_site_pool(self):
        """The silent failure this closes: a shrunken whitelist looks like a working production."""
        with mock.patch.object(crab_sites.urllib.request, "urlopen"), mock.patch.object(
            crab_sites.json, "load", return_value=self.PAYLOAD
        ), self.assertRaises(RuntimeError) as caught:
            processing_sites(cache_path=None)
        self.assertIn("processing sites", str(caught.exception))

    def test_a_short_fresh_cache_is_not_returned_cric_is_asked_instead(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            write_json(cache, ["T2_DE_DESY"])
            with mock.patch.object(
                crab_sites.urllib.request, "urlopen"
            ) as urlopen, mock.patch.object(
                crab_sites.json, "load", side_effect=[["T2_DE_DESY"], self.BIG_PAYLOAD]
            ):
                sites = processing_sites(cache_path=cache)
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(sites, sorted(MANY_SITES))

    def test_a_full_fresh_cache_avoids_the_network(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            write_json(cache, MANY_SITES)
            with mock.patch.object(crab_sites.urllib.request, "urlopen") as urlopen:
                self.assertEqual(processing_sites(cache_path=cache), MANY_SITES)
            urlopen.assert_not_called()

    def test_a_short_cache_is_refused_too_not_only_a_short_fetch(self):
        """Every path out is checked: a truncated cache shrinks the pool just as quietly."""
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            write_json(cache, ["T2_DE_DESY"])
            with mock.patch.object(
                crab_sites.urllib.request, "urlopen", side_effect=OSError("no network")
            ), self.assertRaises(RuntimeError):
                processing_sites(cache_path=cache)

    def test_an_unreachable_cric_falls_back_to_the_cache_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            write_json(cache, MANY_SITES)
            os.utime(cache, (0, 0))  # far older than the 24 h reuse window
            with mock.patch.object(
                crab_sites.urllib.request, "urlopen", side_effect=OSError("no network")
            ), mock.patch("builtins.print") as printed:
                sites = processing_sites(cache_path=cache)
            self.assertEqual(sites, MANY_SITES)
            said = "\n".join(str(c.args[0]) for c in printed.call_args_list if c.args)
            self.assertIn("falling back", said)
            self.assertIn("h ago", said)

    def test_an_unreachable_cric_with_no_cache_raises(self):
        with mock.patch.object(
            crab_sites.urllib.request, "urlopen", side_effect=OSError("no network")
        ), self.assertRaises(RuntimeError) as caught:
            processing_sites(cache_path=None)
        self.assertIn("could not read the CMS site list", str(caught.exception))

    def test_a_real_unreachable_url_with_a_stale_cache_falls_back(self):
        """No mock of the network: the same fallback through a real failed connection."""
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            write_json(cache, MANY_SITES)
            os.utime(cache, (0, 0))
            with mock.patch("builtins.print"):
                self.assertEqual(
                    processing_sites(cache, url=UNREACHABLE, timeout=1), MANY_SITES
                )

    def test_an_unusable_stale_cache_is_a_runtime_error_not_anything_else(self):
        """The caller catches RuntimeError only — that is what lets the quarantine degrade
        with a warning while a configured blacklist aborts. A corrupt or short cache, or one
        removed between the existence check and the read, must not escape as ValueError
        or OSError."""
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            for name, content in (
                ("corrupt", "not json {"),
                ("short", '["T2_CH_CERN"]'),
            ):
                with self.subTest(name):
                    with open(cache, "w") as f:
                        f.write(content)
                    os.utime(cache, (0, 0))
                    with mock.patch("builtins.print"), self.assertRaises(
                        RuntimeError
                    ) as caught:
                        processing_sites(cache, url=UNREACHABLE, timeout=1)
                    self.assertIn(
                        "could not read the CMS site list", str(caught.exception)
                    )
            write_json(cache, MANY_SITES)
            os.utime(cache, (0, 0))
            real_open = open

            def vanishing_open(path, *args, **kwargs):
                if path == cache:
                    raise FileNotFoundError(path)
                return real_open(path, *args, **kwargs)

            with mock.patch("builtins.print"), mock.patch(
                "builtins.open", vanishing_open
            ), self.assertRaises(RuntimeError):
                processing_sites(cache, url=UNREACHABLE, timeout=1)

    def test_a_cache_removed_underneath_does_not_crash_the_freshness_check(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            with mock.patch.object(
                crab_sites.os.path, "getmtime", side_effect=FileNotFoundError(cache)
            ), mock.patch.object(
                crab_sites.urllib.request, "urlopen"
            ), mock.patch.object(
                crab_sites.json, "load", return_value=self.BIG_PAYLOAD
            ):
                self.assertEqual(processing_sites(cache_path=cache), sorted(MANY_SITES))


# -- the whitelist ---------------------------------------------------------------------------


class AnExclusionIsNeverDefeatedByAGlob(unittest.TestCase):
    SITES = ["T1_DE_KIT", "T2_CH_CERN", "T2_EE_Estonia", "T2_US_MIT", "T3_CH_PSI"]

    def test_an_excluded_site_missing_from_the_list_still_removes_the_glob(self):
        """A stale or short list need not hold the excluded site; left in place, `T2_*`
        would have CRAB send jobs to it anyway, since the whitelist has precedence."""
        sites = [s for s in self.SITES if s != "T2_EE_Estonia"]
        out = resolve_whitelist(["T1_*", "T2_*", "T3_*"], ["T2_EE_Estonia"], sites)
        self.assertNotIn("T2_*", out)
        self.assertNotIn("T2_EE_Estonia", out)
        self.assertEqual(out, ["T1_*", "T2_CH_CERN", "T2_US_MIT", "T3_*"])

    def test_a_blacklist_glob_inside_a_whitelist_glob_expands_it(self):
        sites = [s for s in self.SITES if not s.startswith("T2_US_")]
        out = resolve_whitelist(["T2_*"], ["T2_US_*"], sites)
        self.assertEqual(out, ["T2_CH_CERN", "T2_EE_Estonia"])

    def test_entries_covering_nothing_excluded_stay_globs(self):
        out = resolve_whitelist(["T1_*", "T2_*", "T3_*"], ["T2_EE_Estonia"], self.SITES)
        self.assertEqual(out[0], "T1_*")
        self.assertEqual(out[-1], "T3_*")

    def test_a_concrete_entry_absent_from_the_list_is_kept(self):
        """Only an overlap expands an entry; an unrelated name is not dropped for being
        unknown to the list."""
        out = resolve_whitelist(["T2_XX_New", "T2_*"], ["T2_EE_Estonia"], self.SITES)
        self.assertEqual(out[0], "T2_XX_New")

    def test_an_excluded_concrete_entry_absent_from_the_list_still_disappears(self):
        out = resolve_whitelist(["T2_CH_CERN", "T2_XX_Gone"], ["T2_XX_*"], self.SITES)
        self.assertEqual(out, ["T2_CH_CERN"])

    def test_a_blacklist_glob_not_inside_the_entry_expands_it_through_the_list(self):
        """`*_EE_*` is no sub-pattern of `T2_*`, so only the listed sites show the overlap."""
        out = resolve_whitelist(["T1_*", "T2_*"], ["*_EE_*"], self.SITES)
        self.assertEqual(out, ["T1_*", "T2_CH_CERN", "T2_US_MIT"])

    def test_the_glob_aware_blacklist_semantics_are_kept(self):
        self.assertEqual(
            resolve_whitelist(["T1_*", "T2_*", "T3_*"], ["T3_*"], self.SITES),
            ["T1_*", "T2_*"],
        )
        self.assertEqual(
            resolve_whitelist(["T2_CH_CERN", "T2_US_MIT"], ["T2_US_*"], self.SITES),
            ["T2_CH_CERN"],
        )
        with self.assertRaises(RuntimeError):
            resolve_whitelist(["T2_US_MIT"], ["T2_US_MIT"], self.SITES)


# -- the quarantine: escalation ----------------------------------------------------------------

BAD = "T2_XX_Broken"
GOOD = "T2_YY_Fine"


def make_it_fail(st, now, n=6, site=BAD):
    """Enough failures at `site`, against a real baseline elsewhere, to earn a quarantine."""
    for i in range(n):
        st.record(site, False, now=now + i)
    for i in range(40):
        st.record(GOOD, True, now=now + i)


def quarantine_span(st, site, now):
    """Hours the current quarantine of `site` still has to run."""
    return (st.sites[site]["quarantined_until"] - now) / HOUR


class TheFirstQuarantine(unittest.TestCase):
    def test_it_lasts_the_configured_base(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            make_it_fail(st, 0.0)
            self.assertEqual(st.blacklist(now=100.0), [BAD])
            self.assertAlmostEqual(quarantine_span(st, BAD, 100.0), 24.0, places=3)

    def test_the_default_is_a_day_not_six_hours(self):
        self.assertEqual(DEFAULTS["quarantine_hours"], 24.0)
        self.assertEqual(DEFAULTS["max_quarantine_hours"], 768.0)


class EachFurtherOneDoubles(unittest.TestCase):
    """A site that keeps failing is held out for longer each time."""

    def serve_and_reoffend(self, st, rounds, start=0.0):
        """Earn a quarantine, sit it out, fail again -- `rounds` times. Returns the ban lengths."""
        spans, now = [], start
        for _ in range(rounds):
            make_it_fail(st, now)
            self.assertEqual(st.blacklist(now=now + 100.0), [BAD])
            spans.append(quarantine_span(st, BAD, now + 100.0))
            now = st.sites[BAD]["quarantined_until"] + 1.0
            self.assertEqual(st.blacklist(now=now), [])
        return spans

    def test_the_lengths_double(self):
        with tempfile.TemporaryDirectory() as d:
            spans = self.serve_and_reoffend(stats(d), 4)
        self.assertEqual([round(s) for s in spans], [24, 48, 96, 192])

    def test_the_doubling_stops_at_32_days(self):
        with tempfile.TemporaryDirectory() as d:
            spans = self.serve_and_reoffend(stats(d), 8)
        self.assertEqual(
            [round(s) for s in spans], [24, 48, 96, 192, 384, 768, 768, 768]
        )
        self.assertEqual(round(max(spans) / 24), 32, "the cap is 32 days")

    def test_the_cap_is_configurable_and_respected(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, quarantine_hours=1.0, max_quarantine_hours=2.0)
            spans = self.serve_and_reoffend(st, 4)
        self.assertEqual([round(s) for s in spans], [1, 2, 2, 2])

    def test_polling_through_an_active_ban_does_not_escalate_it(self):
        """`blacklist()` re-judges on every submission and a ban lasts many polls:
        escalating per poll rather than per offence would turn 24 h into 96 h in three.
        """
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            make_it_fail(st, 0.0)
            self.assertEqual(st.blacklist(now=100.0), [BAD])
            deadline = st.sites[BAD]["quarantined_until"]
            for poll in range(1, 8):
                self.assertEqual(st.blacklist(now=100.0 + poll * 900.0), [BAD])
            self.assertEqual(st.sites[BAD]["quarantines"], 1)
            self.assertEqual(st.sites[BAD]["quarantined_until"], deadline)

    def test_a_record_from_before_the_escalation_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "crab_site_stats.json")
            write_json(
                path,
                {
                    "version": 1,
                    "sites": {BAD: {"events": [[0.0, 0]], "quarantined_until": 0.0}},
                },
            )
            st = SiteStats(path)
            self.assertEqual(st.sites[BAD]["quarantines"], 0)
            self.assertEqual(st.sites[BAD]["cleared_at"], 0.0)
            self.assertEqual(len(st.sites[BAD]["events"]), 1)


class TheRecordSurvivesTheQuarantine(unittest.TestCase):
    def test_expiry_itself_no_longer_empties_the_record(self):
        """A short ban, so that the rolling window cannot be what removes the evidence."""
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, quarantine_hours=1.0)
            make_it_fail(st, 0.0)
            st.blacklist(now=100.0)
            n_before = len(st.sites[BAD]["events"])
            st.blacklist(now=st.sites[BAD]["quarantined_until"] + 1.0)
            self.assertEqual(
                len(st.sites[BAD]["events"]),
                n_before,
                "the failures that earned the ban were forgotten",
            )

    def test_over_a_long_ban_it_is_the_count_that_carries_the_history(self):
        """With the defaults the ban is as long as the window, so the outcomes age out by
        themselves; the escalation count is what has to survive."""
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            make_it_fail(st, 0.0)
            st.blacklist(now=100.0)
            st.blacklist(now=st.sites[BAD]["quarantined_until"] + 1.0)
            self.assertEqual(st.sites[BAD]["events"], [])
            self.assertEqual(st.sites[BAD]["quarantines"], 1)

    def test_the_count_survives_a_save_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            make_it_fail(st, 0.0)
            st.blacklist(now=100.0)
            st.save()
            with open(st.path) as f:
                self.assertEqual(json.load(f)["sites"][BAD]["quarantines"], 1)
            again = stats(d)
            self.assertEqual(again.sites[BAD]["quarantines"], 1)

    def test_the_moment_a_ban_ended_survives_a_restart_too(self):
        """A driver restarted between the expiry and the next wave would otherwise
        re-quarantine the site instantly, on the pre-ban evidence."""
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, quarantine_hours=1.0)
            make_it_fail(st, 0.0)
            st.blacklist(now=100.0)
            lifted = st.sites[BAD]["quarantined_until"] + 1.0
            st.blacklist(now=lifted)
            st.save()
            again = stats(d)
            self.assertEqual(
                again.sites[BAD]["cleared_at"], st.sites[BAD]["cleared_at"]
            )
            self.assertEqual(
                again.blacklist(now=lifted + 1.0),
                [],
                "a restart re-quarantined the site on the evidence its ban was served for",
            )


class ASecondChanceIsReallyGiven(unittest.TestCase):
    def test_a_lifted_ban_does_not_re_arm_on_the_old_failures(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            make_it_fail(st, 0.0)
            st.blacklist(now=100.0)
            lifted = st.sites[BAD]["quarantined_until"] + 1.0
            self.assertEqual(
                st.blacklist(now=lifted),
                [],
                "re-quarantined on the very evidence the ban was served for",
            )
            self.assertEqual(st.blacklist(now=lifted + HOUR), [])

    def test_but_one_fresh_generation_of_failures_is_enough(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            make_it_fail(st, 0.0)
            st.blacklist(now=100.0)
            lifted = st.sites[BAD]["quarantined_until"] + 1.0
            make_it_fail(st, lifted)
            self.assertEqual(st.blacklist(now=lifted + 100.0), [BAD])
            self.assertAlmostEqual(
                quarantine_span(st, BAD, lifted + 100.0), 48.0, places=3
            )

    def test_a_site_that_comes_back_healthy_is_never_re_quarantined(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            make_it_fail(st, 0.0)
            st.blacklist(now=100.0)
            lifted = st.sites[BAD]["quarantined_until"] + 1.0
            for i in range(30):
                st.record(BAD, True, now=lifted + i)
            for i in range(40):
                st.record(GOOD, True, now=lifted + i)
            self.assertEqual(st.blacklist(now=lifted + 100.0), [])


# -- the quarantine: bursts --------------------------------------------------------------------

HOLE = "T2_XX_Blackhole"

#: "now" of every burst test. Absolute, because the burst window reaches BACKWARDS: with a
#: timeline that starts at zero the window covers the whole history.
NOW = 1_000_000.0


def fail_fast(st, site, n, at=NOW):
    """`n` failures arriving within two minutes, as a black hole's do."""
    for i in range(n):
        st.record(site, False, now=at - 2 * MINUTE + i * (2 * MINUTE / max(n, 1)))


def recent_successes(st, site, n, minutes=20.0):
    """`n` successes spread over the last `minutes`."""
    for i in range(n):
        st.record(
            site, True, now=NOW - minutes * MINUTE + i * (minutes * MINUTE / max(n, 1))
        )


def earlier_successes(st, site, n, first_hours_ago=8.0, last_hours_ago=1.0):
    """`n` successes well before the burst window but inside the 24 h rate window."""
    span = (first_hours_ago - last_hours_ago) * HOUR
    for i in range(n):
        st.record(site, True, now=NOW - first_hours_ago * HOUR + i * span / max(n, 1))


class ABurstIsEnoughOnItsOwn(unittest.TestCase):
    def test_a_site_that_eats_a_wave_is_out_within_the_window(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            recent_successes(st, GOOD, 40)
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [HOLE])

    def test_the_rate_test_alone_would_still_have_been_waiting(self):
        """The site's own successes earlier in the day keep its 24 h ratio under
        `min_failure_rate`."""
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, burst_failures=10**6)  # burst effectively disabled
            recent_successes(st, GOOD, 40)
            earlier_successes(st, HOLE, 40)
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [])

    def test_and_with_the_burst_test_the_same_record_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            recent_successes(st, GOOD, 40)
            earlier_successes(st, HOLE, 40)
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [HOLE])

    def test_the_defaults_are_the_documented_ones(self):
        self.assertEqual(DEFAULTS["burst_failures"], 20)
        self.assertEqual(DEFAULTS["burst_minutes"], 15.0)


class AndOnlyWhenItIsReallyABurst(unittest.TestCase):
    def test_the_same_failures_spread_over_a_day_are_not_one(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            earlier_successes(st, GOOD, 40)
            earlier_successes(st, HOLE, 60)
            # one failure every 40 min: never 20 within a quarter hour
            for i in range(25):
                st.record(HOLE, False, now=NOW - i * 40 * MINUTE)
            self.assertEqual(st.blacklist(now=NOW), [])

    def test_too_few_failures_to_be_one(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            recent_successes(st, GOOD, 40)
            earlier_successes(st, HOLE, 60)
            fail_fast(st, HOLE, 19)
            self.assertEqual(st.blacklist(now=NOW), [])

    def test_jobs_still_running_are_not_what_a_burst_is_measured_against(self):
        """The rolling rate counts every job SENT to a site; the burst counts only what
        ENDED inside its window. 25 failures among 1025 sent is a 2 % rate."""
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            st.set_in_flight({HOLE: 1000})
            recent_successes(st, GOOD, 40)
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [HOLE])
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, burst_failures=10**6)
            st.set_in_flight({HOLE: 1000})
            recent_successes(st, GOOD, 40)
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [])

    def test_a_busy_site_that_mostly_succeeds_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            recent_successes(st, GOOD, 40)
            fail_fast(st, HOLE, 20)
            for i in range(180):
                st.record(HOLE, True, now=NOW - 10 * MINUTE + i * (10 * MINUTE / 180))
            self.assertEqual(st.blacklist(now=NOW), [])


class AFaultEverywhereMustNotBanTheGrid(unittest.TestCase):
    """A payload fault fails everywhere, and looks like a burst everywhere."""

    def test_a_failure_that_hits_every_site_quarantines_none_of_them(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            for site in ("T2_A_A", "T2_B_B", "T2_C_C", "T2_D_D"):
                fail_fast(st, site, 30)
            self.assertEqual(st.blacklist(now=NOW), [])

    def test_but_the_one_site_failing_far_harder_than_the_rest_is(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            for site in ("T2_A_A", "T2_B_B", "T2_C_C"):
                fail_fast(st, site, 4)
                for i in range(36):
                    st.record(
                        site, True, now=NOW - 10 * MINUTE + i * (10 * MINUTE / 36)
                    )
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [HOLE])

    def test_a_site_is_not_its_own_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            earlier_successes(st, HOLE, 400)
            fail_fast(st, HOLE, 300)
            recent_successes(st, GOOD, 30, minutes=10.0)
            self.assertEqual(st.blacklist(now=NOW), [HOLE])

    def test_with_nothing_to_compare_against_nothing_is_quarantined(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            fail_fast(st, HOLE, 50)
            self.assertEqual(st.blacklist(now=NOW), [])

    def test_a_quiet_grid_leaves_the_burst_test_without_a_comparison(self):
        """The burst test needs other sites to have ENDED jobs in the same window; early in a
        wave only the standing rate test can fire, which is the conservative direction.
        """
        with tempfile.TemporaryDirectory() as d:
            st = stats(d)
            earlier_successes(st, GOOD, 40)  # all of it hours ago
            earlier_successes(st, HOLE, 40)
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [])


class TheBurstTestRespectsTheRestOfTheMachinery(unittest.TestCase):
    def test_a_lifted_quarantine_is_not_re_armed_by_the_burst_that_earned_it(self):
        """`cleared_at` bounds the burst window too. The ban is shorter than the window, or
        the failures would age out of it on their own and the clamp would be untested.
        """
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, quarantine_hours=0.1)
            recent_successes(st, GOOD, 40, minutes=8.0)
            fail_fast(st, HOLE, 25)
            self.assertEqual(st.blacklist(now=NOW), [HOLE])
            lifted = st.sites[HOLE]["quarantined_until"] + 1.0
            self.assertEqual(st.blacklist(now=lifted), [])

    def test_a_burst_quarantine_escalates_like_any_other(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, quarantine_hours=1.0)
            earlier_successes(st, GOOD, 40)
            fail_fast(st, HOLE, 25)
            st.blacklist(now=NOW)
            first = st.sites[HOLE]["quarantined_until"] - NOW
            lifted = st.sites[HOLE]["quarantined_until"] + 1.0
            st.blacklist(now=lifted)
            for i in range(40):
                st.record(GOOD, True, now=lifted + i)
            for i in range(25):
                st.record(HOLE, False, now=lifted + 60 + i)
            st.blacklist(now=lifted + 3 * MINUTE)
            second = st.sites[HOLE]["quarantined_until"] - (lifted + 3 * MINUTE)
            self.assertAlmostEqual(second / first, 2.0, places=1)

    def test_a_disabled_record_still_quarantines_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            st = stats(d, enabled=False)
            earlier_successes(st, GOOD, 40)
            fail_fast(st, HOLE, 50)
            self.assertEqual(st.blacklist(now=NOW), [])


# -- loading ---------------------------------------------------------------------------------


class TestLoadingARecordWrittenByAnotherVersion(unittest.TestCase):
    """A file in the analysis data area may not stop a submission: `load` runs while a
    CRAB workflow is being submitted."""

    def write(self, payload):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "stats.json")
        write_json(path, payload)
        return path

    def test_a_site_whose_events_do_not_coerce_is_dropped_and_the_others_kept(self):
        path = self.write(
            {
                "version": 1,
                "sites": {
                    "T2_CH_CERN": {"events": [{"at": 1.0, "ok": 1}]},
                    "T1_DE_KIT": {"events": [[1.0, 0], [2.0, 1]]},
                },
            }
        )
        st = SiteStats(path)
        self.assertEqual(list(st.sites), ["T1_DE_KIT"])
        self.assertEqual(st.sites["T1_DE_KIT"]["events"], [(1.0, 0), (2.0, 1)])

    def test_entries_that_are_not_numbers_are_dropped(self):
        path = self.write(
            {
                "sites": {
                    "T2_CH_CERN": {"events": [["yesterday", 1]]},
                    "T1_DE_KIT": {"events": [], "quarantined_until": "soon"},
                    "T2_IT_Legnaro": {"events": [[1.0, 1, "site"]]},
                    "T2_DE_DESY": {"events": [], "quarantines": "many"},
                    "T2_US_MIT": {"events": [], "cleared_at": "then"},
                }
            }
        )
        self.assertEqual(SiteStats(path).sites, {})

    def test_a_payload_that_is_not_an_object_is_tolerated(self):
        self.assertEqual(SiteStats(self.write([{"T2_CH_CERN": []}])).sites, {})
        self.assertEqual(SiteStats(self.write("nothing")).sites, {})

    def test_what_this_version_writes_still_reads_back(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "stats.json")
        written = SiteStats(path)
        written.record("T1_DE_KIT", False, now=1_000_000.0)
        written.record("T1_DE_KIT", True, now=1_000_001.0)
        written.save()
        self.assertEqual(
            SiteStats(path).sites["T1_DE_KIT"]["events"],
            [(1_000_000.0, 0), (1_000_001.0, 1)],
        )


# -- one record per law process ----------------------------------------------------------------


class SharedRegistryTestCase(unittest.TestCase):
    def setUp(self):
        SiteStats._shared.clear()
        self.addCleanup(SiteStats._shared.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "crab_site_stats.json")


class OneRecordPerFilePerProcess(SharedRegistryTestCase):
    """Several CRAB workflows of one law process each built their own record from the file,
    so a later workflow's `save()` overwrote an earlier one's outcomes and quarantines.
    """

    def test_the_same_path_is_the_same_record(self):
        a = SiteStats.shared(self.path)
        spelled_differently = os.path.join(self.dir, ".", "crab_site_stats.json")
        self.assertIs(SiteStats.shared(spelled_differently), a)

    def test_different_paths_are_different_records(self):
        other = os.path.join(self.dir, "other.json")
        self.assertIsNot(SiteStats.shared(self.path), SiteStats.shared(other))

    def test_the_first_callers_cfg_is_kept(self):
        SiteStats.shared(self.path, {"quarantine_hours": 1.0})
        later = SiteStats.shared(self.path, {"quarantine_hours": 5.0})
        self.assertEqual(later.cfg["quarantine_hours"], 1.0)

    def test_two_workflows_saving_in_turn_keep_both_outcomes(self):
        """The overwrite itself: what one workflow recorded survives the other's save."""
        first = SiteStats.shared(self.path)
        second = SiteStats.shared(self.path)
        first.record("T2_CH_CERN", False, now=NOW)
        first.save()
        second.record("T1_DE_KIT", True, now=NOW)
        second.save()
        self.assertEqual(
            sorted(SiteStats(self.path).sites), ["T1_DE_KIT", "T2_CH_CERN"]
        )

    def test_a_burst_is_caught_through_the_shared_record(self):
        """One workflow sees the healthy sites, another the black hole."""
        healthy_side = SiteStats.shared(self.path)
        hole_side = SiteStats.shared(self.path)
        recent_successes(healthy_side, GOOD, 40)
        fail_fast(hole_side, HOLE, 25)
        self.assertEqual(healthy_side.blacklist(now=NOW), [HOLE])

    def test_an_escalation_is_seen_through_the_shared_record(self):
        one = SiteStats.shared(self.path)
        two = SiteStats.shared(self.path)
        make_it_fail(one, 0.0)
        self.assertEqual(two.blacklist(now=100.0), [BAD])
        lifted = one.sites[BAD]["quarantined_until"] + 1.0
        self.assertEqual(one.blacklist(now=lifted), [])
        make_it_fail(two, lifted)
        self.assertEqual(one.blacklist(now=lifted + 100.0), [BAD])
        self.assertAlmostEqual(
            quarantine_span(one, BAD, lifted + 100.0), 48.0, places=3
        )


class ThePublicMethodsAreSerialised(SharedRegistryTestCase):
    """Job managers harvest from law's query thread pools, so a shared record is used
    from several threads at once; `save()` serialising `sites` while another thread adds a
    site fails with "dictionary changed size during iteration", and two saves of one
    process share one temporary file name."""

    CALLS = {
        "record": lambda st: st.record("T2_CH_CERN", False, now=NOW),
        "set_in_flight": lambda st: st.set_in_flight({"T2_CH_CERN": 1}, source="b"),
        "blacklist": lambda st: st.blacklist(now=NOW),
        "save": lambda st: st.save(),
    }

    def test_each_waits_for_the_lock(self):
        for name, call in self.CALLS.items():
            with self.subTest(name):
                st = SiteStats(os.path.join(self.dir, f"{name}.json"))
                st.record("T1_DE_KIT", True, now=NOW)  # something for save() to write
                done = threading.Event()
                worker = threading.Thread(target=lambda: (call(st), done.set()))
                with st._lock:
                    worker.start()
                    self.assertFalse(
                        done.wait(0.3), f"{name}() ran while the lock was held"
                    )
                worker.join(5.0)
                self.assertTrue(done.is_set(), f"{name}() never ran")

    def test_concurrent_use_loses_nothing_and_raises_nothing(self):
        st = SiteStats.shared(self.path)
        n_threads, n_each = 6, 200
        errors = []

        def work(i):
            try:
                for j in range(n_each):
                    # a new site every few records, so that the dict grows under the others
                    st.record(f"T2_XX_S{i}_{j // 10}", j % 2 == 0, now=NOW + j)
                    if j % 20 == 0:
                        st.set_in_flight({f"T2_XX_S{i}_0": j}, source=i)
                        st.blacklist(now=NOW + j)
                        st.save()
            except Exception as exc:  # surfaced below; a thread's raise is lost
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(
            sum(len(rec["events"]) for rec in st.sites.values()), n_threads * n_each
        )


class InFlightCountsFromSeveralManagers(SharedRegistryTestCase):
    def test_two_sources_add_up(self):
        st = SiteStats.shared(self.path)
        st.set_in_flight({"T2_CH_CERN": 10, "T1_DE_KIT": 3}, source="wf_a")
        st.set_in_flight({"T2_CH_CERN": 5}, source="wf_b")
        self.assertEqual(st.in_flight, {"T2_CH_CERN": 15, "T1_DE_KIT": 3})

    def test_a_source_replaces_only_its_own_counts(self):
        st = SiteStats.shared(self.path)
        st.set_in_flight({"T2_CH_CERN": 10}, source="wf_a")
        st.set_in_flight({"T2_CH_CERN": 5}, source="wf_b")
        st.set_in_flight({"T2_CH_CERN": 1}, source="wf_a")
        self.assertEqual(st.in_flight, {"T2_CH_CERN": 6})
        st.set_in_flight({}, source="wf_b")
        self.assertEqual(st.in_flight, {"T2_CH_CERN": 1})

    def test_without_a_source_the_last_report_replaces_the_previous(self):
        st = SiteStats.shared(self.path)
        st.set_in_flight({"T2_CH_CERN": 10})
        st.set_in_flight({"T1_DE_KIT": 4})
        self.assertEqual(st.in_flight, {"T1_DE_KIT": 4})

    def test_placeholders_are_dropped_per_source(self):
        st = SiteStats.shared(self.path)
        st.set_in_flight({"Unknown": 10, "T2_CH_CERN": 1}, source="wf_a")
        self.assertEqual(st.in_flight, {"T2_CH_CERN": 1})

    def test_the_summed_denominator_is_what_judges_a_site(self):
        """Another workflow's running jobs at a site count towards what was sent there: with
        only one source's count, the rate test would fire on a site the other workflow is
        using successfully."""
        st = SiteStats.shared(self.path)
        for _ in range(10):
            st.record(HOLE, False, now=NOW - 5 * HOUR)
        for _ in range(40):
            st.record(GOOD, True, now=NOW - 5 * HOUR)
        st.set_in_flight({HOLE: 2}, source="wf_a")
        self.assertEqual(st.blacklist(now=NOW), [HOLE])
        SiteStats._shared.clear()
        st = SiteStats.shared(os.path.join(self.dir, "fresh.json"))
        for _ in range(10):
            st.record(HOLE, False, now=NOW - 5 * HOUR)
        for _ in range(40):
            st.record(GOOD, True, now=NOW - 5 * HOUR)
        st.set_in_flight({HOLE: 30}, source="wf_b")
        st.set_in_flight({HOLE: 2}, source="wf_a")
        self.assertEqual(st.blacklist(now=NOW), [])


if __name__ == "__main__":
    unittest.main()
