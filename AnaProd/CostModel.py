"""Cost estimation and job packing for AnaTuple production.

The AnaTuple producer runs a single RDataFrame event loop that fills one output tree per
systematic variation, so the wall time of a file is, to a good approximation,

    seconds(file) = overhead + sec_per_event(dataset) * n_events(file)

`sec_per_event` depends on the analysis selection and varies by more than an order of
magnitude between datasets (a dilepton skim where half the events are selected costs far
more per event than a hadronic one).  It is therefore measured rather than configured:
first by a short probe run at the head of the production, then refined from the durations
of the production jobs themselves.

`n_events` is taken from the input file catalogue when the storage exposes it (DAS for
Rucio-discovered datasets), otherwise it is inferred from the file size, which every
backend reports for free.

Everything is keyed by the law ``--version``: by convention a version fixes the physics
selection, so a calibration measured while producing one era is reused by the others.
"""

import json
import os
import statistics
import tempfile

# Order matters: earlier tiers are better known and are packed less conservatively.
TIERS = ("job", "probe", "catalogue", "process", "group", "default")

DEFAULT_PARAMS = {
    # --- packing -----------------------------------------------------------------
    # Target wall time of one HTCondor job.  Units whose own estimate exceeds it are
    # submitted alone; cheaper ones are combined until the target is reached.
    "target_job_hours": 6.0,
    # Upper bound on the branches per job, independent of cost (bounds the bookkeeping
    # law does inside the job and keeps --branches lists readable).
    "max_units_per_job": 50,
    # A job must fit into max_runtime with this much headroom, so the packing capacity
    # is min(target_job_hours, max_runtime / runtime_safety).
    "runtime_safety": 2.5,
    # Default queue footprint.  A finite value is what creates submission waves, and
    # therefore the opportunity to re-pack the remaining work with better estimates.
    "parallel_jobs": 2000,
    # --- calibration -------------------------------------------------------------
    "probe_enabled": True,
    "probe_events": 5000,
    # Fixed per-job cost (worker setup, RDataFrame JIT, corrections initialisation).
    "overhead_sec": 300.0,
    # Priors, used only until something has been measured.  They are deliberately
    # middle-of-the-road: the "default" tier safety factor covers the spread.
    "default_sec_per_event": 0.02,
    "default_events_per_byte": 4.5e-4,
    # Charged to a file whose size and event count are both unknown.
    "default_file_seconds": 3600.0,
    # Bin capacity is divided by this factor, per tier of the estimate that produced it.
    "tier_safety": {
        "job": 1.0,
        "probe": 1.3,
        "catalogue": 1.3,
        "process": 2.0,
        "group": 3.0,
        "default": 4.0,
    },
    # --- retries -----------------------------------------------------------------
    # A resubmitted job gets more runtime, and more memory when the request is explicit
    # (with request_memory_mb unset the site derives memory from the cpu count).
    "request_memory_mb": None,
    "retry_runtime_factor": 1.5,
    "retry_memory_factor": 1.25,
    "retry_max_factor": 3.0,
    # --- online refinement -------------------------------------------------------
    # Weight of a new measurement in the exponential moving average.
    "measurement_weight": 0.4,
    # Ignore durations shorter than this many overheads: they carry no signal about
    # the event loop.
    "min_measurement_overheads": 1.5,
}


def merged_params(user_params=None):
    """DEFAULT_PARAMS updated with *user_params* (one level deep for `tier_safety`)."""
    params = dict(DEFAULT_PARAMS)
    params["tier_safety"] = dict(DEFAULT_PARAMS["tier_safety"])
    for key, value in (user_params or {}).items():
        if key == "tier_safety" and isinstance(value, dict):
            params["tier_safety"].update(value)
        else:
            params[key] = value
    return params


def entry_key(nano_version, dataset_name):
    """Store key.  The nano source is part of it because the same dataset costs a very
    different amount per event as a channel skim and as central unskimmed NanoAOD."""
    return f"{nano_version}|{dataset_name}"


class CostModel:
    """Per-dataset cost calibration, persisted as a small json keyed by law version."""

    def __init__(self, params=None, entries=None, overhead_sec=None):
        self.params = merged_params(params)
        self.entries = entries or {}
        self.overhead_sec = (
            self.params["overhead_sec"] if overhead_sec is None else overhead_sec
        )
        self.dirty = False

    # ------------------------------------------------------------------ persistence

    @classmethod
    def load(cls, path, params=None):
        model = cls(params=params)
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                print(f"CostModel: ignoring unreadable store {path}: {e}")
                return model
            model.entries = data.get("entries", {})
            model.overhead_sec = data.get("overhead_sec", model.overhead_sec)
        return model

    def save(self, path):
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"overhead_sec": self.overhead_sec, "entries": self.entries}
        # Atomic replace: the store is rewritten while jobs are being submitted.
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        self.dirty = False

    # ------------------------------------------------------------------ ingestion

    def set_from_probe(self, nano_version, dataset_name, probe):
        """Ingest one `AnaTupleCostProbeTask` result."""
        if not probe.get("ok"):
            return False
        n_scanned = probe.get("n_scanned_events") or 0
        loop_sec = probe.get("loop_seconds")
        if n_scanned <= 0 or not loop_sec or loop_sec <= 0:
            return False
        entry = self.entries.setdefault(entry_key(nano_version, dataset_name), {})
        entry["sec_per_event"] = loop_sec / n_scanned
        entry["source"] = "probe"
        entry["n_samples"] = 1
        entry["n_trees"] = probe.get("n_trees")
        entry["era"] = probe.get("era")
        n_total = probe.get("n_original_events")
        size = probe.get("input_size")
        if n_total and size:
            entry["events_per_byte"] = n_total / size
        setup_sec = probe.get("setup_seconds")
        if setup_sec and setup_sec > 0:
            # The probe is the cleanest measurement of the fixed per-job cost we ever
            # get, because its event loop is negligible next to the setup.
            w = self.params["measurement_weight"]
            self.overhead_sec = (1.0 - w) * self.overhead_sec + w * setup_sec
        self.dirty = True
        return True

    def set_events_per_byte_from_catalogue(self, nano_version, dataset_name, file_info):
        """Derive events-per-byte from the file catalogue when it carries event counts."""
        pairs = [
            (info["n_events"], info["size"])
            for info in file_info.values()
            if info.get("n_events") and info.get("size")
        ]
        if not pairs:
            return False
        entry = self.entries.setdefault(entry_key(nano_version, dataset_name), {})
        entry["events_per_byte"] = sum(n for n, _ in pairs) / sum(s for _, s in pairs)
        self.dirty = True
        return True

    def add_measurement(self, nano_version, dataset_name, seconds, n_events):
        """Fold a completed production job into the calibration."""
        if n_events <= 0:
            return False
        floor = self.params["min_measurement_overheads"] * self.overhead_sec
        if seconds <= floor:
            return False
        sec_per_event = (seconds - self.overhead_sec) / n_events
        if sec_per_event <= 0:
            return False
        entry = self.entries.setdefault(entry_key(nano_version, dataset_name), {})
        previous = entry.get("sec_per_event")
        if previous and entry.get("source") == "job":
            w = self.params["measurement_weight"]
            entry["sec_per_event"] = (1.0 - w) * previous + w * sec_per_event
        else:
            entry["sec_per_event"] = sec_per_event
        entry["source"] = "job"
        entry["n_samples"] = entry.get("n_samples", 0) + 1
        self.dirty = True
        return True

    # ------------------------------------------------------------------ estimation

    def _entry(self, nano_version, dataset_name):
        return self.entries.get(entry_key(nano_version, dataset_name))

    def _peer_median(self, nano_version, dataset_names, field):
        values = []
        for name in dataset_names:
            entry = self._entry(nano_version, name)
            if entry and entry.get(field):
                values.append(entry[field])
        return statistics.median(values) if values else None

    def sec_per_event(self, nano_version, dataset_name, peers=None, group_peers=None):
        """(value, tier) for the per-event cost of *dataset_name*."""
        entry = self._entry(nano_version, dataset_name)
        if entry and entry.get("sec_per_event"):
            tier = "job" if entry.get("source") == "job" else "probe"
            return entry["sec_per_event"], tier
        value = self._peer_median(nano_version, peers or [], "sec_per_event")
        if value:
            return value, "process"
        value = self._peer_median(nano_version, group_peers or [], "sec_per_event")
        if value:
            return value, "group"
        return self.params["default_sec_per_event"], "default"

    def events_per_byte(self, nano_version, dataset_name, peers=None, group_peers=None):
        entry = self._entry(nano_version, dataset_name)
        if entry and entry.get("events_per_byte"):
            return entry["events_per_byte"]
        for candidates in (peers or [], group_peers or []):
            value = self._peer_median(nano_version, candidates, "events_per_byte")
            if value:
                return value
        return self.params["default_events_per_byte"]

    def n_events(self, nano_version, dataset_name, info, peers=None, group_peers=None):
        """Event count of one input file: catalogued when known, else from its size."""
        if info:
            if info.get("n_events"):
                return float(info["n_events"]), True
            if info.get("size"):
                rho = self.events_per_byte(
                    nano_version, dataset_name, peers, group_peers
                )
                return float(info["size"]) * rho, False
        return None, False

    def estimate(
        self,
        nano_version,
        dataset_name,
        info,
        peers=None,
        group_peers=None,
        fraction=1.0,
        max_events=None,
    ):
        """(seconds, tier) for one work unit: *fraction* of the file *info* describes.

        *max_events* caps the event count, for runs that process only a prefix of every
        file (``--test``).
        """
        rate, tier = self.sec_per_event(nano_version, dataset_name, peers, group_peers)
        n_events, exact = self.n_events(
            nano_version, dataset_name, info, peers, group_peers
        )
        if n_events is not None and max_events:
            n_events = min(n_events, float(max_events))
        if n_events is None:
            # Neither an event count nor a size: nothing is known about this file, so
            # charge it a whole default file and let the tier safety factor keep the
            # packing conservative.
            return (
                self.overhead_sec + self.params["default_file_seconds"] * fraction,
                "default",
            )
        if not exact and tier in ("job", "probe"):
            # The rate is measured but the event count is inferred from the file size.
            tier = "catalogue"
        return self.overhead_sec + rate * n_events * fraction, tier


_SCAN_WINDOW = 128


def pack_units(units, capacity_sec, max_units_per_job, tier_safety=None):
    """Group work units into jobs of bounded duration.

    *units* is a sequence of ``(key, seconds, tier)``.  Returns a list of lists of keys,
    ordered by decreasing total cost so the longest jobs start first (longest-processing-
    time-first minimises the makespan).  A unit that does not fit on its own gets a job
    to itself rather than being dropped.

    The tier of an estimate scales the capacity, not the cost: a job built entirely from
    well-measured units may fill the whole target, while one built from guesses is kept
    small so that an underestimate cannot produce a job that overruns the wall clock.
    """
    tier_safety = tier_safety or DEFAULT_PARAMS["tier_safety"]
    capacity_sec = max(float(capacity_sec), 1.0)
    max_units_per_job = max(int(max_units_per_job), 1)

    def safety(tier):
        return max(float(tier_safety.get(tier, 1.0)), 1.0)

    ordered = sorted(units, key=lambda u: (-u[1], u[0]))
    bins = []  # [load, worst_safety, [keys]]
    for key, seconds, tier in ordered:
        seconds = max(float(seconds), 0.0)
        placed = False
        # Units arrive largest-first, so the bins with room are the recent ones; scanning
        # only those keeps the packing O(n) instead of O(n^2) for 10k+ units without
        # measurably changing the result.
        for b in bins[-_SCAN_WINDOW:]:
            if len(b[2]) >= max_units_per_job:
                continue
            worst = max(b[1], safety(tier))
            if (b[0] + seconds) * worst <= capacity_sec:
                b[0] += seconds
                b[1] = worst
                b[2].append(key)
                placed = True
                break
        if not placed:
            bins.append([seconds, safety(tier), [key]])
    bins.sort(key=lambda b: (-b[0], b[2][0]))
    return [b[2] for b in bins]


def scaled_bounds(begin, end, n_source, n_target):
    """Map an entry range on a tree of *n_source* entries onto one of *n_target*.

    Used to slice `Events_NotSelected` by the same fraction as `Events`.  Consecutive
    chunks share a boundary by construction, so the mapped ranges are disjoint and their
    union is the whole target tree -- which is what makes the per-chunk denominator sums
    add back up to the whole-file values.
    """
    if n_source <= 0:
        return 0, 0
    return (n_target * begin) // n_source, (n_target * end) // n_source


def chunk_bounds(n_total, chunk_index, n_chunks):
    """Entry range ``[begin, end)`` of chunk *chunk_index* out of *n_chunks*.

    Boundaries are derived from the total entry count alone, so every consumer computes
    the same partition without any shared state, the parts are disjoint and their union
    is the whole file.
    """
    n_total = max(int(n_total), 0)
    n_chunks = max(int(n_chunks), 1)
    chunk_index = min(max(int(chunk_index), 0), n_chunks - 1)
    begin = (n_total * chunk_index) // n_chunks
    end = (n_total * (chunk_index + 1)) // n_chunks
    return begin, end
