#!/usr/bin/env python3
"""Check that the NanoAOD containers the datasets of an analysis read from Rucio exist.

For every era, the datasets the analysis actually uses (the physics model's processes, as
Setup resolves them, with the reused-MC datasets of Run3_2025/2026 included) are mapped to
the NanoAOD source they are read from, the same way the tasks do it:

  - the dataset has its own `fs_nanoAOD`   -> read from that storage, not checked here;
  - era `nanoAODVersions` is not set       -> HLepRare skims from the era's `fs_nanoAOD`
                                              (user configuration), not checked here;
  - otherwise `nanoAOD.<version>` of the dataset is a Rucio container (DAS path).

Each Rucio container must exist, have content, and have every block on a disk replica
(tape-only or absent blocks cannot be read by the tasks). For a container that is missing,
empty or not VALID in DBS, the containers of the same primary dataset and campaign (`/<primary>/<campaign>*/<tier>`)
are listed with their DBS status and disk availability, as candidates for the right name;
nothing is rewritten.

Needs a valid grid proxy (X509_USER_PROXY) and the Rucio client (see RunKit/grid_tools.py);
DBS status additionally needs dasgoclient and is skipped with a notice when it is missing.
GitHub Actions has no grid credentials, so this check is meant for manual runs or a CI
runner that has a proxy.

Usage (from an analysis checkout, or with --analysis):
    python3 FLAF/test/checkDatasetAvailability.py Run3_2024 Run3_2025
    python3 FLAF/test/checkDatasetAvailability.py --flaf-only Run3_2024   # every FLAF config entry
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import yaml

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAPE_SUFFIXES = ("_Tape", "_Buffer", "_Export")
DAS_CLIENT_FALLBACK = "/cvmfs/cms.cern.ch/common/dasgoclient"


def get_client():
    sys.path.insert(0, flaf_repo)
    from RunKit.grid_tools import get_rucio_client

    return get_rucio_client()


def new_client():
    # requests sessions are not shared between threads: one client per thread.
    from rucio.client import Client

    return Client()


def dbs_status(names, das_client):
    """Map dataset name -> DBS status (VALID, INVALID, PRODUCTION, ...) or None if unknown."""
    result = {}
    if das_client is None:
        return result

    def query(name):
        proc = subprocess.run(
            [das_client, "-query", f"dataset dataset={name} status=*", "-json"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return name, None
        try:
            records = json.loads(proc.stdout)
        except ValueError:
            return name, None
        for record in records:
            for entry in record.get("dataset", []):
                if entry.get("name") == name and "status" in entry:
                    return name, entry["status"]
        return name, "NOT_IN_DBS"

    with ThreadPoolExecutor(8) as pool:
        for name, status in pool.map(query, names):
            result[name] = status
    return result


def container_state(name):
    """Return {state, blocks, files, no_disk_blocks} for one Rucio container.

    state: OK (every block on a disk replica), PARTIAL (some blocks only on tape or without
    replicas), EMPTY (no content), MISSING (unknown to Rucio).
    """
    from rucio.common.exception import DataIdentifierNotFound

    client = new_client()
    try:
        blocks = [b["name"] for b in client.list_content("cms", name)]
    except DataIdentifierNotFound:
        return {"state": "MISSING", "blocks": 0, "files": 0, "no_disk_blocks": 0}
    if not blocks:
        return {"state": "EMPTY", "blocks": 0, "files": 0, "no_disk_blocks": 0}

    def block_info(block):
        local = new_client()
        files = 0
        on_disk = False
        for rep in local.list_dataset_replicas("cms", block):
            files = max(files, rep.get("length") or 0)
            if (
                rep.get("state") == "AVAILABLE"
                and rep.get("length")
                and rep.get("available_length") == rep.get("length")
                and not rep["rse"].endswith(TAPE_SUFFIXES)
            ):
                on_disk = True
        if files == 0:
            files = local.get_did("cms", block).get("length") or 0
        return files, on_disk

    with ThreadPoolExecutor(6) as pool:
        infos = list(pool.map(block_info, blocks))
    n_files = sum(f for f, _ in infos)
    n_no_disk = sum(1 for _, d in infos if not d)
    if n_files == 0:
        state = "EMPTY"
    elif n_no_disk:
        state = "PARTIAL"
    else:
        state = "OK"
    return {
        "state": state,
        "blocks": len(blocks),
        "files": n_files,
        "no_disk_blocks": n_no_disk,
    }


def campaign_pattern(name):
    """/primary/processed-vN/tier -> /primary/processed*/tier (None if not parseable)."""
    parts = name.split("/")
    if len(parts) != 4:
        return None
    campaign = re.sub(r"-v\d+$", "", parts[2])
    return f"/{parts[1]}/{campaign}*/{parts[3]}"


def find_candidates(name, client):
    pattern = campaign_pattern(name)
    if pattern is None:
        return []
    return sorted(
        c for c in client.list_dids("cms", {"name": pattern}, did_type="container")
    )


def collect_setup_datasets(ana_path, era):
    """Return {dataset_name: (source_kind, version, das_name)} as the tasks resolve them."""
    sys.path.insert(0, ana_path)
    sys.modules.setdefault("ROOT", mock.MagicMock())
    from FLAF.Common.Setup import Setup

    setup = Setup(ana_path=ana_path, period=era, law_run_version="test")
    versions = setup.global_params.get("nanoAODVersions", {}) or {}
    result = {}
    for name, dataset in setup.datasets.items():
        if "fs_nanoAOD" in dataset:
            result[name] = ("dataset_fs", None, None)
            continue
        is_data = dataset["process_group"] == "data"
        version = versions.get("data" if is_data else "mc", "HLepRare")
        if version == "HLepRare":
            result[name] = ("HLepRare", version, None)
            continue
        das_cfg = dataset.get("nanoAOD", {})
        if isinstance(das_cfg, str):
            das_name = das_cfg
        else:
            das_name = (das_cfg or {}).get(version)
        if das_name is None:
            result[name] = ("no_entry", version, None)
        else:
            result[name] = ("rucio", version, das_name)
    return result


def collect_flaf_datasets(era):
    """Return {dataset_name[tag]: ('rucio', tag, das_name)} for every FLAF config entry."""
    with open(os.path.join(flaf_repo, "config", era, "datasets.yaml")) as f:
        datasets = yaml.safe_load(f) or {}
    result = {}
    for name, desc in datasets.items():
        das_cfg = desc.get("nanoAOD") or {}
        if isinstance(das_cfg, str):
            das_cfg = {"": das_cfg}
        for tag, das_name in das_cfg.items():
            result[f"{name}[{tag}]"] = ("rucio", tag, das_name)
    return result


def describe(info, status):
    text = info["state"]
    if info["state"] in ("OK", "PARTIAL"):
        text += f" ({info['files']} files in {info['blocks']} blocks"
        if info["no_disk_blocks"]:
            text += f", {info['no_disk_blocks']} blocks without disk replica"
        text += ")"
    if status is not None:
        text += f", DBS {status}"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--analysis",
        default=os.path.dirname(flaf_repo),
        help="analysis checkout containing FLAF/, config/ (default: parent of FLAF)",
    )
    parser.add_argument(
        "--flaf-only",
        action="store_true",
        help="check every nanoAOD entry of FLAF/config/<era>/datasets.yaml instead of "
        "the datasets an analysis uses",
    )
    parser.add_argument("--no-dbs", action="store_true", help="skip DBS status")
    parser.add_argument("eras", nargs="+", help="eras to check")
    args = parser.parse_args()
    ana_path = os.path.abspath(args.analysis)

    get_client()
    das_client = None if args.no_dbs else shutil.which("dasgoclient")
    if das_client is None and not args.no_dbs and os.path.exists(DAS_CLIENT_FALLBACK):
        das_client = DAS_CLIENT_FALLBACK
    if das_client is None:
        print("NOTE: dasgoclient not available, DBS status not checked.")

    per_era = {}
    for era in args.eras:
        if args.flaf_only:
            per_era[era] = collect_flaf_datasets(era)
        else:
            per_era[era] = collect_setup_datasets(ana_path, era)

    das_names = sorted(
        {v[2] for items in per_era.values() for v in items.values() if v[0] == "rucio"}
    )
    print(f"Querying Rucio for {len(das_names)} distinct containers...")
    with ThreadPoolExecutor(4) as pool:
        states = dict(zip(das_names, pool.map(container_state, das_names)))
    statuses = dbs_status(das_names, das_client)

    client = new_client()
    n_failed = 0
    for era in args.eras:
        items = per_era[era]
        label = "FLAF config" if args.flaf_only else os.path.basename(ana_path)
        counts = {}
        problems = []
        for name, (kind, version, das_name) in sorted(items.items()):
            if kind == "rucio":
                info = states[das_name]
                status = statuses.get(das_name)
                key = info["state"]
                if key == "OK" and status not in (None, "VALID"):
                    key = "OK_BUT_DBS_" + status
                counts[key] = counts.get(key, 0) + 1
                if key != "OK":
                    problems.append((name, version, das_name, info, status))
            else:
                counts[kind] = counts.get(kind, 0) + 1
                if kind == "no_entry":
                    problems.append((name, version, None, None, None))
        print(f"\n=== {label} {era}: {len(items)} datasets ===")
        print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
        if counts.get("HLepRare") or counts.get("dataset_fs"):
            print(
                "  not checked: HLepRare skims / dataset-level fs_nanoAOD storage "
                "(no DAS name; location is user configuration)"
            )
        for name, version, das_name, info, status in problems:
            if das_name is None:
                print(f"  NO ENTRY  {name}: no nanoAOD.{version} entry")
                n_failed += 1
                continue
            n_failed += 1
            print(f"  {describe(info, status):<40} {name} [{version}] {das_name}")
            if info["state"] in ("MISSING", "EMPTY") or status not in (None, "VALID"):
                candidates = [c for c in find_candidates(das_name, client)]
                cand_states = {
                    c: states.get(c) or container_state(c) for c in candidates
                }
                cand_dbs = dbs_status(candidates, das_client)
                if not candidates:
                    print("      candidates: none")
                for c in candidates:
                    print(
                        f"      candidate: {c}  {describe(cand_states[c], cand_dbs.get(c))}"
                    )
                good = [
                    c
                    for c in candidates
                    if cand_states[c]["state"] == "OK" and cand_dbs.get(c) == "VALID"
                ]
                if len(good) == 1:
                    print(f"      unique VALID candidate: {good[0]}")
                elif candidates:
                    print(
                        f"      {len(good)} VALID candidates: not decidable automatically"
                    )
    print(f"\n{n_failed} problem(s)")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
