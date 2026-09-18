#!/usr/bin/env python3
"""Check that Run 3 nanoAOD config entries point to NanoAOD datasets."""

import os
import unittest

import yaml


flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = os.path.join(flaf_repo, "config")


def iter_run3_dataset_files():
    for era in sorted(os.listdir(config_dir)):
        if not era.startswith("Run3_"):
            continue
        datasets_file = os.path.join(config_dir, era, "datasets.yaml")
        if os.path.isfile(datasets_file):
            yield era, datasets_file


class TestNanoAODDatasetPaths(unittest.TestCase):
    def test_nanoaod_paths_end_in_nanoaod_dataset_tiers(self):
        failures = []

        for era, datasets_file in iter_run3_dataset_files():
            with open(datasets_file, "r") as f:
                datasets = yaml.safe_load(f) or {}

            for dataset_name, dataset_desc in datasets.items():
                nanoaod_versions = dataset_desc.get("nanoAOD") or {}
                for version, dataset_path in nanoaod_versions.items():
                    if not isinstance(dataset_path, str):
                        failures.append(
                            f"{era}/{dataset_name}/nanoAOD/{version}: {dataset_path!r}"
                        )
                        continue
                    dataset_tier = dataset_path.rstrip("/").split("/")[-1]
                    if dataset_tier not in {"NANOAOD", "NANOAODSIM"}:
                        failures.append(
                            f"{era}/{dataset_name}/nanoAOD/{version}: {dataset_path}"
                        )

        self.assertEqual(
            failures,
            [],
            "nanoAOD entries must point to NANOAOD/NANOAODSIM datasets:\n"
            + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
