import sys
import yaml
import os
import re


def load_naming_rules(rules_file):
    """Load dataset naming rules from a YAML file."""
    if not os.path.isfile(rules_file):
        return None

    with open(rules_file, "r") as f:
        rules = yaml.safe_load(f)

    return rules


def check_dataset_name(dataset_name, naming_rules):
    """
    Check if a dataset name complies with naming rules.
    Returns: (is_valid, error_messages)
    """
    errors = []

    if naming_rules is None:
        # No rules specified, allow all names
        return True, errors

    # Check allowed name pattern
    allowed_pattern = naming_rules.get("allowed_name_pattern")
    if allowed_pattern:
        if not re.match(allowed_pattern, dataset_name):
            errors.append(
                f"Dataset name '{dataset_name}' contains invalid characters (must match: {allowed_pattern})"
            )

    # Check for known name variants that should not be used
    known_variants = naming_rules.get("known_name_variants", {})
    for variant, replacement in known_variants.items():
        if re.search(variant, dataset_name):
            errors.append(
                f"Dataset name '{dataset_name}' contains '{variant}', use '{replacement}' instead"
            )

    return len(errors) == 0, errors


def check_dataset_naming(eras, rules_file=None):
    """
    Check that dataset names comply with naming rules.
    """
    all_ok = True
    naming_rules = None

    if rules_file:
        if os.path.isfile(rules_file):
            print(f"Loading naming rules from: {rules_file}")
            naming_rules = load_naming_rules(rules_file)
        else:
            print(
                f"WARNING: Naming rules file '{rules_file}' does not exist, skipping naming checks"
            )
            return True

    for era in eras:
        print(f"Checking dataset naming for era: {era}")

        # Check for Run3 structure (separate datasets.yaml)
        datasets_file = f"config/{era}/datasets.yaml"

        # Check for Run2 structure (single samples.yaml with GLOBAL section)
        samples_file = f"config/{era}/samples.yaml"

        if os.path.isfile(datasets_file):
            # Run3 structure
            with open(datasets_file, "r") as f:
                datasets = yaml.safe_load(f)
        elif os.path.isfile(samples_file):
            # Run2 structure
            with open(samples_file, "r") as f:
                samples = yaml.safe_load(f)
            datasets = {k: v for k, v in samples.items() if k != "GLOBAL"}
        else:
            print(f"ERROR: {era}: Neither datasets.yaml nor samples.yaml found.")
            all_ok = False
            continue

        if datasets is None:
            print(f"WARNING: {era}: No datasets found in configuration files")
            continue

        for ds_name, ds_desc in datasets.items():
            is_valid, errors = check_dataset_name(ds_name, naming_rules)
            if not is_valid:
                for error in errors:
                    print(f"ERROR: {era}: {error}")
                all_ok = False

    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check that dataset names comply with naming rules."
    )
    parser.add_argument("era", type=str, nargs="+", help="eras to check")
    parser.add_argument(
        "--rules",
        type=str,
        required=False,
        default=None,
        help="YAML file with dataset naming rules",
    )
    args = parser.parse_args()

    if check_dataset_naming(args.era, args.rules):
        print("All dataset naming checks passed.")
        sys.exit(0)
    else:
        print("Some dataset naming checks failed.")
        sys.exit(1)
