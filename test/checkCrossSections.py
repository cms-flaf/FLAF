import sys
import yaml
import os


def check_cross_sections(eras):
    """
    Check that cross-sections are well-defined using CrossSectionDB.
    Validates:
    1. All cross-section files can be loaded and parsed
    2. All cross-section entries are valid
    3. All dataset cross-section references exist in the database
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Common.CrossSectionDB import CrossSectionDB

    all_ok = True
    xs_files_checked = set()
    xs_databases = {}

    for era in eras:
        print(f"Checking era: {era}")

        # Check for Run3 structure (separate global.yaml and datasets.yaml)
        global_file = f"config/{era}/global.yaml"
        datasets_file = f"config/{era}/datasets.yaml"

        # Check for Run2 structure (single samples.yaml with GLOBAL section)
        samples_file = f"config/{era}/samples.yaml"

        if os.path.isfile(global_file) and os.path.isfile(datasets_file):
            # Run3 structure
            with open(global_file, "r") as f:
                global_config = yaml.safe_load(f)
            with open(datasets_file, "r") as f:
                datasets = yaml.safe_load(f)
        elif os.path.isfile(samples_file):
            # Run2 structure
            with open(samples_file, "r") as f:
                samples = yaml.safe_load(f)
            global_config = samples.get("GLOBAL", {})
            datasets = {k: v for k, v in samples.items() if k != "GLOBAL"}
        else:
            print(
                f"ERROR: {era}: Neither global.yaml+datasets.yaml nor samples.yaml found."
            )
            all_ok = False
            continue

        xs_file = global_config.get("crossSectionsFile", "")
        if xs_file.startswith("FLAF/"):
            xs_file = xs_file[5:]

        if not xs_file:
            print(f"ERROR: {era}: crossSectionsFile not specified in global.yaml")
            all_ok = False
            continue

        if not os.path.isfile(xs_file):
            print(f"ERROR: {era}: cross-sections file '{xs_file}' does not exist.")
            all_ok = False
            continue

        # Load and validate cross-sections database
        if xs_file not in xs_files_checked:
            print(f"  Validating cross-sections file: {xs_file}")
            try:
                xs_db = CrossSectionDB([xs_file])
                xs_databases[xs_file] = xs_db
                xs_files_checked.add(xs_file)
                print(
                    f"  SUCCESS: Loaded {len(xs_db.entries)} cross-section entries from {xs_file}"
                )
            except Exception as e:
                print(
                    f"ERROR: {era}: Failed to load cross-sections from {xs_file}: {e}"
                )
                all_ok = False
                continue
        else:
            xs_db = xs_databases[xs_file]

        # Check that all referenced cross-sections exist
        for ds_name, ds_desc in datasets.items():
            if "crossSection" in ds_desc:
                xs_name = ds_desc["crossSection"]
                try:
                    xs_value = xs_db.getValue(xs_name)
                    # Check that the cross-section value is valid
                    if xs_value <= 0:
                        print(
                            f"ERROR: {era}/{ds_name}: cross-section '{xs_name}' has invalid value {xs_value} (must be > 0)"
                        )
                        all_ok = False
                except Exception as e:
                    print(
                        f"ERROR: {era}/{ds_name}: cross-section '{xs_name}' not found or invalid: {e}"
                    )
                    all_ok = False

    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check that cross-sections are well-defined."
    )
    parser.add_argument("era", type=str, nargs="+", help="eras to check")
    args = parser.parse_args()

    if check_cross_sections(args.era):
        print("All cross-section checks passed.")
        sys.exit(0)
    else:
        print("Some cross-section checks failed.")
        sys.exit(1)
