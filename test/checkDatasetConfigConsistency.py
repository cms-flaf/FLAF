import yaml
import sys
import re

from yaml import Loader
from yaml.constructor import ConstructorError


def no_duplicates_constructor(loader, node, deep=False):
    """Check for duplicate keys."""

    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        value = loader.construct_object(value_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found duplicate key (%s)" % key,
                key_node.start_mark,
            )
        mapping[key] = value

    return loader.construct_mapping(node, deep)


yaml.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates_constructor
)


class ExceptionMatcher:
    def __init__(self, exceptions):
        self.exceptions = exceptions
        self.used_patterns = set()

    def get_known_exceptions(self, task_name):
        matched_eras = set()
        era_to_pattern_list = {}
        for task_pattern, eras in self.exceptions.items():
            if (
                task_pattern[0] == "^" and re.match(task_pattern, task_name)
            ) or task_pattern == task_name:
                for era in eras:
                    matched_eras.add(era)
                    if era not in era_to_pattern_list:
                        era_to_pattern_list[era] = []
                    era_to_pattern_list[era].append(task_pattern)
                self.used_patterns.add(task_pattern)
        all_ok = True
        era_to_pattern = {}
        for era, patterns in era_to_pattern_list.items():
            if len(patterns) > 1:
                all_ok = False
                patterns_str = ", ".join(patterns)
                print(
                    f"{task_name} is matched by multiple exception patterns that include {era}: {patterns_str}"
                )
            era_to_pattern[era] = patterns[0]
        return all_ok, matched_eras, era_to_pattern_list

    def get_unused_patterns(self):
        return set(self.exceptions.keys()) - self.used_patterns


def get_xs_value(xs_def):
    if isinstance(xs_def, str):
        try:
            xs_value = eval(xs_def)
        except SyntaxError:
            return None
    else:
        xs_value = xs_def
    if not isinstance(xs_value, (int, float)):
        return None
    return xs_value


must_have_properties = {
    "MC": ["crossSection", "generator"],
    "data": ["eraLetter"],
}


def getDatasetType(desc):
    for key, values in must_have_properties.items():
        for item in values:
            if item in desc:
                return key
    return "UNKNOWN"


def check_era_consistency(era, era_desc, xs_db):
    all_ok = True
    sources = {}
    for name, desc in era_desc["datasets"].items():
        datasetType = getDatasetType(desc)
        expected_properties_common = [
            "miniAOD",
            "nanoAOD",
            "dirName",
            "fileNamePattern",
        ]
        if datasetType not in must_have_properties:
            print(f"{era}/{name}: unknown datasetType '{datasetType}'.")
            all_ok = False
            continue
        expected_properties = (
            expected_properties_common + must_have_properties[datasetType]
        )
        for item in desc:
            if item not in expected_properties:
                print(f"{era}/{name}: unexpected property '{item}'.")
                all_ok = False
        dirName = desc.get("dirName", name)
        fileNamePattern = desc.get("fileNamePattern", ".*")
        source_key = (dirName, fileNamePattern)
        if source_key not in sources:
            sources[source_key] = []
        sources[source_key].append(name)

        for item in must_have_properties[datasetType]:
            if item not in desc:
                print(
                    f"{era}/{name}: missing required property '{item}' for {datasetType} dataset."
                )
                all_ok = False
        if datasetType == "MC":
            if "crossSection" in desc:
                xs_name = desc["crossSection"]
                if xs_name not in xs_db:
                    print(
                        f"{era}/{name}: crossSection '{xs_name}' not found in crossSectionsFile."
                    )
                    all_ok = False
                else:
                    xs_entry = xs_db[xs_name]
                    if "crossSec" not in xs_entry:
                        print(
                            f"{era}/{name}: crossSection '{xs_name}' entry missing 'crossSec' field."
                        )
                        all_ok = False
    for source_key, ds_names in sources.items():
        if len(ds_names) > 1:
            dirName, fileNamePattern = source_key
            ds_names_str = ", ".join(ds_names)
            print(
                f"{era}/({ds_names_str}): datasets share the same (dirName, fileNamePattern)=({dirName}, {fileNamePattern})."
            )
            all_ok = False
    return all_ok


def check_dataset_consistency(ds_name, ds_eras_dict, all_eras, exception_matcher):
    datasetTypes = {}
    for era, ds_desc in ds_eras_dict.items():
        datasetType = getDatasetType(ds_desc)
        if datasetType not in datasetTypes:
            datasetTypes[datasetType] = []
        datasetTypes[datasetType].append(era)
    if len(datasetTypes) > 1:
        datasetType_str = ", ".join(
            [f'{k} (in {", ".join(v)})' for k, v in datasetTypes.items()]
        )
        print(f"{ds_name}: inconsistent datasetTypes: {datasetType_str}")
        return False
    datasetType = list(datasetTypes.keys())[0]
    is_data = datasetType == "data"
    if not is_data:
        crossSections = {}
        for era, ds_desc in ds_eras_dict.items():
            crossSection = ds_desc.get("crossSection")
            if crossSection not in crossSections:
                crossSections[crossSection] = []
            crossSections[crossSection].append(era)
        if len(crossSections) > 1:
            crossSection_str = ", ".join(
                [f'{k} (in {", ".join(v)})' for k, v in crossSections.items()]
            )
            print(
                f"{ds_name}: inconsistent crossSection definition: {crossSection_str}"
            )
            return False
    exception_match_ok, known_exceptions, known_exception_to_pattern = (
        exception_matcher.get_known_exceptions(ds_name)
    )
    if not exception_match_ok:
        return False
    ds_eras = set(ds_eras_dict.keys())
    missing_eras = all_eras - ds_eras - known_exceptions
    redundant_exceptions = known_exceptions & ds_eras
    if len(redundant_exceptions) > 0:
        known_exceptions_str = ", ".join(known_exceptions)
        redundant_exceptions_str = ", ".join(redundant_exceptions)
        known_exception_patterns_str = ", ".join(
            set([k for v in known_exception_to_pattern.values() for k in v])
        )
        print(
            f"{ds_name}: listed as exception for [{known_exceptions_str}] in [{known_exception_patterns_str}]"
            f", but it exists for [{redundant_exceptions_str}]"
        )
        return False
    if len(missing_eras) > 0 and not is_data:
        missing_eras_str = ", ".join(missing_eras)
        print(f"{ds_name}: not available in: {missing_eras_str}")
        return False
    return True


def check_cross_era_consistency(era_dict, exception_matcher):
    dataset_to_eras = {}
    for era, era_desc in era_dict.items():
        for ds_name, ds_desc in era_desc["datasets"].items():
            if ds_name not in dataset_to_eras:
                dataset_to_eras[ds_name] = {}
            dataset_to_eras[ds_name][era] = ds_desc
    all_ok = True
    n_eras = len(era_dict)
    all_eras = set(era_dict.keys())
    for ds_name, ds_eras in dataset_to_eras.items():
        if not check_dataset_consistency(ds_name, ds_eras, all_eras, exception_matcher):
            all_ok = False
    return all_ok


def check_consistency(eras, exceptions_file):
    if exceptions_file is not None:
        with open(exceptions_file, "r") as f:
            exceptions = yaml.safe_load(f)
    exceptions = exceptions or {}
    exception_matcher = ExceptionMatcher(exceptions)
    era_dict = {}
    xs_db = {}
    all_ok = True
    for era in eras:
        if era in era_dict:
            print(f"{era}: era specified multiple times.")
            all_ok = False
            continue
        era_dict[era] = {}
        era_dict[era]["file_name"] = f"config/{era}/datasets.yaml"
        with open(era_dict[era]["file_name"], "r") as f:
            era_dict[era]["datasets"] = yaml.safe_load(f)
        with open(f"config/{era}/global.yaml", "r") as f:
            era_dict[era]["GLOBAL"] = yaml.safe_load(f)
        xs_file = era_dict[era]["GLOBAL"]["crossSectionsFile"]
        if xs_file.startswith("FLAF/"):
            xs_file = xs_file[5:]
        era_dict[era]["crossSectionsFile"] = xs_file
        if xs_file not in xs_db:
            with open(xs_file, "r") as f:
                xs_db[xs_file] = yaml.safe_load(f)
        if not check_era_consistency(era, era_dict[era], xs_db[xs_file]):
            all_ok = False
    if len(era_dict) > 1:
        if not check_cross_era_consistency(era_dict, exception_matcher):
            all_ok = False
        unused_patterns = exception_matcher.get_unused_patterns()
        if len(unused_patterns) > 0:
            print(f'unused entries in exceptions: {", ".join(unused_patterns)}')
            all_ok = False
    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check dataset definition.")
    parser.add_argument(
        "--exceptions",
        type=str,
        required=False,
        default=None,
        help="File with known exceptions",
    )
    parser.add_argument("era", type=str, nargs="+", help="eras to consider")
    args = parser.parse_args()

    if check_consistency(args.era, args.exceptions):
        print("All checks passed.")
        exit_code = 0
    else:
        print("Some checks failed.")
        exit_code = 1
    sys.exit(exit_code)
