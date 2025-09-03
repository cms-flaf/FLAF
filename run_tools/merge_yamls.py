import yaml
from FLAF.RunKit.run_tools import natural_sort


def MergeYamls(yaml1name, yaml2name):
    with open(yaml1name, "r") as f:
        yaml1 = yaml.safe_load(f)

    with open(yaml2name, "r") as f:
        yaml2 = yaml.safe_load(f)

    yaml1_keys = yaml1.keys()

    for key in yaml2.keys():
        if key in yaml1_keys:
            yaml1_subkeys = yaml1[key].keys()
            for subkey in yaml2[key].keys():
                if subkey not in yaml1_subkeys:
                    print(f"Subkey {subkey} not in yaml1, WARNING")
                    yaml1[key][subkey] = yaml2[key][subkey]
                elif yaml1[key][subkey] != yaml2[key][subkey]:
                    print(
                        f"Subkey {subkey} does not match! {yaml1[key][subkey]} != {yaml2[key][subkey]}"
                    )
                    print(f"Check this key {key}")
        else:
            print(f"Key {key} did not exist, extending")
            yaml1[key] = yaml2[key]

    new_yaml = {}
    for key in natural_sort(yaml1):
        new_yaml[key] = yaml1[key]
    t = open(f"merged.yaml", "w+")
    yaml.dump(new_yaml, t, allow_unicode=True, width=1000, sort_keys=False)


MergeYamls("Scrape_temp_Run3_2022_xsec.yaml", "Scrape_temp_Run3_2022EE_xsec.yaml")
