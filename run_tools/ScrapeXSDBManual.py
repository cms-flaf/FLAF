from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import time
import yaml
import os
import sys


def ScrapeXSDBManual(input, input_xsec, XSDB_website):
    # with open('temp_Run3_2022.yaml', 'r') as f:
    with open(input, "r") as f:
        yaml_source = yaml.safe_load(f)

    # with open('temp_Run3_2022_xsec.yaml', 'r') as f:
    with open(input_xsec, "r") as f:
        xsec_yaml = yaml.safe_load(f)

    new_yaml = xsec_yaml.copy()

    driver = webdriver.Safari()
    # driver.get('https://xsdb-temp.app.cern.ch/xsdb')
    driver.get(XSDB_website)

    elem = driver.find_element(By.ID, "searchField")

    # for process_name in process_list:
    for key in yaml_source.keys():
        process_name = yaml_source[key]["miniAOD"].split("/")[1]
        das_name = yaml_source[key]["miniAOD"]
        if "crossSection" in yaml_source[key].keys():
            xsec_key = yaml_source[key]["crossSection"]
        elif "crossSectionStitch" in yaml_source[key].keys():
            xsec_key = yaml_source[key]["crossSectionStitch"]
        else:
            print("No xsec in dict?")
            continue

        if xsec_yaml[xsec_key]["reference"] != "fill me!":
            print(f"yaml already has an xsec for process {process_name}")
            continue

        elem.clear()
        elem.send_keys(f"process_name={process_name}")
        elem.send_keys(Keys.RETURN)
        # time.sleep(2)
        not_done = True
        not_done_counter = 0
        sleep_time = 0.5
        while not_done:
            if "Found" in driver.page_source:
                not_done = False
            if "Network Error" in driver.page_source:
                print("Network Error, search again")
                elem.send_keys(Keys.RETURN)
            time.sleep(sleep_time)
            not_done_counter += sleep_time
            if not_done_counter >= 10:
                print(f"Timeout {process_name}")
                not_done = False
        # Read the values!
        dfs = pd.read_html(driver.page_source)
        df = dfs[1]
        xsecs = df["cross_section"]
        uncs = df["total_uncertainty"]
        dasnames = df["DAS"]
        modifiedOns = df["modifiedOn"]

        if len(xsecs) == 0:
            print("No results found!")
            print(f"Search {process_name}")
            continue

        good = True
        xsec = 1.0
        unc = 1.0
        modifiedOn = ""
        for i, tmp_xsec in enumerate(xsecs):
            if dasnames[i].split("/")[1] == das_name.split("/")[1]:
                # Check if new is within unc
                unc_check = abs(tmp_xsec - xsec) > uncs[i]
                if (xsec != 1.0) and (unc_check):
                    good = False
                xsec = tmp_xsec
                unc = uncs[i]
                modifiedOn = modifiedOns[i]

        if not good:
            print("Check this by hand!")
            print(xsecs)
            print(process_name)
            ref = "Check this by hand!!! XSDB has multiple values"
            print(das_name)
            print(df["DAS"])
            continue

        print(f"{process_name} found with xsec {xsec}")
        ref = f"XSDB {process_name} modifiedOn {modifiedOn}"
        new_yaml[xsec_key]["crossSec"] = float(xsec)
        new_yaml[xsec_key]["unc"] = float(unc)
        new_yaml[xsec_key]["reference"] = ref

    # t = open('Run3_2022_xsec.yaml', 'w+')
    t = open(f"Scrape_{input_xsec}", "w+")
    yaml.dump(new_yaml, t, allow_unicode=True, width=1000, sort_keys=False)

    driver.close()


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(file_dir)
    base_dir = os.path.dirname(pkg_dir)
    pkg_dir_name = os.path.split(pkg_dir)[1]
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    __package__ = pkg_dir_name

    import argparse

    parser = argparse.ArgumentParser(
        description="Manually scrape the temporary XSDB website locally when the API website is down"
    )
    parser.add_argument("--input", required=True, type=str, help="input dataset yaml")
    parser.add_argument("--input-xsec", required=True, type=str, help="input xsec yaml")
    parser.add_argument(
        "--XSDB-website",
        required=False,
        default="https://xsdb-temp.app.cern.ch/xsdb",
        type=str,
        help="XSDB website to scrape",
    )

    args = parser.parse_args()
    input = args.input
    input_xsec = args.input_xsec
    XSDB_website = args.XSDB_website

    ScrapeXSDBManual(input, input_xsec, XSDB_website)
