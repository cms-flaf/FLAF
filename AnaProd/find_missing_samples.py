import os
import sys
import yaml


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    args = parser.parse_args()
    config_name = f"/afs/cern.ch/work/v/vdamante/hhbbTauTauRes/prod/Framework/config/samples_Run2_{args.year}.yaml"
    with open(config_name, "r") as f:
        config = yaml.safe_load(f)

    samples_to_drop = []
    missing_samples = []
    samples_available = os.listdir(
        f"/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_{args.year}"
    )
    # print(samples_available)
    for sample, subdict in config.items():
        if sample == "GLOBAL":
            continue
        if sample in samples_available:
            continue
        samples_to_drop.append(sample)
    for sample_av in samples_available:
        if sample_av in config.keys():
            continue
        if "NMSSM" in sample_av:
            continue
        if "UncorrelatedDecay" in sample_av:
            continue
        missing_samples.append(sample_av)
    # print(f"for year {args.year} available samples are : {samples_available}")
    print(f"for year {args.year} samples to drop are : {samples_to_drop}")
    for sample in samples_to_drop:
        print(f"- drop ^{sample}$")
    print(f"for year {args.year} not considered samples are : ")
    for sample in missing_samples:
        print(sample)
