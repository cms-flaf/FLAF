import os


#indir = "/eos/user/d/daebi/ANA_FOLDER_DEV/histograms/Run32022_13Nov24_MediumMuonSF/Run3_2022/merged/"
#indir = "/eos/user/d/daebi/ANA_FOLDER_DEV/histograms/Run32022EE_27Nov24_MuonSF_PU/Run3_2022EE/merged/"
indir = "/eos/user/d/daebi/ANA_FOLDER_DEV/histograms/Run32022_27Nov24_MuonSF_PU/Run3_2022/merged/"
indir = "/eos/user/d/daebi/ANA_FOLDER_DEV/histograms/Run32022EE_27Nov24_MuonSF_PU/Run3_2022EE/merged/"
indir = "/eos/user/d/daebi/ANA_FOLDER_DEV/histograms/Run3_2022EE_11Dec24/Run3_2022EE/merged/"
indir = "/eos/user/d/daebi/ANA_FOLDER_DEV/histograms/shared/Run3_2022/merged/"
indir = "/eos/user/d/daebi/ANA_FOLDER_DEV/histograms/shared/Run3_2022/merged/"

varnames = ["lep1_pt", "lep1_eta", "lep2_pt", "lep2_eta", "bjet1_btagPNetB", "bjet2_btagPNetB"]
varnames = ["lep1_pt", "diLep_mass", "MT_lep1", "MT_lep2", "MT_tot"]
varnames = ["diLep_mass", "Lep1Lep2Jet1Jet2_mass", "Lep1Jet1Jet2_mass"]
varnames = ["dnn_M250_Signal"]
channellist = ["eMu", "mu", "muMu"]

#era = "Run3_2022"
#plotdir = "plots_27Nov_MuonSF_PU_2022/"

era = "Run3_2022"
plotdir = "plots_2022_DNNOut/"
cat = "inclusive"

using_uncertainties = True #When we turn on Up/Down, the file storage changes due to renameHists.py

for var in varnames:
    for channel in channellist:
        filename = os.path.join(indir, var, f"{var}.root")
        print("Loading fname ", filename)
        os.makedirs(plotdir, exist_ok=True)
        outname = os.path.join(plotdir, f"HHbbWW_{channel}_{var}_StackPlot.pdf")

        if not using_uncertainties:
            os.system(f"python3 ../HistPlotter.py --inFile {filename} --bckgConfig ../../config/HH_bbWW/background_samples.yaml --globalConfig ../../config/HH_bbWW/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --rebin False --analysis HH_bbWW --qcdregion OS_Iso --sigConfig ../../config/HH_bbWW/{era}/samples.yaml")

        else:
            filename = os.path.join(indir, var, 'tmp', f"all_histograms_{var}_hadded.root")
            #os.system(f"python3 ../HistPlotter.py --inFile {filename} --bckgConfig ../../config/HH_bbWW/background_samples.yaml --globalConfig ../../config/HH_bbWW/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantSignals --signalMassList [250] --year {era} --rebin False --analysis HH_bbWW --qcdregion OS_Iso --sigConfig ../../config/HH_bbWW/{era}/samples.yaml")
            os.system(f"python3 ../HistPlotter.py --inFile {filename} --bckgConfig ../../config/HH_bbWW/background_samples.yaml --globalConfig ../../config/HH_bbWW/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --signalMassList [250] --year {era} --rebin False --analysis HH_bbWW --qcdregion OS_Iso --sigConfig ../../config/HH_bbWW/{era}/samples.yaml")
