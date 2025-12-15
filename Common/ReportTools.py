import ROOT


def SaveReport(report, reoprtName="Report", verbose=0):
    cuts = [c for c in report]
    hist = ROOT.TH1D(reoprtName, reoprtName, len(cuts) + 1, 0, len(cuts) + 1)
    if len(cuts) > 0:
        hist.GetXaxis().SetBinLabel(1, "Initial")
        hist.SetBinContent(1, cuts[0].GetAll())
        for c_id, cut in enumerate(cuts):
            hist.SetBinContent(c_id + 2, cut.GetPass())
            hist.GetXaxis().SetBinLabel(c_id + 2, cut.GetName())
            if verbose > 0:
                print(
                    f"for the cut {cut.GetName()} there are {cut.GetPass()} events passed over {cut.GetAll()}, resulting in an efficiency of {cut.GetEff()}"
                )
    return hist
