from . import BaselineSelection as Baseline


def skim_RecoLeptons(df):
    Baseline.Initialize()
    df = Baseline.CreateRecoP4(df, "")
    df = Baseline.RecoLeptonsSelection(df)
    return df


def skim_failed_RecoLeptons(df):
    Baseline.Initialize()
    df = Baseline.CreateRecoP4(df, "")
    df, b0_filter = Baseline.RecoLeptonsSelection(df, apply_filter=False)
    df = df.Filter(f"!({b0_filter})")
    return df


def skim_RecoLeptonsJetAcceptance(df):
    Baseline.Initialize()
    df = Baseline.CreateRecoP4(df, "")
    df = Baseline.RecoLeptonsSelection(df)
    df = Baseline.RecoJetAcceptance(df)
    return df


def skim_failed_RecoLeptonsJetAcceptance(df):
    Baseline.Initialize()
    df = Baseline.CreateRecoP4(df, "")
    df, b0_filter = Baseline.RecoLeptonsSelection(df, apply_filter=False)
    df, b1_filter = Baseline.RecoJetAcceptance(df, apply_filter=False)
    df = df.Filter(f"!(({b0_filter}) && ({b1_filter}))")
    return df
