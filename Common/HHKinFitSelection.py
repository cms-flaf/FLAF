import ROOT
import os
from .Utilities import *

initialized = False

def Initialize():
    global initialized
    if initialized:
        raise RuntimeError('HH KinFitSel already initialized')
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    header_path_HHKinFit = os.path.join(headers_dir, "KinFitInterface.h")
    ROOT.gInterpreter.Declare(f'#include "{header_path_HHKinFit}"')
    header_path_SVFit = os.path.join(headers_dir, "SVfitAnaInterface.h")
    ROOT.gInterpreter.Declare(f'#include "{header_path_SVFit}"')
    header_path_MT2 = os.path.join(headers_dir, "MT2.h")
    ROOT.gInterpreter.Declare(f'#include "{header_path_MT2}"')
    header_path_Lester_mt2_bisect = os.path.join(headers_dir, "Lester_mt2_bisect.cpp")
    ROOT.gInterpreter.Declare(f'#include "{header_path_Lester_mt2_bisect}"')
    initialized = True


def GetKinFitConvergence(df):
    if not initialized:
        raise RuntimeError("HH KinFit selection not initialised!")
    df = df.Define('MT2', 'float(analysis::Calculate_MT2(httCand.leg_p4[0], httCand.leg_p4[1], HbbCandidate.leg_p4[0],HbbCandidate.leg_p4[1], MET_p4))')

    df = df.Define("kinFit_result", f"""kin_fit::FitProducer::Fit(HbbCandidate.leg_p4[0],HbbCandidate.leg_p4[1],
                                                       httCand.leg_p4[0], httCand.leg_p4[1],
                                                       MET_p4, MET_covXX, MET_covXY, MET_covYY, Jet_ptRes.at(HbbCandidate.leg_index[0]),
                                                       Jet_ptRes.at(HbbCandidate.leg_index[1]), 0)""")
    df = df.Define('kinFit_convergence', 'kinFit_result.convergence')
    df = df.Define('kinFit_m', 'float(kinFit_result.mass)')
    df = df.Define('kinFit_chi2', 'float(kinFit_result.chi2)')
    #df = df.Define("first_index", "httCand.leg_index[0]")
    #df = df.Define("second_index", "httCand.leg_index[1]")
    #df = df.Define("Tau_DecayMode_size", "Tau_decayMode.size()")
    #df = df.Define("Tau_pt_size", "Tau_pt.size()")
    #df.Display({"first_index","second_index","Tau_DecayMode_size","Tau_pt_size"}).Print()
    for leg_idx in [0,1]:
        df=df.Define(f"Tau_dm_{leg_idx}", f"httCand.leg_type[{leg_idx}] == Leg::tau ? Tau_decayMode.at(httCand.leg_index[{leg_idx}]) : -1;")
    df = df.Define('SVfit_result',
                """sv_fit::FitProducer::Fit(httCand.leg_p4[0], httCand.leg_type[0], Tau_dm_0,
                                            httCand.leg_p4[1], httCand.leg_type[1], Tau_dm_1,
                                            MET_p4, MET_covXX, MET_covXY, MET_covYY)""")

    df = df.Define('SVfit_valid', 'int(SVfit_result.has_valid_momentum)')
    df = df.Define('SVfit_pt', 'float(SVfit_result.momentum.pt())')
    df = df.Define('SVfit_eta', 'float(SVfit_result.momentum.eta())')
    df = df.Define('SVfit_phi', 'float(SVfit_result.momentum.phi())')
    df = df.Define('SVfit_m', 'float(SVfit_result.momentum.mass())')
    df = df.Define('SVfit_pt_error', 'float(SVfit_result.momentum_error.pt())')
    df = df.Define('SVfit_eta_error', 'float(SVfit_result.momentum_error.eta())')
    df = df.Define('SVfit_phi_error', 'float(SVfit_result.momentum_error.phi())')
    df = df.Define('SVfit_m_error', 'float(SVfit_result.momentum_error.mass())')
    df = df.Define('SVfit_mt', 'float(SVfit_result.transverseMass)')
    df = df.Define('SVfit_mt_error', 'float(SVfit_result.transverseMass_error)')
    cols_to_append=['MT2','kinFit_result','kinFit_convergence','kinFit_m','kinFit_chi2', 'SVfit_valid', 'SVfit_pt', 'SVfit_eta', 'SVfit_phi', 'SVfit_m', 'SVfit_pt_error', 'SVfit_eta_error', 'SVfit_phi_error', 'SVfit_m_error', 'SVfit_mt', 'SVfit_mt_error']
    return df,cols_to_append


