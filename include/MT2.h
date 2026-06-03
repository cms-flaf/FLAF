/*! Wrapper for MT2 */

#pragma once

#include "Lester_mt2_bisect.h"

// implementation as LLR and as article
namespace analysis {
    template <typename LVector1, typename LVector2, typename LVector3, typename LVector4, typename LVector5>
    double Calculate_MT2(const LVector1 &lepton1_p4,
                         const LVector2 &lepton2_p4,
                         const LVector3 &bjet_1,
                         const LVector4 &bjet_2,
                         const LVector5 &met_p4) {
        asymm_mt2_lester_bisect::disableCopyrightMessage();
        const double mVisA = bjet_1.mass();
        const double pxA = bjet_1.px();
        const double pyA = bjet_1.py();
        const double mVisB = bjet_2.mass();
        const double pxB = bjet_2.px();
        const double pyB = bjet_2.py();
        const double pxMiss = lepton1_p4.px() + lepton2_p4.px() + met_p4.px();
        const double pyMiss = lepton1_p4.py() + lepton2_p4.py() + met_p4.py();
        double chiA = lepton1_p4.mass();  // hypothesised mass of invisible on side A.  Must be >=0.
        double chiB = lepton2_p4.mass();  // hypothesised mass of invisible on side B.  Must be >=0.
        double MT2 = asymm_mt2_lester_bisect::get_mT2(mVisA, pxA, pyA, mVisB, pxB, pyB, pxMiss, pyMiss, chiA, chiB, 0);
        return MT2;
    }

    // Vague MT2 function to change invisible/visible/hypothesis values directly
    // Did not touch existing MT2 for consistent bbtautau work
    template <typename LVector1, typename LVector2, typename LVector3>
    double Calculate_MT2_func(const LVector1 &visible1,
                              const LVector2 &visible2,
                              const LVector3 &invisible,
                              const double massHypo1,
                              const double massHypo2) {
        asymm_mt2_lester_bisect::disableCopyrightMessage();
        const double mVisA = visible1.mass();
        const double pxA = visible1.px();
        const double pyA = visible1.py();
        const double mVisB = visible2.mass();
        const double pxB = visible2.px();
        const double pyB = visible2.py();
        const double pxMiss = invisible.px();
        const double pyMiss = invisible.py();
        double chiA = massHypo1;  // hypothesised mass of invisible on side A.  Must be >=0.
        double chiB = massHypo2;  // hypothesised mass of invisible on side B.  Must be >=0.
        double MT2 = asymm_mt2_lester_bisect::get_mT2(mVisA, pxA, pyA, mVisB, pxB, pyB, pxMiss, pyMiss, chiA, chiB, 0);
        return MT2;
    }

}  // namespace analysis