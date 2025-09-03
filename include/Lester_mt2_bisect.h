/*
 *  Copyright 2014, Christopher Lester, Univeristy of Cambridge
 *
 *  version 5: arXiv:1411.4312v5
 *    * made more portable by removal of use of __FILE__ and __LINE__ macros in
debug statement
 *    * made fewer demands on poor C++ compilers (ROOT5/CINT) by removal of
certain inline statements
 *    * added this changelog!
 *
 *  version 4: arXiv:1411.4312v4
 *    * added copyright information
 *
 *  version 3: arXiv:1411.4312v3
 *    * added option to turn on/off deci-sectioning
 *    * made code slightly slower for readability gain
 *
 *  version 2: arXiv:1411.4312v2
 *    * no changes w.r.t. version 1
 *
 *  version 1: arXiv:1411.4312v1
 *    * initial public release
 *
 *  This file will let you calculate MT2 or Asymmetric MT2 relatively easily.
 *  An example showing how to do so, may be found below this copyright message.
 *
 *  (Note that this is a low-level library.  Various wrappers exist around
 *   it to allow easier interfacing to ROOT or ATLAS code.)
 *
 *   If you use this implementation, please cite:
 *
 *   http://arxiv.org/abs/1411.4312
 *
 *   as the paper documenting this particular implementation.
 *
 *   You might also need to cite:
 *
 *   http://arxiv.org/abs/hep-ph/9906349
 *   Journal reference: Phys.Lett.B463:99-103,1999
 *   DOI: 10.1016/S0370-2693(99)00945-4
 *
 *   as the paper defining MT2.
 *
 * Here is an example of it's use:


double mVisA = 10; // mass of visible object on side A.  Must be >=0.
double pxA = 20; // x momentum of visible object on side A.
double pyA = 30; // y momentum of visible object on side A.

double mVisB = 10; // mass of visible object on side B.  Must be >=0.
double pxB = -20; // x momentum of visible object on side B.
double pyB = -30; // y momentum of visible object on side B.

double pxMiss = -5; // x component of missing transverse momentum.
double pyMiss = -5; // y component of missing transverse momentum.

double chiA = 4; // hypothesised mass of invisible on side A.  Must be >=0.
double chiB = 7; // hypothesised mass of invisible on side B.  Must be >=0.

double desiredPrecisionOnMt2 = 0; // Must be >=0.  If 0 alg aims for machine
precision.  if >0, MT2 computed to supplied absolute precision.

// asymm_mt2_lester_bisect::disableCopyrightMessage();

double MT2 =  asymm_mt2_lester_bisect::get_mT2(
           mVisA, pxA, pyA,
           mVisB, pxB, pyB,
           pxMiss, pyMiss,
           chiA, chiB,
           desiredPrecisionOnMt2);

 */

#ifndef LESTER_TESTWHETHERELLIPSESAREDISJOINT_H
#define LESTER_TESTWHETHERELLIPSESAREDISJOINT_H

#include <utility>
/*
 * The
 *
 *             bool ellipsesAreDisjoint(const EllipseParams & e1, const
EllipseParams & e2);
 *
 * function determines whether two ellipses (not both singular) are disjoint.
 * Ellipses are assumed to be solid objects with a filled interior.
 * They are disjoint it no part of their interiors overlap.
 * Singular (in this context) is defined below.
 *
 * It uses the method of:

Computer Aided Geometric Design 23 (2006) 324–350
A new approach to characterizing the relative position of two ellipses depending
on one parameter Fernando Etayo 1,3, Laureano Gonzalez-Vega ∗,2,3, Natalia del
Rio 3 Departamento de Matematicas, Estadistica y Computacion, Universidad de
Cantabria, Spain Received 15 September 2004; received in revised form 2 November
2005; accepted 10 January 2006 Available online 28 February 2006

pointed out to me by Gary B. Huges and Mohcine Chraibi authors of

 Comput Visual Sci (2012) 15:291–301 DOI 10.1007/s00791-013-0214-3
 Calculating ellipse overlap areas Gary B. Hughes · Mohcine Chraibi

 * Note:
 *
 * Though the paper above talks only about ellipses, from playing with some test
cases, I (CGL) have conjectured that the algorithm actually works well even if
the conics are parabolas provided that the axx>0&&ayy>0 test is reduced to
axx>=0&&ayy>=0&&axx*ayy!=0 ... which is true is good news for the similicity of
the MT2 calculator ... as the MT2 calculator will not need to distinguish these
two possibilities.  In a private communication between me (CGL) and the  authors
of Computer Aided Geometric Design 23 (2006) 324–350, the authors have indicated
that it is not unreasonable to believe that the code does indeed work on the
parabolica cases too.  This algorithm relies on that generalisation, which may
be the subject of a paper (to appear) from Etayo and Gonzalez-Vega.
 *
 *
 * Definition: an ellipse is defined with respect to cartesian co-ordinates
(x,y) by an equation of the form;
 *
 * xVec^T A xVec = 0                 (1)
 *
 * where xVec is a columnar three vec containing (x,y,1) and where A is a
symmetric matrix having elements:
 *
 *       [ axx axy ax  ]
 *   A = [ axy ayy ay  ]
 *       [ ax  ay  a   ].
 *
 * Therfore the ellipse equation would look like:
 *
 * axx x^2 + 2 axy x y + ayy y^2 + 2 ax x + 2 ay y + a = 0.
 *
 * Note that this parametrisation has one parameter too many ... the "A"-matrix
can be multiplied by a non-zero constant, and the ellipse is not changed.
 * Etayo et al's implementation REQUIRES that axx and ayy be strictly positive.
 * The implementation herein doesn't quite enforce that. The implementation
herein allows axx or ayy to be non-negative .... and it is left to the user to
ensure that axx and ayy are not exactly zero.
 * Note also that (1) is general enough to contain all conic sections, so it is
left to the user to ensure that only values of A consistent
 * with (non-singluar) ellipses are fed into the program below.  For our
purposes, an ellipse is "singular" iff coeffLamPow3 (see below) is zero.
 */

namespace Lester {

    struct EllipseParams {
        // Constructor for non-degenerate ellipses:
        /*
   * Ellipse is represented algebraically by:
   * c_xx x^2 + 2 c_xy x y + c_yy y^2 + 2 c_x x + 2 c_y y + c = 0.
   */
        EllipseParams(const double c_xx2,
                      const double c_yy2,
                      const double c_xy2,
                      const double c_x2,
                      const double c_y2,
                      const double c2);
        EllipseParams() {}
        void setDet();
        // Consstructor for degenerate ellipse (i.e. a "dot" at (x0,y0) ).
        EllipseParams(const double x0, const double y0);
        double lesterFactor(const EllipseParams &e2) const;
        bool operator==(const EllipseParams &other) const;

      public:
        // Data
        double c_xx;
        double c_yy;
        double c_xy;  // note factor of 2 above
        double c_x;   // note factor of 2 above
        double c_y;   // note factor of 2 above
        double c;
        double det;  // The determinant of the 3x3 conic matrix
    };

    // This is the interface: users should call this function:
    bool ellipsesAreDisjoint(const EllipseParams &e1, const EllipseParams &e2);

    // This is an implementation thing: users should not call it:
    bool __private_ellipsesAreDisjoint(const double coeffLamPow3,
                                       const double coeffLamPow2,
                                       const double coeffLamPow1,
                                       const double coeffLamPow0);

    bool ellipsesAreDisjoint(const EllipseParams &e1, const EllipseParams &e2);
    bool __private_ellipsesAreDisjoint(const double coeffLamPow3,
                                       const double coeffLamPow2,
                                       const double coeffLamPow1,
                                       const double coeffLamPow0);

}  // namespace Lester

#endif

#ifndef ASYMM_MT2_BISECT_H
#define ASYMM_MT2_BISECT_H

class asymm_mt2_lester_bisect {
  public:
    static const int MT2_ERROR = -1;

    static double get_mT2(  // returns asymmetric mT2 (which is >=0), or returns a negative
                            // number (such as MT2_ERROR) in the case of an error.
        const double mVis1,
        const double pxVis1,
        const double pyVis1,
        const double mVis2,
        const double pxVis2,
        const double pyVis2,
        const double pxMiss,
        const double pyMiss,
        const double mInvis1,
        const double mInvis2,
        const double desiredPrecisionOnMT2 = 0,     // This must be non-negative.  If set to zero (default) MT2 will be
                                                    // calculated to the highest precision available on the machine (or
                                                    // as close to that as the algorithm permits).  If set to a
                                                    // positive value, MT2 (note that is MT2, not its square) will be
                                                    // calculated to within +- desiredPrecisionOnMT2. Note that by
                                                    // requesting precision of +- 0.01 GeV on an MT2 value of 100 GeV
                                                    // can result in speedups of a factor of ...
        const bool useDeciSectionsInitially = true  // If true, interval is cut at the 10% point until first
                                                    // acceptance, which gives factor 3 increase in speed calculating
                                                    // kinematic min, but 3% slowdown for events in the bulk.  Is on
                                                    // (true) by default, but can be turned off by setting to false.
    );
    static void disableCopyrightMessage(const bool printIfFirst = false);

    static double get_mT2_Sq(  // returns square of asymmetric mT2 (which is >=0), or returns a
                               // negative number (such as MT2_ERROR) in the case of an error.
        const double mVis1,
        const double pxVis1,
        const double pyVis1,
        const double mVis2,
        const double pxVis2,
        const double pyVis2,
        const double pxMiss,
        const double pyMiss,
        const double mInvis1,
        const double mInvis2,
        const double desiredPrecisionOnMT2 = 0,     // This must be non-negative.  If set to zero (default) MT2 will be
                                                    // calculated to the highest precision available on the machine (or
                                                    // as close to that as the algorithm permits).  If set to a
                                                    // positive value, MT2 (note that is MT2, not its square) will be
                                                    // calculated to within +- desiredPrecisionOnMT2. Note that by
                                                    // requesting precision of +- 0.01 GeV on an MT2 value of 100 GeV
                                                    // can resJult in speedups of a factor of ..
        const bool useDeciSectionsInitially = true  // If true, interval is cut at the 10% point until first
                                                    // acceptance, which gives factor 3 increase in speed calculating
                                                    // kinematic min, but 3% slowdown for events in the bulk.  Is on
                                                    // (true) by default, but can be turned off by setting to false.
    );

  private:
    static double lestermax(const double x, const double y);
    static const Lester::EllipseParams helper(const double mSq,  // The test parent-mass value (squared)
                                              const double mtSq,
                                              const double tx,
                                              const double ty,    // The visible particle transverse momentum
                                              const double mqSq,  // The mass of the invisible particle
                                              const double pxmiss,
                                              const double pymiss);

    static void myversion();

    static double MT(double px1, double px2, double py1, double py2, double m1, double m2);

    static std::pair<double, double> ben_findsols(double MT2,
                                                  double px,
                                                  double py,
                                                  double visM,
                                                  double Ma,
                                                  double pxb,
                                                  double pyb,
                                                  double metx,
                                                  double mety,
                                                  double visMb,
                                                  double Mb);
};

#endif