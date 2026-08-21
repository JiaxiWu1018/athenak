//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file trilinear_interp.cpp
//! \brief unit tests for the trilinear particle-interpolation kernels added for the
//! gr_boris low-order geodesic fallback (branch experiment/ST-trilinear-fallback).
//!
//! Everything runs in device kernels on a synthetic 5-D array with the ADM variable
//! layout, so the test exercises exactly the code the pusher calls, on the backend the
//! production run uses. Five checks, each for NGHOST = 2, 3 and 4:
//!
//!   T1  weights: L0+L1 == 1, both in [0,1], dL0 == -dL1 == -1/h. This is the hypothesis
//!       the convexity argument rests on.
//!   T2  stencil offset: with a field set to its own allocated index, the interpolant
//!       must return (base + NGHOST) + t. An offset error of one cell -- the easy
//!       mistake, since the high-order reader hard-wires "+1" for ORDER == NGHOST --
//!       shows up as an error of exactly that many cells.
//!   T3  exactness: a trilinear polynomial and its three first derivatives are
//!       reproduced to round-off, which is what makes the fallback's geometry and its
//!       gradient a consistent pair.
//!   T4  positive definiteness: over a field of independent random positive definite
//!       corner matrices, the trilinear interpolant is positive definite at every
//!       sample, while the 2*NGHOST-node Lagrange interpolant is not. This is the
//!       property the fallback exists for, and the counter-example that motivates it.
//!   T5  InverseMetricGradient: -g^{ja} (d_i g_ab) g^{bk} agrees with a central
//!       difference of the inverse of the same analytic metric.
//!
//! One PASS/FAIL line per check; exits non-zero on the first failure, following
//! src/pgen/unit_tests/gauss_legendre.cpp.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "particles/lagrange_interp.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace {

// deterministic device-side PRNG (a 32-bit integer mix), so the field is reproducible
// across backends and the test cannot pass or fail by luck of the run
KOKKOS_INLINE_FUNCTION
Real Rand01(unsigned int s) {
  s ^= s >> 16;
  s *= 0x7feb352dU;
  s ^= s >> 15;
  s *= 0x846ca68bU;
  s ^= s >> 16;
  return static_cast<Real>(s & 0xffffffU) / static_cast<Real>(0x1000000U);
}

// a trilinear polynomial: reproduced exactly inside one cell by the trilinear
// interpolant, so both its value and its gradient are exact reference data
KOKKOS_INLINE_FUNCTION
Real TriPoly(Real x, Real y, Real z) {
  return 0.37 + 1.3*x - 0.8*y + 0.45*z + 0.6*x*y - 0.25*x*z + 0.9*y*z + 1.7*x*y*z;
}

KOKKOS_INLINE_FUNCTION
void TriPolyGrad(Real x, Real y, Real z, Real g[3]) {
  g[0] = 1.3 + 0.6*y - 0.25*z + 1.7*y*z;
  g[1] = -0.8 + 0.6*x + 0.9*z + 1.7*x*z;
  g[2] = 0.45 - 0.25*x + 0.9*y + 1.7*x*y;
}

// a smooth, strongly varying analytic 3-metric for T5 (conformally flat, puncture-like)
KOKKOS_INLINE_FUNCTION
void AnalyticMetric(Real x, Real y, Real z, Real g[6]) {
  Real r = std::sqrt(x*x + y*y + z*z + 0.09);
  Real psi = 1.0 + 0.5/r;
  Real p4 = psi*psi*psi*psi;
  g[0] = p4*(1.0 + 0.1*x);
  g[1] = p4*(0.05*z);
  g[2] = p4*(0.04*y);
  g[3] = p4*(1.0 + 0.12*y);
  g[4] = p4*(0.03*x);
  g[5] = p4*(1.0 - 0.08*z);
}

KOKKOS_INLINE_FUNCTION
bool SylvesterPD(const Real g[6]) {
  Real minor2 = g[0]*g[3] - g[1]*g[1];
  return (g[0] > 0.0) && (minor2 > 0.0) && (Primitive::GetDeterminant(g) > 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn bool RunChecks
//! \brief all five checks for one NGHOST. Returns false on the first failure.

template <int NG>
bool RunChecks() {
  // deliberately ANISOTROPIC in both cell count and extent: on an isotropic grid an
  // x/y/z mix-up inside the weight kernel or the reader is invisible to every check
  const int nc1 = 16, nc2 = 12, nc3 = 20;
  const int na1 = nc1 + 2*NG, na2 = nc2 + 2*NG, na3 = nc3 + 2*NG;
  const Real x1min = -1.0, x1max = 1.0;
  const Real x2min = -0.5, x2max = 2.5;
  const Real x3min = -3.0, x3max = 1.0;
  const Real h1 = (x1max - x1min)/static_cast<Real>(nc1);
  const Real h2 = (x2max - x2min)/static_cast<Real>(nc2);
  const Real h3 = (x3max - x3min)/static_cast<Real>(nc3);
  const Real mb_par[9] = {x1min, x1max, h1, x2min, x2max, h2, x3min, x3max, h3};
  const int ncell[3] = {nc1, nc2, nc3};
  const Real gmin[3] = {x1min, x2min, x3min};
  const Real gmax[3] = {x1max, x2max, x3max};
  const Real hh[3] = {h1, h2, h3};
  const int nsample = 4096;
  const Real tol = 1.0e-12;

  DvceArray5D<Real> u("ut_trilinear", 1, adm::ADM::nadm, na3, na2, na1);

  // --- T1: weight properties ---------------------------------------------------------
  Real t1_err = 0.0;
  int t1_bad = 0;
  Kokkos::parallel_reduce("ut_tri_T1", Kokkos::RangePolicy<>(0, nsample),
  KOKKOS_LAMBDA(const int n, Real &lerr, int &lbad) {
    Real xp[3];
    for (int d = 0; d < 3; ++d) {
      xp[d] = gmin[d] + (gmax[d] - gmin[d])
              *Rand01(static_cast<unsigned int>(7919u*static_cast<unsigned int>(n)
                                                + 13u*static_cast<unsigned int>(d) + 1u));
    }
    int idcs[4] = {0, -1, -1, -1};
    particles::SetInterpIndices(xp, mb_par, ncell, idcs);
    Real lx[8] = {0.0}, ly[8] = {0.0}, lz[8] = {0.0};
    Real dlx[8] = {0.0}, dly[8] = {0.0}, dlz[8] = {0.0};
    particles::CalcTrilinearWghtAndDrv(xp, mb_par, ncell, idcs,
                                       lx, ly, lz, dlx, dly, dlz);
    const Real *w[3] = {lx, ly, lz};
    const Real *dw[3] = {dlx, dly, dlz};
    for (int d = 0; d < 3; ++d) {
      lerr = fmax(lerr, fabs(w[d][0] + w[d][1] - 1.0));
      lerr = fmax(lerr, fabs(dw[d][0] + dw[d][1]));
      lerr = fmax(lerr, fabs(dw[d][1]*hh[d] - 1.0));
      if (w[d][0] < -1.0e-14 || w[d][0] > 1.0 + 1.0e-14) { lbad += 1; }
      if (w[d][1] < -1.0e-14 || w[d][1] > 1.0 + 1.0e-14) { lbad += 1; }
    }
  }, Kokkos::Max<Real>(t1_err), Kokkos::Sum<int>(t1_bad));
  std::cout << "  [NGHOST=" << NG << "] T1 weights: max identity error " << t1_err
            << ", weights outside [0,1]: " << t1_bad << std::endl;
  if (!(t1_err < tol) || t1_bad != 0) {
    std::cout << "  T1 FAILED" << std::endl;
    return false;
  }

  // --- T2: stencil offset ------------------------------------------------------------
  // variable GXX+d holds its own allocated index in direction d
  par_for("ut_tri_fill_idx", DevExeSpace(), 0, na3-1, 0, na2-1, 0, na1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    u(0, adm::ADM::I_ADM_GXX+0, k, j, i) = static_cast<Real>(i);
    u(0, adm::ADM::I_ADM_GXX+1, k, j, i) = static_cast<Real>(j);
    u(0, adm::ADM::I_ADM_GXX+2, k, j, i) = static_cast<Real>(k);
  });
  Real t2_err = 0.0;
  Kokkos::parallel_reduce("ut_tri_T2", Kokkos::RangePolicy<>(0, nsample),
  KOKKOS_LAMBDA(const int n, Real &lerr) {
    Real xp[3];
    for (int d = 0; d < 3; ++d) {
      xp[d] = gmin[d] + (gmax[d] - gmin[d])
              *Rand01(static_cast<unsigned int>(104729u*static_cast<unsigned int>(n)
                                                + 31u*static_cast<unsigned int>(d) + 5u));
    }
    int idcs[4] = {0, -1, -1, -1};
    particles::SetInterpIndices(xp, mb_par, ncell, idcs);
    Real lx[8] = {0.0}, ly[8] = {0.0}, lz[8] = {0.0};
    Real dlx[8] = {0.0}, dly[8] = {0.0}, dlz[8] = {0.0};
    particles::CalcTrilinearWghtAndDrv(xp, mb_par, ncell, idcs,
                                       lx, ly, lz, dlx, dly, dlz);
    for (int d = 0; d < 3; ++d) {
      Real got = particles::TrilinearInterpolator<NG>(u, adm::ADM::I_ADM_GXX+d,
                                                      idcs, lx, ly, lz);
      Real t = (d == 0) ? lx[1] : ((d == 1) ? ly[1] : lz[1]);
      Real want = static_cast<Real>(idcs[d+1] + NG) + t;
      lerr = fmax(lerr, fabs(got - want));
    }
  }, Kokkos::Max<Real>(t2_err));
  std::cout << "  [NGHOST=" << NG << "] T2 stencil offset: max |index error| "
            << t2_err << " cells" << std::endl;
  if (!(t2_err < 1.0e-10)) {
    std::cout << "  T2 FAILED -- the 2-node reader is not at base+NGHOST" << std::endl;
    return false;
  }

  // --- T3: exactness on a trilinear polynomial ---------------------------------------
  par_for("ut_tri_fill_poly", DevExeSpace(), 0, na3-1, 0, na2-1, 0, na1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    Real xc = CellCenterX(i-NG, nc1, x1min, x1max);
    Real yc = CellCenterX(j-NG, nc2, x2min, x2max);
    Real zc = CellCenterX(k-NG, nc3, x3min, x3max);
    u(0, adm::ADM::I_ADM_ALPHA, k, j, i) = TriPoly(xc, yc, zc);
  });
  Real t3_val = 0.0, t3_grd = 0.0;
  Kokkos::parallel_reduce("ut_tri_T3", Kokkos::RangePolicy<>(0, nsample),
  KOKKOS_LAMBDA(const int n, Real &lval, Real &lgrd) {
    Real xp[3];
    for (int d = 0; d < 3; ++d) {
      // unsigned throughout: 15486071*n overflows a signed int for n > 138 and the
      // wrap is undefined behaviour, not the reproducible mix the test claims
      xp[d] = gmin[d] + (gmax[d] - gmin[d])
              *Rand01(15486071u*static_cast<unsigned int>(n)
                      + 17u*static_cast<unsigned int>(d));
    }
    int idcs[4] = {0, -1, -1, -1};
    particles::SetInterpIndices(xp, mb_par, ncell, idcs);
    Real lx[8] = {0.0}, ly[8] = {0.0}, lz[8] = {0.0};
    Real dlx[8] = {0.0}, dly[8] = {0.0}, dlz[8] = {0.0};
    particles::CalcTrilinearWghtAndDrv(xp, mb_par, ncell, idcs,
                                       lx, ly, lz, dlx, dly, dlz);
    Real got = particles::TrilinearInterpolator<NG>(u, adm::ADM::I_ADM_ALPHA,
                                                    idcs, lx, ly, lz);
    lval = fmax(lval, fabs(got - TriPoly(xp[0], xp[1], xp[2])));
    Real gwant[3];
    TriPolyGrad(xp[0], xp[1], xp[2], gwant);
    Real ggot[3];
    ggot[0] = particles::TrilinearInterpolator<NG>(u, adm::ADM::I_ADM_ALPHA,
                                                   idcs, dlx, ly, lz);
    ggot[1] = particles::TrilinearInterpolator<NG>(u, adm::ADM::I_ADM_ALPHA,
                                                   idcs, lx, dly, lz);
    ggot[2] = particles::TrilinearInterpolator<NG>(u, adm::ADM::I_ADM_ALPHA,
                                                   idcs, lx, ly, dlz);
    for (int d = 0; d < 3; ++d) { lgrd = fmax(lgrd, fabs(ggot[d] - gwant[d])); }
  }, Kokkos::Max<Real>(t3_val), Kokkos::Max<Real>(t3_grd));
  std::cout << "  [NGHOST=" << NG << "] T3 exactness: max |value error| " << t3_val
            << ", max |gradient error| " << t3_grd << std::endl;
  if (!(t3_val < 1.0e-12) || !(t3_grd < 1.0e-10)) {
    std::cout << "  T3 FAILED" << std::endl;
    return false;
  }

  // --- T4: positive definiteness over a random positive definite field ---------------
  par_for("ut_tri_fill_pd", DevExeSpace(), 0, na3-1, 0, na2-1, 0, na1-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    unsigned int s = static_cast<unsigned int>(((k*na2) + j)*na1 + i) + 1u;
    // gamma = L L^T + 0.05 I with a random lower-triangular L: positive definite by
    // construction, and independent from cell to cell, which is the hardest possible
    // field for a wide stencil
    Real l11 = 0.5 + Rand01(s*3u + 1u);
    Real l21 = 2.0*Rand01(s*3u + 2u) - 1.0;
    Real l22 = 0.5 + Rand01(s*3u + 3u);
    Real l31 = 2.0*Rand01(s*5u + 1u) - 1.0;
    Real l32 = 2.0*Rand01(s*5u + 2u) - 1.0;
    Real l33 = 0.5 + Rand01(s*5u + 3u);
    u(0, adm::ADM::I_ADM_GXX+0, k, j, i) = l11*l11 + 0.05;
    u(0, adm::ADM::I_ADM_GXX+1, k, j, i) = l21*l11;
    u(0, adm::ADM::I_ADM_GXX+2, k, j, i) = l31*l11;
    u(0, adm::ADM::I_ADM_GXX+3, k, j, i) = l21*l21 + l22*l22 + 0.05;
    u(0, adm::ADM::I_ADM_GXX+4, k, j, i) = l31*l21 + l32*l22;
    u(0, adm::ADM::I_ADM_GXX+5, k, j, i) = l31*l31 + l32*l32 + l33*l33 + 0.05;
  });
  int t4_tri_bad = 0, t4_hi_bad = 0;
  Kokkos::parallel_reduce("ut_tri_T4", Kokkos::RangePolicy<>(0, nsample),
  KOKKOS_LAMBDA(const int n, int &ltri, int &lhi) {
    Real xp[3];
    for (int d = 0; d < 3; ++d) {
      xp[d] = gmin[d] + (gmax[d] - gmin[d])
              *Rand01(2654435761u*static_cast<unsigned int>(n)
                      + 19u*static_cast<unsigned int>(d));
    }
    int idcs[4] = {0, -1, -1, -1};
    particles::SetInterpIndices(xp, mb_par, ncell, idcs);
    Real lx[8] = {0.0}, ly[8] = {0.0}, lz[8] = {0.0};
    Real dlx[8] = {0.0}, dly[8] = {0.0}, dlz[8] = {0.0};
    particles::CalcTrilinearWghtAndDrv(xp, mb_par, ncell, idcs,
                                       lx, ly, lz, dlx, dly, dlz);
    Real gtri[6];
    for (int m = 0; m < 6; ++m) {
      gtri[m] = particles::TrilinearInterpolator<NG>(u, adm::ADM::I_ADM_GXX+m,
                                                     idcs, lx, ly, lz);
    }
    if (!SylvesterPD(gtri)) { ltri += 1; }
    particles::CalcInterpWghtAndDrv<NG>(xp, mb_par, ncell, idcs,
                                        lx, ly, lz, dlx, dly, dlz);
    Real ghi[6];
    for (int m = 0; m < 6; ++m) {
      ghi[m] = particles::LagrangeInterpolator<NG>(u, adm::ADM::I_ADM_GXX+m,
                                                   idcs, lx, ly, lz);
    }
    if (!SylvesterPD(ghi)) { lhi += 1; }
  }, Kokkos::Sum<int>(t4_tri_bad), Kokkos::Sum<int>(t4_hi_bad));
  std::cout << "  [NGHOST=" << NG << "] T4 positive definiteness over " << nsample
            << " samples of a random positive definite field: trilinear failures "
            << t4_tri_bad << ", " << 2*NG << "-node Lagrange failures " << t4_hi_bad
            << std::endl;
  if (t4_tri_bad != 0) {
    std::cout << "  T4 FAILED -- the convex combination is not positive definite"
              << std::endl;
    return false;
  }
  // The theorem alone would pass on a field where the wide stencil is fine too, which
  // would make the check vacuous. Assert the counter-example as well, for the widths
  // where it is robust: at NGHOST=2 the 4-node stencil fails only a handful of times.
  if (NG >= 3 && t4_hi_bad == 0) {
    std::cout << "  T4 FAILED -- the " << 2*NG << "-node interpolant did not overshoot "
              << "anywhere, so the test is not exercising the case the fallback exists "
              << "for" << std::endl;
    return false;
  }

  // --- T5: InverseMetricGradient against a central difference ------------------------
  Real t5_err = 0.0;
  Kokkos::parallel_reduce("ut_tri_T5", Kokkos::RangePolicy<>(0, nsample),
  KOKKOS_LAMBDA(const int n, Real &lerr) {
    Real xp[3];
    for (int d = 0; d < 3; ++d) {
      xp[d] = -0.8 + 1.6*Rand01(40503u*static_cast<unsigned int>(n)
                                + 23u*static_cast<unsigned int>(d) + 7u);
    }
    Real g[6];
    AnalyticMetric(xp[0], xp[1], xp[2], g);
    Real gu[6];
    Primitive::InvertMatrix(gu, g, Primitive::GetDeterminant(g));
    const Real eps = 1.0e-5;
    for (int i = 0; i < 3; ++i) {
      Real xm[3] = {xp[0], xp[1], xp[2]}, xq[3] = {xp[0], xp[1], xp[2]};
      xm[i] -= eps;
      xq[i] += eps;
      Real gm[6], gq[6], gum[6], guq[6], dgd[6];
      AnalyticMetric(xm[0], xm[1], xm[2], gm);
      AnalyticMetric(xq[0], xq[1], xq[2], gq);
      Primitive::InvertMatrix(gum, gm, Primitive::GetDeterminant(gm));
      Primitive::InvertMatrix(guq, gq, Primitive::GetDeterminant(gq));
      for (int m = 0; m < 6; ++m) { dgd[m] = (gq[m] - gm[m])/(2.0*eps); }
      Real dgu[6];
      particles::InverseMetricGradient(dgu, gu, dgd);
      for (int m = 0; m < 6; ++m) {
        Real fd = (guq[m] - gum[m])/(2.0*eps);
        Real scale = fmax(fabs(fd), 1.0);
        lerr = fmax(lerr, fabs(dgu[m] - fd)/scale);
      }
    }
  }, Kokkos::Max<Real>(t5_err));
  std::cout << "  [NGHOST=" << NG << "] T5 InverseMetricGradient: max relative "
            << "deviation from a central difference " << t5_err << std::endl;
  if (!(t5_err < 1.0e-6)) {
    std::cout << "  T5 FAILED" << std::endl;
    return false;
  }
  return true;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::TrilinearInterp()
//! \brief unit tests for the trilinear particle-interpolation kernels

void ProblemGenerator::TrilinearInterp(ParameterInput *pin, const bool restart) {
  std::cout << "=== trilinear particle-interpolation unit tests ===" << std::endl;
  bool ok = RunChecks<2>() && RunChecks<3>() && RunChecks<4>();
  if (!ok) {
    std::cout << "Trilinear Interpolation Test Failed" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "Test Passed: trilinear interpolation weights, stencil offset, "
            << "exactness, positive definiteness and the inverse-metric gradient"
            << std::endl;
  return;
}
