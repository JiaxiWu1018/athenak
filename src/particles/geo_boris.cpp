//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geo_boris.cpp
//  \brief Boris pusher utilizing DGREM method in Olivares et al.

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "boris_utils.hpp"
#include "lagrange_interp.hpp"
#include "calc_tetrad.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {

template <int NGHOST>
struct GeoBorisFunctor {
  const Real *x_nm1, *x_nmh, *u_nm1, *mb_par;
  const int *ncell;
  const DvceArray5D<Real> &adm_nm1, &adm_n, &w0_nm1, &w0_n, &b_nm1, &b_n;
  const DvceArray5D<Real> &z4c_nm1, &z4c_n;
  const int mb;
  const Real dt, qom;
  const bool use_mhd;
  const bool use_z4c;
  Real flat_dt;
  Real tetrad[4][4] = {0.0}, inv_tetrad[4][4] = {0.0};
  Real Ehat_ul[4][3] = {0.0}, Bhat_uu[4][3] = {0.0};
  Real Bhat_em[3] = {0.0}, Ehat_em[3] = {0.0}, uhat_nm1[3] = {0.0};
  const Real lc3[3][3][3] = {{{0., 0., 0.}, {0., 0., 1.}, {0., -1., 0.}},
                             {{0., 0., -1.}, {0., 0., 0.}, {1., 0., 0.}},
                             {{0., 1., 0.}, {-1., 0., 0.}, {0., 0., 0.}}};

  KOKKOS_INLINE_FUNCTION
  GeoBorisFunctor(const Real x_nm1_[3], const Real x_nmh_[3], const Real u_nm1_[3], const int mb_,
                  const Real mb_par_[9], const int ncell_[3], const Real dt_, const Real qom_,
                  const DvceArray5D<Real> &adm_nm1_, const DvceArray5D<Real> &adm_n_,
                  const DvceArray5D<Real> &w0_nm1_, const DvceArray5D<Real> &w0_n_,
                  const DvceArray5D<Real> &b_nm1_, const DvceArray5D<Real> &b_n_,
                  const bool use_mhd_,
                  const DvceArray5D<Real> &z4c_nm1_, const DvceArray5D<Real> &z4c_n_,
                  const bool use_z4c_)
    : x_nm1(x_nm1_), x_nmh(x_nmh_), u_nm1(u_nm1_), mb(mb_),
      mb_par(mb_par_), ncell(ncell_), dt(dt_), qom(qom_),
      use_mhd(use_mhd_), use_z4c(use_z4c_),
      adm_nm1(adm_nm1_), adm_n(adm_n_),
      w0_nm1(w0_nm1_), w0_n(w0_n_),
      b_nm1(b_nm1_), b_n(b_n_),
      z4c_nm1(z4c_nm1_), z4c_n(z4c_n_) {
    // Calculate Lagrangian weight at x^{n-1/2}
    constexpr int N = 2 * NGHOST;
    int interp_indcs_nmh[4] = {mb, -1, -1, -1};
    SetInterpIndices(x_nmh, mb_par, ncell, interp_indcs_nmh);
    Real Lx_nmh[N] = {0.0}, Ly_nmh[N] = {0.0}, Lz_nmh[N] = {0.0};
    Real dLx_nmh[N] = {0.0}, dLy_nmh[N] = {0.0}, dLz_nmh[N] = {0.0};
    CalcInterpWghtAndDrv<NGHOST>(x_nmh, mb_par, ncell, interp_indcs_nmh,
                                 Lx_nmh, Ly_nmh, Lz_nmh, dLx_nmh, dLy_nmh, dLz_nmh);
    // Calculate Lagrangian weight at x^{n-1}
    int interp_indcs_nm1[4] = {mb, -1, -1, -1};
    SetInterpIndices(x_nm1, mb_par, ncell, interp_indcs_nm1);
    Real Lx_nm1[N] = {0.0}, Ly_nm1[N] = {0.0}, Lz_nm1[N] = {0.0};
    CalcInterpWght<NGHOST>(x_nm1, mb_par, ncell, interp_indcs_nm1, Lx_nm1, Ly_nm1, Lz_nm1);

    // Interpolate B, v at x=x^{n-1/2} (only when MHD is enabled)
    Real B_nmh[3] = {0.0}, v_nmh[3] = {0.0}, E_nmh[3] = {0.0};
    if (use_mhd) {
      for (int i = 0; i < 3; ++i) {
        Real B_nm1 = LagrangeInterpolator<NGHOST>(b_nm1, i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        Real B_n = LagrangeInterpolator<NGHOST>(b_n, i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        B_nmh[i] = 0.5 * (B_nm1 + B_n);
        Real v_nm1 = LagrangeInterpolator<NGHOST>(w0_nm1, IVX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        Real v_n = LagrangeInterpolator<NGHOST>(w0_n, IVX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        v_nmh[i] = 0.5 * (v_nm1 + v_n);
      }
    }
    // Interpolate adm variables at x=x^{n-1} t=x^{n-1} and x=x^{n-1/2}, t=t^{n-1/2}.
    // When Z4c is on the lapse and shift live in the Z4c array (adm has those
    // slots removed); g_ij stays in the ADM array either way.
    Real alp_nmh, beta_nmh[3], g3d_nmh[6];
    Real alp_nm1, beta_nm1[3], g3d_nm1[6];
    if (use_z4c) {
      alp_nm1 = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nm1, Lx_nm1, Ly_nm1, Lz_nm1);
      Real alp_mid_nm1 = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      Real alp_mid_n = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      alp_nmh = 0.5 * (alp_mid_n + alp_mid_nm1);
      for (int i = 0; i < 3; ++i) {
        beta_nm1[i] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nm1, Lx_nm1, Ly_nm1, Lz_nm1);
        Real beta_mid_nm1 = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        Real beta_mid_n = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        beta_nmh[i] = 0.5 * (beta_mid_n + beta_mid_nm1);
      }
    } else {
      alp_nm1 = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_ALPHA, interp_indcs_nm1, Lx_nm1, Ly_nm1, Lz_nm1);
      Real alp_mid_nm1 = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      Real alp_mid_n = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      alp_nmh = 0.5 * (alp_mid_n + alp_mid_nm1);
      for (int i = 0; i < 3; ++i) {
        beta_nm1[i] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_BETAX + i, interp_indcs_nm1, Lx_nm1, Ly_nm1, Lz_nm1);
        Real beta_mid_nm1 = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        Real beta_mid_n = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        beta_nmh[i] = 0.5 * (beta_mid_n + beta_mid_nm1);
      }
    }
    flat_dt = dt * alp_nmh;
    for (int i = 0; i < 6; ++i) {
      g3d_nm1[i] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_GXX + i, interp_indcs_nm1, Lx_nm1, Ly_nm1, Lz_nm1);
      Real g3d_mid_nm1 = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      Real g3d_mid_n = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      g3d_nmh[i] = 0.5 * (g3d_mid_n + g3d_mid_nm1);
    }
    // Always compute the spatial determinant at x^{n-1/2}: the magnetic part of
    // the gravito-Faraday tensor below uses 1/sqrt(det) regardless of MHD.
    Real det_nmh = Primitive::GetDeterminant(g3d_nmh);
    Real sqrtdet_nmh = std::sqrt(det_nmh);

    // Calculate tetrad at x^{n-1/2}; always needed for the gravito-fields.
    CalcTetrad(alp_nmh, beta_nmh, g3d_nmh, tetrad, inv_tetrad);

    // Calculate E field assuming ideal MHD and project EM fields onto the tetrad.
    // Skipped when MHD is absent (Ehat_em, Bhat_em remain zero).
    if (use_mhd) {
      E_nmh[0] = sqrtdet_nmh * (B_nmh[1] * v_nmh[2] - B_nmh[2] * v_nmh[1]);
      E_nmh[1] = sqrtdet_nmh * (B_nmh[2] * v_nmh[0] - B_nmh[0] * v_nmh[2]);
      E_nmh[2] = sqrtdet_nmh * (B_nmh[0] * v_nmh[1] - B_nmh[1] * v_nmh[0]);
      TetradCvrtL(Ehat_em, E_nmh, inv_tetrad);
      TetradCvrtU(Bhat_em, B_nmh, tetrad);
    }
    // Calculate tetrad at x^{n-1} and turn velocity into tetrad frame
    Real tetrad_nm1[4][4] = {0.0}, inv_tetrad_nm1[4][4] = {0.0};
    CalcTetrad(alp_nm1, beta_nm1, g3d_nm1, tetrad_nm1, inv_tetrad_nm1);
    TetradCvrtL(uhat_nm1, u_nm1, inv_tetrad_nm1);

    // Interpolate the spacetime derivatives of adm variables at x=x^{n-1/2}.
    // Lapse/shift come from Z4c when active; g_ij always from ADM.
    Real dalp[4] = {0.0}, dbeta[4][3] = {0.0}, dg3d[4][6] = {0.0};
    Real dalp_nm1[4] = {0.0}, dalp_n[4] = {0.0};
    Real idt = 1.0 / dt;
    if (use_z4c) {
      dalp_nm1[0] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      dalp_n[0] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      dalp[0] = (dalp_n[0] - dalp_nm1[0]) * idt;
      dalp_nm1[1] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
      dalp_n[1] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
      dalp[1] = 0.5 * (dalp_nm1[1] + dalp_n[1]);
      dalp_nm1[2] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
      dalp_n[2] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
      dalp[2] = 0.5 * (dalp_nm1[2] + dalp_n[2]);
      dalp_nm1[3] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
      dalp_n[3] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
      dalp[3] = 0.5 * (dalp_nm1[3] + dalp_n[3]);
      for (int i = 0; i < 3; ++i) {
        Real dbeta_nm1[4] = {0.0}, dbeta_n[4] = {0.0};
        dbeta_nm1[0] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        dbeta_n[0] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        dbeta[0][i] = (dbeta_n[0] - dbeta_nm1[0]) * idt;
        dbeta_nm1[1] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
        dbeta_n[1] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
        dbeta[1][i] = 0.5 * (dbeta_nm1[1] + dbeta_n[1]);
        dbeta_nm1[2] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
        dbeta_n[2] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
        dbeta[2][i] = 0.5 * (dbeta_nm1[2] + dbeta_n[2]);
        dbeta_nm1[3] = LagrangeInterpolator<NGHOST>(z4c_nm1, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
        dbeta_n[3] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
        dbeta[3][i] = 0.5 * (dbeta_nm1[3] + dbeta_n[3]);
      }
    } else {
      dalp_nm1[0] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      dalp_n[0] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      dalp[0] = (dalp_n[0] - dalp_nm1[0]) * idt;
      dalp_nm1[1] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
      dalp_n[1] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
      dalp[1] = 0.5 * (dalp_nm1[1] + dalp_n[1]);
      dalp_nm1[2] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
      dalp_n[2] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
      dalp[2] = 0.5 * (dalp_nm1[2] + dalp_n[2]);
      dalp_nm1[3] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
      dalp_n[3] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
      dalp[3] = 0.5 * (dalp_nm1[3] + dalp_n[3]);
      for (int i = 0; i < 3; ++i) {
        Real dbeta_nm1[4] = {0.0}, dbeta_n[4] = {0.0};
        dbeta_nm1[0] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        dbeta_n[0] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
        dbeta[0][i] = (dbeta_n[0] - dbeta_nm1[0]) * idt;
        dbeta_nm1[1] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
        dbeta_n[1] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
        dbeta[1][i] = 0.5 * (dbeta_nm1[1] + dbeta_n[1]);
        dbeta_nm1[2] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
        dbeta_n[2] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
        dbeta[2][i] = 0.5 * (dbeta_nm1[2] + dbeta_n[2]);
        dbeta_nm1[3] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
        dbeta_n[3] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
        dbeta[3][i] = 0.5 * (dbeta_nm1[3] + dbeta_n[3]);
      }
    }
    for (int i = 0; i < 6; ++i) {
      Real dg3d_nm1[4] = {0.0}, dg3d_n[4] = {0.0};
      dg3d_nm1[0] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      dg3d_n[0] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, Lz_nmh);
      dg3d[0][i] = (dg3d_n[0] - dg3d_nm1[0]) * idt;
      dg3d_nm1[1] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
      dg3d_n[1] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, dLx_nmh, Ly_nmh, Lz_nmh);
      dg3d[1][i] = 0.5 * (dg3d_nm1[1] + dg3d_n[1]);
      dg3d_nm1[2] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
      dg3d_n[2] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, dLy_nmh, Lz_nmh);
      dg3d[2][i] = 0.5 * (dg3d_nm1[2] + dg3d_n[2]);
      dg3d_nm1[3] = LagrangeInterpolator<NGHOST>(adm_nm1, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
      dg3d_n[3] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs_nmh, Lx_nmh, Ly_nmh, dLz_nmh);
      dg3d[3][i] = 0.5 * (dg3d_nm1[3] + dg3d_n[3]);
    }

    // Calculate Faraday tensor
    Real Faraday[4][4][4] = {0.0};
    CalcFaraday(alp_nmh, beta_nmh, g3d_nmh, dalp, dbeta, dg3d, tetrad, Faraday);

    // Calculate normal observer.
    Real ialp = 1. / alp_nmh;
    Real n_u[4] = {0.0};
    for (int i = 1; i < 4; ++i) {
      n_u[i] = -1. * ialp * beta_nmh[i - 1];
    }
    n_u[0] = ialp;

    // Calculate E^{\hat{a}}_i
    Real E_uhat_l[4][3] = {0.0};
    for (int a = 0; a < 4; ++a) {
      for (int i = 0; i < 3; ++i) {
        for (int b = 0; b < 4; ++b) {
          E_uhat_l[a][i] += Faraday[a][i + 1][b] * n_u[b];
        }
      }
    }
    // Calculate B^{\hat{a}i}
    Real B_uhat_u[4][3] = {0.0};
    Real isqrtdet = 1. / sqrtdet_nmh;
    for (int a = 0; a < 4; ++a) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (j == i) continue;
          for (int k = 0; k < 3; ++k) {
            if (k == i || k == j) continue;
            B_uhat_u[a][i] += 0.5 * isqrtdet * lc3[i][j][k] * Faraday[a][j + 1][k + 1];
          }
        }
      }
    }
    // Calculate E^{\hat{a}}_{\hat{i}} and B^{\hat{a}\hat{i}}
    for (int a = 0; a < 4; ++a) {
      for (int i = 0; i < 3; ++i) {
        Ehat_ul[a][i] = 0.0;
        Bhat_uu[a][i] = 0.0;
        for (int j = 0; j < 3; ++j) {
          Ehat_ul[a][i] += inv_tetrad[j + 1][i + 1] * E_uhat_l[a][j];
          Bhat_uu[a][i] += tetrad[i + 1][j + 1] * B_uhat_u[a][j];
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Real xin[3], const Real uin[3], Real xout[3], Real uout[3]) const {
    Real x_mid[3] = {0.0}, u_mid[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      x_mid[i] = 0.5 * (xin[i] + x_nmh[i]);
      u_mid[i] = 0.5 * (uin[i] + u_nm1[i]);
    }
    Real uhat_mid_3[3] = {0.0};
    TetradCvrtL(uhat_mid_3, u_mid, inv_tetrad);
    Real uhat_mid[4] = {0.0};
    for (int i = 1; i < 4; ++i) {
      uhat_mid[i] = uhat_mid_3[i - 1];
    }
    uhat_mid[0] = -1. * std::sqrt(1. + Primitive::Contract(uhat_mid_3, uhat_mid_3));

    // Step 1: Update velocity
    // (i) Calculate gravitatioal electric and magnetic field
    Real Ehat_gr[3] = {0.0}, Bhat_gr[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      for (int a = 0; a < 4; ++a) {
        Ehat_gr[i] += Ehat_ul[a][i] * uhat_mid[a];
        Bhat_gr[i] += Bhat_uu[a][i] * uhat_mid[a];
      }
    }
    Real Ehat[3] = {0.0}, Bhat[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      Ehat[i] = qom * Ehat_em[i] + Ehat_gr[i];
      Bhat[i] = qom * Bhat_em[i] + Bhat_gr[i];
    }
    // (ii) Do flat spacetime push
    Real uhat_n[3] = {0.0};
    FlatBorisPush(uhat_n, uhat_nm1, Ehat, Bhat, 1, flat_dt);
    // (iii) Interpolate adm variables at t=n, x=x_mid.
    // Lapse and shift come from Z4c when active; g_ij always from ADM.
    int interp_indcs[4] = {mb, -1, -1, -1};
    SetInterpIndices(x_mid, mb_par, ncell, interp_indcs);
    constexpr int N = 2 * NGHOST;
    Real Lx[N] = {0.0}, Ly[N] = {0.0}, Lz[N] = {0.0};
    CalcInterpWght<NGHOST>(x_mid, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
    Real alp_n;
    Real beta_n[3] = {0.0}, g3d_n[6] = {0.0};
    if (use_z4c) {
      alp_n = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_n[i] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs, Lx, Ly, Lz);
      }
    } else {
      alp_n = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_n[i] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs, Lx, Ly, Lz);
      }
    }
    for (int i = 0; i < 6; ++i) {
      g3d_n[i] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs, Lx, Ly, Lz);
    }
    // (iv) Calculate tetrad at x^n and turn velocity back to global frame
    Real tetrad_n[4][4] = {0.0}, inv_tetrad_n[4][4] = {0.0};
    CalcTetrad(alp_n, beta_n, g3d_n, tetrad_n, inv_tetrad_n);
    TetradCvrtL(uout, uhat_n, tetrad_n);

    // Step 2: Update position
    // (i) calculate transport velocity
    Real det_n = Primitive::GetDeterminant(g3d_n);
    Real g3u_n[6] = {0.0};
    Primitive::InvertMatrix(g3u_n, g3d_n, det_n);
    Real uin_u[3] = {0.0};
    Primitive::RaiseForm(uin_u, uin, g3u_n);
    Real Lorentz_in = std::sqrt(1.0 + Primitive::Contract(uin_u, uin));
    Real iLorentz_in = 1. / Lorentz_in;
    Real v[3] = {0.0};
    for (int i = 0; i < 3 ; ++i) {
      v[i] = alp_n * iLorentz_in * uin_u[i] - beta_n[i];
    }
    // (ii) update position
    for (int i = 0; i < 3; ++i) {
      xout[i] = x_nmh[i] + dt * v[i];
    }
  }
};

template<class F>
KOKKOS_INLINE_FUNCTION
bool FixedPointIteration(const F& f, const Real x0[3], const Real u0[3],
                         Real x[3], Real u[3], Real tol=1e-7, int maxIter=50) {
  Real x_new[3], u_new[3];
  for (int i = 0; i < 3; ++i) {
    x_new[i] = x0[i];
    u_new[i] = u0[i];
  }
  Real x_next[3], u_next[3];
  for (int iter = 0; iter < maxIter; ++iter) {
    f(x_new, u_new, x_next, u_next);
    bool to_break = false;
    for (int i = 0; i < 3; ++i) {
      if (!isfinite(x_next[i]) || !isfinite(u_next[i])) {
        to_break = true;
      }
    }
    if (to_break) {
      break;
    }
    Real err = 0.0;
    for (int i = 0; i < 3; ++i) {
      Real dx = x_next[i] - x_new[i];
      Real du = u_next[i] - u_new[i];
      err = fmax(err, fabs(dx));
      err = fmax(err, fabs(du));
    }
    if (err < tol) {
      for (int i = 0; i < 3; ++i) {
        x[i] = x_next[i];
        u[i] = u_next[i];
      }
      return true;
    }
    for (int i = 0; i < 3; ++i) {
      x_new[i] = x_next[i];
      u_new[i] = u_next[i];
    }
  }
  f(x0, u0, x, u);
  for (int i = 0; i < 3; ++i) {
    if (!isfinite(x[i]) || !isfinite(u[i])) {
      for (int j = 0; j < 3; ++j) {
        x[j] = x0[j];
        u[j] = u0[j];
      }
      break;
    }
  }
  return false;
}

void Particles::Geo_BorisPush() {
  // Extract MHD variables
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  Real dt_ = pmy_pack->pmesh->dt;
  Real qom = q_over_m;

  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &adm_nm1 = adm_last;
  auto &adm_n = pmy_pack->padm->u_adm;

  // MHD is optional: dust particles on a Z4c-only background do not need it.
  // Skip all MHD interpolations / projections / stash copies in that case.
  const bool use_mhd = (pmy_pack->pmhd != nullptr);
  if ((!use_mhd) && (std::abs(qom) > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Charged geo_boris particles require MHD variables"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DvceArray5D<Real> w0_nm1, bcc0_nm1, w0_n, bcc0_n;
  if (use_mhd) {
    w0_nm1 = w0_last;
    bcc0_nm1 = bcc0_last;
    w0_n = pmy_pack->pmhd->w0;
    bcc0_n = pmy_pack->pmhd->bcc0;
  }

  // Z4c is optional: stationary-spacetime tests don't need it.  When Z4c is
  // on, lapse and shift are stored in pz4c->u0 (and the ADM array's alpha/beta
  // slots are removed), so we must read them from z4c_n / z4c_nm1.
  const bool use_z4c = (pmy_pack->pz4c != nullptr);
  DvceArray5D<Real> z4c_nm1, z4c_n;
  if (use_z4c) {
    z4c_nm1 = z4c_last;
    z4c_n = pmy_pack->pz4c->u0;
  }

  // Loop over all particles
  par_for("geo_boris_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    // Retrieve particle position and velocity
    Real x_nm1[3] = {pr(IPLX, p), pr(IPLY, p), pr(IPLZ, p)};
    Real x_nmh[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    Real u_nm1[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};

    // Implicitly solve the momentum equation
    Real u_n[3] = {0.0}, x_nph[3] = {0.0};
    bool find_root = false;
    switch (ng) {
    case 2: {
      GeoBorisFunctor<2> geoboris(x_nm1, x_nmh, u_nm1, mb, mb_par, ncell, dt_, qom,
                                  adm_nm1, adm_n, w0_nm1, w0_n, bcc0_nm1, bcc0_n,
                                  use_mhd, z4c_nm1, z4c_n, use_z4c);
      find_root = FixedPointIteration(geoboris, x_nmh, u_nm1, x_nph, u_n);
      break;
    }
    case 3: {
      GeoBorisFunctor<3> geoboris(x_nm1, x_nmh, u_nm1, mb, mb_par, ncell, dt_, qom,
                                  adm_nm1, adm_n, w0_nm1, w0_n, bcc0_nm1, bcc0_n,
                                  use_mhd, z4c_nm1, z4c_n, use_z4c);
      find_root = FixedPointIteration(geoboris, x_nmh, u_nm1, x_nph, u_n);
      break;
    }
    case 4: {
      GeoBorisFunctor<4> geoboris(x_nm1, x_nmh, u_nm1, mb, mb_par, ncell, dt_, qom,
                                  adm_nm1, adm_n, w0_nm1, w0_n, bcc0_nm1, bcc0_n,
                                  use_mhd, z4c_nm1, z4c_n, use_z4c);
      find_root = FixedPointIteration(geoboris, x_nmh, u_nm1, x_nph, u_n);
      break;
    }}

    // Update particle position and speed into device memory
    if (!find_root) {
      Kokkos::printf("Root finding of geo_boris pusher failed; using one explicit update.\n");
    }
    pr(IPLX, p) = 0.5 * (x_nmh[0] + x_nph[0]);
    pr(IPLY, p) = 0.5 * (x_nmh[1] + x_nph[1]);
    pr(IPLZ, p) = 0.5 * (x_nmh[2] + x_nph[2]);
    pr(IPX, p) = x_nph[0];
    pr(IPY, p) = x_nph[1];
    pr(IPZ, p) = x_nph[2];
    pr(IPVX, p) = u_n[0];
    pr(IPVY, p) = u_n[1];
    pr(IPVZ, p) = u_n[2];
  });

  // Stash current state as the "previous step" data for the next call.
  if (use_mhd) {
    Kokkos::deep_copy(DevExeSpace(), w0_last, pmy_pack->pmhd->w0);
    Kokkos::deep_copy(DevExeSpace(), bcc0_last, pmy_pack->pmhd->bcc0);
  }
  Kokkos::deep_copy(DevExeSpace(), adm_last, pmy_pack->padm->u_adm);
  if (use_z4c) {
    Kokkos::deep_copy(DevExeSpace(), z4c_last, pmy_pack->pz4c->u0);
  }
} // end BorisPush

template<int NG>
KOKKOS_INLINE_FUNCTION
void GetADMVariables(Real &alp, Real *beta, Real *g3d, const Real *x_mid,
                     const int mb, const Real *mb_par, const int *ncell,
                     const DvceArray5D<Real> &adm_n,
                     const bool use_z4c,
                     const DvceArray5D<Real> &z4c_n) {
  int interp_indcs[4] = {mb, -1, -1, -1};
  SetInterpIndices(x_mid, mb_par, ncell, interp_indcs);
  constexpr int N = 2 * NG;
  Real Lx[N] = {0.0}, Ly[N] = {0.0}, Lz[N] = {0.0};
  particles::CalcInterpWght<NG>(x_mid, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
  // Lapse and shift live in Z4c when active, otherwise in ADM.
  if (use_z4c) {
    alp = LagrangeInterpolator<NG>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
    for (int i = 0; i < 3; ++i) {
      beta[i] = LagrangeInterpolator<NG>(z4c_n, z4c::Z4c::I_Z4C_BETAX + i, interp_indcs, Lx, Ly, Lz);
    }
  } else {
    alp = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
    for (int i = 0; i < 3; ++i) {
      beta[i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_BETAX + i, interp_indcs, Lx, Ly, Lz);
    }
  }
  // The 3-metric g_ij is always stored in the ADM array.
  for (int i = 0; i < 6; ++i) {
    g3d[i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_GXX + i, interp_indcs, Lx, Ly, Lz);
  }
}

KOKKOS_INLINE_FUNCTION
int Sym3Index(const int i, const int j) {
  const int a = (i < j) ? i : j;
  const int b = (i < j) ? j : i;
  if (a == 0 && b == 0) return 0;
  if (a == 0 && b == 1) return 1;
  if (a == 0 && b == 2) return 2;
  if (a == 1 && b == 1) return 3;
  if (a == 1 && b == 2) return 4;
  return 5;
}

KOKKOS_INLINE_FUNCTION
Real Sym3Value(const Real g[6], const int i, const int j) {
  return g[Sym3Index(i, j)];
}

KOKKOS_INLINE_FUNCTION
void FourVelocityFromCovMomentum(const Real alp, const Real beta[3], const Real g3d[6],
                                 const Real u_l[3], Real U[4]) {
  Real g3u[6] = {0.0};
  Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));
  Real u_u[3] = {0.0};
  Primitive::RaiseForm(u_u, u_l, g3u);
  const Real W = std::sqrt(1.0 + Primitive::Contract(u_u, u_l));
  U[0] = W / alp;
  for (int i = 0; i < 3; ++i) {
    U[i + 1] = u_u[i] - beta[i] * U[0];
  }
}

KOKKOS_INLINE_FUNCTION
void BuildMetricAndChristoffel(const Real alp, const Real beta[3], const Real g3d[6],
                               const Real dalp[4], const Real dbeta[4][3],
                               const Real dg3d[4][6],
                               Real gcov[4][4], Real gcon[4][4],
                               Real Gamma[4][4][4]) {
  Real g3u[6] = {0.0};
  Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));

  Real beta_l[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      beta_l[i] += Sym3Value(g3d, i, j) * beta[j];
    }
  }

  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      gcov[mu][nu] = 0.0;
      gcon[mu][nu] = 0.0;
    }
  }

  gcov[0][0] = -alp * alp;
  for (int i = 0; i < 3; ++i) {
    gcov[0][0] += beta_l[i] * beta[i];
    gcov[0][i + 1] = beta_l[i];
    gcov[i + 1][0] = beta_l[i];
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      gcov[i + 1][j + 1] = Sym3Value(g3d, i, j);
    }
  }

  const Real ialp2 = 1.0 / (alp * alp);
  gcon[0][0] = -ialp2;
  for (int i = 0; i < 3; ++i) {
    gcon[0][i + 1] = beta[i] * ialp2;
    gcon[i + 1][0] = beta[i] * ialp2;
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      gcon[i + 1][j + 1] = Sym3Value(g3u, i, j) - beta[i] * beta[j] * ialp2;
    }
  }

  Real dgcov[4][4][4] = {0.0};
  for (int d = 1; d < 4; ++d) {
    Real dbeta_l[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        dbeta_l[i] += Sym3Value(dg3d[d], i, j) * beta[j]
                    + Sym3Value(g3d, i, j) * dbeta[d][j];
      }
    }

    Real dshift2 = 0.0;
    for (int i = 0; i < 3; ++i) {
      dshift2 += dbeta_l[i] * beta[i] + beta_l[i] * dbeta[d][i];
    }
    dgcov[d][0][0] = -2.0 * alp * dalp[d] + dshift2;
    for (int i = 0; i < 3; ++i) {
      dgcov[d][0][i + 1] = dbeta_l[i];
      dgcov[d][i + 1][0] = dbeta_l[i];
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        dgcov[d][i + 1][j + 1] = Sym3Value(dg3d[d], i, j);
      }
    }
  }

  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      for (int lam = 0; lam < 4; ++lam) {
        Real sum = 0.0;
        for (int sig = 0; sig < 4; ++sig) {
          sum += gcon[mu][sig] * (dgcov[nu][sig][lam] + dgcov[lam][sig][nu]
                                - dgcov[sig][nu][lam]);
        }
        Gamma[mu][nu][lam] = 0.5 * sum;
      }
    }
  }
}

template<int NG>
KOKKOS_INLINE_FUNCTION
void GetStationaryADMAndChristoffel(Real &alp, Real beta[3], Real g3d[6],
                                    Real gcov[4][4], Real gcon[4][4],
                                    Real Gamma[4][4][4], const Real x[3],
                                    const int mb, const Real mb_par[9],
                                    const int ncell[3],
                                    const DvceArray5D<Real> &adm_n) {
  int interp_indcs[4] = {mb, -1, -1, -1};
  SetInterpIndices(x, mb_par, ncell, interp_indcs);
  constexpr int N = 2 * NG;
  Real Lx[N] = {0.0}, Ly[N] = {0.0}, Lz[N] = {0.0};
  Real dLx[N] = {0.0}, dLy[N] = {0.0}, dLz[N] = {0.0};
  CalcInterpWghtAndDrv<NG>(x, mb_par, ncell, interp_indcs,
                           Lx, Ly, Lz, dLx, dLy, dLz);

  Real dalp[4] = {0.0}, dbeta[4][3] = {0.0}, dg3d[4][6] = {0.0};
  alp = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs,
                                 Lx, Ly, Lz);
  dalp[1] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs,
                                     dLx, Ly, Lz);
  dalp[2] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs,
                                     Lx, dLy, Lz);
  dalp[3] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs,
                                     Lx, Ly, dLz);
  for (int i = 0; i < 3; ++i) {
    beta[i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_BETAX + i,
                                       interp_indcs, Lx, Ly, Lz);
    dbeta[1][i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_BETAX + i,
                                           interp_indcs, dLx, Ly, Lz);
    dbeta[2][i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_BETAX + i,
                                           interp_indcs, Lx, dLy, Lz);
    dbeta[3][i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_BETAX + i,
                                           interp_indcs, Lx, Ly, dLz);
  }
  for (int i = 0; i < 6; ++i) {
    g3d[i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_GXX + i,
                                      interp_indcs, Lx, Ly, Lz);
    dg3d[1][i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_GXX + i,
                                          interp_indcs, dLx, Ly, Lz);
    dg3d[2][i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_GXX + i,
                                          interp_indcs, Lx, dLy, Lz);
    dg3d[3][i] = LagrangeInterpolator<NG>(adm_n, adm::ADM::I_ADM_GXX + i,
                                          interp_indcs, Lx, Ly, dLz);
  }

  BuildMetricAndChristoffel(alp, beta, g3d, dalp, dbeta, dg3d, gcov, gcon, Gamma);
}

KOKKOS_INLINE_FUNCTION
void LowerFourVelocity(const Real gcov[4][4], const Real U[4], Real U_l[4]) {
  for (int mu = 0; mu < 4; ++mu) {
    U_l[mu] = 0.0;
    for (int nu = 0; nu < 4; ++nu) {
      U_l[mu] += gcov[mu][nu] * U[nu];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void FrameComponentsFromFourVelocity(const Real e[4][4], const Real gcov[4][4],
                                     const Real U[4], Real Uhat[4]) {
  Real U_l[4] = {0.0};
  LowerFourVelocity(gcov, U, U_l);
  for (int a = 0; a < 4; ++a) {
    Real comp = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      comp += e[a][mu] * U_l[mu];
    }
    Uhat[a] = (a == 0) ? -comp : comp;
  }
  Uhat[0] = std::sqrt(1.0 + Uhat[1] * Uhat[1]
                          + Uhat[2] * Uhat[2]
                          + Uhat[3] * Uhat[3]);
}

KOKKOS_INLINE_FUNCTION
void FourVelocityFromFrameComponents(const Real e[4][4], const Real Uhat[4], Real U[4]) {
  for (int mu = 0; mu < 4; ++mu) {
    U[mu] = 0.0;
    for (int a = 0; a < 4; ++a) {
      U[mu] += e[a][mu] * Uhat[a];
    }
  }
}

KOKKOS_INLINE_FUNCTION
Real FourDot(const Real gcov[4][4], const Real a[4], const Real b[4]) {
  Real dot = 0.0;
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      dot += gcov[mu][nu] * a[mu] * b[nu];
    }
  }
  return dot;
}

KOKKOS_INLINE_FUNCTION
void OrthonormalizeTetrad(Real e[4][4], const Real gcov[4][4]) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < a; ++b) {
      Real eb[4] = {e[b][0], e[b][1], e[b][2], e[b][3]};
      const Real eta_b = (b == 0) ? -1.0 : 1.0;
      const Real proj = FourDot(gcov, e[a], eb) / eta_b;
      for (int mu = 0; mu < 4; ++mu) {
        e[a][mu] -= proj * e[b][mu];
      }
    }
    const Real eta_a = (a == 0) ? -1.0 : 1.0;
    const Real norm = FourDot(gcov, e[a], e[a]);
    const Real scale = 1.0 / std::sqrt(fabs(norm));
    for (int mu = 0; mu < 4; ++mu) {
      e[a][mu] *= scale;
    }
    if (eta_a * FourDot(gcov, e[a], e[a]) < 0.0) {
      for (int mu = 0; mu < 4; ++mu) {
        e[a][mu] *= -1.0;
      }
    }
  }
  if (e[0][0] < 0.0) {
    for (int mu = 0; mu < 4; ++mu) {
      e[0][mu] *= -1.0;
    }
  }
}

template<int NG>
KOKKOS_INLINE_FUNCTION
void GeoBorisFWDeriv(const Real x[3], const Real e[4][4], const Real Uhat[4],
                     const int mb, const Real mb_par[9], const int ncell[3],
                     const DvceArray5D<Real> &adm_n,
                     Real dxdt[3], Real dedt[4][4]) {
  Real alp = 0.0, beta[3] = {0.0}, g3d[6] = {0.0};
  Real gcov[4][4] = {0.0}, gcon[4][4] = {0.0}, Gamma[4][4][4] = {0.0};
  GetStationaryADMAndChristoffel<NG>(alp, beta, g3d, gcov, gcon, Gamma,
                                     x, mb, mb_par, ncell, adm_n);
  Real U[4] = {0.0};
  FourVelocityFromFrameComponents(e, Uhat, U);
  const Real iUt = 1.0 / U[0];
  for (int i = 0; i < 3; ++i) {
    dxdt[i] = U[i + 1] * iUt;
  }
  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      dedt[a][mu] = 0.0;
      for (int nu = 0; nu < 4; ++nu) {
        for (int lam = 0; lam < 4; ++lam) {
          dedt[a][mu] -= Gamma[mu][nu][lam] * e[a][nu] * U[lam] * iUt;
        }
      }
    }
  }
}

template<int NG>
KOKKOS_INLINE_FUNCTION
void GeoBorisFWAdvance(const Real x0[3], const Real e0[4][4], const Real Uhat[4],
                       const int mb, const Real mb_par[9], const int ncell[3],
                       const Real dt, const DvceArray5D<Real> &adm_n,
                       Real x1[3], Real e1[4][4]) {
  Real k1x[3] = {0.0}, k2x[3] = {0.0}, k3x[3] = {0.0}, k4x[3] = {0.0};
  Real k1e[4][4] = {0.0}, k2e[4][4] = {0.0};
  Real k3e[4][4] = {0.0}, k4e[4][4] = {0.0};
  Real xt[3] = {0.0}, et[4][4] = {0.0};

  GeoBorisFWDeriv<NG>(x0, e0, Uhat, mb, mb_par, ncell, adm_n, k1x, k1e);
  for (int i = 0; i < 3; ++i) {
    xt[i] = x0[i] + 0.5 * dt * k1x[i];
  }
  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      et[a][mu] = e0[a][mu] + 0.5 * dt * k1e[a][mu];
    }
  }

  GeoBorisFWDeriv<NG>(xt, et, Uhat, mb, mb_par, ncell, adm_n, k2x, k2e);
  for (int i = 0; i < 3; ++i) {
    xt[i] = x0[i] + 0.5 * dt * k2x[i];
  }
  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      et[a][mu] = e0[a][mu] + 0.5 * dt * k2e[a][mu];
    }
  }

  GeoBorisFWDeriv<NG>(xt, et, Uhat, mb, mb_par, ncell, adm_n, k3x, k3e);
  for (int i = 0; i < 3; ++i) {
    xt[i] = x0[i] + dt * k3x[i];
  }
  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      et[a][mu] = e0[a][mu] + dt * k3e[a][mu];
    }
  }

  GeoBorisFWDeriv<NG>(xt, et, Uhat, mb, mb_par, ncell, adm_n, k4x, k4e);
  for (int i = 0; i < 3; ++i) {
    x1[i] = x0[i] + dt * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]) / 6.0;
  }
  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      e1[a][mu] = e0[a][mu]
                + dt * (k1e[a][mu] + 2.0 * k2e[a][mu]
                      + 2.0 * k3e[a][mu] + k4e[a][mu]) / 6.0;
    }
  }
}

template<int NG>
KOKKOS_INLINE_FUNCTION
void GeoBorisFWBorisMap(const Real x0[3], const Real e0[4][4],
                        const Real Uhat0[4], const Real x_guess[3],
                        const Real e_guess[4][4], const int mb,
                        const Real mb_par[9], const int ncell[3],
                        const Real dt, const DvceArray5D<Real> &adm_n,
                        Real x1[3], Real e1[4][4], Real Uhat1[4]) {
  Real x_mid[3] = {0.0}, e_mid[4][4] = {0.0};
  for (int i = 0; i < 3; ++i) {
    x_mid[i] = 0.5 * (x0[i] + x_guess[i]);
  }
  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      e_mid[a][mu] = 0.5 * (e0[a][mu] + e_guess[a][mu]);
    }
  }

  Real alp = 0.0, beta[3] = {0.0}, g3d[6] = {0.0};
  Real gcov[4][4] = {0.0}, gcon[4][4] = {0.0}, Gamma[4][4][4] = {0.0};
  GetStationaryADMAndChristoffel<NG>(alp, beta, g3d, gcov, gcon, Gamma,
                                     x_mid, mb, mb_par, ncell, adm_n);
  OrthonormalizeTetrad(e_mid, gcov);

  Real Uhat_in[3] = {Uhat0[1], Uhat0[2], Uhat0[3]};
  Real Ehat[3] = {0.0}, Bhat[3] = {0.0}, Uhat_out[3] = {0.0};
  Real U_mid_pre[4] = {0.0};
  FourVelocityFromFrameComponents(e_mid, Uhat0, U_mid_pre);
  const Real dt_hat = dt * Uhat0[0] / U_mid_pre[0];
  FlatBorisPush(Uhat_out, Uhat_in, Ehat, Bhat, 1.0, dt_hat);

  Uhat1[0] = std::sqrt(1.0 + Uhat_out[0] * Uhat_out[0]
                            + Uhat_out[1] * Uhat_out[1]
                            + Uhat_out[2] * Uhat_out[2]);
  for (int i = 0; i < 3; ++i) {
    Uhat1[i + 1] = Uhat_out[i];
  }

  Real Uhat_mid[4] = {0.0};
  for (int a = 0; a < 4; ++a) {
    Uhat_mid[a] = 0.5 * (Uhat0[a] + Uhat1[a]);
  }
  Real U_mid[4] = {0.0};
  FourVelocityFromFrameComponents(e_mid, Uhat_mid, U_mid);
  const Real iUt = 1.0 / U_mid[0];

  for (int i = 0; i < 3; ++i) {
    x1[i] = x0[i] + dt * U_mid[i + 1] * iUt;
  }

  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      Real dedt = 0.0;
      for (int nu = 0; nu < 4; ++nu) {
        for (int lam = 0; lam < 4; ++lam) {
          dedt -= Gamma[mu][nu][lam] * e_mid[a][nu] * U_mid[lam] * iUt;
        }
      }
      e1[a][mu] = e0[a][mu] + dt * dedt;
    }
  }
}

template<int NG>
KOKKOS_INLINE_FUNCTION
bool GeoBorisFWBorisFixedPoint(const Real x0[3], const Real e0[4][4],
                               const Real Uhat0[4], const int mb,
                               const Real mb_par[9], const int ncell[3],
                               const Real dt, const DvceArray5D<Real> &adm_n,
                               Real x1[3], Real e1[4][4], Real Uhat1[4],
                               Real tol=1e-10, int maxIter=40) {
  Real x_guess[3] = {0.0}, e_guess[4][4] = {0.0};
  GeoBorisFWAdvance<NG>(x0, e0, Uhat0, mb, mb_par, ncell, dt, adm_n,
                        x_guess, e_guess);

  Real x_next[3] = {0.0}, e_next[4][4] = {0.0}, Uhat_next[4] = {0.0};
  for (int iter = 0; iter < maxIter; ++iter) {
    GeoBorisFWBorisMap<NG>(x0, e0, Uhat0, x_guess, e_guess, mb, mb_par, ncell,
                           dt, adm_n, x_next, e_next, Uhat_next);

    bool bad = false;
    Real err = 0.0;
    for (int i = 0; i < 3; ++i) {
      if (!isfinite(x_next[i])) {
        bad = true;
      }
      err = fmax(err, fabs(x_next[i] - x_guess[i]));
    }
    for (int a = 0; a < 4; ++a) {
      if (!isfinite(Uhat_next[a])) {
        bad = true;
      }
      for (int mu = 0; mu < 4; ++mu) {
        if (!isfinite(e_next[a][mu])) {
          bad = true;
        }
        err = fmax(err, fabs(e_next[a][mu] - e_guess[a][mu]));
      }
    }
    if (bad) {
      break;
    }
    if (err < tol) {
      for (int i = 0; i < 3; ++i) {
        x1[i] = x_next[i];
      }
      for (int a = 0; a < 4; ++a) {
        Uhat1[a] = Uhat_next[a];
        for (int mu = 0; mu < 4; ++mu) {
          e1[a][mu] = e_next[a][mu];
        }
      }
      return true;
    }
    for (int i = 0; i < 3; ++i) {
      x_guess[i] = x_next[i];
    }
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        e_guess[a][mu] = e_next[a][mu];
      }
    }
  }

  GeoBorisFWBorisMap<NG>(x0, e0, Uhat0, x0, e0, mb, mb_par, ncell,
                         dt, adm_n, x1, e1, Uhat1);
  return false;
}

KOKKOS_INLINE_FUNCTION
void StoreCovMomentumAndHalfPosition(const Real x[3], const Real e[4][4],
                                     const Real Uhat[4], const Real gcov[4][4],
                                     const Real dt, Real u_l[3],
                                     Real x_half[3]) {
  Real U[4] = {0.0}, U_cov[4] = {0.0};
  FourVelocityFromFrameComponents(e, Uhat, U);
  LowerFourVelocity(gcov, U, U_cov);
  const Real iUt = 1.0 / U[0];
  for (int i = 0; i < 3; ++i) {
    u_l[i] = U_cov[i + 1];
    x_half[i] = x[i] + 0.5 * dt * U[i + 1] * iUt;
  }
}

KOKKOS_INLINE_FUNCTION
Real TetradOrthonormalityError(const Real e[4][4], const Real gcov[4][4]) {
  Real err = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      Real dot = 0.0;
      for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
          dot += gcov[mu][nu] * e[a][mu] * e[b][nu];
        }
      }
      const Real target = (a == b) ? ((a == 0) ? -1.0 : 1.0) : 0.0;
      err = fmax(err, fabs(dot - target));
    }
  }
  return err;
}

void Particles::Geo_BorisFWPush() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  Real dt_ = pmy_pack->pmesh->dt;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &adm_n = pmy_pack->padm->u_adm;

  if (std::abs(q_over_m) > 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "geo_boris_fw is geodesic-only in this experimental version"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pz4c != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "geo_boris_fw currently supports stationary ADM metrics only"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  par_for("geo_boris_fw_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    Real x0[3] = {pr(IPLX, p), pr(IPLY, p), pr(IPLZ, p)};
    Real u0_l[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};
    Real e0[4][4] = {0.0};
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        e0[a][mu] = pr(IPFW + 4 * a + mu, p);
      }
    }

    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};

    Real alp0 = 0.0, beta0[3] = {0.0}, g3d0[6] = {0.0};
    Real gcov0[4][4] = {0.0}, gcon0[4][4] = {0.0}, Gamma0[4][4][4] = {0.0};
    switch (ng) {
    case 2:
      GetStationaryADMAndChristoffel<2>(alp0, beta0, g3d0, gcov0, gcon0, Gamma0,
                                        x0, mb, mb_par, ncell, adm_n);
      break;
    case 3:
      GetStationaryADMAndChristoffel<3>(alp0, beta0, g3d0, gcov0, gcon0, Gamma0,
                                        x0, mb, mb_par, ncell, adm_n);
      break;
    case 4:
      GetStationaryADMAndChristoffel<4>(alp0, beta0, g3d0, gcov0, gcon0, Gamma0,
                                        x0, mb, mb_par, ncell, adm_n);
      break;
    }

    Real U0[4] = {0.0}, Uhat[4] = {0.0};
    FourVelocityFromCovMomentum(alp0, beta0, g3d0, u0_l, U0);
    FrameComponentsFromFourVelocity(e0, gcov0, U0, Uhat);

    Real x1[3] = {0.0}, e1[4][4] = {0.0};
    switch (ng) {
    case 2:
      GeoBorisFWAdvance<2>(x0, e0, Uhat, mb, mb_par, ncell, dt_, adm_n, x1, e1);
      break;
    case 3:
      GeoBorisFWAdvance<3>(x0, e0, Uhat, mb, mb_par, ncell, dt_, adm_n, x1, e1);
      break;
    case 4:
      GeoBorisFWAdvance<4>(x0, e0, Uhat, mb, mb_par, ncell, dt_, adm_n, x1, e1);
      break;
    }

    Real alp1 = 0.0, beta1[3] = {0.0}, g3d1[6] = {0.0};
    Real gcov1[4][4] = {0.0}, gcon1[4][4] = {0.0}, Gamma1[4][4][4] = {0.0};
    switch (ng) {
    case 2:
      GetStationaryADMAndChristoffel<2>(alp1, beta1, g3d1, gcov1, gcon1, Gamma1,
                                        x1, mb, mb_par, ncell, adm_n);
      break;
    case 3:
      GetStationaryADMAndChristoffel<3>(alp1, beta1, g3d1, gcov1, gcon1, Gamma1,
                                        x1, mb, mb_par, ncell, adm_n);
      break;
    case 4:
      GetStationaryADMAndChristoffel<4>(alp1, beta1, g3d1, gcov1, gcon1, Gamma1,
                                        x1, mb, mb_par, ncell, adm_n);
      break;
    }

    Real u1_l[3] = {0.0}, x_half[3] = {0.0};
    StoreCovMomentumAndHalfPosition(x1, e1, Uhat, gcov1, dt_, u1_l, x_half);

    bool bad = false;
    for (int i = 0; i < 3; ++i) {
      if (!isfinite(x1[i]) || !isfinite(x_half[i]) || !isfinite(u1_l[i])) {
        bad = true;
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        if (!isfinite(e1[a][mu])) {
          bad = true;
        }
      }
    }
    if (bad) {
      Kokkos::printf("geo_boris_fw transported-tetrad step failed; leaving particle unchanged.\n");
      return;
    }

    pr(IPLX, p) = x1[0];
    pr(IPLY, p) = x1[1];
    pr(IPLZ, p) = x1[2];
    pr(IPX, p) = x_half[0];
    pr(IPY, p) = x_half[1];
    pr(IPZ, p) = x_half[2];
    pr(IPVX, p) = u1_l[0];
    pr(IPVY, p) = u1_l[1];
    pr(IPVZ, p) = u1_l[2];
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        pr(IPFW + 4 * a + mu, p) = e1[a][mu];
      }
    }
  });

  Real max_ortho_err = 0.0;
  Kokkos::parallel_reduce("geo_boris_fw_ortho_error",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nprtcl_thispack),
  KOKKOS_LAMBDA(const int p, Real &lmax) {
    Real x[3] = {pr(IPLX, p), pr(IPLY, p), pr(IPLZ, p)};
    Real e[4][4] = {0.0};
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        e[a][mu] = pr(IPFW + 4 * a + mu, p);
      }
    }
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    Real alp = 0.0, beta[3] = {0.0}, g3d[6] = {0.0};
    Real gcov[4][4] = {0.0}, gcon[4][4] = {0.0}, Gamma[4][4][4] = {0.0};
    switch (ng) {
    case 2:
      GetStationaryADMAndChristoffel<2>(alp, beta, g3d, gcov, gcon, Gamma,
                                        x, mb, mb_par, ncell, adm_n);
      break;
    case 3:
      GetStationaryADMAndChristoffel<3>(alp, beta, g3d, gcov, gcon, Gamma,
                                        x, mb, mb_par, ncell, adm_n);
      break;
    case 4:
      GetStationaryADMAndChristoffel<4>(alp, beta, g3d, gcov, gcon, Gamma,
                                        x, mb, mb_par, ncell, adm_n);
      break;
    }
    lmax = fmax(lmax, TetradOrthonormalityError(e, gcov));
  }, Kokkos::Max<Real>(max_ortho_err));

  if (global_variable::my_rank == 0 && max_ortho_err > 1.0e-6) {
    std::cout << "geo_boris_fw max tetrad orthonormality error = "
              << max_ortho_err << std::endl;
  }
}

void Particles::Geo_BorisFWBorisPush() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  Real dt_ = pmy_pack->pmesh->dt;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &adm_n = pmy_pack->padm->u_adm;

  if (std::abs(q_over_m) > 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "geo_boris_fw_boris is geodesic-only in this experimental version"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pz4c != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "geo_boris_fw_boris currently supports stationary ADM metrics only"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  par_for("geo_boris_fw_boris_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    Real x0[3] = {pr(IPLX, p), pr(IPLY, p), pr(IPLZ, p)};
    Real u0_l[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};
    Real e0[4][4] = {0.0};
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        e0[a][mu] = pr(IPFW + 4 * a + mu, p);
      }
    }

    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};

    Real alp0 = 0.0, beta0[3] = {0.0}, g3d0[6] = {0.0};
    Real gcov0[4][4] = {0.0}, gcon0[4][4] = {0.0}, Gamma0[4][4][4] = {0.0};
    switch (ng) {
    case 2:
      GetStationaryADMAndChristoffel<2>(alp0, beta0, g3d0, gcov0, gcon0, Gamma0,
                                        x0, mb, mb_par, ncell, adm_n);
      break;
    case 3:
      GetStationaryADMAndChristoffel<3>(alp0, beta0, g3d0, gcov0, gcon0, Gamma0,
                                        x0, mb, mb_par, ncell, adm_n);
      break;
    case 4:
      GetStationaryADMAndChristoffel<4>(alp0, beta0, g3d0, gcov0, gcon0, Gamma0,
                                        x0, mb, mb_par, ncell, adm_n);
      break;
    }

    Real U0[4] = {0.0}, Uhat0[4] = {0.0};
    FourVelocityFromCovMomentum(alp0, beta0, g3d0, u0_l, U0);
    FrameComponentsFromFourVelocity(e0, gcov0, U0, Uhat0);

    Real x_start[3] = {x0[0], x0[1], x0[2]};
    Real e_start[4][4] = {0.0}, Uhat_start[4] = {0.0};
    for (int a = 0; a < 4; ++a) {
      Uhat_start[a] = Uhat0[a];
      for (int mu = 0; mu < 4; ++mu) {
        e_start[a][mu] = e0[a][mu];
      }
    }

    Real x1[3] = {0.0}, e1[4][4] = {0.0}, Uhat1[4] = {0.0};
    bool find_root = true;
    const Real sub_dt = 0.25 * dt_;
    for (int sub = 0; sub < 4; ++sub) {
      bool sub_root = false;
      switch (ng) {
      case 2:
        sub_root = GeoBorisFWBorisFixedPoint<2>(x_start, e_start, Uhat_start,
                                                mb, mb_par, ncell, sub_dt,
                                                adm_n, x1, e1, Uhat1);
        break;
      case 3:
        sub_root = GeoBorisFWBorisFixedPoint<3>(x_start, e_start, Uhat_start,
                                                mb, mb_par, ncell, sub_dt,
                                                adm_n, x1, e1, Uhat1);
        break;
      case 4:
        sub_root = GeoBorisFWBorisFixedPoint<4>(x_start, e_start, Uhat_start,
                                                mb, mb_par, ncell, sub_dt,
                                                adm_n, x1, e1, Uhat1);
        break;
      }
      find_root = find_root && sub_root;

      Real alp_sub = 0.0, beta_sub[3] = {0.0}, g3d_sub[6] = {0.0};
      Real gcov_sub[4][4] = {0.0}, gcon_sub[4][4] = {0.0};
      Real Gamma_sub[4][4][4] = {0.0};
      switch (ng) {
      case 2:
        GetStationaryADMAndChristoffel<2>(alp_sub, beta_sub, g3d_sub, gcov_sub,
                                          gcon_sub, Gamma_sub, x1, mb, mb_par,
                                          ncell, adm_n);
        break;
      case 3:
        GetStationaryADMAndChristoffel<3>(alp_sub, beta_sub, g3d_sub, gcov_sub,
                                          gcon_sub, Gamma_sub, x1, mb, mb_par,
                                          ncell, adm_n);
        break;
      case 4:
        GetStationaryADMAndChristoffel<4>(alp_sub, beta_sub, g3d_sub, gcov_sub,
                                          gcon_sub, Gamma_sub, x1, mb, mb_par,
                                          ncell, adm_n);
        break;
      }
      OrthonormalizeTetrad(e1, gcov_sub);

      for (int i = 0; i < 3; ++i) {
        x_start[i] = x1[i];
      }
      for (int a = 0; a < 4; ++a) {
        Uhat_start[a] = Uhat1[a];
        for (int mu = 0; mu < 4; ++mu) {
          e_start[a][mu] = e1[a][mu];
        }
      }
    }
    if (!find_root) {
      Kokkos::printf("Root finding of geo_boris_fw_boris pusher failed; using explicit centered substep fallback.\n");
    }

    Real alp1 = 0.0, beta1[3] = {0.0}, g3d1[6] = {0.0};
    Real gcov1[4][4] = {0.0}, gcon1[4][4] = {0.0}, Gamma1[4][4][4] = {0.0};
    switch (ng) {
    case 2:
      GetStationaryADMAndChristoffel<2>(alp1, beta1, g3d1, gcov1, gcon1, Gamma1,
                                        x1, mb, mb_par, ncell, adm_n);
      break;
    case 3:
      GetStationaryADMAndChristoffel<3>(alp1, beta1, g3d1, gcov1, gcon1, Gamma1,
                                        x1, mb, mb_par, ncell, adm_n);
      break;
    case 4:
      GetStationaryADMAndChristoffel<4>(alp1, beta1, g3d1, gcov1, gcon1, Gamma1,
                                        x1, mb, mb_par, ncell, adm_n);
      break;
    }

    Real u1_l[3] = {0.0}, x_half[3] = {0.0};
    OrthonormalizeTetrad(e1, gcov1);
    StoreCovMomentumAndHalfPosition(x1, e1, Uhat1, gcov1, dt_, u1_l, x_half);

    bool bad = false;
    for (int i = 0; i < 3; ++i) {
      if (!isfinite(x1[i]) || !isfinite(x_half[i]) || !isfinite(u1_l[i])) {
        bad = true;
      }
    }
    for (int a = 0; a < 4; ++a) {
      if (!isfinite(Uhat1[a])) {
        bad = true;
      }
      for (int mu = 0; mu < 4; ++mu) {
        if (!isfinite(e1[a][mu])) {
          bad = true;
        }
      }
    }
    if (bad) {
      Kokkos::printf("geo_boris_fw_boris transported-frame step failed; leaving particle unchanged.\n");
      return;
    }

    pr(IPLX, p) = x1[0];
    pr(IPLY, p) = x1[1];
    pr(IPLZ, p) = x1[2];
    pr(IPX, p) = x_half[0];
    pr(IPY, p) = x_half[1];
    pr(IPZ, p) = x_half[2];
    pr(IPVX, p) = u1_l[0];
    pr(IPVY, p) = u1_l[1];
    pr(IPVZ, p) = u1_l[2];
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        pr(IPFW + 4 * a + mu, p) = e1[a][mu];
      }
    }
  });

  Real max_ortho_err = 0.0;
  Kokkos::parallel_reduce("geo_boris_fw_boris_ortho_error",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nprtcl_thispack),
  KOKKOS_LAMBDA(const int p, Real &lmax) {
    Real x[3] = {pr(IPLX, p), pr(IPLY, p), pr(IPLZ, p)};
    Real e[4][4] = {0.0};
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        e[a][mu] = pr(IPFW + 4 * a + mu, p);
      }
    }
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    Real alp = 0.0, beta[3] = {0.0}, g3d[6] = {0.0};
    Real gcov[4][4] = {0.0}, gcon[4][4] = {0.0}, Gamma[4][4][4] = {0.0};
    switch (ng) {
    case 2:
      GetStationaryADMAndChristoffel<2>(alp, beta, g3d, gcov, gcon, Gamma,
                                        x, mb, mb_par, ncell, adm_n);
      break;
    case 3:
      GetStationaryADMAndChristoffel<3>(alp, beta, g3d, gcov, gcon, Gamma,
                                        x, mb, mb_par, ncell, adm_n);
      break;
    case 4:
      GetStationaryADMAndChristoffel<4>(alp, beta, g3d, gcov, gcon, Gamma,
                                        x, mb, mb_par, ncell, adm_n);
      break;
    }
    lmax = fmax(lmax, TetradOrthonormalityError(e, gcov));
  }, Kokkos::Max<Real>(max_ortho_err));

  if (global_variable::my_rank == 0 && max_ortho_err > 1.0e-6) {
    std::cout << "geo_boris_fw_boris max tetrad orthonormality error = "
              << max_ortho_err << std::endl;
  }
}

void Particles::Geo_BorisInitPush() {
  // Stagger the particle position and velocity if using geo_boris pusher
  auto &pi_ = prtcl_idata;
  auto &pr_ = prtcl_rdata;
  int &npart = nprtcl_thispack;
  int &pgid = pmy_pack->gids;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  auto &adm_n = pmy_pack->padm->u_adm;
  Real dt_ = pmy_pack->pmesh->dt;

  // Read lapse/shift from Z4c when active; otherwise from ADM.
  const bool use_z4c = (pmy_pack->pz4c != nullptr);
  DvceArray5D<Real> z4c_n;
  if (use_z4c) {
    z4c_n = pmy_pack->pz4c->u0;
  }

  par_for("geo_boris_init_push", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(int n) {
    int mb = pi_(PGID, n) - pgid;
    Real x_mid[3] = {pr_(IPX, n), pr_(IPY, n), pr_(IPZ, n)};
    Real u_l[3] = {pr_(IPVX, n), pr_(IPVY, n), pr_(IPVZ, n)};
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};

    Real alp, beta[3], g3d[6];
    switch (ng) {
    case 2: {
      GetADMVariables<2>(alp, beta, g3d, x_mid, mb, mb_par, ncell, adm_n, use_z4c, z4c_n);
      break;
    }
    case 3: {
      GetADMVariables<3>(alp, beta, g3d, x_mid, mb, mb_par, ncell, adm_n, use_z4c, z4c_n);
      break;
    }
    case 4: {
      GetADMVariables<4>(alp, beta, g3d, x_mid, mb, mb_par, ncell, adm_n, use_z4c, z4c_n);
      break;
    }}

    Real det = Primitive::GetDeterminant(g3d);
    Real g3u[6] = {0.0};
    Primitive::InvertMatrix(g3u, g3d, det);
    Real u_u[3] = {0.0};
    Primitive::RaiseForm(u_u, u_l, g3u);
    Real Lorentz = std::sqrt(1.0 + Primitive::Contract(u_u, u_l));
    Real iLorentz = 1. / Lorentz;
    Real v[3] = {0.0};
    for (int i = 0; i < 3 ; ++i) {
      v[i] = alp * iLorentz * u_u[i] - beta[i];
    }
    for (int i = 0; i < 3; ++i) {
      pr_(IPX + i, n) = x_mid[i] + 0.5 * dt_ * v[i];
      pr_(IPLX + i, n) = x_mid[i];
    }
  });

}

void Particles::Geo_BorisFWInitPush() {
  auto &pi_ = prtcl_idata;
  auto &pr_ = prtcl_rdata;
  int &npart = nprtcl_thispack;
  int &pgid = pmy_pack->gids;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  auto &adm_n = pmy_pack->padm->u_adm;
  Real dt_ = pmy_pack->pmesh->dt;

  if (std::abs(q_over_m) > 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "geo_boris_fw is geodesic-only in this experimental version"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pz4c != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "geo_boris_fw currently supports stationary ADM metrics only"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  par_for("geo_boris_fw_init_push", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(int n) {
    int mb = pi_(PGID, n) - pgid;
    Real x_int[3] = {pr_(IPX, n), pr_(IPY, n), pr_(IPZ, n)};
    Real u_l[3] = {pr_(IPVX, n), pr_(IPVY, n), pr_(IPVZ, n)};
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};

    Real alp = 0.0, beta[3] = {0.0}, g3d[6] = {0.0};
    switch (ng) {
    case 2:
      GetADMVariables<2>(alp, beta, g3d, x_int, mb, mb_par, ncell, adm_n, false, adm_n);
      break;
    case 3:
      GetADMVariables<3>(alp, beta, g3d, x_int, mb, mb_par, ncell, adm_n, false, adm_n);
      break;
    case 4:
      GetADMVariables<4>(alp, beta, g3d, x_int, mb, mb_par, ncell, adm_n, false, adm_n);
      break;
    }

    Real tetrad[4][4] = {0.0}, inv_tetrad[4][4] = {0.0};
    CalcTetrad(alp, beta, g3d, tetrad, inv_tetrad);
    for (int a = 0; a < 4; ++a) {
      for (int mu = 0; mu < 4; ++mu) {
        pr_(IPFW + 4 * a + mu, n) = inv_tetrad[mu][a];
      }
    }

    Real U[4] = {0.0};
    FourVelocityFromCovMomentum(alp, beta, g3d, u_l, U);
    const Real iUt = 1.0 / U[0];
    for (int i = 0; i < 3; ++i) {
      pr_(IPLX + i, n) = x_int[i];
      pr_(IPX + i, n) = x_int[i] + 0.5 * dt_ * U[i + 1] * iUt;
    }
  });
}
} // end namespace particles
