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
} // end namespace particles
