//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boris_pusher.cpp
//  \brief New Boris pusher function, which uses routines outlined in Zou+2024

#include <cmath>
#include <functional>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "lagrange_interp.hpp"
#include "calc_tetrad.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {

KOKKOS_INLINE_FUNCTION
void FlatPush(Real uhat_pushed[3], const Real uhat[3], const Real Ehat[3],
              const Real Bhat[3], const Real qom, const Real dt) {
  // First half electric field acceleration
  Real u_minus[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    u_minus[i] = uhat[i] + 0.5 * qom * dt * Ehat[i];
  }
  // Rotation step
  Real gamma_minus = std::sqrt(1.0 + u_minus[0] * u_minus[0] + u_minus[1] * u_minus[1] + u_minus[2] * u_minus[2]);
  Real t[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    t[i] = 0.5 * qom * dt / gamma_minus * Bhat[i];
  }
  Real tsqr = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
  Real s[3] = {0.0};
  for (int i = 0; i < 3; ++i) {
    s[i] = 2. / (1. + tsqr) * t[i];
  }
  Real s_dot_t = s[0] * t[0] + s[1] * t[1] + s[2] * t[2];
  Real s_dot_u_minus = s[0] * u_minus[0] + s[1] * u_minus[1] + s[2] * u_minus[2];
  Real u_plus[3] = {0.0};
  u_plus[0] = u_minus[0] + u_minus[1] * s[2] - u_minus[2] * s[1] - s_dot_t * u_minus[0] + s_dot_u_minus * t[0];
  u_plus[1] = u_minus[1] + u_minus[2] * s[0] - u_minus[0] * s[2] - s_dot_t * u_minus[1] + s_dot_u_minus * t[1];
  u_plus[2] = u_minus[2] + u_minus[0] * s[1] - u_minus[1] * s[0] - s_dot_t * u_minus[2] + s_dot_u_minus * t[2];
  // Second half electric field acceleration
  for (int i = 0; i < 3; ++i) {
    uhat_pushed[i] = u_plus[i] + 0.5 * dt * qom * Ehat[i];
  }
}

template <int NG>
struct GeodesicPush {
  const Real *x_old, *u_old, *mb_par;
  const int *ncell;
  const DvceArray5D<Real> adm_old, adm_new, z4c_old, z4c_new;
  const int mb;
  const Real dt;
  const bool use_z4c;

  KOKKOS_INLINE_FUNCTION
  GeodesicPush(const Real x_[3], const Real u_[3], const int mb_, const Real mb_par_[9], const int ncell_[3],
               const Real dt_,
               const DvceArray5D<Real>& adm_old_, const DvceArray5D<Real>& adm_new_,
               const bool use_z4c_,
               const DvceArray5D<Real>& z4c_old_, const DvceArray5D<Real>& z4c_new_)
    : x_old(x_), u_old(u_), mb_par(mb_par_), ncell(ncell_),
      adm_old(adm_old_), adm_new(adm_new_), z4c_old(z4c_old_), z4c_new(z4c_new_),
      mb(mb_), dt(dt_), use_z4c(use_z4c_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Real xin[3], const Real uin[3], Real xout[3], Real uout[3], bool Euler) const {
    Real x_mid[3] = {0.0}, u_mid[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      x_mid[i] = 0.5 * (xin[i] + x_old[i]); // x_mid = x_old for Euler step
      u_mid[i] = 0.5 * (uin[i] + u_old[i]); // u_mid = u_old for Euler step
    }
    int interp_indcs[4] = {mb, -1, -1, -1};
    interp_indcs[1] = static_cast<int>(std::floor((x_mid[0] - (mb_par[0] + mb_par[2] / 2.0)) / mb_par[2]));
    interp_indcs[2] = static_cast<int>(std::floor((x_mid[1] - (mb_par[3] + mb_par[5] / 2.0)) / mb_par[5]));
    interp_indcs[3] = static_cast<int>(std::floor((x_mid[2] - (mb_par[6] + mb_par[8] / 2.0)) / mb_par[8]));
    Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};
    Real dLx[8] = {0.0}, dLy[8] = {0.0}, dLz[8] = {0.0};
    CalcInterpWghtAndDrv<NG>(x_mid, mb_par, ncell, interp_indcs, Lx, Ly, Lz, dLx, dLy, dLz);

    // Step 1: Update position
    // (i) Interpolate adm variables at t=n+1/2, x=x_mid
    Real alp_old = 0.0, alp_new = 0.0, alp = 0.0;
    Real beta_old[3] = {0.0}, beta_new[3] = {0.0}, beta[3] = {0.0};
    Real g3d_old[6] = {0.0}, g3d_new[6] = {0.0}, g3d[6] = {0.0};
    if (use_z4c) {
      alp_old = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      alp_new = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_old[i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, Lx, Ly, Lz);
        beta_new[i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, Lx, Ly, Lz);
      }
    } else {
      alp_old = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
      alp_new = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
      for (int i = 0; i < 3; ++i) {
        beta_old[i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i, interp_indcs, Lx, Ly, Lz);
        beta_new[i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i, interp_indcs, Lx, Ly, Lz);
      }
    }
    alp = 0.5 * (alp_old + alp_new);
    for (int i = 0; i < 3; ++i) {
      beta[i] = 0.5 * (beta_old[i] + beta_new[i]);
    }
    for (int i = 0; i < 6; ++i) {
      g3d_old[i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX+i, interp_indcs, Lx, Ly, Lz);
      g3d_new[i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX+i, interp_indcs, Lx, Ly, Lz);
      g3d[i] = 0.5 * (g3d_old[i] + g3d_new[i]);
    }
    // (ii) calculate transport velocity
    if (Euler) {
      alp = alp_old;
      for (int i = 0; i < 3; ++i) {
        beta[i] = beta_old[i];
      }
      for (int i = 0; i < 6; ++i) {
        g3d[i] = g3d_old[i];
      }
    }
    Real g3u[6] = {0.0};
    Real det = Primitive::GetDeterminant(g3d);
    Primitive::InvertMatrix(g3u, g3d, det);
    Real u_mid_u[3] = {0.0};
    Primitive::RaiseForm(u_mid_u, u_mid, g3u);
    Real Lorentz = std::sqrt(1.0 + Primitive::Contract(u_mid_u, u_mid));
    Real v[3] = {0.0};
    for (int i = 0; i < 3; ++i) {
      v[i] = alp * u_mid_u[i] / Lorentz - beta[i];
    }
    // (iii) update position
    for (int i = 0; i < 3; ++i) {
      xout[i] = x_old[i] + dt * v[i];
    }

    // Step 2: Update velocity
    // (i) Interpolate the derivatives of adm variables at t=n+1/2, x=x_mid
    Real dalp_old[3] = {0.0}, dalp_new[3] = {0.0}, dalp[3] = {0.0};
    Real dbeta_old[3][3] = {0.0}, dbeta_new[3][3] = {0.0}, dbeta[3][3] = {0.0};
    Real dg3u_old[3][6] = {0.0}, dg3u_new[3][6] = {0.0}, dg3u[3][6] = {0.0};
    if (use_z4c) {
      dalp_old[0] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, dLx, Ly, Lz);
      dalp_old[1] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, dLy, Lz);
      dalp_old[2] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, dLz);
      dalp_new[0] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, dLx, Ly, Lz);
      dalp_new[1] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, dLy, Lz);
      dalp_new[2] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, dLz);
      for (int i = 0; i < 3; ++i) {
        dbeta_old[0][i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, dLx, Ly, Lz);
        dbeta_old[1][i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, Lx, dLy, Lz);
        dbeta_old[2][i] = LagrangeInterpolator<NG>(z4c_old, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, Lx, Ly, dLz);
        dbeta_new[0][i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, dLx, Ly, Lz);
        dbeta_new[1][i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, Lx, dLy, Lz);
        dbeta_new[2][i] = LagrangeInterpolator<NG>(z4c_new, z4c::Z4c::I_Z4C_BETAX+i, interp_indcs, Lx, Ly, dLz);
      }
    } else {
      dalp_old[0] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA, interp_indcs, dLx, Ly, Lz);
      dalp_old[1] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, dLy, Lz);
      dalp_old[2] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, dLz);
      dalp_new[0] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA, interp_indcs, dLx, Ly, Lz);
      dalp_new[1] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, dLy, Lz);
      dalp_new[2] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, dLz);
      for (int i = 0; i < 3; ++i) {
        dbeta_old[0][i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i, interp_indcs, dLx, Ly, Lz);
        dbeta_old[1][i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i, interp_indcs, Lx, dLy, Lz);
        dbeta_old[2][i] = LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_BETAX+i, interp_indcs, Lx, Ly, dLz);
        dbeta_new[0][i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i, interp_indcs, dLx, Ly, Lz);
        dbeta_new[1][i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i, interp_indcs, Lx, dLy, Lz);
        dbeta_new[2][i] = LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_BETAX+i, interp_indcs, Lx, Ly, dLz);
      }
    }
    for (int i = 0; i < 3; ++i) {
      dalp[i] = 0.5 * (dalp_old[i] + dalp_new[i]);
      for (int j = 0; j < 3; ++j) {
        dbeta[i][j] = 0.5 * (dbeta_old[i][j] + dbeta_new[i][j]);
      }
    }
    LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX, interp_indcs, dLx, Ly, Lz, dg3u_old[0]);
    LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX, interp_indcs, Lx, dLy, Lz, dg3u_old[1]);
    LagrangeInterpolator<NG>(adm_old, adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, dLz, dg3u_old[2]);
    LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX, interp_indcs, dLx, Ly, Lz, dg3u_new[0]);
    LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX, interp_indcs, Lx, dLy, Lz, dg3u_new[1]);
    LagrangeInterpolator<NG>(adm_new, adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, dLz, dg3u_new[2]);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        dg3u[i][j] = 0.5 * (dg3u_old[i][j] + dg3u_new[i][j]);
      }
    }
    // (ii) Calculate geodesic force
    Real g[3] = {0.0};
    if (Euler) { // alp already changed
      for (int i = 0; i < 3; ++i) {
        dalp[i] = dalp_old[i];
        for (int j = 0; j < 3; ++j) {
          dbeta[i][j] = dbeta_old[i][j];
        }
        for (int j = 0; j < 6; ++j) {
          dg3u[i][j] = dg3u_old[i][j];
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      g[i] = -1. * Lorentz * dalp[i] + u_mid[0] * dbeta[i][0] + u_mid[1] * dbeta[i][1] + u_mid[2] * dbeta[i][2] -
             0.5 * alp / Lorentz * (u_mid[0] * u_mid[0] * dg3u[i][0] + u_mid[1] * u_mid[1] * dg3u[i][3] +
                                    u_mid[2] * u_mid[2] * dg3u[i][5] + 2. * u_mid[0] * u_mid[1] * dg3u[i][1] +
                                    2. * u_mid[0] * u_mid[2] * dg3u[i][2] + 2. * u_mid[1] * u_mid[2] * dg3u[i][4]);
    }
    // (iii) update velocity
    for (int i = 0; i < 3; ++i) {
      uout[i] = u_old[i] + dt * g[i];
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
  Real err = 0.0;
  for (int iter = 0; iter < maxIter; ++iter) {
    f(x_new, u_new, x_next, u_next, false);
    bool to_break = false;
    if (!isfinite(x_next[0]) || !isfinite(x_next[1]) || !isfinite(x_next[2])) {
      to_break = true;
    }
    if (!isfinite(u_next[0]) || !isfinite(u_next[1]) || !isfinite(u_next[2])) {
      to_break = true;
    }
    if (to_break) {
      break;
    }
    err = 0.0;
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
  f(x0, u0, x, u, true);
  return false;
}

void Particles::GR_BorisPush() {
  // Extract MHD variables
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ng = indcs.ng;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  auto dt_ = pmy_pack->pmesh->dt;
  auto qom = q_over_m;

  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &w0_n = w0_last;
  auto &bcc0_n = bcc0_last;
  auto &adm_n = adm_last;

  DvceArray5D<Real> w0_np1, bcc0_np1;
  bool use_mhd = false;
  if (pmy_pack->pmhd != nullptr) {
    use_mhd = true;
    w0_np1 = pmy_pack->pmhd->w0;
    bcc0_np1 = pmy_pack->pmhd->bcc0;
  }

  auto &adm_np1 = pmy_pack->padm->u_adm;

  DvceArray5D<Real> z4c_n, z4c_np1;
  bool use_z4c = false;
  if (pmy_pack->pz4c != nullptr) {
    use_z4c = true;
    z4c_n = z4c_last;
    z4c_np1 = pmy_pack->pz4c->u0;
  }
  // Loop over all particles
  par_for("gr_boris_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    // Extract interpolation info
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};
    int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};

    // Retrieve particle position and velocity
    Real x_n[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    Real u_n[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};
    // Initialize interpolation array
    Real B_interp_n[3] = {0.0}, v_interp_n[3] = {0.0};
    Real alp_n = 0.0, beta_n[3] = {0.0}, g3d_n[6] = {0.0};

    // Step 1: Interpolate field variables at t=n and x=x^n
    Real u_p[3] = {0.0};
    if (use_mhd) {
      // (i) Interpolate B field and fluid velocity
      int interp_indcs[4] = {mb, -1, -1, -1};
      interp_indcs[1] = static_cast<int>(std::floor((x_n[0] - (mb_par[0] + mb_par[2] / 2.0)) / mb_par[2]));
      interp_indcs[2] = static_cast<int>(std::floor((x_n[1] - (mb_par[3] + mb_par[5] / 2.0)) / mb_par[5]));
      interp_indcs[3] = static_cast<int>(std::floor((x_n[2] - (mb_par[6] + mb_par[8] / 2.0)) / mb_par[8]));
      Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};
      switch (ng) {
      case 2: {
        CalcInterpWght<2>(x_n, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_n[idx] = LagrangeInterpolator<2>(bcc0_n, idx, interp_indcs, Lx, Ly, Lz);
          v_interp_n[idx] = LagrangeInterpolator<2>(w0_n, idx+IVX, interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_n = LagrangeInterpolator<2>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<2>(z4c_n, idx+z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_n = LagrangeInterpolator<2>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<2>(adm_n, idx+adm::ADM::I_ADM_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_n[idx] = LagrangeInterpolator<2>(adm_n, idx+adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 3: {
        CalcInterpWght<3>(x_n, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_n[idx] = LagrangeInterpolator<3>(bcc0_n, idx, interp_indcs, Lx, Ly, Lz);
          v_interp_n[idx] = LagrangeInterpolator<3>(w0_n, idx+IVX, interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_n = LagrangeInterpolator<3>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<3>(z4c_n, idx+z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_n = LagrangeInterpolator<3>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<3>(adm_n, idx+adm::ADM::I_ADM_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_n[idx] = LagrangeInterpolator<3>(adm_n, idx+adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 4: {
        CalcInterpWght<4>(x_n, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_n[idx] = LagrangeInterpolator<4>(bcc0_n, idx, interp_indcs, Lx, Ly, Lz);
          v_interp_n[idx] = LagrangeInterpolator<4>(w0_n, idx+IVX, interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_n = LagrangeInterpolator<4>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<4>(z4c_n, idx+z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_n = LagrangeInterpolator<4>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_n[idx] = LagrangeInterpolator<4>(adm_n, idx+adm::ADM::I_ADM_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_n[idx] = LagrangeInterpolator<4>(adm_n, idx+adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, Lz);
        }
        break;
      }}
      // (ii) calculate E field assuming ideal MHD
      Real det = Primitive::GetDeterminant(g3d_n);
      Real sqrtdet = std::sqrt(det);
      Real E_interp_n[3] = {0.0};
      E_interp_n[0] = sqrtdet * (B_interp_n[1] * v_interp_n[2] - B_interp_n[2] * v_interp_n[1]);
      E_interp_n[1] = sqrtdet * (B_interp_n[2] * v_interp_n[0] - B_interp_n[0] * v_interp_n[2]);
      E_interp_n[2] = sqrtdet * (B_interp_n[0] * v_interp_n[1] - B_interp_n[1] * v_interp_n[0]);

      // Step 2: Perform the first flat Boris push
      // (i) Convert to local tetrad basis at x=x^n
      Real tetrad[4][4] = {0.0}, inv_tetrad[4][4] = {0.0};
      CalcTetrad(alp_n, beta_n, g3d_n, tetrad, inv_tetrad);
      Real uhat_n[3] = {0.0};
      Real Ehat_interp_n[3] = {0.0}, Bhat_interp_n[3] = {0.0};
      TetradCvrtL(uhat_n, u_n, inv_tetrad);
      TetradCvrtL(Ehat_interp_n, E_interp_n, inv_tetrad);
      TetradCvrtU(Bhat_interp_n, B_interp_n, tetrad);
      // (ii) Push the particle
      Real uhat_p[3] = {0.0};
      FlatPush(uhat_p, uhat_n, Ehat_interp_n, Bhat_interp_n, qom, 0.5 * dt_);
      // (iii) Convert back to coordinate basis at x=x^n
      TetradCvrtL(u_p, uhat_p, tetrad);
    } else {
      // If MHD variables are not available, just do a geodesic push
      for (int i = 0; i < 3; ++i) {
        u_p[i] = u_n[i];
      }
    }

    // Step 3: Implicitly solve the geodesic motion
    Real x_np1[3] = {0.0}, u_pp[3] = {0.0};
    bool find_root = false;
    switch (ng) {
    case 2: {
      GeodesicPush<2> geodesicpush(x_n, u_p, mb, mb_par, ncell, dt_,
                                   adm_n, adm_np1, use_z4c, z4c_n, z4c_np1);
      find_root = FixedPointIteration(geodesicpush, x_n, u_p, x_np1, u_pp);
      break;
    }
    case 3: {
      GeodesicPush<3> geodesicpush(x_n, u_p, mb, mb_par, ncell, dt_,
                                   adm_n, adm_np1, use_z4c, z4c_n, z4c_np1);
      find_root = FixedPointIteration(geodesicpush, x_n, u_p, x_np1, u_pp);
      break;
    }
    case 4: {
      GeodesicPush<4> geodesicpush(x_n, u_p, mb, mb_par, ncell, dt_,
                                   adm_n, adm_np1, use_z4c, z4c_n, z4c_np1);
      find_root = FixedPointIteration(geodesicpush, x_n, u_p, x_np1, u_pp);
      break;
    }}
    if (!find_root) {
      Kokkos::printf("Root finding failed, forward Euler used.\n");
    }

    // Step 4: Interpolate field variables at t=n+1 and x=x^{n+1}
    Real u_np1[3] = {0.0};
    if (use_mhd) {
      int interp_indcs[4] = {mb, -1, -1, -1};
      // (i) Interpolate B field and fluid velocity
      interp_indcs[1] = static_cast<int>(std::floor((x_np1[0] - (mb_par[0] + mb_par[2] / 2.0)) / mb_par[2]));
      interp_indcs[2] = static_cast<int>(std::floor((x_np1[1] - (mb_par[3] + mb_par[5] / 2.0)) / mb_par[5]));
      interp_indcs[3] = static_cast<int>(std::floor((x_np1[2] - (mb_par[6] + mb_par[8] / 2.0)) / mb_par[8]));
      // Initialize interpolation array
      Real B_interp_np1[3] = {0.0}, v_interp_np1[3] = {0.0};
      Real alp_np1 = 0.0, beta_np1[3] = {0.0}, g3d_np1[6] = {0.0};
      Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};
      switch (ng) {
      case 2: {
        CalcInterpWght<2>(x_np1, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_np1[idx] = LagrangeInterpolator<2>(bcc0_np1, idx, interp_indcs, Lx, Ly, Lz);
          v_interp_np1[idx] = LagrangeInterpolator<2>(w0_np1, idx+IVX, interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_np1 = LagrangeInterpolator<2>(z4c_np1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<2>(z4c_np1, idx+z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_np1 = LagrangeInterpolator<2>(adm_np1, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<2>(adm_np1, idx+adm::ADM::I_ADM_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_np1[idx] = LagrangeInterpolator<2>(adm_np1, idx+adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 3: {
        CalcInterpWght<3>(x_np1, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_np1[idx] = LagrangeInterpolator<3>(bcc0_np1, idx, interp_indcs, Lx, Ly, Lz);
          v_interp_np1[idx] = LagrangeInterpolator<3>(w0_np1, idx+IVX, interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_np1 = LagrangeInterpolator<3>(z4c_np1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<3>(z4c_np1, idx+z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_np1 = LagrangeInterpolator<3>(adm_np1, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<3>(adm_np1, idx+adm::ADM::I_ADM_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_np1[idx] = LagrangeInterpolator<3>(adm_np1, idx+adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, Lz);
        }
        break;
      }
      case 4: {
        CalcInterpWght<4>(x_np1, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
        for (int idx = 0; idx < 3; ++idx) {
          B_interp_np1[idx] = LagrangeInterpolator<4>(bcc0_np1, idx, interp_indcs, Lx, Ly, Lz);
          v_interp_np1[idx] = LagrangeInterpolator<4>(w0_np1, idx+IVX, interp_indcs, Lx, Ly, Lz);
        }
        if (use_z4c) {
          alp_np1 = LagrangeInterpolator<4>(z4c_np1, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<4>(z4c_np1, idx+z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        } else {
          alp_np1 = LagrangeInterpolator<4>(adm_np1, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
          for (int idx = 0; idx < 3; ++idx) {
            beta_np1[idx] = LagrangeInterpolator<4>(adm_np1, idx+adm::ADM::I_ADM_BETAX, interp_indcs, Lx, Ly, Lz);
          }
        }
        for (int idx = 0; idx < 6; ++idx) {
          g3d_np1[idx] = LagrangeInterpolator<4>(adm_np1, idx+adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, Lz);
        }
        break;
      }}
      // (iii) calculate E field assuming ideal MHD
      Real det = Primitive::GetDeterminant(g3d_np1);
      Real sqrtdet = std::sqrt(det);
      Real E_interp_np1[3] = {0.0};
      E_interp_np1[0] = sqrtdet * (B_interp_np1[1] * v_interp_np1[2] - B_interp_np1[2] * v_interp_np1[1]);
      E_interp_np1[1] = sqrtdet * (B_interp_np1[2] * v_interp_np1[0] - B_interp_np1[0] * v_interp_np1[2]);
      E_interp_np1[2] = sqrtdet * (B_interp_np1[0] * v_interp_np1[1] - B_interp_np1[1] * v_interp_np1[0]);

      // Step 5: Perform the second flat Boris push
      // (i) Convert to local tetrad basis at x=x^n+1
      Real tetrad[4][4] = {0.0}, inv_tetrad[4][4] = {0.0};
      CalcTetrad(alp_np1, beta_np1, g3d_np1, tetrad, inv_tetrad);
      Real uhat_pp[3] = {0.0};
      Real Ehat_interp_np1[3] = {0.0}, Bhat_interp_np1[3] = {0.0};
      TetradCvrtL(uhat_pp, u_pp, inv_tetrad);
      TetradCvrtL(Ehat_interp_np1, E_interp_np1, inv_tetrad);
      TetradCvrtU(Bhat_interp_np1, B_interp_np1, tetrad);
      // (ii) Push the particle
      Real uhat_np1[3] = {0.0};
      FlatPush(uhat_np1, uhat_pp, Ehat_interp_np1, Bhat_interp_np1, qom, 0.5 * dt_);
      // (iii) Convert back to coordinate basis at x=x^{n+1}
      TetradCvrtL(u_np1, uhat_np1, tetrad);
    } else {
      // If MHD variables are not available, just set u^{n+1} = u^{n+}
      for (int i = 0; i < 3; ++i) {
        u_np1[i] = u_pp[i];
      }
    }

    // Step 6: Update particle position and speed into device memory
    pr(IPX, p) = x_np1[0];
    pr(IPY, p) = x_np1[1];
    pr(IPZ, p) = x_np1[2];
    pr(IPVX, p) = u_np1[0];
    pr(IPVY, p) = u_np1[1];
    pr(IPVZ, p) = u_np1[2];
  });

  // Update primitive variables, magnetic field and adm variables
  if (use_mhd) {
    Kokkos::deep_copy(DevExeSpace(), w0_last, pmy_pack->pmhd->w0);
    Kokkos::deep_copy(DevExeSpace(), bcc0_last, pmy_pack->pmhd->bcc0);
  }

  Kokkos::deep_copy(DevExeSpace(), adm_last, pmy_pack->padm->u_adm);
  if (use_z4c) {
    Kokkos::deep_copy(DevExeSpace(), z4c_last, pmy_pack->pz4c->u0);
  }

  Kokkos::fence();
} // end BorisPush
} // end namespace particles