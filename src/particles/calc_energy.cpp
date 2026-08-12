//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file calc_energy.cpp
//! \brief compute the conserved specific energy E = -u_t for each particle. The ADM
//! metric (lapse alpha, shift beta^i, 3-metric gamma_ij) is interpolated to the particle
//! position and the timelike normalization g^{mu nu} u_mu u_nu = -1 is solved as a
//! quadratic in u_t. The stored spatial velocity slots IPVX/IPVY/IPVZ hold the COVARIANT
//! spatial 4-velocity u_i; the result -u_t (per unit mass) is written to IPEN.
//!
//! CAVEAT: E = -u_t is conserved only in a STATIONARY spacetime, where d_t is a Killing
//! vector. On a dynamical / non-stationary spacetime (e.g. Oppenheimer-Snyder collapse,
//! any live-Z4c run) -u_t is NOT conserved and is a diagnostic only; the energy
//! definition will likely need revision in a later stage.

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "particles.hpp"
#include "gr_monopole.hpp"
#include "lagrange_interp.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn template<int NGHOST> void Particles::calc_prtcl_energy()
//! \brief interpolate the metric at each particle and store -u_t into IPEN.

template <int NGHOST>
void Particles::calc_prtcl_energy() {
  // -u_t requires a metric; no-op if no ADM variables are present
  // (e.g. flat-space SR tests)
  if (pmy_pack->padm == nullptr) {return;}

  if (gr_boris_live_monopole) {
    if (nprtcl_thispack == 0) {return;}
    DvceArray5D<Real> adm_metric = pmy_pack->padm->u_adm;
    DvceArray5D<Real> z4c_metric;
    bool use_z4c_metric = (pmy_pack->pz4c != nullptr);
    if (use_z4c_metric) {z4c_metric = pmy_pack->pz4c->u0;}
    if (!gr_boris_monopole_profile_valid) {
      BuildGRBorisMonopoleProfiles(
          adm_metric, adm_metric, use_z4c_metric,
          z4c_metric, z4c_metric, true);
    }
    auto &pr_mono = prtcl_rdata;
    auto profile = gr_boris_monopole_profile_new;
    int nr = gr_boris_monopole_nr;
    Real dr = gr_boris_monopole_dr;
    Real c0 = gr_boris_monopole_center[0];
    Real c1 = gr_boris_monopole_center[1];
    Real c2 = gr_boris_monopole_center[2];
    par_for("calc_prtcl_energy_monopole", DevExeSpace(), 0, nprtcl_thispack-1,
    KOKKOS_LAMBDA(const int p) {
      Real xr[3] = {pr_mono(IPX,p)-c0, pr_mono(IPY,p)-c1, pr_mono(IPZ,p)-c2};
      Real r = sqrt(xr[0]*xr[0] + xr[1]*xr[1] + xr[2]*xr[2]);
      Real rsafe = (r > 1.0e-14) ? r : 1.0e-14;
      Real n[3] = {xr[0]/rsafe, xr[1]/rsafe, xr[2]/rsafe};
      Real u[3] = {pr_mono(IPVX,p), pr_mono(IPVY,p), pr_mono(IPVZ,p)};
      Real q = n[0]*u[0] + n[1]*u[1] + n[2]*u[2];
      Real usq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
      Real metric[N_GR_MONO_PROFILE];
      InterpolateGRMonopoleProfile(profile, nr, dr, r, metric);
      Real A = metric[MONO_GAMMA_R];
      Real B = metric[MONO_GAMMA_T];
      Real W = sqrt(1.0 + B*usq + (A-B)*q*q);
      pr_mono(IPEN,p) = metric[MONO_ALPHA]*W - metric[MONO_BETA_R]*q;
    });
    return;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  DvceArray5D<Real> adm_n = gr_boris_freeze_metric
      ? adm_last : pmy_pack->padm->u_adm;
  // lapse/shift source: when Z4c is live the ADM storage drops the alpha/beta slots
  // (u_adm holds nadm-4 variables) and the gauge lives in the Z4c state vector -- the
  // same source switch as the gr_boris pusher. Reading I_ADM_ALPHA from the trimmed
  // u_adm was out of bounds (latent Stage-1 bug, first reachable with z4c+particles).
  DvceArray5D<Real> z4c_u0;
  bool use_z4c = (pmy_pack->pz4c != nullptr);
  if (use_z4c) {
    z4c_u0 = gr_boris_freeze_metric ? z4c_last : pmy_pack->pz4c->u0;
  }

  par_for("calc_prtcl_energy", DevExeSpace(), 0, (nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    Real x[3]   = {pr(IPX,p),  pr(IPY,p),  pr(IPZ,p)};
    Real u_d[3] = {pr(IPVX,p), pr(IPVY,p),
                   pr(IPVZ,p)};   // covariant spatial 4-velocity u_i
    int m = pi(PGID,p) - gids;
    const Real mb_par[9] = {
      size.d_view(m).x1min, size.d_view(m).x1max, size.d_view(m).dx1,
      size.d_view(m).x2min, size.d_view(m).x2max, size.d_view(m).dx2,
      size.d_view(m).x3min, size.d_view(m).x3max, size.d_view(m).dx3};

    // stencil indices + Lagrange weights at the particle
    int interp_indcs[4] = {m, -1, -1, -1};
    SetInterpIndices(x, mb_par, ncell, interp_indcs);
    Real Lx[2*NGHOST] = {0.0}, Ly[2*NGHOST] = {0.0}, Lz[2*NGHOST] = {0.0};
    CalcInterpWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);

    // interpolate lapse + shift (from Z4c when live, else ADM) and the 3-metric (ADM)
    Real alp;
    Real beta[3];
    if (use_z4c) {
      alp     = LagrangeInterpolator<NGHOST>(z4c_u0, z4c::Z4c::I_Z4C_ALPHA,
                                             interp_indcs, Lx,Ly,Lz);
      beta[0] = LagrangeInterpolator<NGHOST>(z4c_u0, z4c::Z4c::I_Z4C_BETAX,
                                             interp_indcs, Lx,Ly,Lz);
      beta[1] = LagrangeInterpolator<NGHOST>(z4c_u0, z4c::Z4c::I_Z4C_BETAY,
                                             interp_indcs, Lx,Ly,Lz);
      beta[2] = LagrangeInterpolator<NGHOST>(z4c_u0, z4c::Z4c::I_Z4C_BETAZ,
                                             interp_indcs, Lx,Ly,Lz);
    } else {
      alp     = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA,
                                             interp_indcs, Lx,Ly,Lz);
      beta[0] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX,
                                             interp_indcs, Lx,Ly,Lz);
      beta[1] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAY,
                                             interp_indcs, Lx,Ly,Lz);
      beta[2] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAZ,
                                             interp_indcs, Lx,Ly,Lz);
    }
    Real g3d[6];
    g3d[0]    = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX,
                                             interp_indcs, Lx,Ly,Lz);
    g3d[1]    = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXY,
                                             interp_indcs, Lx,Ly,Lz);
    g3d[2]    = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXZ,
                                             interp_indcs, Lx,Ly,Lz);
    g3d[3]    = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GYY,
                                             interp_indcs, Lx,Ly,Lz);
    g3d[4]    = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GYZ,
                                             interp_indcs, Lx,Ly,Lz);
    g3d[5]    = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GZZ,
                                             interp_indcs, Lx,Ly,Lz);

    // inverse 3-metric gamma^{ij}
    Real g3u[6];
    Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));

    // Solve g^{mu nu} u_mu u_nu = -1 for u_t, with
    //   g^{tt} = -1/alpha^2,  g^{ti} = beta^i/alpha^2,
    //   g^{ij} = gamma^{ij} - beta^i beta^j/alpha^2.
    // => a*u_t^2 + 2*b*u_t + c = 0, using the stored covariant u_i.
    Real ialp2 = 1.0/(alp*alp);
    Real a = -ialp2;                                    // g^{tt}
    Real b = 0.0;
    for (int i=0; i<3; ++i) {b += beta[i]*ialp2*u_d[i];}   // g^{ti} u_i
    Real c = 1.0;                                       // g^{ij} u_i u_j + 1
    c += (g3u[0] - beta[0]*beta[0]*ialp2)*u_d[0]*u_d[0];
    c += (g3u[3] - beta[1]*beta[1]*ialp2)*u_d[1]*u_d[1];
    c += (g3u[5] - beta[2]*beta[2]*ialp2)*u_d[2]*u_d[2];
    c += 2.0*(g3u[1] - beta[0]*beta[1]*ialp2)*u_d[0]*u_d[1];
    c += 2.0*(g3u[2] - beta[0]*beta[2]*ialp2)*u_d[0]*u_d[2];
    c += 2.0*(g3u[4] - beta[1]*beta[2]*ialp2)*u_d[1]*u_d[2];

    // future-pointing root; energy E = -u_t (=1 in flat space with u_i=0)
    pr(IPEN,p) = -(-b + std::sqrt(b*b - a*c))/a;
  });
  return;
}

// explicit instantiations for the supported ghost-zone counts
template void Particles::calc_prtcl_energy<2>();
template void Particles::calc_prtcl_energy<3>();
template void Particles::calc_prtcl_energy<4>();

} // namespace particles
