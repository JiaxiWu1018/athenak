//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file calc_energy.cpp
//  \brief Calculate conserved particle energy u_t

#include <cmath>
#include <functional>

#include "athena.hpp"
#include "particles.hpp"
#include "lagrange_interp.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {

template <int NGHOST>
void Particles::calc_prtcl_energy() {
  // Extract MHD variables
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  bool is_geo_boris = pusher == ParticlesPusher::geo_boris;
  bool use_z4c = pmy_pack->pz4c != nullptr;

  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &adm_n = pmy_pack->padm->u_adm;
  DvceArray5D<Real> z4c_n;
  if (use_z4c) {
    z4c_n = pmy_pack->pz4c->u0;
  }

  // Loop over all particles
  par_for("geo_boris_push", DevExeSpace(), 0, nprtcl_thispack - 1,
  KOKKOS_LAMBDA(const int p) {
    // particle position should be at the same time as velocity
    Real x[3] = {pr(IPX, p), pr(IPY, p), pr(IPZ, p)};
    if (is_geo_boris) {
      x[0] = pr(IPLX, p);
      x[1] = pr(IPLY, p);
      x[2] = pr(IPLZ, p);
    }
    Real u[3] = {pr(IPVX, p), pr(IPVY, p), pr(IPVZ, p)};
    int mb = pi(PGID, p) - gids;
    const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max, size.d_view(mb).dx1,
                            size.d_view(mb).x2min, size.d_view(mb).x2max, size.d_view(mb).dx2,
                            size.d_view(mb).x3min, size.d_view(mb).x3max, size.d_view(mb).dx3};

    // interpolate adm variables at the particle position
    int interp_indcs[4] = {mb, -1, -1, -1};
    interp_indcs[1] = static_cast<int>(std::floor((x[0] - (mb_par[0] + mb_par[2] / 2.0)) / mb_par[2]));
    interp_indcs[2] = static_cast<int>(std::floor((x[1] - (mb_par[3] + mb_par[5] / 2.0)) / mb_par[5]));
    interp_indcs[3] = static_cast<int>(std::floor((x[2] - (mb_par[6] + mb_par[8] / 2.0)) / mb_par[8]));

    Real alp = 0.0;
    Real beta[3] = {0.0};
    Real g3d[6] = {0.0};

    Real Lx[2 * NGHOST] = {0.0}, Ly[2 * NGHOST] = {0.0}, Lz[2 * NGHOST] = {0.0};
    CalcInterpWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
    if (use_z4c) {
      alp = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_ALPHA, interp_indcs, Lx, Ly, Lz);
      beta[0] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAX, interp_indcs, Lx, Ly, Lz);
      beta[1] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAY, interp_indcs, Lx, Ly, Lz);
      beta[2] = LagrangeInterpolator<NGHOST>(z4c_n, z4c::Z4c::I_Z4C_BETAZ, interp_indcs, Lx, Ly, Lz);
    } else {
      alp = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_ALPHA, interp_indcs, Lx, Ly, Lz);
      beta[0] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAX, interp_indcs, Lx, Ly, Lz);
      beta[1] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAY, interp_indcs, Lx, Ly, Lz);
      beta[2] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_BETAZ, interp_indcs, Lx, Ly, Lz);
    }
    g3d[0] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX, interp_indcs, Lx, Ly, Lz);
    g3d[1] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXY, interp_indcs, Lx, Ly, Lz);
    g3d[2] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXZ, interp_indcs, Lx, Ly, Lz);
    g3d[3] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GYY, interp_indcs, Lx, Ly, Lz);
    g3d[4] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GYZ, interp_indcs, Lx, Ly, Lz);
    g3d[5] = LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GZZ, interp_indcs, Lx, Ly, Lz);

    Real g3u[6] = {0.0};
    Real det = Primitive::GetDeterminant(g3d);
    Primitive::InvertMatrix(g3u, g3d, det);

    // calculate the coefficients of quadratic equaiton of u_t
    Real ialp_sqr = 1. / (alp * alp);
    Real a = -ialp_sqr; // a = g^{tt}
    Real b = 0;
    for (int i = 0; i < 3; ++i) {
      b += beta[i] * ialp_sqr * u[i]; // b = g^{ti}u_i
    }
    Real c = 1.0; // c = g^{ij}u_iu_j + 1
    c += (g3u[0] - beta[0] * beta[0] * ialp_sqr) * u[0] * u[0];
    c += (g3u[3] - beta[1] * beta[1] * ialp_sqr) * u[1] * u[1];
    c += (g3u[5] - beta[2] * beta[2] * ialp_sqr) * u[2] * u[2];
    c += 2 * (g3u[1] - beta[0] * beta[1] * ialp_sqr) * u[0] * u[1];
    c += 2 * (g3u[2] - beta[0] * beta[2] * ialp_sqr) * u[0] * u[2];
    c += 2 * (g3u[4] - beta[1] * beta[2] * ialp_sqr) * u[1] * u[2];

    // get u_t, energy is -u_t
    pr(IPEN, p) = -1 * (-b + std::sqrt(b * b - a * c)) / a;
  });

  return;
}

// Explicit template instantiations
template void Particles::calc_prtcl_energy<2>();
template void Particles::calc_prtcl_energy<3>();
template void Particles::calc_prtcl_energy<4>();

} // end namespace particles