//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_excise.cpp
//! \brief parameterized particle excision: the MarkExcised task runs between Push and
//! NewGID (only scheduled when a criterion is enabled) and writes the per-particle
//! excise_flag/excise_crit arrays that SetNewPrtclGID's destruction marking consumes.
//! Three independent criteria (see particles.hpp):
//!   sphere  (PrtclDeathSphere):  |x - c| < excise_radius -- pure geometry, any pusher;
//!   lapse   (PrtclDeathLapse):   alpha(x_p) < excise_lapse, with alpha Lagrange-
//!     interpolated at the (post-push) particle position from the live arrays with the
//!     gr_boris source switch: Z4c present => I_Z4C_ALPHA from u0, else I_ADM_ALPHA;
//!   horizon (PrtclDeathHorizon): the particle is inside a converged FastFlow apparent
//!     horizon that FastFlow published (z4c/fastflow.cpp, SnapshotSurface).
//! They are checked in that order, so the recorded reason follows the precedence
//! exit > sphere > lapse > horizon (exit is applied later, in SetNewPrtclGID); every
//! enabled criterion still destroys. Marking uses the pre-wrap position: a particle that
//! periodically wraps INTO the excision region this cycle is caught at the next cycle's
//! marking. The criteria evaluate the post-step (n+1) fields, consistent with the
//! post-push position.
//!
//! WHICH horizon surface. Per cycle the driver runs the RK stage loop ("stagen") and
//! then, once, "after_timeintegrator". Z4c::FindHorizon is a stagen task gated to the
//! final stage, after Z4c::ConvertZ4cToADM publishes t^{n+1}; Push and MarkExcised
//! run afterwards. So MarkExcised sees the horizon found THIS cycle from the SAME t^{n+1}
//! geometry the push used. If the find did not converge, or FastFlow refused to publish
//! it, StageHorizons falls back to the last published surface -- never to
//! FastFlow::ah_found, which is also restored from the restart parameter dump without its
//! shape coefficients.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "lagrange_interp.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/horizon_query.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::MarkExcised
//! \brief size the marking arrays and dispatch the kernel on the active ghost count
//! (the lapse interpolation stencil is NGHOST-dependent; sphere-only runs need no
//! interpolation, so any instantiation serves)

//----------------------------------------------------------------------------------------
//! \fn int Particles::StageHorizons()
//! \brief copy the published FastFlow surfaces into the device-readable staging views
//!
//! Reads FastFlow::ah_surf_*, the sticky snapshot of the last converged find that was
//! wholly on the mesh -- never FastFlow::ah_found (see fastflow.hpp). The surfaces are
//! rank-replicated, so this is a local host copy with no collective; it is a few hundred
//! bytes per cycle.

int Particles::StageHorizons() {
  auto &pff = pmy_pack->pz4c->pfastflow;
  const int lmax1 = ah_lmax + 1;
  int nvalid = 0;
  for (int h=0; h<ah_nhorizon; ++h) {
    const bool ok = pff[h]->ah_surf_valid;
    ah_par.h_view(h, IAHVALID) = ok ? 1.0 : 0.0;
    if (!ok) {continue;}
    nvalid += 1;
    ah_par.h_view(h, IAHCX) = pff[h]->ah_surf_center[0];
    ah_par.h_view(h, IAHCY) = pff[h]->ah_surf_center[1];
    ah_par.h_view(h, IAHCZ) = pff[h]->ah_surf_center[2];
    ah_par.h_view(h, IAHRMIN) = pff[h]->ah_surf_rmin;
    ah_par.h_view(h, IAHRMAX) = pff[h]->ah_surf_rmax;
    for (int l=0; l<lmax1; ++l) {ah_coef.h_view(h, l) = pff[h]->a0_surf.h_view(l);}
    for (int i=0; i<ah_lmpoints; ++i) {
      ah_coef.h_view(h, lmax1 + i) = pff[h]->ac_surf.h_view(i);
      ah_coef.h_view(h, lmax1 + ah_lmpoints + i) = pff[h]->as_surf.h_view(i);
    }
  }
  ah_par.template modify<HostMemSpace>();
  ah_par.template sync<DevExeSpace>();
  ah_coef.template modify<HostMemSpace>();
  ah_coef.template sync<DevExeSpace>();
  return nvalid;
}

TaskStatus Particles::MarkExcised(Driver *pdrive, int stage) {
  int npart = nprtcl_thispack;
  if (npart > static_cast<int>(excise_flag.extent(0))) {
    Kokkos::realloc(excise_flag, npart);
    Kokkos::realloc(excise_crit, npart);
  }
  ah_nvalid = excise_ah ? StageHorizons() : 0;
  if (npart == 0) {return TaskStatus::complete;}

  if (excise_lapse > 0.0) {
    int ng = pmy_pack->pmesh->mb_indcs.ng;
    switch (ng) {
      case 2: mark_excised<2>(); break;
      case 3: mark_excised<3>(); break;
      case 4: mark_excised<4>(); break;
      default:
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "lapse excision supports NGHOST=2,3,4 only (got "
                  << ng << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
  } else {
    mark_excised<2>();   // sphere only: the interpolation branch is never taken
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn template <int NGHOST> void Particles::mark_excised()
//! \brief the marking kernel: writes excise_flag(p) (a ParticlesDeathReason, 0 = keep)
//! and excise_crit(p) -- r for sphere, alpha for lapse, r/R_horizon for horizon

template <int NGHOST>
void Particles::mark_excised() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &eflag = excise_flag;
  auto &ecrit = excise_crit;
  bool three_d = pmy_pack->pmesh->three_d;

  bool sphere_on = (excise_radius > 0.0);
  Real rexc = excise_radius;
  Real cx1 = excise_x1, cx2 = excise_x2, cx3 = excise_x3;
  bool lapse_on = (excise_lapse > 0.0);
  Real alpha_thr = excise_lapse;
  bool ah_on = (excise_ah && ah_nvalid > 0);
  int nh = ah_nhorizon, ah_lmax_ = ah_lmax, ah_lmpts_ = ah_lmpoints;
  auto ahpar = ah_par;
  auto ahcoef = ah_coef;

  // alpha source: live (post-step) arrays, gr_boris convention
  DvceArray5D<Real> alpha_arr;
  int alpha_idx = 0;
  if (lapse_on) {
    if (pmy_pack->pz4c != nullptr) {
      alpha_arr = pmy_pack->pz4c->u0;
      alpha_idx = z4c::Z4c::I_Z4C_ALPHA;
    } else {
      alpha_arr = pmy_pack->padm->u_adm;
      alpha_idx = adm::ADM::I_ADM_ALPHA;
    }
  }

  par_for("part_excise",DevExeSpace(),0,(nprtcl_thispack-1), KOKKOS_LAMBDA(const int p) {
    int flag = 0;
    Real crit = 0.0;
    Real x1 = pr(IPX,p);
    Real x2 = pr(IPY,p);
    Real x3 = three_d ? pr(IPZ,p) : 0.0;
    if (sphere_on) {
      Real dx1 = x1 - cx1, dx2 = x2 - cx2, dx3 = x3 - cx3;
      Real r = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3);
      if (r < rexc) {
        flag = PrtclDeathSphere;
        crit = r;
      }
    }
    if (flag == 0 && lapse_on) {
      int mb = pi(PGID,p) - gids;
      const Real mb_par[9] = {size.d_view(mb).x1min, size.d_view(mb).x1max,
                              size.d_view(mb).dx1,
                              size.d_view(mb).x2min, size.d_view(mb).x2max,
                              size.d_view(mb).dx2,
                              size.d_view(mb).x3min, size.d_view(mb).x3max,
                              size.d_view(mb).dx3};
      int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
      Real xp[3] = {x1, x2, x3};
      int interp_indcs[4] = {mb, -1, -1, -1};
      SetInterpIndices(xp, mb_par, ncell, interp_indcs);
      Real Lx[8] = {0.0}, Ly[8] = {0.0}, Lz[8] = {0.0};
      CalcInterpWght<NGHOST>(xp, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      Real alpha = LagrangeInterpolator<NGHOST>(alpha_arr, alpha_idx, interp_indcs,
                                                Lx, Ly, Lz);
      if (alpha < alpha_thr) {
        flag = PrtclDeathLapse;
        crit = alpha;
      }
    }
    if (flag == 0 && ah_on) {
      // inside ANY published horizon; record the smallest containment ratio, i.e. the
      // horizon the particle is deepest inside
      Real best = AH_CRIT_FAR;
      for (int h=0; h<nh; ++h) {
        Real c;
        if (AHInside(ahpar.d_view, ahcoef.d_view, h, ah_lmax_, ah_lmpts_,
                     x1, x2, x3, c)) {flag = PrtclDeathHorizon;}
        if (c < best) {best = c;}
      }
      if (flag == PrtclDeathHorizon) {crit = best;}
    }
    eflag(p) = flag;
    ecrit(p) = crit;
  });
  return;
}

// explicit instantiations for the supported ghost counts
template void Particles::mark_excised<2>();
template void Particles::mark_excised<3>();
template void Particles::mark_excised<4>();

} // namespace particles
