//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_excise.cpp
//! \brief parameterized particle excision (NRPIC Stage 3c(b), extended with the
//! experimental apparent-horizon criterion): the MarkExcised task runs between Push and
//! NewGID (only scheduled when a criterion is enabled) and writes the per-particle
//! excise_flag/excise_crit arrays that SetNewPrtclGID's destruction marking consumes.
//! Three independent criteria (see particles.hpp):
//!   sphere  (flag 1): |x - c| < excise_radius -- pure geometry, any pusher;
//!   lapse   (flag 2): alpha(x_p) < excise_lapse -- alpha Lagrange-interpolated at the
//!     (post-push) particle position from the live arrays with exactly the gr_boris
//!     source switch: Z4c present => I_Z4C_ALPHA from u0, else I_ADM_ALPHA from u_adm.
//!   horizon (flag 3): the particle is inside a converged FastFlow apparent horizon,
//!     shrunk by the fractional margin excise_ah_margin. EXPERIMENTAL, default OFF.
//! They are checked in that order, so the reason precedence is
//!   exit > sphere > lapse > horizon
//! (exit is applied later, in SetNewPrtclGID). Precedence only decides which single
//! reason is recorded for a particle that trips several at once; every enabled criterion
//! still destroys.
//!
//! Marking uses the pre-wrap position: a particle that periodically wraps INTO the
//! excision region this cycle is caught at the next cycle's marking (one-cycle delay;
//! irrelevant for the horizon-excision use case, where the region is far from periodic
//! boundaries). The criterion evaluates the post-step (n+1) fields, consistent with the
//! post-push position; for static backgrounds these equal the *_last snapshots by
//! construction.
//!
//! WHICH horizon surface is used, exactly. Per cycle the driver runs the RK stage loop
//! ("stagen") and then, once, "after_timeintegrator". Z4c::FindHorizon is a stagen task
//! gated to stage == nexp_stages, so it runs on the FINAL stage, after
//! Z4c::ConvertZ4cToADM has published the t^{n+1} geometry. The particle tasks -- Push,
//! then MarkExcised -- run afterwards in "after_timeintegrator". MarkExcised therefore
//! sees the horizon found THIS cycle from the SAME t^{n+1} geometry the push just used:
//! the freshest surface that exists, not a stale one. (FindHorizon labels that find with
//! pmesh->time, i.e. t^n, because the driver advances time after the task lists; the
//! geometry is t^{n+1}. That one-dt labelling offset affects reported AH times, not which
//! surface is used.) If the find did not converge this cycle, StageHorizons falls back to
//! the last one that did -- never to FastFlow::ah_found, which is also restored from the
//! restart parameter dump without its shape coefficients.

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
//! \brief copy the currently valid FastFlow surfaces into the device-readable staging
//! views and return how many there are.
//!
//! Reads FastFlow::ah_surf_* (the sticky snapshot of the last converged find), never
//! FastFlow::ah_found -- see the file docstring. The surfaces are rank-replicated by
//! construction (FastFlow flows on MPI-reduced surface integrals), so this is a purely
//! local host-side copy with no collective. Cost is nhorizon * ((lmax+1) + 2*(lmax+1)^2)
//! Reals per cycle, i.e. a few hundred bytes; not worth caching behind a change counter.

int Particles::StageHorizons() {
  auto &pff = pmy_pack->pz4c->pfastflow;
  const int nh = ah_nhorizon;
  const int lmax1 = ah_lmax + 1;
  int nvalid = 0;
  for (int h=0; h<nh; ++h) {
    const bool ok = pff[h]->ah_surf_valid;
    ah_par.h_view(h, IAHVALID) = ok ? 1.0 : 0.0;
    if (!ok) {continue;}
    nvalid += 1;
    ah_par.h_view(h, IAHCX) = pff[h]->ah_surf_center[0];
    ah_par.h_view(h, IAHCY) = pff[h]->ah_surf_center[1];
    ah_par.h_view(h, IAHCZ) = pff[h]->ah_surf_center[2];
    ah_par.h_view(h, IAHRMIN) = pff[h]->ah_surf_rmin;
    ah_par.h_view(h, IAHRMAX) = pff[h]->ah_surf_rmax;
    for (int l=0; l<lmax1; ++l) {
      ah_coef.h_view(h, l) = pff[h]->a0_surf.h_view(l);
    }
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
  // Refresh the staged horizons even when this rank holds no particles: the call is
  // rank-local and cheap, and keeping it unconditional makes the staged state a pure
  // function of the cycle rather than of the local particle count.
  int nah_valid = 0;
  if (excise_ah) {nah_valid = StageHorizons();}
  ah_nvalid_thiscycle = nah_valid;
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
//! and excise_crit(p) -- r for sphere, alpha for lapse, the containment ratio r/R_horizon
//! for horizon -- for every particle

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
  // horizon criterion: live only once a find has converged (ah_nvalid_thiscycle > 0),
  // so before the first horizon the branch costs one predicate per particle
  bool ah_on = (excise_ah && ah_nvalid_thiscycle > 0);
  int nh = ah_nhorizon;
  int ah_lmax_ = ah_lmax, ah_lmpts_ = ah_lmpoints;
  bool ah_surf_ = excise_ah_use_surface;
  Real ah_margin_ = excise_ah_margin;
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
      // "inside ANY valid horizon". Report the SMALLEST containment ratio over the
      // horizons, i.e. the one the particle is deepest inside (or, if outside them all,
      // the one it came closest to) -- so excise_crit is a monotone depth measure whether
      // or not the particle was destroyed.
      Real best = AH_CRIT_FAR;
      for (int h=0; h<nh; ++h) {
        Real c;
        int in = AHContainment(ahpar.d_view, ahcoef.d_view, h, ah_lmax_, ah_lmpts_,
                               ah_surf_, ah_margin_, x1, x2, x3, c);
        if (c < best) {best = c;}
        if (in == kAHInside) {flag = PrtclDeathHorizon;}
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
