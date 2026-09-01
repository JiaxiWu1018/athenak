//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_excise.cpp
//! \brief parameterized particle excision (NRPIC Stage 3c(b)): the MarkExcised task runs
//! between Push and NewGID (only scheduled when a criterion is enabled) and writes the
//! per-particle excise_flag/excise_crit arrays that SetNewPrtclGID's destruction marking
//! consumes. Two independent criteria (see particles.hpp):
//!   sphere (flag 1): |x - c| < excise_radius -- pure geometry, any pusher;
//!   lapse  (flag 2): alpha(x_p) < excise_lapse -- alpha Lagrange-interpolated at the
//!     (post-push) particle position from the live arrays with exactly the gr_boris
//!     source switch: Z4c present => I_Z4C_ALPHA from u0, else I_ADM_ALPHA from u_adm.
//! Sphere is checked first (reason precedence exit > sphere > lapse). Marking uses the
//! pre-wrap position: a particle that periodically wraps INTO the excision region this
//! cycle is caught at the next cycle's marking (one-cycle delay; irrelevant for the
//! horizon-excision use case, where the region is far from periodic boundaries).
//! The criterion evaluates the post-step (n+1) fields, consistent with the post-push
//! position; for static backgrounds these equal the *_last snapshots by construction.

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

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::MarkExcised
//! \brief size the marking arrays and dispatch the kernel on the active ghost count
//! (the lapse interpolation stencil is NGHOST-dependent; sphere-only runs need no
//! interpolation, so any instantiation serves)

TaskStatus Particles::MarkExcised(Driver *pdrive, int stage) {
  int npart = nprtcl_thispack;
  if (npart > static_cast<int>(excise_flag.extent(0))) {
    Kokkos::realloc(excise_flag, npart);
    Kokkos::realloc(excise_crit, npart);
  }
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
//! \brief the marking kernel: writes excise_flag(p) in {0,1,2} and excise_crit(p)
//! (= r for sphere, alpha for lapse) for every particle

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
  const bool tri = (interp_method == ParticleInterpMethod::trilinear);

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
        flag = 1;
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
      if (tri) {
        CalcTriWght<NGHOST>(xp, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      } else {
        CalcInterpWght<NGHOST>(xp, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      }
      Real alpha = LagrangeInterpolator<NGHOST>(alpha_arr, alpha_idx, interp_indcs,
                                                Lx, Ly, Lz);
      if (alpha < alpha_thr) {
        flag = 2;
        crit = alpha;
      }
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
