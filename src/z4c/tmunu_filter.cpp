//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tmunu_filter.cpp
//! \brief Entity-style digital filtering of the deposited particle stress-energy
//! (NRPIC tmunu-filter campaign, 2026-08-14).
//!
//! Each pass applies the exact separable outer product of the 1D binomial kernel
//! [1/4, 1/2, 1/4] -- Hakobyan et al., arXiv:2511.17710 Eq. (11):
//!
//!     T~(i,j,k) = sum_{|di|,|dj|,|dk| <= 1} 2^{-(3+|di|+|dj|+|dk|)} T(i+di,j+dj,k+dk)
//!
//! i.e. center 1/8, 6 faces 1/16, 12 edges 1/32, 8 corners 1/64 (sum exactly 1). NOTE:
//! the Entity MASTER source (kernels/digital_filter.hpp, 3D Cartesian interior branch)
//! deviates from its own Eq. (11): its 1/32 list double-counts the two i3 faces and
//! omits the (0,-1,+1)/(0,+1,-1) edges. Eq. (11) is the source of truth here, NOT the
//! Entity code; the implementation below follows the exact separable form.
//!
//! DESIGN (deliberate choices, in order of the campaign spec):
//!  * The filter is a SEPARATE post-deposition kernel. The CIC deposit, its ghost-image
//!    transport and its exact conservation identity (particles_tmunu.cpp) run first and
//!    are completely untouched; Particles::SetPrtclTmunu calls ApplyDigitalFilter after
//!    they finish, so the filter runs on every deposit path (per-cycle task, the
//!    init/restart seed, and the post-regrid re-deposit).
//!  * RAW (undensitized) components are filtered, all ten independently. Entity filters
//!    conformal (densitized) currents; filtering raw q instead changes the proper-volume
//!    integral sum q sqrt(gamma) dV by O(dx^2 laplacian(sqrt(gamma))/sqrt(gamma)) per
//!    pass (the kernel's second moment), which the diagnostics below MEASURE every
//!    ndiag cycles as the pre/post integrals of all ten components. For the target
//!    cluster (dx_fine = 0.0859M, metric scale ~ r_iso = 4.44M) the expected fractional
//!    change is ~1e-4; densitized filtering (multiply by sqrt(gamma), filter, divide)
//!    would conserve the integral exactly on uniform levels and remains the documented
//!    fallback if the measured change is not acceptably small.
//!  * Ghost fill: u_tmunu ghosts are exactly zero after deposition (the deposit writes
//!    physical cells only). Before each pass the ghosts are filled with the standard
//!    cell-centered machinery via a dedicated MeshBoundaryValuesCC object, using the
//!    proven synchronous driver-init sequence (Driver::InitBoundaryValuesAndPrimitives):
//!    Restrict -> InitRecv -> PackAndSend -> ClearSend -> ClearRecv -> RecvAndUnpack ->
//!    FillCoarseInBndry -> Prolongate. ClearRecv blocks in MPI_Wait, so RecvAndUnpackCC
//!    completes on its first call.
//!  * SMR seams: each level filters its own cells; fine ghosts across a seam hold
//!    2nd-order prolongated coarse data and coarse ghosts hold restricted fine data --
//!    exactly how every other cell-centered stencil in this code sees a seam. Constant
//!    and linear fields cross a seam unchanged; conservation at a seam is approximate
//!    and is covered by the measured integrals. In the R/M=5.5 baseline the cluster is
//!    fully inside the finest box
//!    (12.3 fine cells of clearance) until R99 crosses +-5.5M at t ~ 1.7P.
//!  * Physical (non-periodic) outer boundaries: ghosts there stay ZERO -- an explicit
//!    vacuum continuation. The matter sits at r < 10M while the boundary is at 1408M,
//!    so every stencil that reaches a physical boundary acts on exact zeros anyway.
//!    No BC function is applied to u_tmunu (Z4cBCs-style extrapolation of a source
//!    field would be wrong; periodic faces are ordinary neighbor exchanges).
//!  * Memory: scratch is ONE component (the ten components filter independently), not a
//!    ten-component copy; the coarse buffer and bvals buffers exist only when
//!    tmunu_filter_passes > 0. The constructor prints the exact per-rank byte count.
//!
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "bvals/bvals.hpp"
#include "z4c/tmunu.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void Tmunu::FillTmunuGhosts()
//! \brief synchronous cell-centered ghost fill of u_tmunu (driver-init sequence).

void Tmunu::FillTmunuGhosts() {
  auto pm = pmy_pack->pmesh;
  if (pm->multilevel) {
    pm->pmr->RestrictCC(u_tmunu, coarse_u_tmunu);
  }
  (void) pbval_tmunu->InitRecv(N_Tmunu);
  (void) pbval_tmunu->PackAndSendCC(u_tmunu, coarse_u_tmunu);
  (void) pbval_tmunu->ClearSend();
  (void) pbval_tmunu->ClearRecv();
  (void) pbval_tmunu->RecvAndUnpackCC(u_tmunu, coarse_u_tmunu);
  if (pm->multilevel) {
    pbval_tmunu->FillCoarseInBndryCC(u_tmunu, coarse_u_tmunu);
    pbval_tmunu->ProlongateCC(u_tmunu, coarse_u_tmunu);
  }
  // physical (non-periodic) outer boundaries: ghosts stay exactly zero (vacuum
  // continuation; see the file docblock). Periodic faces were served above by the
  // ordinary neighbor exchange.
}

//----------------------------------------------------------------------------------------
//! \fn void Tmunu::FilterOnePass()
//! \brief one Eq. (11) pass over all ten components: stencil into the one-component
//! scratch (never in place -- neighbors must read pre-pass data), then copy back into
//! the physical cells. Ghosts keep pre-pass values; they are refilled before the next
//! pass and no consumer reads them.

void Tmunu::FilterOnePass() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &u = u_tmunu;
  auto &scr = u_filt_scratch;
  for (int v=0; v<N_Tmunu; ++v) {
    par_for("tmunu_filter_stencil", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real w[4] = {0.125, 0.0625, 0.03125, 0.015625};
      Real s = 0.0;
      for (int dk=-1; dk<=1; ++dk) {
        for (int dj=-1; dj<=1; ++dj) {
          for (int di=-1; di<=1; ++di) {
            s += w[abs(di)+abs(dj)+abs(dk)]*u(m, v, k+dk, j+dj, i+di);
          }
        }
      }
      scr(m, k, j, i) = s;
    });
    par_for("tmunu_filter_copyback", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u(m, v, k, j, i) = scr(m, k, j, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Tmunu::ComputeSourceIntegrals(int slot)
//! \brief proper-volume integrals sum_cells q sqrt(gamma) dV of all ten components over
//! physical cells, into filt_sums[slot*10 .. slot*10+9] (slot 0 = pre, 1 = post).
//! Mirrors the deposit identity bookkeeping (same SpatialDet at cell centers, same
//! E==0 skip: the deposit makes E > 0 wherever anything was deposited, and the filter's
//! non-negative weights preserve that support property).

void Tmunu::ComputeSourceIntegrals(int slot) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  auto &g_dd = pmy_pack->padm->adm.g_dd;
  auto &tm = tmunu;
  auto &sums = filt_sums;
  int base = slot*10;
  par_for("tmunu_filter_zero_sums", DevExeSpace(), 0, 9,
  KOKKOS_LAMBDA(const int c) {
    sums(base + c) = 0.0;
  });
  par_for("tmunu_filter_integrals", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (tm.E(m,k,j,i) == 0.0) {return;}
    Real detg = adm::SpatialDet(g_dd(m,0,0,k,j,i), g_dd(m,0,1,k,j,i),
                                g_dd(m,0,2,k,j,i), g_dd(m,1,1,k,j,i),
                                g_dd(m,1,2,k,j,i), g_dd(m,2,2,k,j,i));
    const RegionSize &sz = size.d_view(m);
    Real w = sqrt(detg)*sz.dx1*sz.dx2*sz.dx3;
    Kokkos::atomic_add(&sums(base + 0), tm.E(m,k,j,i)*w);
    for (int a=0; a<3; ++a) {
      Kokkos::atomic_add(&sums(base + 1 + a), tm.S_d(m,a,k,j,i)*w);
      for (int b=a; b<3; ++b) {
        int c = 4 + (a*(7-a))/2 + (b-a);
        Kokkos::atomic_add(&sums(base + c), tm.S_dd(m,a,b,k,j,i)*w);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Tmunu::ReportFilterDiagnostics(int ncycle, bool full)
//! \brief global (MPI-reduced) pre/post-filter integral report. Component order matches
//! the deposit identity: {E, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz}. Fractional
//! changes are quoted per component against |pre_c| and, uniformly, against |pre_E|
//! (the S_i integrals are ~0 by the sampler's exact momentum pairing, so their
//! self-relative change is not meaningful).

void Tmunu::ReportFilterDiagnostics(int ncycle, bool full) {
  auto hs = Kokkos::create_mirror_view(filt_sums);
  Kokkos::deep_copy(hs, filt_sums);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, hs.data(), 20, MPI_ATHENA_REAL, MPI_SUM,
                pbval_tmunu->comm_vars);
#endif
  if (global_variable::my_rank != 0) {return;}
  static char const * const comp[10] = {"E","Sx","Sy","Sz","Sxx","Sxy","Sxz",
                                        "Syy","Syz","Szz"};
  Real e_scale = fabs(hs(0));
  if (e_scale == 0.0) {e_scale = std::numeric_limits<Real>::min();}
  Real max_rel = 0.0;
  int cbad = 0;
  for (int c=0; c<10; ++c) {
    Real r = fabs(hs(10 + c) - hs(c))/e_scale;
    if (r > max_rel) {max_rel = r; cbad = c;}
  }
  std::streamsize op = std::cout.precision(15);
  std::cout << "[tmunu-filter] cycle=" << ncycle << " passes=" << nfilter_passes
            << " intE_pre=" << hs(0) << " intE_post=" << hs(10)
            << " dE/E=" << (hs(10) - hs(0))/e_scale
            << " max|dQ|/E=" << max_rel << " (" << comp[cbad] << ")" << std::endl;
  if (full) {
    for (int c=0; c<10; ++c) {
      Real pre = hs(c), post = hs(10 + c);
      Real self = fabs(pre) > 0.0 ? (post - pre)/fabs(pre) : 0.0;
      std::cout << "[tmunu-filter]   " << comp[c] << ": pre=" << pre
                << " post=" << post << " d=" << post - pre
                << " d/|pre|=" << self << " d/E=" << (post - pre)/e_scale << std::endl;
    }
  }
  std::cout.precision(op);
}

//----------------------------------------------------------------------------------------
//! \fn void Tmunu::ApplyDigitalFilter(int ncycle, int debug_lvl)
//! \brief filter driver: conservation diagnostics around nfilter_passes applications
//! of (ghost fill + Eq. (11) pass).

void Tmunu::ApplyDigitalFilter(int ncycle, int debug_lvl) {
  if (nfilter_passes <= 0) {return;}
  bool diag = (debug_lvl >= 1) || (ncycle % filter_diag_cadence == 0);
  if (diag) {ComputeSourceIntegrals(0);}
  for (int p=0; p<nfilter_passes; ++p) {
    FillTmunuGhosts();
    FilterOnePass();
  }
  if (diag) {
    ComputeSourceIntegrals(1);
    ReportFilterDiagnostics(ncycle, (debug_lvl >= 1) || (ncycle == 0));
  }
}
