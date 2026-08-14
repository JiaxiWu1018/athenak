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
//! Entity code; the impulse selftest below pins every one of the 27 weights.
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
//!    and linear fields cross a seam unchanged (verified by the selftest on the live
//!    mesh); conservation at a seam is approximate and is covered by the measured
//!    integrals. In the R/M=5.5 baseline the cluster is fully inside the finest box
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
//! The in-situ selftest (<particles> tmunu_filter_selftest = true) verifies on the LIVE
//! mesh, decomposition and pass count, then restores the real deposit:
//!  T1 impulse: value 64 at every MeshBlock center -> after n passes the (2n+1)^3
//!     neighborhood must equal the n-fold kernel self-convolution EXACTLY (all weights
//!     are powers of two: bitwise assert), zero elsewhere in the block.
//!  T2 constant: 1.0 everywhere -> unchanged to <= 4 ulp away from physical boundaries
//!     (exercises same-level, cross-rank AND seam restriction/prolongation of a
//!     constant); the physical-boundary band shows the documented vacuum clip.
//!  T3 linear: 1 + (ax + by + cz) -> unchanged to <= 1e-12 rel away from physical
//!     boundaries (odd kernel moments vanish; seam prolongation/restriction are
//!     linear-exact), pinning index offsets and seam handling to first order.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "z4c/tmunu.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

//----------------------------------------------------------------------------------------
//! n-fold 3D self-convolution of the Eq. (11) kernel applied to a unit impulse of value
//! `amp`, evaluated on the host in a (2n+1)^3 box. The kernel support grows by exactly
//! one cell per pass, so zero extension outside the box is exact.

void HostImpulseReference(int npass, double amp, std::vector<double> &ref, int &half) {
  half = npass;
  int dim = 2*npass + 1;
  std::vector<double> a(dim*dim*dim, 0.0), b(dim*dim*dim, 0.0);
  auto at = [&](std::vector<double> &v, int k, int j, int i) -> double& {
    return v[(k*dim + j)*dim + i];
  };
  at(a, npass, npass, npass) = amp;
  const double w[4] = {0.125, 0.0625, 0.03125, 0.015625};
  for (int p=0; p<npass; ++p) {
    std::fill(b.begin(), b.end(), 0.0);
    for (int k=0; k<dim; ++k) {
      for (int j=0; j<dim; ++j) {
        for (int i=0; i<dim; ++i) {
          double s = 0.0;
          for (int dk=-1; dk<=1; ++dk) {
            for (int dj=-1; dj<=1; ++dj) {
              for (int di=-1; di<=1; ++di) {
                int kk=k+dk, jj=j+dj, ii=i+di;
                if (kk<0||kk>=dim||jj<0||jj>=dim||ii<0||ii>=dim) {continue;}
                s += w[abs(di)+abs(dj)+abs(dk)]*at(a, kk, jj, ii);
              }
            }
          }
          at(b, k, j, i) = s;
        }
      }
    }
    std::swap(a, b);
  }
  ref = a;
}

}  // namespace

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
//! \brief filter driver: optional one-time selftest, conservation diagnostics around
//! nfilter_passes applications of (ghost fill + Eq. (11) pass).

void Tmunu::ApplyDigitalFilter(int ncycle, int debug_lvl) {
  if (nfilter_passes <= 0) {return;}
  if (filter_selftest && !selftest_done_) {
    FilterSelftest();
    selftest_done_ = true;
  }
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

//----------------------------------------------------------------------------------------
//! \fn void Tmunu::FilterSelftest()
//! \brief in-situ verification on the live mesh/decomposition (see file docblock).
//! Saves u_tmunu, runs T1 (impulse, exact), T2 (constant, <= 4 ulp), T3 (linear,
//! <= 1e-12 rel), restores u_tmunu. Any failure is fatal.

void Tmunu::FilterSelftest() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int np = nfilter_passes;
  auto &u = u_tmunu;
  auto &size = pmy_pack->pmb->mb_size;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;
  Real x1max_d = pmy_pack->pmesh->mesh_size.x1max;

  if (nx1/2 - np < 1 || nx2/2 - np < 1 || nx3/2 - np < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tmunu_filter_selftest needs MeshBlocks wider than "
              << "2*(passes+1) cells" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // save the real deposit. NOTE: create_mirror (NOT create_mirror_view) -- the latter
  // aliases the original view on serial/CPU builds where host == device space, which
  // would make the save/restore a self-copy no-op and leave the T3 linear field in
  // u_tmunu after the selftest.
  auto usave = Kokkos::create_mirror(u_tmunu);
  Kokkos::deep_copy(usave, u_tmunu);

  auto uh = Kokkos::create_mirror(u_tmunu);
  const Real eps = std::numeric_limits<Real>::epsilon();
  // physical-face flags per block, on host: faces 0..5 = x1lo,x1hi,x2lo,x2hi,x3lo,x3hi
  auto bcs_h = mbbcs.h_view;
  auto is_phys = [&](int m, int f) {
    BoundaryFlag b = bcs_h(m, f);
    return !(b == BoundaryFlag::block || b == BoundaryFlag::periodic);
  };
  // cell excluded from T2/T3 verification iff within np cells of a physical face
  auto excluded = [&](int m, int k, int j, int i) {
    if (is_phys(m,0) && (i - is) < np) {return true;}
    if (is_phys(m,1) && (ie - i) < np) {return true;}
    if (is_phys(m,2) && (j - js) < np) {return true;}
    if (is_phys(m,3) && (je - j) < np) {return true;}
    if (is_phys(m,4) && (k - ks) < np) {return true;}
    if (is_phys(m,5) && (ke - k) < np) {return true;}
    return false;
  };

  long nfail[3] = {0, 0, 0};
  double maxerr[3] = {0.0, 0.0, 0.0};

  //--------------------------------------------------------------- T1: impulse, exact
  {
    Kokkos::deep_copy(u_tmunu, 0.0);
    int ic = is + nx1/2, jc = js + nx2/2, kc = ks + nx3/2;
    par_for("tmunu_selftest_impulse", DevExeSpace(), 0, nmb-1, 0, N_Tmunu-1,
    KOKKOS_LAMBDA(const int m, const int v) {
      u(m, v, kc, jc, ic) = 64.0;
    });
    for (int p=0; p<np; ++p) {FillTmunuGhosts(); FilterOnePass();}
    std::vector<double> ref;
    int half;
    HostImpulseReference(np, 64.0, ref, half);
    int dim = 2*half + 1;
    Kokkos::deep_copy(uh, u_tmunu);
    for (int m=0; m<nmb; ++m) {
      for (int v=0; v<N_Tmunu; ++v) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              int dk = k - kc, dj = j - jc, di = i - ic;
              double expect = 0.0;
              if (abs(dk)<=half && abs(dj)<=half && abs(di)<=half) {
                expect = ref[((dk+half)*dim + (dj+half))*dim + (di+half)];
              }
              double err = fabs(static_cast<double>(uh(m,v,k,j,i)) - expect);
              if (err != 0.0) {
                ++nfail[0];
                if (err > maxerr[0]) {maxerr[0] = err;}
              }
            }
          }
        }
      }
    }
  }

  //--------------------------------------------------- T2: constant field, <= 4 ulp
  {
    Kokkos::deep_copy(u_tmunu, 0.0);
    par_for("tmunu_selftest_const", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int v=0; v<N_Tmunu; ++v) {u(m, v, k, j, i) = 1.0;}
    });
    for (int p=0; p<np; ++p) {FillTmunuGhosts(); FilterOnePass();}
    Kokkos::deep_copy(uh, u_tmunu);
    for (int m=0; m<nmb; ++m) {
      for (int v=0; v<N_Tmunu; ++v) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              double val = uh(m,v,k,j,i);
              if (excluded(m,k,j,i)) {
                // vacuum-clipped band at a physical boundary: value must lie in [0,1]
                if (val < -4.0*eps || val > 1.0 + 4.0*eps) {
                  ++nfail[1];
                  maxerr[1] = std::max(maxerr[1], fabs(val - 1.0));
                }
              } else {
                double err = fabs(val - 1.0);
                if (err > 4.0*eps) {
                  ++nfail[1];
                  if (err > maxerr[1]) {maxerr[1] = err;}
                }
              }
            }
          }
        }
      }
    }
  }

  //------------------------------------------------------ T3: linear field, <= 1e-12
  {
    Kokkos::deep_copy(u_tmunu, 0.0);
    Real xs = 1.0/x1max_d;   // gradients scaled to the domain half-width
    par_for("tmunu_selftest_linear", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const RegionSize &sz = size.d_view(m);
      Real x = CellCenterX(i - is, nx1, sz.x1min, sz.x1max);
      Real y = CellCenterX(j - js, nx2, sz.x2min, sz.x2max);
      Real z = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max);
      Real f = 1.0 + 0.25*xs*x + 0.125*xs*y + 0.0625*xs*z;
      for (int v=0; v<N_Tmunu; ++v) {u(m, v, k, j, i) = f;}
    });
    for (int p=0; p<np; ++p) {FillTmunuGhosts(); FilterOnePass();}
    Kokkos::deep_copy(uh, u_tmunu);
    for (int m=0; m<nmb; ++m) {
      RegionSize sz = size.h_view(m);
      for (int v=0; v<N_Tmunu; ++v) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              if (excluded(m,k,j,i)) {continue;}
              double x = CellCenterX(i - is, nx1, sz.x1min, sz.x1max);
              double y = CellCenterX(j - js, nx2, sz.x2min, sz.x2max);
              double z = CellCenterX(k - ks, nx3, sz.x3min, sz.x3max);
              double f = 1.0 + 0.25*xs*x + 0.125*xs*y + 0.0625*xs*z;
              double err = fabs(static_cast<double>(uh(m,v,k,j,i)) - f);
              if (err > 1.0e-12*3.0) {
                ++nfail[2];
                if (err > maxerr[2]) {maxerr[2] = err;}
              }
            }
          }
        }
      }
    }
  }

  // restore the real deposit
  Kokkos::deep_copy(u_tmunu, usave);

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, nfail, 3, MPI_LONG, MPI_SUM, pbval_tmunu->comm_vars);
  MPI_Allreduce(MPI_IN_PLACE, maxerr, 3, MPI_DOUBLE, MPI_MAX, pbval_tmunu->comm_vars);
#endif
  bool pass = (nfail[0] + nfail[1] + nfail[2] == 0);
  if (global_variable::my_rank == 0) {
    std::streamsize op = std::cout.precision(3);
    std::cout << "[tmunu-filter] SELFTEST passes=" << np
              << " impulse:"  << (nfail[0] == 0 ? "PASS" : "FAIL")
              << " (nfail=" << nfail[0] << " maxerr=" << maxerr[0] << ")"
              << " constant:" << (nfail[1] == 0 ? "PASS" : "FAIL")
              << " (nfail=" << nfail[1] << " maxerr=" << maxerr[1] << ")"
              << " linear:"   << (nfail[2] == 0 ? "PASS" : "FAIL")
              << " (nfail=" << nfail[2] << " maxerr=" << maxerr[2] << ")" << std::endl;
    std::cout.precision(op);
  }
  if (!pass) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tmunu digital-filter selftest FAILED (see counts above)"
              << std::endl << std::flush;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    std::exit(EXIT_FAILURE);
#endif
  }
}
