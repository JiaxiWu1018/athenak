//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_tmunu.cpp
//! \brief deposit the particle stress-energy into Tmunu (NRPIC Stage 4a). For a
//! collisionless ensemble of point masses T^{mu nu} = sum_p m_p u^mu u^nu
//! delta^3(x-x_p) / (u^0 sqrt(-g)), so with W = -n.u = sqrt(1 + gamma^{ij} u_i u_j)
//! the ADM matter sources consumed by z4c_calcrhs/z4c_adm are, per particle and cell:
//!
//!   E    += m W     S_ijk / (sqrt(gamma)_ijk dV)
//!   S_a  += m u_a   S_ijk / (sqrt(gamma)_ijk dV)
//!   S_ab += m u_a u_b S_ijk / (W sqrt(gamma)_ijk dV)     [b >= a: SYM2 storage]
//!
//! (stage4_feedback/tmunu_deposition.tex; the legacy prototype e317c931 was missing the
//! 1/(W sqrt(gamma)) in every component -- an O(1) error in the OS interior). S_ijk is
//! the first-order cloud-in-cell weight, dV the coordinate cell volume of the target
//! block, and sqrt(gamma) is evaluated at the CELL CENTER (dyn_grmhd SetTmunu pattern),
//! which makes  sum_cells E sqrt(gamma) dV == sum_p m_p W_p  an EXACT discrete identity
//! -- the debug >= 1 diagnostic at the bottom of this file. The sources are deposited
//! undensitized; the consumer applies the 4pi/8pi/16pi factors.
//!
//! GHOST-IMAGE ARCHITECTURE (user-locked 2026-06-12; cross-level added in Stage 5b(a)):
//! the kernel writes ONLY its own MeshBlock's cells. The share of a boundary-band
//! particle's cloud that falls in a neighbor is delivered by a TmunuImage record
//! (particles.hpp). A SAME-LEVEL share carries the source CIC stencil and routes by
//! off_code alone (index spaces align -- no wrapped-position arithmetic, periodic wrap
//! exact). A CROSS-LEVEL share (a cloud spanning a seam) carries the particle's absolute
//! position x[3], the target + source levels, and (scheme A) the fine stencil + source
//! origin sxmin[3]. The <particles> cross_level_deposit flag selects the kernel: scheme B
//! (native) rebuilds the target-frame stencil and deposits at the target resolution
//! (DepositCloudNative); scheme A (conservative, DEFAULT) deposits the whole cloud at the
//! finest level it touches and RESTRICTS fine cells over coarser leaves into the coarse
//! cell (DepositCloudRestrict). Cross-level records are made UNIQUE per (tag, target gid)
//! at generation (EnumerateParticleTargets). Every contribution -- each particle's own
//! cloud (self record, off_code 13), its same-rank neighbor images, and images from other
//! ranks -- deposits in ONE pass in canonical (target_m, tag, off_code, lev) order; since
//! tag is globally unique the per-cell sums are independent of how blocks are distributed
//! over ranks (the Stage-4c bitwise np-invariance criterion, CPU/serial-host).
//! Kokkos::atomic_add on every write keeps the kernel GPU-correct (harmless on serial).
//!
//! Cross-level deposition is supported on STATIC refinement since Stage 5b(a). With
//! scheme A (5b(b)) the per-cycle identity Sum E sqrt(gamma) dV == Sum m W is EXACT even
//! across a seam (a residual above tol is fatal); with scheme B it is a measured
//! O(straddle) non-conservation, NOT a fatal (the diagnostic at the bottom). Bands at
//! non-periodic physical mesh boundaries generate no image; the lost share is exactly the
//! per-dim clip factor f_p accounted by that diagnostic. (Dynamic AMR + feedback is
//! supported as of Stage 5c: the regrid relabels/ships particles and re-deposits Tmunu on
//! the new grid -- see mesh_refinement.cpp AdaptiveMeshRefinement.)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#include "bvals/prtcl_search.hpp"
#include "particles.hpp"
#include "lagrange_interp.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {

namespace {  // file-local device helpers

//----------------------------------------------------------------------------------------
//! \struct CicDim
//! \brief per-dimension CIC stencil + boundary-band classification of one particle.

struct CicDim {
  int idx;      // left-center index: largest i with CellCenterX(i) <= x, in [-1, n-1]
  Real delta;   // (x - CellCenterX(idx))/dx, clamped to the closed interval [0, 1]
  int band;     // -1: lower half of first cell, +1: upper half of last cell, 0: interior
  bool open;    // banded dims only: true iff the adjacent face is block|periodic
                // (an image will deliver the out-of-block share); physical faces clip
};

//----------------------------------------------------------------------------------------
//! \fn void CicClassify()
//! \brief the SINGLE definition of the CIC stencil/band/clip predicates (count pass,
//! deposit pass and the identity bookkeeping must all call this so they cannot drift --
//! the Stage-3 shared-predicate rule). delta = 1.0 is reachable by roundoff when x sits
//! within an ulp of the upper node; the clamp keeps both weights non-negative (band
//! membership is decided by idx, never by delta).

KOKKOS_INLINE_FUNCTION
void CicClassify(Real x, int n, Real xmin, Real xmax,
                 BoundaryFlag f_lo, BoundaryFlag f_hi, CicDim &c) {
  c.idx = LeftCenterIndex(x, n, xmin, xmax);
  Real dx = (xmax - xmin)/static_cast<Real>(n);
  Real d = (x - CellCenterX(c.idx, n, xmin, xmax))/dx;
  c.delta = fmin(fmax(d, 0.0), 1.0);
  c.band = (c.idx == -1) ? -1 : ((c.idx == n-1) ? 1 : 0);
  if (c.band == 0) {
    c.open = true;
  } else {
    BoundaryFlag f = (c.band < 0) ? f_lo : f_hi;
    c.open = (f == BoundaryFlag::block || f == BoundaryFlag::periodic);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TmunuAmplitudes()
//! \brief the 10 per-particle deposit amplitudes, in the identity-sum component order
//! {E, Sx, Sy, Sz, Sxx, Sxy, Sxz, Syy, Syz, Szz}. The ONLY place these floating-point
//! expressions exist: the local pass, the image pass and the identity bookkeeping all
//! consume this one helper, so their contributions are bitwise-consistent.

KOKKOS_INLINE_FUNCTION
void TmunuAmplitudes(Real mass, Real lorentz, const Real u_d[3], Real amp[10]) {
  amp[0] = mass*lorentz;
  amp[1] = mass*u_d[0];
  amp[2] = mass*u_d[1];
  amp[3] = mass*u_d[2];
  Real moW = mass/lorentz;
  amp[4] = moW*u_d[0]*u_d[0];
  amp[5] = moW*u_d[0]*u_d[1];
  amp[6] = moW*u_d[0]*u_d[2];
  amp[7] = moW*u_d[1]*u_d[1];
  amp[8] = moW*u_d[1]*u_d[2];
  amp[9] = moW*u_d[2]*u_d[2];
}

//----------------------------------------------------------------------------------------
//! \fn void DepositCloud()
//! \brief deposit one CIC cloud (a local particle or a ghost image) into the physical
//! cells of block tm. Per dimension the image offset off[d] selects the cells:
//!   off ==  0 : cells {idx, idx+1} clipped to [0, n-1], weights {1-delta, delta}
//!               (a same-level neighbor at offset 0 in d spans the identical index
//!                range, so the rule applies verbatim to images);
//!   off == -1 : the single target cell n-1 with weight 1-delta (the share that fell
//!               below the source block);
//!   off == +1 : the single target cell 0 with weight delta.
//! The union of the local (off=0,0,0) deposit and every generated image covers each of
//! the <= 8 stencil cells exactly once. Writes are atomic; S_dd is written for b >= a
//! only (SYM2 aliased storage -- a full 3x3 loop double-counts the off-diagonals).

KOKKOS_INLINE_FUNCTION
void DepositCloud(const Tmunu::Tmunu_vars &tmunu,
                  const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
                  int tm, int is, int js, int ks, const int ncell[3], Real dv,
                  const int off[3], const int idx[3], const Real delta[3],
                  const Real amp[10]) {
  int cells[3][2];
  Real wght[3][2];
  int ncl[3];
  for (int d=0; d<3; ++d) {
    if (off[d] == 0) {
      ncl[d] = 0;
      if (idx[d] >= 0) {
        cells[d][ncl[d]] = idx[d];
        wght[d][ncl[d]] = 1.0 - delta[d];
        ncl[d]++;
      }
      if (idx[d]+1 <= ncell[d]-1) {
        cells[d][ncl[d]] = idx[d]+1;
        wght[d][ncl[d]] = delta[d];
        ncl[d]++;
      }
    } else if (off[d] == -1) {
      cells[d][0] = ncell[d]-1;
      wght[d][0] = 1.0 - delta[d];
      ncl[d] = 1;
    } else {
      cells[d][0] = 0;
      wght[d][0] = delta[d];
      ncl[d] = 1;
    }
  }
  for (int kk=0; kk<ncl[2]; ++kk) {
    for (int jj=0; jj<ncl[1]; ++jj) {
      for (int ii=0; ii<ncl[0]; ++ii) {
        Real s = wght[0][ii]*wght[1][jj]*wght[2][kk];
        int ci = is + cells[0][ii];
        int cj = js + cells[1][jj];
        int ck = ks + cells[2][kk];
        Real detg = adm::SpatialDet(g_dd(tm,0,0,ck,cj,ci), g_dd(tm,0,1,ck,cj,ci),
                                    g_dd(tm,0,2,ck,cj,ci), g_dd(tm,1,1,ck,cj,ci),
                                    g_dd(tm,1,2,ck,cj,ci), g_dd(tm,2,2,ck,cj,ci));
        Real fac = s/(sqrt(detg)*dv);
        Kokkos::atomic_add(&tmunu.E(tm,ck,cj,ci), amp[0]*fac);
        for (int a=0; a<3; ++a) {
          Kokkos::atomic_add(&tmunu.S_d(tm,a,ck,cj,ci), amp[1+a]*fac);
          for (int b=a; b<3; ++b) {
            int c = 4 + (a*(7-a))/2 + (b-a);   // SYM2 row-major slot {xx,xy,xz,yy,yz,zz}
            Kokkos::atomic_add(&tmunu.S_dd(tm,a,b,ck,cj,ci), amp[c]*fac);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DepositCloudNative()
//! \brief Stage-5b scheme-B cross-level deposit: deposit one CIC cloud into block tm at
//! tm's OWN resolution, from the particle's ABSOLUTE position x[3]. Where DepositCloud
//! routes a source-frame stencil by off_code (valid only when the index spaces align),
//! this recomputes the left-center index and CIC weight in tm's frame -- the SAME
//! LeftCenterIndex/CellCenterX predicates as CicClassify, so it is bitwise-consistent --
//! and keeps only the stencil cells inside tm's physical range [0, n-1]. The dropped
//! (out-of-range) cells carry the share owned by the source side / sibling blocks, which
//! deposit it at THEIR resolution; the kept weights therefore do NOT sum to 1 across the
//! seam -- the O(straddle) non-conservation that defines scheme B. dV and sqrt(gamma) are
//! tm's own (cell-center metric), exactly as in the same-level kernel.

KOKKOS_INLINE_FUNCTION
void DepositCloudNative(const Tmunu::Tmunu_vars &tmunu,
                        const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
                        int tm, int is, int js, int ks, const int ncell[3],
                        const RegionSize &tsz, const Real x[3], const Real amp[10]) {
  Real xmin[3] = {tsz.x1min, tsz.x2min, tsz.x3min};
  Real xmax[3] = {tsz.x1max, tsz.x2max, tsz.x3max};
  int cells[3][2];
  Real wght[3][2];
  int ncl[3];
  for (int d=0; d<3; ++d) {
    int idxt = LeftCenterIndex(x[d], ncell[d], xmin[d], xmax[d]);
    Real dxd = (xmax[d] - xmin[d])/static_cast<Real>(ncell[d]);
    Real delt = fmin(fmax((x[d] - CellCenterX(idxt, ncell[d], xmin[d], xmax[d]))/dxd,
                          0.0), 1.0);
    ncl[d] = 0;                                  // keep only the in-block stencil cells
    if (idxt >= 0 && idxt <= ncell[d]-1) {
      cells[d][ncl[d]] = idxt;   wght[d][ncl[d]] = 1.0 - delt; ncl[d]++;
    }
    if (idxt+1 >= 0 && idxt+1 <= ncell[d]-1) {
      cells[d][ncl[d]] = idxt+1; wght[d][ncl[d]] = delt;       ncl[d]++;
    }
  }
  Real dv = tsz.dx1*tsz.dx2*tsz.dx3;             // tm's native cell volume
  for (int kk=0; kk<ncl[2]; ++kk) {
    for (int jj=0; jj<ncl[1]; ++jj) {
      for (int ii=0; ii<ncl[0]; ++ii) {
        Real s = wght[0][ii]*wght[1][jj]*wght[2][kk];
        int ci = is + cells[0][ii];
        int cj = js + cells[1][jj];
        int ck = ks + cells[2][kk];
        Real detg = adm::SpatialDet(g_dd(tm,0,0,ck,cj,ci), g_dd(tm,0,1,ck,cj,ci),
                                    g_dd(tm,0,2,ck,cj,ci), g_dd(tm,1,1,ck,cj,ci),
                                    g_dd(tm,1,2,ck,cj,ci), g_dd(tm,2,2,ck,cj,ci));
        Real fac = s/(sqrt(detg)*dv);
        Kokkos::atomic_add(&tmunu.E(tm,ck,cj,ci), amp[0]*fac);
        for (int a=0; a<3; ++a) {
          Kokkos::atomic_add(&tmunu.S_d(tm,a,ck,cj,ci), amp[1+a]*fac);
          for (int b=a; b<3; ++b) {
            int c = 4 + (a*(7-a))/2 + (b-a);   // SYM2 row-major slot {xx,xy,xz,yy,yz,zz}
            Kokkos::atomic_add(&tmunu.S_dd(tm,a,b,ck,cj,ci), amp[c]*fac);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DepositCloudRestrict()
//! \brief Stage-5b(b) scheme-A CONSERVATIVE cross-level deposit. The user rule: a cloud
//! is always deposited at the FINEST level it touches; fine cells over a COARSER
//! leaf are RESTRICTED (their integrated source summed) into the covering coarse cell --
//! never prolonged. The source cloud is carried as its FINE-resolution CIC stencil
//! (idx,delta) in the source frame anchored at sxmin (fine spacing dxf = 0.5 * tm's
//! spacing; 2:1 balance => source = tm + 1). For each of the <= 8 fine stencil cells we
//! form its EXACT center c = sxmin + (i+0.5) dxf (clamp-independent, from the carried
//! integer index -- NOT from a clamped delta), keep it iff c lies in tm's half-open bbox
//! [x?min, x?max), and add its integrated source amp*s into the coarse cell of tm that
//! CONTAINS c, as the coarse density amp*s/(sqrt(gamma_c) dV_c). Fine cells outside tm
//! are owned by another leaf (the source self, a fine neighbor's DepositCloudNative, or
//! another restrict image) and deposited there. Because the carried fine weights are
//! clamp-consistent with the source self and sum to 1 over the whole cloud, and each fine
//! cell is deposited exactly once, Sum E sqrt(gamma) dV == Sum m W is EXACT at a seam.
//!
//! Serves three roles, distinguished only by (tm, sxmin): a FINE->COARSE image (tm = the
//! coarse neighbor, sxmin = source fine block origin); a COARSE->FINE self (tm = the
//! particle's own coarse block, sxmin = its origin, cloud carried at the fine sublevel);
//! and a coarse->fine transverse restrict onto a same-level coarse neighbor. The metric
//! and volume are tm's coarse-cell values (the metric divides out of the integral
//! Q = q sqrt(gamma) dV, so the coarse cell's sqrt(gamma) is exact). The containing-cell
//! floor is robust: a fine-cell center sits at a coarse-cell quarter point, interior by
//! 0.25 dxc, so it can neither cross tm's bbox nor a coarse-cell face by roundoff.

KOKKOS_INLINE_FUNCTION
void DepositCloudRestrict(const Tmunu::Tmunu_vars &tmunu,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
                          int tm, int is, int js, int ks, const int ncell[3],
                          const RegionSize &tsz, const Real sxmin[3],
                          const int idx[3], const Real delta[3], const Real amp[10]) {
  Real xmin[3] = {tsz.x1min, tsz.x2min, tsz.x3min};
  Real xmax[3] = {tsz.x1max, tsz.x2max, tsz.x3max};
  Real dxc[3]  = {tsz.dx1, tsz.dx2, tsz.dx3};    // tm's (coarse) spacing
  int  ccell[3][2];                          // coarse cell of tm for each kept fine cell
  Real wght[3][2];
  int  ncl[3];
  for (int d=0; d<3; ++d) {
    Real dxf = 0.5*dxc[d];                       // fine spacing (source one level finer)
    ncl[d] = 0;
    for (int t=0; t<2; ++t) {                    // the two CIC fine cells idx, idx+1
      Real c = sxmin[d] + (static_cast<Real>(idx[d]+t) + 0.5)*dxf;   // exact fine center
      if (c < xmin[d] || c >= xmax[d]) {continue;}             // owned by another leaf
      int ic = static_cast<int>(floor((c - xmin[d])/dxc[d]));  // containing coarse cell
      if (ic < 0) {ic = 0;} else if (ic > ncell[d]-1) {ic = ncell[d]-1;}  // defensive
      ccell[d][ncl[d]] = ic;
      wght[d][ncl[d]] = (t == 0) ? (1.0 - delta[d]) : delta[d];
      ncl[d]++;
    }
  }
  Real dv = tsz.dx1*tsz.dx2*tsz.dx3;             // tm's coarse cell volume
  for (int kk=0; kk<ncl[2]; ++kk) {
    for (int jj=0; jj<ncl[1]; ++jj) {
      for (int ii=0; ii<ncl[0]; ++ii) {
        Real s = wght[0][ii]*wght[1][jj]*wght[2][kk];
        int ci = is + ccell[0][ii];
        int cj = js + ccell[1][jj];
        int ck = ks + ccell[2][kk];
        Real detg = adm::SpatialDet(g_dd(tm,0,0,ck,cj,ci), g_dd(tm,0,1,ck,cj,ci),
                                    g_dd(tm,0,2,ck,cj,ci), g_dd(tm,1,1,ck,cj,ci),
                                    g_dd(tm,1,2,ck,cj,ci), g_dd(tm,2,2,ck,cj,ci));
#ifdef NRPIC_BUG_RESTRICT
        Real fac = s;   // SEEDED BUG (drill): drop the 1/(sqrt(g) dV) coarse-cell norm ->
                        // the restricted density is wrong -> the A identity FATALs (RED)
#else
        Real fac = s/(sqrt(detg)*dv);
#endif
        Kokkos::atomic_add(&tmunu.E(tm,ck,cj,ci), amp[0]*fac);
        for (int a=0; a<3; ++a) {
          Kokkos::atomic_add(&tmunu.S_d(tm,a,ck,cj,ci), amp[1+a]*fac);
          for (int b=a; b<3; ++b) {
            int c = 4 + (a*(7-a))/2 + (b-a);   // SYM2 row-major slot {xx,xy,xz,yy,yz,zz}
            Kokkos::atomic_add(&tmunu.S_dd(tm,a,b,ck,cj,ci), amp[c]*fac);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \struct SortTmunuImage
//! \brief canonical order (target_m, tag, off_code, lev): per-block grouping, then a
//! total order that makes the deposit independent of generation/arrival order (the
//! duplicate check is lev-conditional -- see the deposit pass below).

struct SortTmunuImage {
  bool operator()(const TmunuImage &a, const TmunuImage &b) const {
    if (a.target_m != b.target_m) {return a.target_m < b.target_m;}
    if (a.tag != b.tag) {return a.tag < b.tag;}
    if (a.off_code != b.off_code) {return a.off_code < b.off_code;}
    return a.lev < b.lev;   // Stage 5b: total order incl. cross-level images
  }
};

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn template<int NGHOST> void Particles::set_prtcl_tmunu()
//! \brief zero Tmunu, deposit all local particles, generate + deposit same-rank ghost
//! images, and (debug >= 1) verify the exact conservation identities.

template <int NGHOST>
void Particles::set_prtcl_tmunu() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbpar = pmy_pack->pmb->mb_parity;
  int gids = pmy_pack->gids;
  int nmb = pmy_pack->nmb_thispack;
  int npart = nprtcl_thispack;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &adm = pmy_pack->padm->adm;
  auto &adm_n = pmy_pack->padm->u_adm;
  auto &tmunu = pmy_pack->ptmunu->tmunu;
  auto &u_tmunu = pmy_pack->ptmunu->u_tmunu;
  auto &g_dd = adm.g_dd;
  int ncell[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int dbg = debug_lvl;
  // cross-level deposition scheme: 0 = conservative (A, restrict), 1 = native (B). Picks
  // the cross-level kernel in the deposit pass + the conservation regime in the identity.
  int xl_scheme = (xlevel_deposit == CrossLevelDeposit::native) ? 1 : 0;
  int myrank = global_variable::my_rank;
  int ncycle = pmy_pack->pmesh->ncycle;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;

  // ---- (a) zero pass: full array including ghosts (the deposit below touches physical
  // cells only and nothing else writes u_tmunu in the feedback configuration -- dyn_grmhd
  // is fatal-guarded). Runs even with npart == 0: an all-excised ensemble must source
  // vacuum, which is exactly the trumpet end state of the OS collapse.
  Kokkos::deep_copy(u_tmunu, 0.0);

  auto &psum = tmunu_psums;
  if (dbg >= 1) {
    Kokkos::deep_copy(psum, 0.0);
  }

  nimages_thispack = 0;
  nimg_send_thispack = 0;
  int n_cross_thispack = 0;   // Stage 5b: cross-level images this pack (= derr(1))
  if (npart > 0) {
    // ---- (b1) count pass: cross-block images per particle = nonempty offset subsets of
    // the banded-and-open dims (same predicates as the deposit pass: CicClassify). The
    // per-particle self record (own-block cloud) is generated below, so it
    // is NOT counted here -- the queue is sized for npart self records plus these.
    int nimg_need = 0;
    Kokkos::parallel_reduce("tmunu_count_img",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &sum) {
        int m = pi(PGID,p) - gids;
        Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
        const RegionSize &sz = size.d_view(m);
        CicDim cd[3];
        CicClassify(x[0], ncell[0], sz.x1min, sz.x1max,
                    mbbcs.d_view(m,0), mbbcs.d_view(m,1), cd[0]);
        CicClassify(x[1], ncell[1], sz.x2min, sz.x2max,
                    mbbcs.d_view(m,2), mbbcs.d_view(m,3), cd[1]);
        CicClassify(x[2], ncell[2], sz.x3min, sz.x3max,
                    mbbcs.d_view(m,4), mbbcs.d_view(m,5), cd[2]);
        int beff[3];
        for (int d=0; d<3; ++d) {
          beff[d] = (cd[d].band != 0 && cd[d].open) ? cd[d].band : 0;
        }
        if (beff[0] == 0 && beff[1] == 0 && beff[2] == 0) {return;}
        int mylev = mblev.d_view(m);
        int fx = (x[0] < 0.5*(sz.x1min + sz.x1max)) ? 0 : 1;
        int fy = (x[1] < 0.5*(sz.x2min + sz.x2max)) ? 0 : 1;
        int fz = (x[2] < 0.5*(sz.x3min + sz.x3max)) ? 0 : 1;
        int px = mbpar.d_view(m,0), py = mbpar.d_view(m,1), pz = mbpar.d_view(m,2);
        // count = the per-particle DEDUPED ghost-image target count, via the SAME helper
        // the fill pass uses (EnumerateParticleTargets) so cap == appends: same-level
        // 1 per subset, coarse->fine up to 4 children, fine->coarse unique per coarse gid
        // (a demoted diagonal that lands on an already-targeted coarse face is dropped).
        PartImageTarget tgt[24];
        int nmiss = 0, ov = 0;
        sum += EnumerateParticleTargets(nghbr.d_view, m, mylev, beff, fx,fy,fz, px,py,pz,
                                        multi_d, three_d, tgt, 24, nmiss, ov);
      }, Kokkos::Sum<int>(nimg_need));
    // size the queue for npart self records (slots [0,npart)) plus all cross-block images
    // if they were all same-rank (the upper bound); cross-rank-bound images go to the
    // separate send-staging array.
    if (npart + nimg_need > static_cast<int>(tmunu_images.extent(0))) {
      Kokkos::realloc(tmunu_images, npart + nimg_need);
    }
#if MPI_PARALLEL_ENABLED
    if (nimg_need > static_cast<int>(tmunu_img_send.extent(0))) {
      Kokkos::realloc(tmunu_img_send, nimg_need);
    }
#endif
    Kokkos::deep_copy(tmunu_nimg, 0);   // {0: same-rank imgs beyond npart, 1: cross-rank}

    // device counters: slots {0: no-neighbor, 2: bad local pack range, 3: image-list
    // overflow, 4: cross-level image through a PERIODIC boundary (NRPIC Stage 5c,
    // unsupported -- see the host check below)} are fatal errors; slot {1: cross-level
    // image count} is a Stage-5b DIAGNOSTIC (not an error): a nonzero global value flips
    // the conservation identity below to a measured report (scheme B non-conservative).
    DvceArray1D<int> derr("tmunu_err",5);   // zero-initialized

    // ---- (b2) record-generation pass: emit one self record per particle (its own-block
    // cloud) into slot p, append same-rank neighbor images beyond npart, stage cross-rank
    // neighbor images for transport, and (debug) accumulate the particle-side identity
    // sums with the boundary-clip factor f_p. Nothing is deposited here -- the single
    // canonical pass below deposits every record in rank-invariant order.
    auto &img = tmunu_images;
    auto &img_send = tmunu_img_send;
    auto &nimg_ctr = tmunu_nimg;
    int img_cap = static_cast<int>(tmunu_images.extent(0));
    int send_cap = static_cast<int>(tmunu_img_send.extent(0));
    par_for("tmunu_gen_records", DevExeSpace(), 0, (npart-1),
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID,p) - gids;
      int mylev = mblev.d_view(m);
      Real x[3]   = {pr(IPX,p),  pr(IPY,p),  pr(IPZ,p)};
      Real u_d[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};   // covariant u_i
      Real mp = pr(IPM,p);
      const RegionSize &sz = size.d_view(m);

      // CIC stencil + band/clip classification (the shared predicate helper)
      CicDim cd[3];
      CicClassify(x[0], ncell[0], sz.x1min, sz.x1max,
                  mbbcs.d_view(m,0), mbbcs.d_view(m,1), cd[0]);
      CicClassify(x[1], ncell[1], sz.x2min, sz.x2max,
                  mbbcs.d_view(m,2), mbbcs.d_view(m,3), cd[1]);
      CicClassify(x[2], ncell[2], sz.x3min, sz.x3max,
                  mbbcs.d_view(m,4), mbbcs.d_view(m,5), cd[2]);
      int idx[3]   = {cd[0].idx, cd[1].idx, cd[2].idx};
      Real dlt[3]  = {cd[0].delta, cd[1].delta, cd[2].delta};

      // normal-frame Lorentz factor W = sqrt(1 + gamma^{ij} u_i u_j), with gamma^{ij}
      // from the per-node inverse-metric interpolation (the gr_boris machinery); the
      // 3-metric slots are the leading 6 of u_adm in both the full and the z4c-trimmed
      // layouts, so this never touches the (possibly absent) gauge slots.
      const Real mb_par[9] = {sz.x1min, sz.x1max, sz.dx1,
                              sz.x2min, sz.x2max, sz.dx2,
                              sz.x3min, sz.x3max, sz.dx3};
      int interp_indcs[4] = {m, -1, -1, -1};
      SetInterpIndices(x, mb_par, ncell, interp_indcs);
      Real Lx[2*NGHOST] = {0.0}, Ly[2*NGHOST] = {0.0}, Lz[2*NGHOST] = {0.0};
      CalcInterpWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      Real g3u[6];
      LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX, interp_indcs,
                                   Lx, Ly, Lz, g3u);
      Real usq = g3u[0]*u_d[0]*u_d[0] + g3u[3]*u_d[1]*u_d[1] + g3u[5]*u_d[2]*u_d[2]
               + 2.0*(g3u[1]*u_d[0]*u_d[1] + g3u[2]*u_d[0]*u_d[2]
                      + g3u[4]*u_d[1]*u_d[2]);
      Real lor = sqrt(1.0 + usq);

      Real amp[10];
      TmunuAmplitudes(mp, lor, u_d, amp);

      // ---- enumerate this cloud's cross-block image targets ONCE (shared by the self
      // record's encoding and the image emission). For scheme A, a particle whose cloud
      // reaches a FINER neighbor (touches_finer) deposits the WHOLE cloud at the fine
      // sublevel: self + any same-level overhang RESTRICT fine cells into coarse leaves
      // and only the finer-neighbor part is native (the "deposit on the finest level the
      // cloud touches; restrict, never prolong" rule).
      int beff[3];
      for (int d=0; d<3; ++d) {
        beff[d] = (cd[d].band != 0 && cd[d].open) ? cd[d].band : 0;
      }
      bool banded = (beff[0] != 0 || beff[1] != 0 || beff[2] != 0);
      PartImageTarget tgt[24];
      int ntgt = 0;
      bool touches_finer = false;
      if (banded) {
        int fx = (x[0] < 0.5*(sz.x1min + sz.x1max)) ? 0 : 1;
        int fy = (x[1] < 0.5*(sz.x2min + sz.x2max)) ? 0 : 1;
        int fz = (x[2] < 0.5*(sz.x3min + sz.x3max)) ? 0 : 1;
        int px = mbpar.d_view(m,0), py = mbpar.d_view(m,1), pz = mbpar.d_view(m,2);
        int nmiss = 0, ov = 0;
        ntgt = EnumerateParticleTargets(nghbr.d_view, m, mylev, beff, fx,fy,fz,
                                        px,py,pz, multi_d, three_d, tgt, 24, nmiss, ov);
        if (nmiss > 0) {                          // banded-open dir(s) with no neighbor
          Kokkos::atomic_add(&derr(0), nmiss);
          Kokkos::printf("[tmunu-debug] rank=%d cycle=%d tag=%d gid=%d NO NEIGHBOR "
                         "(missing=%d) pos=(%.16e,%.16e,%.16e)\n", myrank, ncycle,
                         pi(PTAG,p), pi(PGID,p), nmiss, x[0], x[1], x[2]);
        }
        if (ov) {Kokkos::atomic_add(&derr(3), 1);}   // dedup overflow (impossible: <=19)
        for (int s=0; s<ntgt; ++s) {
          if (nghbr.d_view(m, tgt[s].slot).lev > mylev) {touches_finer = true;}
        }
      }
      // scheme A + a finer neighbor touched: deposit the whole cloud at the fine sublevel
      bool cfine = (xl_scheme == 0) && touches_finer;

      // the FINE-resolution CIC stencil of x in THIS block refined x2 -- the source
      // stencil for every restrict record a c->f cloud emits (the self over own block,
      // and any transverse overhang onto a coarse neighbor). Aligned with the fine
      // neighbor's cells, so the restrict and the fine-neighbor native deposit see the
      // same fine cells (weights sum to 1 -> exact conservation).
      int idxf[3] = {0, 0, 0};
      Real dltf[3] = {0.0, 0.0, 0.0};
      Real sxmin[3] = {sz.x1min, sz.x2min, sz.x3min};
      if (cfine) {
        Real fxmin[3] = {sz.x1min, sz.x2min, sz.x3min};
        Real fxmax[3] = {sz.x1max, sz.x2max, sz.x3max};
        for (int d=0; d<3; ++d) {
          int nf = 2*ncell[d];
          idxf[d] = LeftCenterIndex(x[d], nf, fxmin[d], fxmax[d]);
          Real dxf = (fxmax[d] - fxmin[d])/static_cast<Real>(nf);
          dltf[d] = fmin(fmax((x[d] - CellCenterX(idxf[d], nf, fxmin[d], fxmax[d]))/dxf,
                              0.0), 1.0);
        }
      }

      // self record: the particle's own cloud (off_code 13). Same-level DepositCloud
      // (lev = -1) UNLESS scheme A and the cloud reaches a finer neighbor, in which case
      // the self RESTRICTS the fine cells that land over THIS coarse block (lev=mylev,
      // slev=mylev+1). A first-class record at slot p so the cloud + every neighbor image
      // deposit in the one canonical (target_m,tag,off_code,lev) pass below.
      {
        TmunuImage self;
        self.target_m = m;
        self.tag = pi(PTAG,p);
        self.off_code = 13;
        for (int d=0; d<3; ++d) {
          self.x[d] = x[d];
          self.sxmin[d] = sxmin[d];
          self.u_d[d] = u_d[d];
        }
        self.mass = mp;
        self.lorentz = lor;
        if (cfine) {
          self.lev = mylev;
          self.slev = mylev + 1;       // restrict the fine sublevel into this block
          for (int d=0; d<3; ++d) {self.idx[d] = idxf[d]; self.delta[d] = dltf[d];}
        } else {
          self.lev = -1;               // ordinary same-level cloud (DepositCloud)
          self.slev = -1;
          for (int d=0; d<3; ++d) {self.idx[d] = idx[d]; self.delta[d] = dlt[d];}
        }
        img.d_view(p) = self;
      }

      // image generation: one record per enumerated target. The (lev, slev) pair picks
      // the deposit kernel below: lev<0 same-level off_code; scheme-B native (lev=target,
      // slev=mylev); scheme-A restrict (slev>lev: f->c, or a c->f overhang onto a coarse
      // leaf); scheme-A native at a finer target (slev<lev: the c->f fine-neighbor part).
      if (banded) {
        for (int s=0; s<ntgt; ++s) {
          const NeighborBlock &nb = nghbr.d_view(m, tgt[s].slot);
          int oc = tgt[s].oc;
          int rlev, rslev, ridx[3];
          Real rdlt[3];
          if (xl_scheme == 1) {                   // scheme B (native across the seam)
            rlev = (nb.lev == mylev) ? -1 : nb.lev;
            rslev = mylev;
            for (int d=0; d<3; ++d) {ridx[d] = idx[d]; rdlt[d] = dlt[d];}
          } else if (nb.lev > mylev) {            // scheme A, finer neighbor: c->f native
            rlev = nb.lev;
            rslev = mylev;
            for (int d=0; d<3; ++d) {ridx[d] = idx[d]; rdlt[d] = dlt[d];}
          } else if (nb.lev < mylev) {            // scheme A, coarser nbr: f->c restrict
            rlev = nb.lev;
            rslev = mylev;                         // idx/dlt = the fine source stencil
            for (int d=0; d<3; ++d) {ridx[d] = idx[d]; rdlt[d] = dlt[d];}
          } else if (cfine) {                     // scheme A, c->f coarse overhang
            rlev = mylev;
            rslev = mylev + 1;                     // restrict the fine sublevel stencil
            for (int d=0; d<3; ++d) {ridx[d] = idxf[d]; rdlt[d] = dltf[d];}
          } else {                                // scheme A, plain same-level neighbor
            rlev = -1;
            rslev = -1;
            for (int d=0; d<3; ++d) {ridx[d] = idx[d]; rdlt[d] = dlt[d];}
          }
          if (rlev >= 0) {
            Kokkos::atomic_add(&derr(1), 1);   // cross-level image (Stage-5b diagnostic)
            // NRPIC Stage 5c: a cross-level image rebuilds the deposit stencil in the
            // TARGET block's frame from the particle's absolute position
            // (DepositCloudNative / Restrict), which assumes the target's coordinates are
            // adjacent to the particle. That is FALSE for a coarse-fine seam that ALSO
            // wraps a PERIODIC boundary (the wrap neighbor sits a domain-length away) ->
            // unsupported. Detect it (crossed face is periodic) and flag a fatal below;
            // INTERIOR seams cross block faces, not periodic ones, so old tests pass.
            int bxo = (oc % 3) - 1, byo = ((oc / 3) % 3) - 1, bzo = (oc / 9) - 1;
            if ((bxo < 0 && mbbcs.d_view(m,0) == BoundaryFlag::periodic) ||
                (bxo > 0 && mbbcs.d_view(m,1) == BoundaryFlag::periodic) ||
                (byo < 0 && mbbcs.d_view(m,2) == BoundaryFlag::periodic) ||
                (byo > 0 && mbbcs.d_view(m,3) == BoundaryFlag::periodic) ||
                (bzo < 0 && mbbcs.d_view(m,4) == BoundaryFlag::periodic) ||
                (bzo > 0 && mbbcs.d_view(m,5) == BoundaryFlag::periodic)) {
              Kokkos::atomic_add(&derr(4), 1);
            }
          }
          if (nb.rank == myrank) {
            // same-rank neighbor: append into the local queue (slots beyond npart self
            // records). Its gid must map into this pack -- a violation is corruption.
            int tm = nb.gid - gids;
            if (tm < 0 || tm >= nmb) {
              Kokkos::atomic_add(&derr(2), 1);
              Kokkos::printf("[tmunu-debug] rank=%d cycle=%d tag=%d gid=%d BAD LOCAL "
                             "TARGET nbr_gid=%d\n", myrank, ncycle, pi(PTAG,p),
                             pi(PGID,p), nb.gid);
              continue;
            }
            int slot = npart + Kokkos::atomic_fetch_add(&nimg_ctr(0), 1);
            if (slot >= img_cap) {
              Kokkos::atomic_add(&derr(3), 1);
              continue;
            }
            TmunuImage rec;
            rec.target_m = tm;
            rec.tag = pi(PTAG,p);
            rec.off_code = oc;
            rec.lev = rlev;
            rec.slev = rslev;
            for (int d=0; d<3; ++d) {
              rec.idx[d] = ridx[d];
              rec.delta[d] = rdlt[d];
              rec.x[d] = x[d];
              rec.sxmin[d] = sxmin[d];
              rec.u_d[d] = u_d[d];
            }
            rec.mass = mp;
            rec.lorentz = lor;
            img.d_view(slot) = rec;
          } else {
            // cross-rank neighbor: stage a wire record for MPI transport. The target is
            // named by its GLOBAL gid (the receiver converts it to a local index and
            // re-sorts the image into its own queue -- Stage 4c ExchangeTmunuImages).
            int slot = Kokkos::atomic_fetch_add(&nimg_ctr(1), 1);
            if (slot >= send_cap) {
              Kokkos::atomic_add(&derr(3), 1);
              continue;
            }
            TmunuImageWire w;
            w.target_gid = nb.gid;
            w.tag = pi(PTAG,p);
            w.off_code = oc;
            w.lev = rlev;
            w.slev = rslev;
            for (int d=0; d<3; ++d) {
              w.idx[d] = ridx[d];
              w.delta[d] = rdlt[d];
              w.x[d] = x[d];
              w.sxmin[d] = sxmin[d];
              w.u_d[d] = u_d[d];
            }
            w.mass = mp;
            w.lorentz = lor;
            img_send.d_view(slot) = w;
          }
        }
      }

      // particle-side identity sums with the exact boundary-clip factor f_p = prod_d s_d:
      // s_d = 1 except at a closed (no-image) band, where only the in-domain weight was
      // deposited. The clip MUST use the same resolution as the deposit: scheme A with a
      // finer neighbor (cfine) deposited the cloud at the FINE sublevel, so a closed
      // transverse face clips the FINE ghost cell across it, not the coarse half-band.
      if (dbg >= 1) {
        Real f = 1.0;
        for (int d=0; d<3; ++d) {
          if (cd[d].band != 0 && !cd[d].open) {
            if (cfine) {
              // the lost share is the single fine ghost cell across the closed face:
              // idxf=-1 below (its center is 0.5 dx_fine < xmin, dropped; cell 0 kept, wt
              // dltf) or idxf+1=nf above (cell nf dropped; cell nf-1 kept, wt 1-dltf) --
              // exactly the cell the restrict/native deposit drops by its bbox test.
              // idxf is the compare-corrected LeftCenterIndex, so this index test matches
              // the kernel to the bit (fine centers never sit on the edge). The cfine dim
              // itself is an OPEN (coarse-fine) face, so only transverse closed dims do.
              int nf = 2*ncell[d];
              f *= (cd[d].band < 0) ? ((idxf[d] == -1) ? dltf[d] : 1.0)
                                    : ((idxf[d] + 1 == nf) ? (1.0 - dltf[d]) : 1.0);
            } else {
              f *= (cd[d].band < 0) ? dlt[d] : (1.0 - dlt[d]);
            }
          }
        }
        for (int c=0; c<10; ++c) {
          Kokkos::atomic_add(&psum(c), amp[c]*f);
        }
      }
    });

    // read the image counters + error counters back (deep_copy fences the kernel). The
    // queue holds npart self records (slots [0,npart)) plus n_local same-rank images;
    // n_remote cross-rank images were staged in tmunu_img_send for the MPI transport.
    auto hcnt = Kokkos::create_mirror_view(tmunu_nimg);
    Kokkos::deep_copy(hcnt, tmunu_nimg);
    int n_local_img = hcnt(0);
    int n_remote_img = hcnt(1);
    nimages_thispack = npart + n_local_img;
    nimg_send_thispack = n_remote_img;
    auto herr = Kokkos::create_mirror_view(derr);
    Kokkos::deep_copy(herr, derr);
    n_cross_thispack = herr(1);   // diagnostic (cross-level images), NOT an error
    if (herr(4) > 0) {  // NRPIC Stage 5c: periodic cross-level seam (unsupported)
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "cross-level Tmunu deposition through a PERIODIC boundary is not "
                << "supported (Stage 5c): " << herr(4) << " image(s) at cycle " << ncycle
                << "." << std::endl
                << "A cloud straddling a coarse-fine seam that ALSO wraps a periodic "
                << "boundary would mis-deposit: the cross-level image carries the "
                << "particle's absolute position and rebuilds the stencil in the wrap "
                << "neighbor's frame, a domain away. Use non-periodic boundaries in the "
                << "refined directions, or keep refinement away from periodic faces."
                << std::endl << std::flush;
#if MPI_PARALLEL_ENABLED
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      std::exit(EXIT_FAILURE);
#endif
    }
    if (herr(0) + herr(2) + herr(3) > 0 ||
        (n_local_img + n_remote_img) != nimg_need) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Tmunu deposit failed at cycle " << ncycle
                << ": no_neighbor=" << herr(0)
                << " bad_local_target=" << herr(2) << " overflow=" << herr(3)
                << " (cross_level=" << herr(1) << ")"
                << " images=" << (n_local_img + n_remote_img) << "/" << nimg_need
                << " (see offender dump above)" << std::endl << std::flush;
#if MPI_PARALLEL_ENABLED
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      std::exit(EXIT_FAILURE);
#endif
    }
  }   // end (npart > 0): per-particle records generated

  // ---- cross-rank ghost-image transport (Stage 4c): ship the staged images and append
  // the received ones into tmunu_images, growing nimages_thispack. Serial no-op.
  // Runs on EVERY rank even with npart==0 (a neighbor may image into this rank's blocks),
  // so the collective census never deadlocks.
#if MPI_PARALLEL_ENABLED
  pbval_part->ExchangeTmunuImages();
#endif

  // ---- (c) canonical deposit order: sort the merged queue (self records + same-rank +
  // received cross-rank images) on host by (target_m,tag,off_code,lev) -- a total order;
  // tag is globally unique, so per-cell accumulation order is identical for every rank
  // decomposition (the Stage-4c bitwise rank-invariance criterion). A duplicate key means
  // duplicate particle tags (an init=pgen contract violation) or a generation bug: fatal.
  if (nimages_thispack > 0) {
    tmunu_images.template modify<DevExeSpace>();
    tmunu_images.template sync<HostMemSpace>();
    TmunuImage *ibeg = tmunu_images.h_view.data();
    std::sort(ibeg, ibeg + nimages_thispack, SortTmunuImage());
    for (int g=1; g<nimages_thispack; ++g) {
      // duplicate-key invariant (records sharing (target_m, tag) are adjacent under the
      // sort). CROSS-LEVEL (lev>=0): exactly ONE record per (target_m, tag) is allowed --
      // DepositCloudNative deposits the whole clipped cloud, so a duplicate is a double-
      // deposit and EnumerateParticleTargets should have deduped it (gid). SAME-LEVEL
      // (lev<0): off_code distinguishes the disjoint cells, so the key is (.., off_code).
      bool same_mt = (ibeg[g-1].target_m == ibeg[g].target_m &&
                      ibeg[g-1].tag == ibeg[g].tag);
      bool xlevel_dup = same_mt && (ibeg[g-1].lev >= 0 || ibeg[g].lev >= 0);
      bool samelev_dup = same_mt && ibeg[g-1].lev < 0 && ibeg[g].lev < 0 &&
                         ibeg[g-1].off_code == ibeg[g].off_code;
      if (xlevel_dup || samelev_dup) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << (xlevel_dup ? "duplicate CROSS-LEVEL Tmunu target (one record per "
                                   "(target_m, tag) required -- dedup failed): "
                                 : "duplicate same-level Tmunu image key: ")
                  << "target_m=" << ibeg[g].target_m << ", tag=" << ibeg[g].tag
                  << ", off_code=" << ibeg[g].off_code << ", lev=" << ibeg[g].lev
                  << " at cycle " << ncycle << std::endl << std::flush;
#if MPI_PARALLEL_ENABLED
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        std::exit(EXIT_FAILURE);
#endif
      }
    }
    tmunu_images.template modify<HostMemSpace>();
    tmunu_images.template sync<DevExeSpace>();

    // ---- (d) one unified deposit pass over every record (self + same-rank images +
    // received cross-rank images), in canonical order (serial RangePolicy executes in
    // index order; on GPU the atomics keep it correct, not bitwise-reproducible).
    auto &imgd = tmunu_images;
    int nimg = nimages_thispack;
    par_for("tmunu_deposit", DevExeSpace(), 0, (nimg-1),
    KOKKOS_LAMBDA(const int g) {
      const TmunuImage rec = imgd.d_view(g);
      int tm = rec.target_m;
      Real amp[10];
      TmunuAmplitudes(rec.mass, rec.lorentz, rec.u_d, amp);
      const RegionSize &tsz = size.d_view(tm);
      if (rec.lev < 0) {
        // same-level (self or same-level neighbor): off_code routing at the shared dx
        int off[3];
        off[0] = rec.off_code % 3 - 1;
        off[1] = (rec.off_code / 3) % 3 - 1;
        off[2] = rec.off_code / 9 - 1;
        Real dv = tsz.dx1*tsz.dx2*tsz.dx3;
        DepositCloud(tmunu, g_dd, tm, is, js, ks, ncell, dv, off, rec.idx, rec.delta,
                     amp);
      } else if (xl_scheme == 1) {
        // scheme B (native): deposit the cross-level cloud at tm's resolution from the
        // absolute position -- smooth, O(straddle) non-conservative (Stage 5b(a)).
        DepositCloudNative(tmunu, g_dd, tm, is, js, ks, ncell, tsz, rec.x, amp);
      } else if (rec.slev > rec.lev) {
        // scheme A (conservative): RESTRICT the fine source into tm's coarse cells
        // (f->c, or a c->f overhang onto a coarse leaf incl. the own-block self). Exact.
        DepositCloudRestrict(tmunu, g_dd, tm, is, js, ks, ncell, tsz, rec.sxmin,
                             rec.idx, rec.delta, amp);
      } else {
        // scheme A, c->f part where tm is the FINER target: deposit at tm's own (fine)
        // resolution from the absolute position -- the finest level the cloud touches.
        DepositCloudNative(tmunu, g_dd, tm, is, js, ks, ncell, tsz, rec.x, amp);
      }
    });
  }

  // ---- (e) identity diagnostics (debug >= 1): sum_cells E sqrt(gamma) dV must equal
  // sum_p m W f_p to machine precision, and likewise the 9 S_d/S_dd combinations. The
  // multiply-back telescopes against the deposit divisor (same SpatialDet expression at
  // the same cell centers), so any lost/duplicated share, wrong factor or misplaced
  // write shows as an O(weight) residual. Tolerance scales with the all-positive
  // summation length (~eps*N), not a flat 1e-12, to avoid spurious failures at large N.
  if (dbg >= 1) {
    auto &csum = tmunu_csums;
    Kokkos::deep_copy(csum, 0.0);
    int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    par_for("tmunu_id_cells", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // E == 0 implies no particle deposited here (amplitudes mW are strictly positive
      // for the supported m > 0), so every component received only exact zeros: skip
      if (tmunu.E(m,k,j,i) == 0.0) {return;}
      Real detg = adm::SpatialDet(g_dd(m,0,0,k,j,i), g_dd(m,0,1,k,j,i),
                                  g_dd(m,0,2,k,j,i), g_dd(m,1,1,k,j,i),
                                  g_dd(m,1,2,k,j,i), g_dd(m,2,2,k,j,i));
      const RegionSize &sz = size.d_view(m);
      Real w = sqrt(detg)*sz.dx1*sz.dx2*sz.dx3;
      Kokkos::atomic_add(&csum(0), tmunu.E(m,k,j,i)*w);
      for (int a=0; a<3; ++a) {
        Kokkos::atomic_add(&csum(1+a), tmunu.S_d(m,a,k,j,i)*w);
        for (int b=a; b<3; ++b) {
          int c = 4 + (a*(7-a))/2 + (b-a);
          Kokkos::atomic_add(&csum(c), tmunu.S_dd(m,a,b,k,j,i)*w);
        }
      }
    });
    auto hps = Kokkos::create_mirror_view(tmunu_psums);
    auto hcs = Kokkos::create_mirror_view(tmunu_csums);
    Kokkos::deep_copy(hps, tmunu_psums);
    Kokkos::deep_copy(hcs, tmunu_csums);
    int npart_tot = npart;
    int ncross_tot = n_cross_thispack;   // global: did ANY cloud cross a level seam?
#if MPI_PARALLEL_ENABLED
    // the identity closes only GLOBALLY: a particle on one rank deposits (via an image)
    // into cells that may be owned by another. Reduce both sides and the count (the
    // tolerance scale) across ranks. Collective; debug_lvl is input-uniform so every rank
    // reaches it. This is the end-to-end transport oracle: a share lost or duplicated in
    // flight shows as a residual (a misroute is killed earlier by the recv bounds check).
    MPI_Allreduce(MPI_IN_PLACE, hps.data(), 10, MPI_ATHENA_REAL, MPI_SUM,
                  pbval_part->mpi_comm_part);
    MPI_Allreduce(MPI_IN_PLACE, hcs.data(), 10, MPI_ATHENA_REAL, MPI_SUM,
                  pbval_part->mpi_comm_part);
    MPI_Allreduce(MPI_IN_PLACE, &npart_tot, 1, MPI_INT, MPI_SUM,
                  pbval_part->mpi_comm_part);
    MPI_Allreduce(MPI_IN_PLACE, &ncross_tot, 1, MPI_INT, MPI_SUM,
                  pbval_part->mpi_comm_part);
#endif
    Real scale = 0.0, resid = 0.0;
    int cbad = -1;
    for (int c=0; c<10; ++c) {
      scale = fmax(scale, fabs(hps(c)));
      Real r = fabs(hps(c) - hcs(c));
      if (r > resid) {resid = r; cbad = c;}
    }
    Real eps = std::numeric_limits<Real>::epsilon();
    Real tol = scale*fmax(1.0e-12, 32.0*eps*static_cast<Real>(npart_tot));
    static char const * const comp[10] = {"E","Sx","Sy","Sz","Sxx","Sxy","Sxz",
                                          "Syy","Syz","Szz"};
    // EXACT-conservation regime: scheme A (conservative) restores the identity across
    // a seam, so a residual above tol is always a transport/deposit bug -- fatal, exactly
    // as Stage 4. Scheme B (native) is INTENTIONALLY O(straddle) non-conservative once a
    // cloud crosses a seam (ncross>0): then report the measured residual but do NOT abort
    // (Stage 5b README sec 2.2 / test 8). No seam (ncross==0) is exact for either scheme.
    bool exact_regime = (xl_scheme == 0) || (ncross_tot == 0);
    if (exact_regime && resid > tol) {
      if (myrank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Tmunu conservation identity violated at cycle "
                  << ncycle << " (global): worst component " << comp[cbad] << std::endl;
        for (int c=0; c<10; ++c) {
          std::cout << "  " << comp[c] << ": particle-side " << hps(c)
                    << "  cell-side " << hcs(c) << "  resid " << fabs(hps(c)-hcs(c))
                    << std::endl;
        }
        std::cout << "  npart=" << npart_tot << " tol=" << tol << std::endl;
      }
      std::cout << std::flush;
#if MPI_PARALLEL_ENABLED
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      std::exit(EXIT_FAILURE);
#endif
    }
    if (myrank == 0) {
      if (xl_scheme == 1 && ncross_tot > 0) {
        std::streamsize op = std::cout.precision(12);   // resid is verified vs the closed
        std::cout << "[tmunu-debug] cycle=" << ncycle << " npart=" << npart_tot
                  << " cross_level=" << ncross_tot     // form, so print enough digits
                  << " scheme-B non-conservation max_resid=" << resid << " (worst "
                  << comp[cbad] << "; measured, not fatal)" << std::endl;
        std::cout.precision(op);
      } else {
        // scheme A (any seam) or no seam: the identity holds exactly (asserted above).
        // cross_level reports whether a cloud spanned a seam (the test-7 oracle: >0 means
        // the conservative restrict ran and STILL conserved).
        std::cout << "[tmunu-debug] cycle=" << ncycle << " npart=" << npart_tot
                  << " cross_level=" << ncross_tot
                  << " identity max_resid=" << resid << " (tol " << tol << ") (global)"
                  << std::endl;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::SetPrtclTmunu
//! \brief Wrapper task that dispatches set_prtcl_tmunu<NGHOST> on the active ghost
//! count. Queued after EnergyCalculation when <particles> feedback = true, plus one
//! seed call from Driver::Initialize (fresh starts AND restarts -- Tmunu is derived
//! state and is not stored in restart files).

TaskStatus Particles::SetPrtclTmunu(Driver *pdrive, int stage) {
  if (pmy_pack->ptmunu == nullptr) {   // wiring error: feedback without the container
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "SetPrtclTmunu called but no Tmunu object exists"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int ng = pmy_pack->pmesh->mb_indcs.ng;
  switch (ng) {
    case 2: set_prtcl_tmunu<2>(); break;
    case 3: set_prtcl_tmunu<3>(); break;
    case 4: set_prtcl_tmunu<4>(); break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Tmunu deposition supports NGHOST=2,3,4 only (got " << ng << ")"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  // optional Entity-style digital filtering of the deposited sources (tmunu_filter.cpp;
  // <particles> tmunu_filter_passes, default 0 = no-op). Placed HERE, after the deposit
  // and its conservation identity, so it covers every deposit path through this
  // wrapper: the per-cycle task, the Driver::Initialize seed (fresh starts and
  // restarts -- the subsequent ADMConstraints refresh then sees the filtered source),
  // and the post-regrid re-deposit in mesh_refinement.cpp.
  pmy_pack->ptmunu->ApplyDigitalFilter(pmy_pack->pmesh->ncycle, debug_lvl);
  return TaskStatus::complete;
}

// explicit instantiations for the supported ghost-zone counts
template void Particles::set_prtcl_tmunu<2>();
template void Particles::set_prtcl_tmunu<3>();
template void Particles::set_prtcl_tmunu<4>();

} // namespace particles
