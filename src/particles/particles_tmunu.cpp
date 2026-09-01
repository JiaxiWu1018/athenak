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
//! GHOST-IMAGE ARCHITECTURE: the kernel writes only its target MeshBlock's physical
//! cells. Same-level images carry a source-computed CIC stencil and route by off_code.
//! Cross-level images additionally carry source/target levels and the source origin.
//! Conservative mode deposits on the finest touched level and restricts fine cells over
//! coarse leaves; native mode rebuilds the CIC stencil at each target's resolution. All
//! contributions deposit in canonical order, preserving CPU rank-count invariance.
//!
//! Conservative mode is the default and restores the exact identity across a seam.
//! Native mode remains smooth but intentionally non-conservative, so its seam residual
//! is reported rather than fatal. Dynamic AMR re-deposits Tmunu after every regrid.

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

// Deposit a cross-level image at the target block's native resolution. The particle
// position is reclassified in the target frame and the resulting stencil is clipped to
// the target's physical cells; neighboring source/sibling blocks own the dropped share.
KOKKOS_INLINE_FUNCTION
void DepositCloudNative(const Tmunu::Tmunu_vars &tmunu,
                        const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
                        int tm, int is, int js, int ks, const int ncell[3],
                        const RegionSize &tsz, const Real x[3], const Real amp[10]) {
  Real xmin[3] = {tsz.x1min, tsz.x2min, tsz.x3min};
  Real xmax[3] = {tsz.x1max, tsz.x2max, tsz.x3max};
  int cells[3][2];
  Real weight[3][2];
  int kept[3];
  for (int d=0; d<3; ++d) {
    int idx = LeftCenterIndex(x[d], ncell[d], xmin[d], xmax[d]);
    Real dx = (xmax[d] - xmin[d])/static_cast<Real>(ncell[d]);
    Real delta = fmin(fmax((x[d] - CellCenterX(idx, ncell[d], xmin[d], xmax[d]))/dx,
                           0.0), 1.0);
    kept[d] = 0;
    if (idx >= 0 && idx < ncell[d]) {
      cells[d][kept[d]] = idx;
      weight[d][kept[d]++] = 1.0 - delta;
    }
    if (idx + 1 >= 0 && idx + 1 < ncell[d]) {
      cells[d][kept[d]] = idx + 1;
      weight[d][kept[d]++] = delta;
    }
  }
  Real dv = tsz.dx1*tsz.dx2*tsz.dx3;
  for (int kk=0; kk<kept[2]; ++kk) {
    for (int jj=0; jj<kept[1]; ++jj) {
      for (int ii=0; ii<kept[0]; ++ii) {
        Real s = weight[0][ii]*weight[1][jj]*weight[2][kk];
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
            int c = 4 + (a*(7-a))/2 + (b-a);
            Kokkos::atomic_add(&tmunu.S_dd(tm,a,b,ck,cj,ci), amp[c]*fac);
          }
        }
      }
    }
  }
}

// Deposit a fine-resolution stencil into the target's covering coarse cells. Each fine
// cell contributes its integrated source exactly once, so restriction preserves the
// particle-to-cell Tmunu identity across a balanced 2:1 refinement interface.
KOKKOS_INLINE_FUNCTION
void DepositCloudRestrict(const Tmunu::Tmunu_vars &tmunu,
                          const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
                          int tm, int is, int js, int ks, const int ncell[3],
                          const RegionSize &tsz, const Real sxmin[3],
                          const int idx[3], const Real delta[3], const Real amp[10]) {
  Real xmin[3] = {tsz.x1min, tsz.x2min, tsz.x3min};
  Real xmax[3] = {tsz.x1max, tsz.x2max, tsz.x3max};
  Real dxc[3] = {tsz.dx1, tsz.dx2, tsz.dx3};
  int coarse_cell[3][2];
  Real weight[3][2];
  int kept[3];
  for (int d=0; d<3; ++d) {
    Real dxf = 0.5*dxc[d];
    kept[d] = 0;
    for (int t=0; t<2; ++t) {
      Real center = sxmin[d] + (static_cast<Real>(idx[d] + t) + 0.5)*dxf;
      if (center < xmin[d] || center >= xmax[d]) {continue;}
      int ic = static_cast<int>(floor((center - xmin[d])/dxc[d]));
      if (ic < 0) {
        ic = 0;
      } else if (ic >= ncell[d]) {
        ic = ncell[d] - 1;
      }
      coarse_cell[d][kept[d]] = ic;
      weight[d][kept[d]++] = (t == 0) ? (1.0 - delta[d]) : delta[d];
    }
  }

  Real dv = tsz.dx1*tsz.dx2*tsz.dx3;
  for (int kk=0; kk<kept[2]; ++kk) {
    for (int jj=0; jj<kept[1]; ++jj) {
      for (int ii=0; ii<kept[0]; ++ii) {
        Real s = weight[0][ii]*weight[1][jj]*weight[2][kk];
        int ci = is + coarse_cell[0][ii];
        int cj = js + coarse_cell[1][jj];
        int ck = ks + coarse_cell[2][kk];
        Real detg = adm::SpatialDet(g_dd(tm,0,0,ck,cj,ci), g_dd(tm,0,1,ck,cj,ci),
                                    g_dd(tm,0,2,ck,cj,ci), g_dd(tm,1,1,ck,cj,ci),
                                    g_dd(tm,1,2,ck,cj,ci), g_dd(tm,2,2,ck,cj,ci));
        Real fac = s/(sqrt(detg)*dv);
        Kokkos::atomic_add(&tmunu.E(tm,ck,cj,ci), amp[0]*fac);
        for (int a=0; a<3; ++a) {
          Kokkos::atomic_add(&tmunu.S_d(tm,a,ck,cj,ci), amp[1+a]*fac);
          for (int b=a; b<3; ++b) {
            int c = 4 + (a*(7-a))/2 + (b-a);
            Kokkos::atomic_add(&tmunu.S_dd(tm,a,b,ck,cj,ci), amp[c]*fac);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \struct SortTmunuImage
//! \brief canonical image order independent of generation and arrival order.

struct SortTmunuImage {
  bool operator()(const TmunuImage &a, const TmunuImage &b) const {
    if (a.target_m != b.target_m) {return a.target_m < b.target_m;}
    if (a.tag != b.tag) {return a.tag < b.tag;}
    if (a.off_code != b.off_code) {return a.off_code < b.off_code;}
    return a.lev < b.lev;
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
  int n_cross_thispack = 0;
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
        PartImageTarget target[24];
        int missing = 0, overflow = 0;
        sum += EnumerateParticleTargets(nghbr.d_view, m, mylev, beff, fx, fy, fz,
                                        px, py, pz, multi_d, three_d, target, 24,
                                        missing, overflow);
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

    // Counters 0, 2, 3, and 4 are fatal. Counter 1 records cross-level images for the
    // conservation diagnostic; counter 4 rejects an unwrapped cross-level periodic seam.
    DvceArray1D<int> derr("tmunu_err",5);   // zero-initialized

    // ---- (b2) record-generation pass: emit one self record per particle (its own-block
    // cloud) into slot p, append same-rank neighbor images beyond npart, stage cross-rank
    // neighbor images for transport, and (debug) accumulate the particle-side identity
    // sums with the boundary-clip factor f_p. Nothing is deposited here -- the single
    // canonical pass below deposits every record in rank-invariant order.
    auto &img = tmunu_images;
    auto &img_send = tmunu_img_send;
    auto &nimg_ctr = tmunu_nimg;
    // gather-scheme selector for the Lorentz-factor metric evaluation below; the CIC
    // deposit stencil itself is unaffected by this choice
    const bool tri = (interp_method == ParticleInterpMethod::trilinear);
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
      if (tri) {
        CalcTriWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      } else {
        CalcInterpWght<NGHOST>(x, mb_par, ncell, interp_indcs, Lx, Ly, Lz);
      }
      Real g3u[6];
      LagrangeInterpolator<NGHOST>(adm_n, adm::ADM::I_ADM_GXX, interp_indcs,
                                   Lx, Ly, Lz, g3u);
      Real usq = g3u[0]*u_d[0]*u_d[0] + g3u[3]*u_d[1]*u_d[1] + g3u[5]*u_d[2]*u_d[2]
               + 2.0*(g3u[1]*u_d[0]*u_d[1] + g3u[2]*u_d[0]*u_d[2]
                      + g3u[4]*u_d[1]*u_d[2]);
      Real lor = sqrt(1.0 + usq);

      Real amp[10];
      TmunuAmplitudes(mp, lor, u_d, amp);

      // Enumerate once so scheme A can detect whether the cloud touches a finer block.
      int beff[3];
      for (int d=0; d<3; ++d) {
        beff[d] = (cd[d].band != 0 && cd[d].open) ? cd[d].band : 0;
      }
      bool banded = (beff[0] != 0 || beff[1] != 0 || beff[2] != 0);
      PartImageTarget target[24];
      int ntarget = 0;
      bool touches_finer = false;
      if (banded) {
        int fx = (x[0] < 0.5*(sz.x1min + sz.x1max)) ? 0 : 1;
        int fy = (x[1] < 0.5*(sz.x2min + sz.x2max)) ? 0 : 1;
        int fz = (x[2] < 0.5*(sz.x3min + sz.x3max)) ? 0 : 1;
        int px = mbpar.d_view(m,0), py = mbpar.d_view(m,1), pz = mbpar.d_view(m,2);
        int missing = 0, overflow = 0;
        ntarget = EnumerateParticleTargets(nghbr.d_view, m, mylev, beff,
                                           fx, fy, fz, px, py, pz,
                                           multi_d, three_d, target, 24,
                                           missing, overflow);
        if (missing > 0) {
          Kokkos::atomic_add(&derr(0), missing);
          Kokkos::printf("[tmunu-debug] rank=%d cycle=%d tag=%d gid=%d NO NEIGHBOR "
                         "(missing=%d) pos=(%.16e,%.16e,%.16e)\n", myrank, ncycle,
                         pi(PTAG,p), pi(PGID,p), missing, x[0], x[1], x[2]);
        }
        if (overflow) {Kokkos::atomic_add(&derr(3), 1);}
        for (int s=0; s<ntarget; ++s) {
          if (nghbr.d_view(m, target[s].slot).lev > mylev) {touches_finer = true;}
        }
      }

      // A coarse source touching a finer neighbor represents its whole cloud on the fine
      // sublevel. The portion remaining over coarse leaves is then restricted back.
      bool use_fine_stencil = (xl_scheme == 0) && touches_finer;
      int fine_idx[3] = {0, 0, 0};
      Real fine_delta[3] = {0.0, 0.0, 0.0};
      Real source_min[3] = {sz.x1min, sz.x2min, sz.x3min};
      if (use_fine_stencil) {
        Real source_max[3] = {sz.x1max, sz.x2max, sz.x3max};
        for (int d=0; d<3; ++d) {
          int nfine = 2*ncell[d];
          fine_idx[d] = LeftCenterIndex(x[d], nfine, source_min[d], source_max[d]);
          Real dxf = (source_max[d] - source_min[d])/static_cast<Real>(nfine);
          fine_delta[d] =
              fmin(fmax((x[d] - CellCenterX(fine_idx[d], nfine, source_min[d],
                                            source_max[d]))/dxf, 0.0), 1.0);
        }
      }

      // Self records normally use the same-level stencil. Under conservative c->f,
      // restrict the fine representation back into the source block's coarse cells.
      {
        TmunuImage self;
        self.target_m = m;
        self.tag = pi(PTAG,p);
        self.off_code = 13;
        for (int d=0; d<3; ++d) {
          self.x[d] = x[d];
          self.sxmin[d] = source_min[d];
          self.u_d[d] = u_d[d];
        }
        self.mass = mp;
        self.lorentz = lor;
        if (use_fine_stencil) {
          self.lev = mylev;
          self.slev = mylev + 1;
          for (int d=0; d<3; ++d) {
            self.idx[d] = fine_idx[d];
            self.delta[d] = fine_delta[d];
          }
        } else {
          self.lev = -1;
          self.slev = -1;
          for (int d=0; d<3; ++d) {
            self.idx[d] = idx[d];
            self.delta[d] = dlt[d];
          }
        }
        img.d_view(p) = self;
      }

      // Emit one record per enumerated target. The target/source level pair selects
      // same-level routing, target-native deposition, or conservative restriction.
      if (banded) {
        for (int s=0; s<ntarget; ++s) {
          const NeighborBlock &nb = nghbr.d_view(m, target[s].slot);
          int oc = target[s].oc;
          int record_level, source_level, record_idx[3];
          Real record_delta[3];
          if (xl_scheme == 1) {
            record_level = (nb.lev == mylev) ? -1 : nb.lev;
            source_level = mylev;
            for (int d=0; d<3; ++d) {
              record_idx[d] = idx[d];
              record_delta[d] = dlt[d];
            }
          } else if (nb.lev > mylev) {
            record_level = nb.lev;
            source_level = mylev;
            for (int d=0; d<3; ++d) {
              record_idx[d] = idx[d];
              record_delta[d] = dlt[d];
            }
          } else if (nb.lev < mylev) {
            record_level = nb.lev;
            source_level = mylev;
            for (int d=0; d<3; ++d) {
              record_idx[d] = idx[d];
              record_delta[d] = dlt[d];
            }
          } else if (use_fine_stencil) {
            record_level = mylev;
            source_level = mylev + 1;
            for (int d=0; d<3; ++d) {
              record_idx[d] = fine_idx[d];
              record_delta[d] = fine_delta[d];
            }
          } else {
            record_level = -1;
            source_level = -1;
            for (int d=0; d<3; ++d) {
              record_idx[d] = idx[d];
              record_delta[d] = dlt[d];
            }
          }
          if (record_level >= 0) {
            Kokkos::atomic_add(&derr(1), 1);
            // Absolute positions are not shifted by a domain length in cross-level
            // records, so rebuilding a stencil in a periodic wrap neighbor would target
            // the wrong coordinates. Reject that unsupported geometry explicitly.
            int bx = oc % 3 - 1;
            int by = (oc / 3) % 3 - 1;
            int bz = oc / 9 - 1;
            if ((bx < 0 && mbbcs.d_view(m,0) == BoundaryFlag::periodic) ||
                (bx > 0 && mbbcs.d_view(m,1) == BoundaryFlag::periodic) ||
                (by < 0 && mbbcs.d_view(m,2) == BoundaryFlag::periodic) ||
                (by > 0 && mbbcs.d_view(m,3) == BoundaryFlag::periodic) ||
                (bz < 0 && mbbcs.d_view(m,4) == BoundaryFlag::periodic) ||
                (bz > 0 && mbbcs.d_view(m,5) == BoundaryFlag::periodic)) {
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
            rec.lev = record_level;
            rec.slev = source_level;
            for (int d=0; d<3; ++d) {
              rec.idx[d] = record_idx[d];
              rec.delta[d] = record_delta[d];
              rec.x[d] = x[d];
              rec.sxmin[d] = source_min[d];
              rec.u_d[d] = u_d[d];
            }
            rec.mass = mp;
            rec.lorentz = lor;
            img.d_view(slot) = rec;
          } else {
            // cross-rank neighbor: stage a wire record for the MPI transport. The target
            // is named by its GLOBAL gid (the receiver converts it to a local index and
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
            w.lev = record_level;
            w.slev = source_level;
            for (int d=0; d<3; ++d) {
              w.idx[d] = record_idx[d];
              w.delta[d] = record_delta[d];
              w.x[d] = x[d];
              w.sxmin[d] = source_min[d];
              w.u_d[d] = u_d[d];
            }
            w.mass = mp;
            w.lorentz = lor;
            img_send.d_view(slot) = w;
          }
        }
      }

      // Closed-boundary clipping must use the same resolution as the deposited stencil.
      if (dbg >= 1) {
        Real f = 1.0;
        for (int d=0; d<3; ++d) {
          if (cd[d].band != 0 && !cd[d].open) {
            if (use_fine_stencil) {
              int nfine = 2*ncell[d];
              f *= (cd[d].band < 0)
                       ? ((fine_idx[d] == -1) ? fine_delta[d] : 1.0)
                       : ((fine_idx[d] + 1 == nfine) ? (1.0 - fine_delta[d]) : 1.0);
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
    n_cross_thispack = herr(1);
    if (herr(4) > 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "cross-level Tmunu deposition through a periodic "
                << "boundary is unsupported: " << herr(4) << " image(s) at cycle "
                << ncycle << ". Keep refinement away from periodic faces or use "
                << "non-periodic boundaries in refined directions." << std::endl
                << std::flush;
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
  // received cross-rank images) on host by (target_m,tag,off_code,lev). The key is total
  // and tag is globally unique, so per-cell accumulation order is identical for every
  // rank decomposition (the Stage-4c bitwise rank-invariance criterion). A duplicate key
  // means duplicate particle tags (an init=pgen contract violation) or a generation bug:
  // fatal.
  if (nimages_thispack > 0) {
    tmunu_images.template modify<DevExeSpace>();
    tmunu_images.template sync<HostMemSpace>();
    TmunuImage *ibeg = tmunu_images.h_view.data();
    std::sort(ibeg, ibeg + nimages_thispack, SortTmunuImage());
    for (int g=1; g<nimages_thispack; ++g) {
      bool same_target_tag = (ibeg[g-1].target_m == ibeg[g].target_m &&
                              ibeg[g-1].tag == ibeg[g].tag);
      bool cross_level_duplicate = same_target_tag &&
                                   (ibeg[g-1].lev >= 0 || ibeg[g].lev >= 0);
      bool same_level_duplicate = same_target_tag && ibeg[g-1].lev < 0 &&
                                  ibeg[g].lev < 0 &&
                                  ibeg[g-1].off_code == ibeg[g].off_code;
      if (cross_level_duplicate || same_level_duplicate) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << (cross_level_duplicate ? "duplicate cross-level Tmunu target: "
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
        int off[3];
        off[0] = rec.off_code % 3 - 1;
        off[1] = (rec.off_code / 3) % 3 - 1;
        off[2] = rec.off_code / 9 - 1;
        Real dv = tsz.dx1*tsz.dx2*tsz.dx3;
        DepositCloud(tmunu, g_dd, tm, is, js, ks, ncell, dv, off, rec.idx, rec.delta,
                     amp);
      } else if (xl_scheme == 1) {
        DepositCloudNative(tmunu, g_dd, tm, is, js, ks, ncell, tsz, rec.x, amp);
      } else if (rec.slev > rec.lev) {
        DepositCloudRestrict(tmunu, g_dd, tm, is, js, ks, ncell, tsz, rec.sxmin,
                             rec.idx, rec.delta, amp);
      } else {
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
    int ncross_tot = n_cross_thispack;
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
    int cbad = 0;
    for (int c=0; c<10; ++c) {
      scale = fmax(scale, fabs(hps(c)));
      Real r = fabs(hps(c) - hcs(c));
      if (r > resid) {resid = r; cbad = c;}
    }
    Real eps = std::numeric_limits<Real>::epsilon();
    Real tol = scale*fmax(1.0e-12, 32.0*eps*static_cast<Real>(npart_tot));
    static char const * const comp[10] = {"E","Sx","Sy","Sz","Sxx","Sxy","Sxz",
                                          "Syy","Syz","Szz"};
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
        std::streamsize old_precision = std::cout.precision(12);
        std::cout << "[tmunu-debug] cycle=" << ncycle << " npart=" << npart_tot
                  << " cross_level=" << ncross_tot
                  << " scheme-B non-conservation max_resid=" << resid << " (worst "
                  << comp[cbad] << "; measured, not fatal)" << std::endl;
        std::cout.precision(old_precision);
      } else {
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
