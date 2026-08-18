#ifndef BVALS_PRTCL_SEARCH_HPP_
#define BVALS_PRTCL_SEARCH_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prtcl_search.hpp
//! \brief destination-MeshBlock search for particle migration (NRPIC Stage 3a(c)).
//!
//! Given the integer offsets (ix,iy,iz) of a particle that left its MeshBlock, find the
//! index in the 56-slot nghbr array (mesh/nghbr_index.hpp) of the neighbor that contains
//! it, by DIRECT indexed lookups only -- never by walking slots, which is what the legacy
//! code did and what produced the Stage-3a(b) failure catalog (walks exited empty slot
//! groups onto unrelated neighbors; empty corner slots had no fallback at all).
//!
//! The lookups follow the population contract of MeshBlock::SetNeighbors:
//!  - same-level neighbor: subslot 0 of its face/edge/corner group;
//!  - FINER neighbors: every subslot populated; the correct one is selected by which
//!    half of the block the particle sits in transversely (fx/fy/fz bits);
//!  - COARSER neighbor: stored at the subslot given by THIS block's parity within its
//!    parent (px,py,pz = lloc.lx&1, available on device as MeshBlock::mb_parity), with
//!    the subslot axes:  x1 face (myfx2,myfx3) | x2 face (myfx1,myfx3) |
//!    x3 face (myfx1,myfx2) | x1x2 edge myfx3 | x3x1 edge myfx2 | x2x3 edge myfx1;
//!  - EXTERIOR-EDGE/CORNER RULE: a coarser diagonal neighbor is stored at an edge/corner
//!    slot only when this block sits at the matching exterior edge/corner of its parent
//!    (myox == offset for every offset direction, myox = 2*parity-1). Otherwise the slot
//!    is EMPTY and the region beyond the edge/corner is covered by a coarser FACE (or
//!    edge) neighbor -- handled here by DEMOTION:
//!      a coarser face neighbor in direction d covers a diagonal region only on the side
//!      where the parent extends past this block, i.e. for every nonzero transverse
//!      offset i_t the COVERAGE CONDITION  i_t == -myox_t  must hold. A coarser edge
//!      neighbor (corner demotion) spanning (d1,d2) covers the corner iff the remaining
//!      offset i_3 == -myox_3.
//!    With 2:1 balance (enforced by MeshBlockTree::Refine across faces, edges, AND
//!    corners, incl. periodic wrap) exactly one demotion candidate passes whenever
//!    demotion is legitimately reached: a coarser neighbor in direction d implies
//!    myox_d == d (otherwise the d-neighbor would be a same-level sibling), so two
//!    coarse faces force the exterior-corner case in which the diagonal slot was
//!    populated in the first place (see the stage-3 README section 4.1).
//!
//! Returns -1 when no destination exists; the caller must NOT update the particle GID.
//! Since Stage 3c, particles exiting the mesh through a physical boundary are destroyed
//! BEFORE the search runs (ExitsMeshBoundary below), so a -1 for a surviving particle
//! is unconditionally a logic/balance/overspeed error, which the <particles> debug
//! validation makes fatal.
//!
//! All functions are templated on the nghbr view type so the same code runs in device
//! kernels (pass nghbr.d_view) and in the host enumeration audit (pass nghbr.h_view).

#include "athena.hpp"
#include "mesh/mesh.hpp"        // RegionSize
#include "mesh/nghbr_index.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn void ComputeBlockOffsets()
//! \brief integer offset (ix,iy,iz), each in {-2,-1,0,1,2}, of position (x1,x2,x3)
//! relative to MeshBlock bbox sz, by DIRECT COMPARISON with the same half-open [min,max)
//! predicates that define block ownership everywhere else (FindContainingMeshBlock,
//! CheckMigration containment). Arithmetic forms like floor((x-xmin)/lx) are NOT exactly
//! consistent with those predicates: when x sits within half an ulp of lx BELOW a
//! boundary, the subtraction rounds up and classifies a crossing that ownership denies
//! (found by the Stage-3a(c) lattice soak on the nested grid, where x0 + n*dt
//! accumulation landed at -3.5e-18 and was sent to the x>=0 block). Comparisons also
//! DETECT a particle more than one block width away (|offset| = 2 fails the destination
//! search) instead of mislabeling it. |offset| = 2 stands for "2 or more".
//! This is the SINGLE definition of the crossing predicate: the migration kernel (both
//! the counting and the search/fill passes) and the host enumeration audit must all call
//! this -- never hand-copy the comparisons -- so they cannot drift apart.
//! In 2D (three_d false) x3 is ignored and iz = 0 (the trimmed particle layout has no z).

KOKKOS_INLINE_FUNCTION
void ComputeBlockOffsets(const RegionSize &sz, Real x1, Real x2, Real x3, bool three_d,
                         int &ix, int &iy, int &iz) {
  Real lx = (sz.x1max - sz.x1min);
  Real ly = (sz.x2max - sz.x2min);
  Real lz = (sz.x3max - sz.x3min);
  ix = 0; iy = 0; iz = 0;
  if (x1 <  sz.x1min) {
    ix = (x1 <  sz.x1min - lx) ? -2 : -1;
  } else if (x1 >= sz.x1max) {
    ix = (x1 >= sz.x1max + lx) ?  2 :  1;
  }
  if (x2 <  sz.x2min) {
    iy = (x2 <  sz.x2min - ly) ? -2 : -1;
  } else if (x2 >= sz.x2max) {
    iy = (x2 >= sz.x2max + ly) ?  2 :  1;
  }
  if (three_d) {
    if (x3 <  sz.x3min) {
      iz = (x3 <  sz.x3min - lz) ? -2 : -1;
    } else if (x3 >= sz.x3max) {
      iz = (x3 >= sz.x3max + lz) ?  2 :  1;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool ExitsMeshBoundary()
//! \brief true iff a crossing with block offsets (ix,iy,iz) leaves the computational
//! domain through a NON-periodic mesh boundary of block m -- the destruction predicate
//! (reason=exit) of Stage 3c. Decided per direction from this block's own face flags
//! (MeshBlock::mb_bcs, faces 0..5 = inner/outer x1,x2,x3): `block` means the face is
//! interior (a neighbor exists), `periodic` means the crossing wraps; ANY other flag
//! (outflow, diode, vacuum, user, and the unimplemented-for-particles reflect/inflow)
//! destroys. A single physical direction kills a mixed corner/edge exit -- the periodic
//! or interior components do not save it. |offset| >= 2 toward a physical face also
//! exits (the particle is already outside the mesh on that side). Like
//! ComputeBlockOffsets this is the SINGLE definition of the predicate: the counting
//! pass and the marking pass must both call it so capacity == appends by construction.
//! Templated on the mb_bcs view type (device kernels pass d_view, host audits h_view).

template <typename BcsView>
KOKKOS_INLINE_FUNCTION
bool ExitsMeshBoundary(const BcsView &bcs, int m, int ix, int iy, int iz) {
  if (ix != 0) {
    BoundaryFlag f = bcs(m, (ix < 0) ? 0 : 1);
    if (f != BoundaryFlag::block && f != BoundaryFlag::periodic) {return true;}
  }
  if (iy != 0) {
    BoundaryFlag f = bcs(m, (iy < 0) ? 2 : 3);
    if (f != BoundaryFlag::block && f != BoundaryFlag::periodic) {return true;}
  }
  if (iz != 0) {
    BoundaryFlag f = bcs(m, (iz < 0) ? 4 : 5);
    if (f != BoundaryFlag::block && f != BoundaryFlag::periodic) {return true;}
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn int CoarseProbe()
//! \brief accept slot indx only if it holds a COARSER neighbor (gid set, lev < mylev)

template <typename NghbrView>
KOKKOS_INLINE_FUNCTION
int CoarseProbe(const NghbrView &ngh, int m, int mylev, int indx) {
  if (ngh(m,indx).gid >= 0 && ngh(m,indx).lev < mylev) {return indx;}
  return -1;
}

//----------------------------------------------------------------------------------------
//! \fn int FaceIndex()
//! \brief destination slot for a face crossing with offsets (ix,iy,iz) (one nonzero),
//! position-half bits (f1,f2) and parity bits (p1,p2) on the two transverse axes in
//! ascending order. Returns -1 only at a physical boundary (no neighbor in the group).

template <typename NghbrView>
KOKKOS_INLINE_FUNCTION
int FaceIndex(const NghbrView &ngh, int m, int mylev, int ix, int iy, int iz,
              int f1, int f2, int p1, int p2) {
  int indx = NeighborIndex(ix,iy,iz,0,0);
  if (ngh(m,indx).lev > mylev) {            // finer: subslot from the position halves
    return NeighborIndex(ix,iy,iz,f1,f2);
  }
  if (ngh(m,indx).gid >= 0) {return indx;}  // same level (or coarse with parity (0,0))
  return CoarseProbe(ngh,m,mylev, NeighborIndex(ix,iy,iz,p1,p2));   // coarser
}

//----------------------------------------------------------------------------------------
//! \fn int EdgeIndex()
//! \brief destination slot for an edge crossing (two nonzero offsets); f1/p1 are the
//! position-half/parity bit of the single transverse axis. Returns -1 when the edge
//! region is covered by a coarser FACE neighbor (exterior-edge rule) -- the caller
//! demotes -- or at a physical boundary.

template <typename NghbrView>
KOKKOS_INLINE_FUNCTION
int EdgeIndex(const NghbrView &ngh, int m, int mylev, int ix, int iy, int iz,
              int f1, int p1) {
  int indx = NeighborIndex(ix,iy,iz,0,0);
  if (ngh(m,indx).lev > mylev) {            // finer: sub-edge from the position half
    return NeighborIndex(ix,iy,iz,f1,0);
  }
  if (ngh(m,indx).gid >= 0) {return indx;}  // same level (or coarse with parity 0)
  return CoarseProbe(ngh,m,mylev, NeighborIndex(ix,iy,iz,p1,0));    // exterior coarse
}

//----------------------------------------------------------------------------------------
//! \fn int FindDestinationIndex()
//! \brief nghbr-array index of the MeshBlock a particle with block-relative integer
//! offsets (ix,iy,iz) belongs to, or -1 (see file docstring). (fx,fy,fz) = which half of
//! the block the particle occupies per dimension; (px,py,pz) = the block's parity bits.

template <typename NghbrView>
KOKKOS_INLINE_FUNCTION
int FindDestinationIndex(const NghbrView &ngh, int m, int mylev,
                         int ix, int iy, int iz, int fx, int fy, int fz,
                         int px, int py, int pz) {
  int d = abs(ix) + abs(iy) + abs(iz);
  int ox = 2*px - 1, oy = 2*py - 1, oz = 2*pz - 1;
  int indx = -1;

  if (d == 1) {                             // ---- faces (no demotion possible)
    if (ix != 0) {return FaceIndex(ngh,m,mylev, ix,0,0, fy,fz, py,pz);}
    if (iy != 0) {return FaceIndex(ngh,m,mylev, 0,iy,0, fx,fz, px,pz);}
    return FaceIndex(ngh,m,mylev, 0,0,iz, fx,fy, px,py);

  } else if (d == 2) {                      // ---- edges
    if (iz == 0) {                          // x1x2 edge
      indx = EdgeIndex(ngh,m,mylev, ix,iy,0, fz, pz);
      if (indx >= 0) {return indx;}
      // demote to the coarser covering face (coverage: i_transverse == -myox)
      if (iy == -oy) {indx = CoarseProbe(ngh,m,mylev, NeighborIndex(ix,0,0,py,pz));}
      if (indx < 0 && ix == -ox) {
        indx = CoarseProbe(ngh,m,mylev, NeighborIndex(0,iy,0,px,pz));
      }
    } else if (iy == 0) {                   // x3x1 edge
      indx = EdgeIndex(ngh,m,mylev, ix,0,iz, fy, py);
      if (indx >= 0) {return indx;}
      if (iz == -oz) {indx = CoarseProbe(ngh,m,mylev, NeighborIndex(ix,0,0,py,pz));}
      if (indx < 0 && ix == -ox) {
        indx = CoarseProbe(ngh,m,mylev, NeighborIndex(0,0,iz,px,py));
      }
    } else {                                // x2x3 edge
      indx = EdgeIndex(ngh,m,mylev, 0,iy,iz, fx, px);
      if (indx >= 0) {return indx;}
      if (iz == -oz) {indx = CoarseProbe(ngh,m,mylev, NeighborIndex(0,iy,0,px,pz));}
      if (indx < 0 && iy == -oy) {
        indx = CoarseProbe(ngh,m,mylev, NeighborIndex(0,0,iz,px,py));
      }
    }
    return indx;
  }

  // ---- corner (d == 3): single slot holds same-level, finer, or exterior-coarse
  indx = NeighborIndex(ix,iy,iz,0,0);
  if (ngh(m,indx).gid >= 0) {return indx;}
  // demote to a coarser covering EDGE (coverage: remaining offset == -myox)
  indx = -1;
  if (iz == -oz) {indx = CoarseProbe(ngh,m,mylev, NeighborIndex(ix,iy,0,pz,0));}
  if (indx < 0 && iy == -oy) {
    indx = CoarseProbe(ngh,m,mylev, NeighborIndex(ix,0,iz,py,0));
  }
  if (indx < 0 && ix == -ox) {
    indx = CoarseProbe(ngh,m,mylev, NeighborIndex(0,iy,iz,px,0));
  }
  if (indx >= 0) {return indx;}
  // demote to a coarser covering FACE (coverage in BOTH transverse directions)
  if (iy == -oy && iz == -oz) {
    indx = CoarseProbe(ngh,m,mylev, NeighborIndex(ix,0,0,py,pz));
  }
  if (indx < 0 && ix == -ox && iz == -oz) {
    indx = CoarseProbe(ngh,m,mylev, NeighborIndex(0,iy,0,px,pz));
  }
  if (indx < 0 && ix == -ox && iy == -oy) {
    indx = CoarseProbe(ngh,m,mylev, NeighborIndex(0,0,iz,px,py));
  }
  return indx;
}

// Enumerate the neighbor slots that receive one cloud overhang. A same-level or
// coarser neighbor is a single target. A finer face/edge can expose up to four/two
// children, all of which must clip the source cloud in their native frame.
template <typename NghbrView>
KOKKOS_INLINE_FUNCTION
int EnumerateImageTargets(const NghbrView &ngh, int m, int mylev,
                          int ox, int oy, int oz, int fx, int fy, int fz,
                          int px, int py, int pz, bool multi_d, bool three_d,
                          int slots[4]) {
  int indx = FindDestinationIndex(ngh, m, mylev, ox, oy, oz, fx, fy, fz, px, py, pz);
  if (indx < 0) {return 0;}
  if (ngh(m, indx).lev <= mylev) {slots[0] = indx; return 1;}

  int refx = 2, refy = multi_d ? 2 : 1, refz = three_d ? 2 : 1;
  int d = abs(ox) + abs(oy) + abs(oz);
  int ns = 0;
  if (d == 1) {
    int r1, r2;
    if (ox != 0) {
      r1 = refy; r2 = refz;
    } else if (oy != 0) {
      r1 = refx; r2 = refz;
    } else {
      r1 = refx; r2 = refy;
    }
    for (int s2=0; s2<r2; ++s2) {
      for (int s1=0; s1<r1; ++s1) {
        int sl = NeighborIndex(ox, oy, oz, s1, s2);
        if (ngh(m, sl).gid >= 0 && ngh(m, sl).lev > mylev) {slots[ns++] = sl;}
      }
    }
  } else if (d == 2) {
    int r1 = (oz == 0) ? refz : ((oy == 0) ? refy : refx);
    for (int s1=0; s1<r1; ++s1) {
      int sl = NeighborIndex(ox, oy, oz, s1, 0);
      if (ngh(m, sl).gid >= 0 && ngh(m, sl).lev > mylev) {slots[ns++] = sl;}
    }
  } else {
    int sl = NeighborIndex(ox, oy, oz, 0, 0);
    if (ngh(m, sl).gid >= 0 && ngh(m, sl).lev > mylev) {slots[ns++] = sl;}
  }
  return ns;
}

struct PartImageTarget {
  int slot;
  int oc;
};

// Build the complete per-particle target list. Same-level offsets remain distinct;
// cross-level targets are unique by gid because a native deposit covers the whole
// target-clipped cloud and would otherwise be counted twice after diagonal demotion.
template <typename NghbrView>
KOKKOS_INLINE_FUNCTION
int EnumerateParticleTargets(const NghbrView &ngh, int m, int mylev, const int beff[3],
                             int fx, int fy, int fz, int px, int py, int pz,
                             bool multi_d, bool three_d, PartImageTarget *out,
                             int out_cap, int &n_missing, int &overflow) {
  constexpr int MAXSEEN = 24;
  int seen[MAXSEEN];
  int n_seen = 0;
  int n = 0;
  n_missing = 0;
  overflow = 0;
  for (int code=1; code<8; ++code) {
    int sx = code & 1, sy = (code >> 1) & 1, sz = (code >> 2) & 1;
    if ((sx && beff[0] == 0) || (sy && beff[1] == 0) || (sz && beff[2] == 0)) {continue;}
    int ox = sx ? beff[0] : 0;
    int oy = sy ? beff[1] : 0;
    int oz = sz ? beff[2] : 0;
    int oc = (ox + 1) + 3*(oy + 1) + 9*(oz + 1);
    int slots[4];
    int ns = EnumerateImageTargets(ngh, m, mylev, ox, oy, oz, fx, fy, fz, px, py, pz,
                                   multi_d, three_d, slots);
    if (ns == 0) {++n_missing; continue;}
    for (int s=0; s<ns; ++s) {
      int slot = slots[s];
      if (ngh(m, slot).lev != mylev) {
        int gid = ngh(m, slot).gid;
        bool duplicate = false;
        for (int t=0; t<n_seen; ++t) {
          if (seen[t] == gid) {duplicate = true; break;}
        }
        if (duplicate) {continue;}
        if (n_seen < MAXSEEN) {seen[n_seen++] = gid;} else {overflow = 1;}
      }
      if (n < out_cap) {
        out[n].slot = slot;
        out[n].oc = oc;
        ++n;
      } else {
        overflow = 1;
      }
    }
  }
  return n;
}

} // namespace particles
#endif // BVALS_PRTCL_SEARCH_HPP_
