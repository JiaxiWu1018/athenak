//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_debug.cpp
//! \brief migration debug instrumentation (<particles> debug = 1|2): CheckMigration is a
//! post-migration validation task that runs after the particle communication tasks each
//! cycle and verifies the invariants the migration machinery must preserve:
//!   (1) every particle's PGID refers to a MeshBlock of this pack: PGID in [gids, gide];
//!   (2) every particle's position lies inside its MeshBlock's bounding box, with the
//!       half-open [min,max) ownership convention (a particle exactly on the max edge
//!       belongs to the neighbor);
//!   (3) no destination-search failures were recorded by SetNewPrtclGID;
//!   (4) the particle count is conserved (no destruction is implemented yet, so on a
//!       single rank the count must stay exactly equal to its initial value; the
//!       multi-rank sum check is a Stage-3 session-B extension).
//! Any violation is FATAL: every offending particle is printed (tag, gid, position,
//! velocity, owning-block bbox), then the code exits nonzero. A per-cycle summary of
//! face/edge/corner crossings (counted in SetNewPrtclGID) is printed when nonzero.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "bvals/prtcl_search.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::CheckMigration
//! \brief validate the post-migration particle state (no-op unless debug >= 1)

TaskStatus Particles::CheckMigration(Driver *pdrive, int stage) {
  if (debug_lvl < 1) {return TaskStatus::complete;}

  // capture the initial particle count lazily (first call), so this works for any init
  // path (ppc/file/pgen-filled) and across restarts
  if (nprtcl_initial < 0) {nprtcl_initial = nprtcl_thispack;}

  int ncycle = pmy_pack->pmesh->ncycle;
  int npart = nprtcl_thispack;

  // per-cycle migration summary (counters filled by SetNewPrtclGID this cycle)
  if ((nmigr_face + nmigr_edge + nmigr_corner + nsearch_fail) > 0) {
    std::cout << "[prtcl-debug] cycle=" << ncycle << " migrations: face=" << nmigr_face
              << " edge=" << nmigr_edge << " corner=" << nmigr_corner
              << " search_fail=" << nsearch_fail << " npart=" << npart << std::endl;
  }

  // validation pass: count GID-range and bbox-containment violations
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &size = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  int gide = pmy_pack->gide;
  bool three_d = pmy_pack->pmesh->three_d;

  int nbad_gid = 0, nbad_box = 0;
  Kokkos::parallel_reduce("part_check",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &bad_gid, int &bad_box) {
      int gid = pi(PGID,p);
      if (gid < gids || gid > gide) {
        bad_gid += 1;
      } else {
        const RegionSize &sz = size.d_view(gid - gids);
        bool in = (pr(IPX,p) >= sz.x1min) && (pr(IPX,p) < sz.x1max)
               && (pr(IPY,p) >= sz.x2min) && (pr(IPY,p) < sz.x2max);
        if (three_d) {
          in = in && (pr(IPZ,p) >= sz.x3min) && (pr(IPZ,p) < sz.x3max);
        }
        if (!in) {bad_box += 1;}
      }
    }, Kokkos::Sum<int>(nbad_gid), Kokkos::Sum<int>(nbad_box));

  // count conservation (exact, single rank only: cross-rank sends change the per-rank
  // count legitimately; the global-sum check is added with the multi-rank session)
  bool bad_count = (global_variable::nranks == 1) && (npart != nprtcl_initial);

  if ((nbad_gid + nbad_box + nsearch_fail) > 0 || bad_count) {
    // print every offending particle, then die
    par_for("part_check_dump",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
      int gid = pi(PGID,p);
      Real x3 = three_d ? pr(IPZ,p)  : 0.0;
      Real v3 = three_d ? pr(IPVZ,p) : 0.0;
      if (gid < gids || gid > gide) {
        Kokkos::printf("[prtcl-debug] BAD GID: tag=%d gid=%d (pack range %d..%d) "
                       "pos=(%.16e,%.16e,%.16e) vel=(%.16e,%.16e,%.16e)\n",
                       pi(PTAG,p), gid, gids, gide,
                       pr(IPX,p), pr(IPY,p), x3, pr(IPVX,p), pr(IPVY,p), v3);
      } else {
        const RegionSize &sz = size.d_view(gid - gids);
        bool in = (pr(IPX,p) >= sz.x1min) && (pr(IPX,p) < sz.x1max)
               && (pr(IPY,p) >= sz.x2min) && (pr(IPY,p) < sz.x2max);
        if (three_d) {
          in = in && (pr(IPZ,p) >= sz.x3min) && (pr(IPZ,p) < sz.x3max);
        }
        if (!in) {
          Kokkos::printf("[prtcl-debug] OUT OF BBOX: tag=%d gid=%d "
                         "pos=(%.16e,%.16e,%.16e) vel=(%.16e,%.16e,%.16e) "
                         "bbox x1=[%.16e,%.16e) x2=[%.16e,%.16e) x3=[%.16e,%.16e)\n",
                         pi(PTAG,p), gid,
                         pr(IPX,p), pr(IPY,p), x3, pr(IPVX,p), pr(IPVY,p), v3,
                         sz.x1min, sz.x1max, sz.x2min, sz.x2max, sz.x3min, sz.x3max);
        }
      }
    });
    Kokkos::fence();

    // host-side reverse lookup: for every offender, report the local MeshBlock that
    // actually contains its position (the correct destination), turning each failure
    // into a (wrong gid -> right gid) pair for the migration failure catalog
    auto hr = Kokkos::create_mirror_view(pr);
    auto hi = Kokkos::create_mirror_view(pi);
    Kokkos::deep_copy(hr, pr);
    Kokkos::deep_copy(hi, pi);
    for (int p=0; p<npart; ++p) {
      int gid = hi(PGID,p);
      bool bad = (gid < gids || gid > gide);
      if (!bad) {
        const RegionSize &sz = size.h_view(gid - gids);
        bad = !((hr(IPX,p) >= sz.x1min) && (hr(IPX,p) < sz.x1max)
             && (hr(IPY,p) >= sz.x2min) && (hr(IPY,p) < sz.x2max));
        if (three_d && !bad) {
          bad = !((hr(IPZ,p) >= sz.x3min) && (hr(IPZ,p) < sz.x3max));
        }
      }
      if (bad) {
        Real z = three_d ? hr(IPZ,p) : 0.0;
        int mok = FindContainingMeshBlock(hr(IPX,p), hr(IPY,p), z);
        if (mok >= 0) {
          std::cout << "[prtcl-debug] tag=" << hi(PTAG,p) << " has gid=" << gid
                    << " but should be gid=" << gids + mok << std::endl;
        } else {
          std::cout << "[prtcl-debug] tag=" << hi(PTAG,p) << " has gid=" << gid
                    << " but no local MeshBlock contains its position" << std::endl;
        }
      }
    }

    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "particle migration check failed at cycle " << ncycle
              << " (rank " << global_variable::my_rank << "): bad_gid=" << nbad_gid
              << " out_of_bbox=" << nbad_box << " search_fail=" << nsearch_fail;
    if (bad_count) {
      std::cout << " count=" << npart << " (initial " << nprtcl_initial << ")";
    }
    std::cout << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::AuditDestinationSearch()
//! \brief exhaustive host-side enumeration audit of the migration destination search.
//! For every local MeshBlock, probe points are generated on/around every face, edge, and
//! corner -- transverse fractions {0, 1/4, 1/2, 3/4, 1} x outward distances
//! {0 (exactly ON the boundary), 1e-6 dx, 0.4 dx} -- and the full kernel pipeline is
//! mirrored on the host (comparison-based offsets identical to the ownership predicates,
//! half bits, parity -> FindDestinationIndex). The result is compared against the ground
//! truth: a brute-force scan of every block's bbox (FindContainingMeshBlock) on the
//! periodically wrapped probe. Any mismatch is printed
//! and the audit is FATAL. This is a proof by enumeration of the search for the given
//! grid, independent of particle dynamics. Single rank + strictly periodic only.

void Particles::AuditDestinationSearch() {
  Mesh *pm = pmy_pack->pmesh;
  if (global_variable::nranks != 1 || !(pm->strictly_periodic)) {
    std::cout << "[prtcl-audit] SKIPPED (requires 1 rank and strictly periodic mesh)"
              << std::endl;
    return;
  }
  bool three_d = pm->three_d;
  auto &size  = pmy_pack->pmb->mb_size;
  auto &ngh   = pmy_pack->pmb->nghbr;
  auto &mbpar = pmy_pack->pmb->mb_parity;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int nmb  = pmy_pack->nmb_thispack;
  int gids = pmy_pack->gids;
  auto &ms = pm->mesh_size;

  const Real frac[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
  const int nfrac = 5, ndist = 3;
  std::int64_t nprobe = 0;
  int nbad = 0;

  for (int m=0; m<nmb; ++m) {
    const RegionSize &sz = size.h_view(m);
    Real bmin[3] = {sz.x1min, sz.x2min, sz.x3min};
    Real bmax[3] = {sz.x1max, sz.x2max, sz.x3max};
    Real dmin = std::fmin(sz.dx1, sz.dx2);
    if (three_d) {dmin = std::fmin(dmin, sz.dx3);}
    const Real dist[3] = {0.0, 1.0e-6*dmin, 0.4*dmin};
    int mylevel = mblev.h_view(m);
    int par[3] = {mbpar.h_view(m,0), mbpar.h_view(m,1), mbpar.h_view(m,2)};

    for (int iz0=-1; iz0<=1; ++iz0) {
      if (!three_d && iz0 != 0) {continue;}
      for (int iy0=-1; iy0<=1; ++iy0) {
        for (int ix0=-1; ix0<=1; ++ix0) {
          int off0[3] = {ix0, iy0, iz0};
          if (abs(ix0) + abs(iy0) + abs(iz0) == 0) {continue;}
          for (int kd=0; kd<ndist; ++kd) {
            for (int kb=0; kb<nfrac; ++kb) {
              for (int ka=0; ka<nfrac; ++ka) {
                // probe: outward of each crossed boundary by dist; transverse dims at the
                // fraction lattice (0 and 1 land exactly on the lateral boundaries, so a
                // nominal face probe can legitimately become an edge/corner probe -- the
                // classification below follows the actual position, like the kernel)
                Real fr2[2] = {frac[ka], frac[kb]};
                Real pos[3];
                int kf = 0;
                for (int dim=0; dim<3; ++dim) {
                  Real len = bmax[dim] - bmin[dim];
                  if (off0[dim] == 0) {
                    Real f = (dim == 2 && !three_d) ? 0.5 : fr2[kf++];
                    pos[dim] = bmin[dim] + f*len;
                  } else {
                    pos[dim] = (off0[dim] > 0) ? bmax[dim] + dist[kd]
                                               : bmin[dim] - dist[kd];
                  }
                }
                nprobe++;

                // ---- mirror the SetNewPrtclGID kernel pipeline (comparison-based
                // offsets, exactly consistent with the ownership predicates)
                Real x3k = three_d ? pos[2] : bmin[2];
                int ix = 0, iy = 0, iz = 0;
                if (pos[0] <  bmin[0]) {ix = -1;} else if (pos[0] >= bmax[0]) {ix = 1;}
                if (pos[1] <  bmin[1]) {iy = -1;} else if (pos[1] >= bmax[1]) {iy = 1;}
                if (x3k    <  bmin[2]) {iz = -1;} else if (x3k    >= bmax[2]) {iz = 1;}
                int got_gid;
                if ((abs(ix) + abs(iy) + abs(iz)) == 0) {
                  got_gid = gids + m;     // still inside: no migration
                } else {
                  int fx = (pos[0] < 0.5*(bmin[0] + bmax[0])) ? 0 : 1;
                  int fy = (pos[1] < 0.5*(bmin[1] + bmax[1])) ? 0 : 1;
                  int fz = (x3k    < 0.5*(bmin[2] + bmax[2])) ? 0 : 1;
                  fz = three_d ? fz : 0;
                  int indx = FindDestinationIndex(ngh.h_view, m, mylevel, ix,iy,iz,
                                                  fx,fy,fz, par[0],par[1],par[2]);
                  got_gid = (indx >= 0) ? ngh.h_view(m,indx).gid : -1;
                }

                // ---- ground truth: brute-force bbox scan of the wrapped probe
                Real wx = pos[0], wy = pos[1], wz = pos[2];
                if (wx <  ms.x1min) {wx += (ms.x1max - ms.x1min);}
                if (wx >= ms.x1max) {wx -= (ms.x1max - ms.x1min);}
                if (wy <  ms.x2min) {wy += (ms.x2max - ms.x2min);}
                if (wy >= ms.x2max) {wy -= (ms.x2max - ms.x2min);}
                if (three_d) {
                  if (wz <  ms.x3min) {wz += (ms.x3max - ms.x3min);}
                  if (wz >= ms.x3max) {wz -= (ms.x3max - ms.x3min);}
                }
                int mok = FindContainingMeshBlock(wx, wy, three_d ? wz : 0.0);
                int want_gid = (mok >= 0) ? gids + mok : -2;

                if (got_gid != want_gid) {
                  nbad++;
                  if (nbad <= 50) {
                    std::cout << "[prtcl-audit] MISMATCH block gid=" << gids + m
                              << " dir=(" << ix0 << "," << iy0 << "," << iz0 << ")"
                              << " probe=(" << pos[0] << "," << pos[1] << "," << pos[2]
                              << ") off=(" << ix << "," << iy << "," << iz << ")"
                              << " got gid=" << got_gid << " want gid=" << want_gid
                              << std::endl;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::cout << "[prtcl-audit] " << nprobe << " probes on " << nmb << " MeshBlocks: "
            << nbad << " mismatches" << std::endl;
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "destination-search audit failed" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles
