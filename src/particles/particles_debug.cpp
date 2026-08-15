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

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

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

} // namespace particles
