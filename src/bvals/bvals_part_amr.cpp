//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_part_amr.cpp
//! \brief NRPIC Stage 5a: the particle-redistribution remap that follows a dynamic-AMR
//! regrid (refine / derefine / load-balance). One regrid renumbers every gid in Z-order,
//! creates/destroys blocks, and re-ranks them, so every particle's PGID goes stale while
//! no particle row moves. SetPrtclGIDForAMR rewrites each PGID from its old block's
//! fate using the (old gid -> new gid) maps the regrid already built, then hands the
//! cross-rank movers to the SAME send/compaction/ledger chain the per-cycle migration
//! uses (bvals_part.cpp). It is a SetNewPrtclGID-class device pass with a different
//! destination rule and NO neighbor-array involvement: the regrid covers the domain, so
//! the old->new map is total -- no search, no periodic wrap, no destruction.
//!
//! The map (per particle, from the reconciled refine_flag + oldtonew built in
//! mesh_refinement.cpp::RedistAndRefineMeshBlocks):
//!   unchanged block -> oldtonew[old_gid]                       (pure relabel)
//!   derefined child -> oldtonew[old_gid]  (the parent's new gid; all nleaf children of a
//!                                          derefine group map to the same parent)
//!   refined parent  -> oldtonew[old_gid] + (fx + 2*fy + 4*fz), where (fx,fy,fz) are the
//!                      half-tests of the particle position against the OLD parent center
//!                      -- the Z-order child offset (meshblock_tree.cpp child ordering
//!                      n = i + 2j + 4k). The chosen child's new bbox is exactly the
//!                      parent octant on that side of the center, so the relabel is
//!                      position-consistent with CheckMigration's bbox + gid-range tests.
//!
//! Called at the hook point in RedistAndRefineMeshBlocks where the OLD MeshBlock geometry
//! (mb_size, for the half-tests) and both maps are still live; the resulting sendlist is
//! shipped after the NEW MeshBlockPack gids/ranks are installed.

#include <cstdlib>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlesBoundaryValues::SetPrtclGIDForAMR()
//! \brief Rewrite every particle's PGID from its old block's fate after a regrid, and
//! build the sendlist of cross-rank movers for the existing migration chain. Mirrors the
//! two-pass (count+realloc, then atomic-append) structure of SetNewPrtclGID, but with the
//! direct old->new relabel rule instead of a neighbor search.
//!
//! \param oldtonew  DualArray, indexed by OLD gid, value = base NEW gid (new gid for an
//!                  unchanged block, the parent's new gid for a derefined child, the
//!                  FIRST child's new gid for a refined parent).
//! \param newrank   DualArray, indexed by NEW gid, value = owning rank (load balance).
//! \param refflag   DualArray, indexed by OLD global gid, value = block fate (+1 refined,
//!                  -nleaf derefined, 0 unchanged); the reconciled flag from the regrid.
//! \param old_gids  this rank's starting OLD global gid (so m = old_gid - old_gids is the
//!                  local index into the OLD mb_size, still live at the call site).

TaskStatus ParticlesBoundaryValues::SetPrtclGIDForAMR(const DualArray1D<int> &oldtonew,
                                                      const DualArray1D<int> &newrank,
                                                      const DualArray1D<int> &refflag,
                                                      int old_gids) {
  auto &pr = pmy_part->prtcl_rdata;
  auto &pi = pmy_part->prtcl_idata;
  int npart = pmy_part->nprtcl_thispack;
  // OLD MeshBlock geometry: the regrid has not yet deleted the old MeshBlocks at the call
  // site, so mb_size still holds the pre-regrid block extents needed by the half-tests.
  auto &mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto myrank = global_variable::my_rank;
  bool &multi_d = pmy_part->pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_part->pmy_pack->pmesh->three_d;
  auto &psendl = sendlist;
  auto o2n = oldtonew.d_view;
  auto nrk = newrank.d_view;
  auto rfl = refflag.d_view;

  // A regrid relabels particles but creates/destroys NONE of them and walks no neighbors.
  // Zero the destruction + debug census this pass would otherwise leave stale, so the
  // reused chain (RecvAndUnpackPrtcls' merged hole compaction reads nprtcl_destroy; the
  // census Allgather reads ndestroy_thisrank) and the debug ledger see a clean state.
  nprtcl_destroy = 0;
  ndest_global[0] = ndest_global[1] = ndest_global[2] = 0;
  for (int k=0; k<3; ++k) {pmy_part->ndestroy_thisrank[k] = 0;}
  pmy_part->nmigr_face = 0;
  pmy_part->nmigr_edge = 0;
  pmy_part->nmigr_corner = 0;
  pmy_part->nsearch_fail = 0;

  // Exact sizing, pass 1 of 2: count this rank's particles whose new owner is another
  // rank (the same predicate the append in pass 2 uses, so the capacity grown here cannot
  // be exceeded). Serial builds never append (UpdateGID is MPI-only), so skip the growth.
  int ncross = 0;
  Kokkos::parallel_reduce("amr_count_cross",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &csum) {
      int old_gid = pi(PGID,p);
      int newgid = o2n(old_gid);
      if (rfl(old_gid) > 0) {                       // refined parent: pick the child
        int m = old_gid - old_gids;
        Real xc = 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max);
        Real yc = 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max);
        Real zc = 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max);
        int fx = (pr(IPX,p) >= xc) ? 1 : 0;
        int fy = (multi_d && (pr(IPY,p) >= yc)) ? 1 : 0;
        int fz = (three_d && (pr(IPZ,p) >= zc)) ? 1 : 0;
        newgid += fx + 2*fy + 4*fz;
      }
      if (nrk(newgid) != myrank) {csum += 1;}
    }, Kokkos::Sum<int>(ncross));
#if MPI_PARALLEL_ENABLED
  if (ncross > static_cast<int>(sendlist.extent(0))) {
    Kokkos::realloc(sendlist, ncross);
  }
#else
  (void)ncross;
#endif

  // GPU-safe device send counter (zero-initialized), read back after the kernel
  DvceArray1D<int> scnt("amr_send_cnt",1);
  int *pcounter = scnt.data();

  // Pass 2 of 2: rewrite PGID into NEW gid space; append off-rank movers to the sendlist.
  par_for("amr_relabel",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int old_gid = pi(PGID,p);
    int newgid = o2n(old_gid);
    if (rfl(old_gid) > 0) {                         // refined parent: pick the child
      int m = old_gid - old_gids;
      Real xc = 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max);
      Real yc = 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max);
      Real zc = 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max);
      int fx = (pr(IPX,p) >= xc) ? 1 : 0;
      int fy = (multi_d && (pr(IPY,p) >= yc)) ? 1 : 0;
      int fz = (three_d && (pr(IPZ,p) >= zc)) ? 1 : 0;
      newgid += fx + 2*fy + 4*fz;
    }
    pi(PGID,p) = newgid;
#if MPI_PARALLEL_ENABLED
    if (nrk(newgid) != myrank) {
      int index = Kokkos::atomic_fetch_add(pcounter,1);
      psendl.d_view(index).prtcl_indx = p;
      psendl.d_view(index).dest_gid   = newgid;
      psendl.d_view(index).dest_rank  = nrk(newgid);
    }
#else
    (void)pcounter;
#endif
  });

  // read the device send counter back (the deep_copy also fences the kernel above)
  auto hscnt = Kokkos::create_mirror_view(scnt);
  Kokkos::deep_copy(hscnt, scnt);
  nprtcl_send = hscnt(0);
#if MPI_PARALLEL_ENABLED
  // senders <= crossers by construction (same predicate both passes); a violation means
  // the sendlist was overrun -- fatal, never ship garbage entries
  if (nprtcl_send > ncross) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "AMR sendlist overflow: " << nprtcl_send
              << " off-rank movers > " << ncross << " counted" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  Kokkos::resize(sendlist, nprtcl_send);
  // sync sendlist device array with host (CountSendsAndRecvs sorts the host view)
  sendlist.template modify<DevExeSpace>();
  sendlist.template sync<HostMemSpace>();

  return TaskStatus::complete;
}

} // namespace particles
