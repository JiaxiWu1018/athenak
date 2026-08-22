//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_part_amr.cpp
//! \brief Relabel particles after dynamic AMR changes MeshBlock GIDs and owners.

#include <cstdlib>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlesBoundaryValues::SetPrtclGIDForAMR()
//! \brief Map old particle GIDs to the new AMR tree and collect cross-rank movers.

TaskStatus ParticlesBoundaryValues::SetPrtclGIDForAMR(
    const DualArray1D<int> &oldtonew, const DualArray1D<int> &newrank,
    const DualArray1D<int> &refflag, int old_gids) {
  auto &pr = pmy_part->prtcl_rdata;
  auto &pi = pmy_part->prtcl_idata;
  auto &mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto &psendl = sendlist;
  auto o2n = oldtonew.d_view;
  auto nrk = newrank.d_view;
  auto rfl = refflag.d_view;
  int npart = pmy_part->nprtcl_thispack;
  int myrank = global_variable::my_rank;
  bool multi_d = pmy_part->pmy_pack->pmesh->multi_d;
  bool three_d = pmy_part->pmy_pack->pmesh->three_d;

  // A regrid does not destroy particles. Clear the per-cycle state consumed by the
  // regular migration compaction, census, and debug validator before reusing that chain.
  nprtcl_destroy = 0;
  for (int n=0; n<NPRTCL_DEATH_REASON; ++n) {
    ndest_global[n] = 0;
    pmy_part->ndestroy_thisrank[n] = 0;
  }
  pmy_part->nmigr_face = 0;
  pmy_part->nmigr_edge = 0;
  pmy_part->nmigr_corner = 0;
  pmy_part->nsearch_fail = 0;

  // Count off-rank destinations before appending so the device send list is exact-sized.
  int ncross = 0;
  Kokkos::parallel_reduce("amr_count_cross",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &count) {
      int old_gid = pi(PGID,p);
      int new_gid = o2n(old_gid);
      if (rfl(old_gid) > 0) {
        int m = old_gid - old_gids;
        Real xc = 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max);
        Real yc = 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max);
        Real zc = 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max);
        int fx = (pr(IPX,p) >= xc) ? 1 : 0;
        int fy = (multi_d && pr(IPY,p) >= yc) ? 1 : 0;
        int fz = (three_d && pr(IPZ,p) >= zc) ? 1 : 0;
        new_gid += fx + 2*fy + 4*fz;
      }
      if (nrk(new_gid) != myrank) {
        count += 1;
      }
    }, Kokkos::Sum<int>(ncross));
#if MPI_PARALLEL_ENABLED
  if (ncross > sendlist.extent_int(0)) {
    Kokkos::realloc(sendlist, ncross);
  }
#else
  (void)ncross;
#endif

  DvceArray1D<int> send_count("amr_send_count", 1);
  int *pcounter = send_count.data();
  par_for("amr_relabel",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int old_gid = pi(PGID,p);
    int new_gid = o2n(old_gid);
    if (rfl(old_gid) > 0) {
      int m = old_gid - old_gids;
      Real xc = 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max);
      Real yc = 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max);
      Real zc = 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max);
      int fx = (pr(IPX,p) >= xc) ? 1 : 0;
      int fy = (multi_d && pr(IPY,p) >= yc) ? 1 : 0;
      int fz = (three_d && pr(IPZ,p) >= zc) ? 1 : 0;
      new_gid += fx + 2*fy + 4*fz;
    }
    pi(PGID,p) = new_gid;
#if MPI_PARALLEL_ENABLED
    if (nrk(new_gid) != myrank) {
      int index = Kokkos::atomic_fetch_add(pcounter, 1);
      psendl.d_view(index).prtcl_indx = p;
      psendl.d_view(index).dest_gid = new_gid;
      psendl.d_view(index).dest_rank = nrk(new_gid);
    }
#else
    (void)pcounter;
#endif
  });

  auto host_count = Kokkos::create_mirror_view(send_count);
  Kokkos::deep_copy(host_count, send_count);
  nprtcl_send = host_count(0);
#if MPI_PARALLEL_ENABLED
  if (nprtcl_send > ncross) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "AMR sendlist overflow: " << nprtcl_send
              << " sends > " << ncross << " counted" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  Kokkos::resize(sendlist, nprtcl_send);
  sendlist.template modify<DevExeSpace>();
  sendlist.template sync<HostMemSpace>();
  return TaskStatus::complete;
}

} // namespace particles
