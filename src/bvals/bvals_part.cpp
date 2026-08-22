//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_part.cpp
//! \brief

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"
#include "prtcl_search.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::UpdateGID()
//! \brief Updates GID of particles that cross boundary of their parent MeshBlock.  If
//! the new GID is on a different rank, then store in sendlist_buf DvceArray: (1) index of
//! particle in prtcl array, (2) destination GID, and (3) destination rank.

KOKKOS_INLINE_FUNCTION
void UpdateGID(int &newgid, NeighborBlock nghbr, int myrank, int *pcounter,
               DualArray1D<ParticleLocationData> slist, int p) {
  newgid = nghbr.gid;
#if MPI_PARALLEL_ENABLED
  if (nghbr.rank != myrank) {
    int index = Kokkos::atomic_fetch_add(pcounter,1);
    slist.d_view(index).prtcl_indx = p;
    slist.d_view(index).dest_gid   = nghbr.gid;
    slist.d_view(index).dest_rank  = nghbr.rank;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::SetNewGID()
//! \brief

TaskStatus ParticlesBoundaryValues::SetNewPrtclGID() {
  // create local references for variables in kernel
  auto gids = pmy_part->pmy_pack->gids;
  auto &pr = pmy_part->prtcl_rdata;
  auto &pi = pmy_part->prtcl_idata;
  int npart = pmy_part->nprtcl_thispack;
  auto &mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto &mblev = pmy_part->pmy_pack->pmb->mb_lev;
  auto &meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto myrank = global_variable::my_rank;
  auto &nghbr = pmy_part->pmy_pack->pmb->nghbr;
  auto &mbpar = pmy_part->pmy_pack->pmb->mb_parity;
  auto &mbbcs = pmy_part->pmy_pack->pmb->mb_bcs;
  auto &psendl = sendlist;
  // GPU-safe device send counter (the legacy code atomically incremented a HOST stack
  // address from inside the device kernel, and the host read of it had no fence)
  DvceArray1D<int> scnt("psend_cnt",1);   // zero-initialized
  int *pcounter = scnt.data();
  bool &multi_d = pmy_part->pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_part->pmy_pack->pmesh->three_d;

  // Per-cycle crossing counters accumulated on device. {0,1,2} classify placed crossings
  // by face/edge/corner and are published for CheckMigration only under <particles>
  // debug; {3} counts crossings the search could NOT place and {4} the overspeed subset
  // of those, and both are read back and acted on unconditionally (see the fatal check
  // after the kernel). debug >= 2 adds a per-event log.
  int dbg = pmy_part->debug_lvl;
  int ncycle = pmy_part->pmy_pack->pmesh->ncycle;
  Real mtime = pmy_part->pmy_pack->pmesh->time;
  Real mdt = pmy_part->pmy_pack->pmesh->dt;
  // Cap on per-particle detail lines per rank per cycle, so a whole population of
  // offenders cannot bury the summary in the log.
  const int kMigrDetail = 8;
  DvceArray1D<int> dcnt("pdbg_cnt",5);   // zero-initialized

  // destruction marking: device counters {0: append slot, 1..N: per-reason counts,
  // indexed by ParticlesDeathReason} and the destroyed-side {sum tag, sum tag^2} checksum
  // accumulators of the two-sided conservation ledger (written only when debug >= 1).
  // The kernel does atomic_add(&dstc(1+reason)), so this must be 1+NPRTCL_DEATH_REASON.
  DvceArray1D<int> dstc("pdest_cnt",1+NPRTCL_DEATH_REASON);   // zero-initialized
  DvceArray1D<uint64_t> dsums("pdest_sums",2);             // zero-initialized
  auto &pdestl = destroylist;
  auto &drr = destroy_rec_r;
  auto &dri = destroy_rec_i;
  // excision flags written by the MarkExcised task this cycle (C(b) criteria)
  bool exc_any = pmy_part->excise_any;
  auto &eflag = pmy_part->excise_flag;
  auto &ecrit = pmy_part->excise_crit;

  // Exact list sizing, pass 1 of 2: count (i) the particles that crossed a MeshBlock
  // boundary and (ii) the particles to destroy (mesh exits through non-periodic
  // boundaries; the C(b) excision criteria join this predicate via their flag array).
  // Cheap ownership comparisons only, no neighbor lookups. Crossers bound off-rank
  // senders from above, and both passes classify with the SAME predicates
  // (ComputeBlockOffsets + ExitsMeshBoundary), so the capacities grown here cannot be
  // exceeded by the pass-2 appends. The legacy guess of 0.1*npart overflowed the device
  // atomic appends (out-of-bounds writes) whenever more than 10% of a rank's particles
  // left in one cycle, and was zero for npart < 10.
  int ncross = 0, ndest_ub = 0;
  Kokkos::parallel_reduce("part_count_cross",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &csum, int &dsum) {
      int m = pi(PGID,p) - gids;
      Real x3 = three_d ? pr(IPZ,p) : 0.0;
      int cix, ciy, ciz;
      ComputeBlockOffsets(mbsize.d_view(m), pr(IPX,p), pr(IPY,p), x3, three_d,
                          cix, ciy, ciz);
      bool crossed = ((abs(cix) + abs(ciy) + abs(ciz)) != 0);
      if (crossed) {csum += 1;}
      if ((crossed && ExitsMeshBoundary(mbbcs.d_view, m, cix, ciy, ciz))
          || (exc_any && eflag(p) != 0)) {dsum += 1;}
    }, Kokkos::Sum<int>(ncross), Kokkos::Sum<int>(ndest_ub));
#if MPI_PARALLEL_ENABLED
  // serial builds never append to sendlist (UpdateGID is MPI-only): skip the growth
  if (ncross > static_cast<int>(sendlist.extent(0))) {
    Kokkos::realloc(sendlist, ncross);
  }
#else
  (void)ncross;
#endif
  if (ndest_ub > static_cast<int>(destroylist.extent(0))) {
    Kokkos::realloc(destroylist, ndest_ub);
  }
  if (ndest_ub > destroy_rec_r.extent_int(1)) {
    Kokkos::realloc(destroy_rec_r, 7, ndest_ub);
    Kokkos::realloc(destroy_rec_i, 3, ndest_ub);
  }
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int mylevel = mblev.d_view(m);
    Real x1 = pr(IPX,p);
    Real x2 = pr(IPY,p);
    // 2D problems use the trimmed 6-real layout where IPZ/IPVZ do not exist; substitute a
    // position inside the block so all x3 logic below is a no-op (iz=0, fz=0, no wrap)
    Real x3 = three_d ? pr(IPZ,p) : mbsize.d_view(m).x3min;

    // Integer offset of the particle relative to its MeshBlock, by the shared ownership
    // predicate (prtcl_search.hpp::ComputeBlockOffsets) -- the same comparisons used by
    // the sizing pass above, the containment validator, and the search audit. See the
    // helper's docstring for why this must never be an arithmetic (floor) form.
    int ix, iy, iz;
    ComputeBlockOffsets(mbsize.d_view(m), x1, x2, x3, three_d, ix, iy, iz);

    bool crossed = ((abs(ix) + abs(iy) + abs(iz)) != 0);

    // destruction marking: a crossing that leaves the mesh through any non-periodic
    // boundary destroys the particle (PrtclDeathExit), and the excision criteria destroy
    // via the MarkExcised flags (sphere/lapse/horizon; excised particles need not have
    // crossed anything). Exit takes precedence.
    // Marked particles are excluded from the search, sendlist, and wrap; the actual
    // removal is the merged hole compaction in RecvAndUnpackPrtcls.
    int reason = -1;
    if (crossed && ExitsMeshBoundary(mbbcs.d_view, m, ix, iy, iz)) {
      reason = PrtclDeathExit;
    } else if (exc_any && eflag(p) != 0) {
      reason = eflag(p);
    }
    if (reason >= 0) {
      int slot = Kokkos::atomic_fetch_add(&dstc(0), 1);
      Kokkos::atomic_add(&dstc(1+reason), 1);
      pdestl.d_view(slot) = p;
      // death record: the exact state at marking (post-push, pre-wrap -- for an exit
      // this is the first position OUTSIDE the mesh, within one v*dt of the crossing;
      // for excision the first position past the criterion surface)
      drr(0,slot) = x1;
      drr(1,slot) = x2;
      drr(2,slot) = three_d ? x3 : 0.0;
      drr(3,slot) = pr(IPVX,p);
      drr(4,slot) = pr(IPVY,p);
      drr(5,slot) = three_d ? pr(IPVZ,p) : 0.0;
      drr(6,slot) = (reason > 0) ? ecrit(p) : 0.0;  // r or alpha at marking
      dri(0,slot) = pi(PTAG,p);
      dri(1,slot) = pi(PGID,p);
      dri(2,slot) = reason;
      if (dbg > 0) {
        // destroyed-side checksums of the two-sided conservation ledger (cast BEFORE
        // multiplying: int tag*tag overflows at tag >= 46341)
        uint64_t t = static_cast<uint64_t>(pi(PTAG,p));
        Kokkos::atomic_add(&dsums(0), t);
        Kokkos::atomic_add(&dsums(1), t*t);
      }
      if (dbg > 1) {
        Kokkos::printf("[prtcl-debug] rank=%d cycle=%d tag=%d gid=%d DESTROY "
                       "reason=%d off=(%d,%d,%d) pos=(%.6e,%.6e,%.6e)\n", myrank,
                       ncycle, pi(PTAG,p), pi(PGID,p), reason, ix, iy, iz, x1, x2, x3);
      }
      return;   // skip search/sendlist/wrap: this particle is gone
    }

    // sublock indices for faces and edges with S/AMR
    int fx = (x1 < 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max))? 0 : 1;
    int fy = (x2 < 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max))? 0 : 1;
    int fz = (x3 < 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max))? 0 : 1;
    fy = multi_d ? fy : 0;
    fz = three_d ? fz : 0;

    // only update particle GID if it has crossed MeshBlock boundary
    if (crossed) {
      int oldgid = pi(PGID,p);
      // resolve the destination by direct parity-indexed lookups (bvals/prtcl_search.hpp;
      // replaces the legacy slot walk, see the Stage-3a(b) failure catalog)
      int indx = -1;
      if (abs(ix) <= 1 && abs(iy) <= 1 && abs(iz) <= 1) {
        indx = FindDestinationIndex(nghbr.d_view, m, mylevel, ix,iy,iz, fx,fy,fz,
                                    mbpar.d_view(m,0), mbpar.d_view(m,1),
                                    mbpar.d_view(m,2));
      }
      if (indx >= 0) {
        if (dbg > 0) {
          int d = abs(ix) + abs(iy) + abs(iz);   // 1: face, 2: edge, 3: corner crossing
          Kokkos::atomic_add(&dcnt(d-1), 1);
        }
        UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
      } else {
        // No destination: unrecoverable, the host aborts below (the GID is left alone
        // only to avoid shipping a dangling one to MPI). Mesh exits are destroyed before
        // the search, so two causes reach here, both fatal by design and reported
        // unconditionally: (a) overspeed -- some |offset| == 2, the particle moved
        // beyond the 26-neighbour migration range (re-homing it would hide the broken
        // assumption); (b) offsets within +-1 but no neighbour slot -- the mesh violates
        // its 2:1-balance / SetNeighbors contract.
        bool overspeed = (abs(ix) > 1 || abs(iy) > 1 || abs(iz) > 1);
        int fslot = Kokkos::atomic_fetch_add(&dcnt(3), 1);
        if (overspeed) {Kokkos::atomic_add(&dcnt(4), 1);}
        if (fslot < kMigrDetail) {
          const RegionSize &sz = mbsize.d_view(m);
          Real bmin[3] = {sz.x1min, sz.x2min, sz.x3min};
          Real bmax[3] = {sz.x1max, sz.x2max, sz.x3max};
          // fmax guards the 2D case, where the z extent is unused and may be zero
          Real bw[3]   = {fmax(sz.x1max - sz.x1min, 1.0e-300),
                          fmax(sz.x2max - sz.x2min, 1.0e-300),
                          fmax(sz.x3max - sz.x3min, 1.0e-300)};
          Real dxc[3]  = {fmax(sz.dx1, 1.0e-300), fmax(sz.dx2, 1.0e-300),
                          fmax(sz.dx3, 1.0e-300)};
          Real xp[3]   = {x1, x2, x3};
          // Overshoot beyond the owning block. The particle started inside [bmin,bmax),
          // so this is a lower bound on the update's displacement, tight to within one
          // block width. The exact step-n position is not retained on purpose.
          Real ov[3];
          for (int d=0; d<3; ++d) {
            ov[d] = (xp[d] >= bmax[d]) ? (xp[d] - bmax[d])
                  : ((xp[d] <  bmin[d]) ? (bmin[d] - xp[d]) : 0.0);
          }
          // Two literal format strings, not one with a "%s": the HIP device printf
          // drops the segment carrying a %s conversion (measured on gfx90a / ROCm
          // 6.4.1), which silently swallowed the line naming the cause.
          if (overspeed) {
            Kokkos::printf(
              "### FATAL ERROR particle migration: particle moved MORE THAN ONE "
              "MeshBlock width in one update\n"
              "    rank=%d cycle=%d time=%.8e dt=%.8e tag=%d gid=%d\n"
              "    x_new=(% .8e,% .8e,% .8e)  u_i/v=(% .8e,% .8e,% .8e)\n"
              "    owning block bbox x1=[% .8e,% .8e) x2=[% .8e,% .8e) "
              "x3=[% .8e,% .8e)\n"
              "    block offsets=(%d,%d,%d)  [+-2 means 'two or more block widths']\n"
              "    displacement >= (%.6f,%.6f,%.6f) MeshBlock widths"
              " = (%.4f,%.4f,%.4f) cells\n",
              myrank, ncycle, mtime, mdt, pi(PTAG,p), oldgid,
              x1, x2, x3, pr(IPVX,p), pr(IPVY,p), three_d ? pr(IPVZ,p) : 0.0,
              bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2],
              ix, iy, iz,
              ov[0]/bw[0], ov[1]/bw[1], ov[2]/bw[2],
              ov[0]/dxc[0], ov[1]/dxc[1], ov[2]/dxc[2]);
          } else {
            Kokkos::printf(
              "### FATAL ERROR particle migration: supported motion, but the mesh has "
              "NO neighbour to move into\n"
              "    rank=%d cycle=%d time=%.8e dt=%.8e tag=%d gid=%d\n"
              "    x_new=(% .8e,% .8e,% .8e)  u_i/v=(% .8e,% .8e,% .8e)\n"
              "    owning block bbox x1=[% .8e,% .8e) x2=[% .8e,% .8e) "
              "x3=[% .8e,% .8e)\n"
              "    block offsets=(%d,%d,%d)  [all within +-1: the neighbour array is "
              "missing an entry]\n"
              "    displacement >= (%.6f,%.6f,%.6f) MeshBlock widths"
              " = (%.4f,%.4f,%.4f) cells\n",
              myrank, ncycle, mtime, mdt, pi(PTAG,p), oldgid,
              x1, x2, x3, pr(IPVX,p), pr(IPVY,p), three_d ? pr(IPVZ,p) : 0.0,
              bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2],
              ix, iy, iz,
              ov[0]/bw[0], ov[1]/bw[1], ov[2]/bw[2],
              ov[0]/dxc[0], ov[1]/dxc[1], ov[2]/dxc[2]);
          }
        }
      }

      // per-event migration log (<particles> debug = 2); position is pre-wrap
      if (dbg > 1) {
        Kokkos::printf("[prtcl-debug] rank=%d cycle=%d tag=%d gid %d -> %d off=(%d,%d,%d)"
                       " pos=(%.6e,%.6e,%.6e)\n", myrank, ncycle, pi(PTAG,p), oldgid,
                       pi(PGID,p), ix, iy, iz, x1, x2, x3);
      }

      // reset x,y,z positions if particle crosses Mesh boundary using periodic BCs.
      // (>= at the max edge: a particle exactly on the mesh boundary has migrated, so its
      // wrapped position must land exactly on the min edge, consistent with the half-open
      // [min,max) block-ownership convention.) The position test alone is per-direction
      // correct: a surviving particle can only be beyond a mesh edge in a direction
      // whose face is periodic -- any non-periodic exit was destroyed above (Stage 3c),
      // which is what replaced the legacy unconditional wrap (bug A5/B7).
      if (x1 < meshsize.x1min) {
        pr(IPX,p) += (meshsize.x1max - meshsize.x1min);
      } else if (x1 >= meshsize.x1max) {
        pr(IPX,p) -= (meshsize.x1max - meshsize.x1min);
      }
      if (x2 < meshsize.x2min) {
        pr(IPY,p) += (meshsize.x2max - meshsize.x2min);
      } else if (x2 >= meshsize.x2max) {
        pr(IPY,p) -= (meshsize.x2max - meshsize.x2min);
      }
      if (three_d) {
        if (x3 < meshsize.x3min) {
          pr(IPZ,p) += (meshsize.x3max - meshsize.x3min);
        } else if (x3 >= meshsize.x3max) {
          pr(IPZ,p) -= (meshsize.x3max - meshsize.x3min);
        }
      }
    }
  });
  // read the device send counter back (the deep_copy also fences the kernel above)
  auto hscnt = Kokkos::create_mirror_view(scnt);
  Kokkos::deep_copy(hscnt, scnt);
  nprtcl_send = hscnt(0);
#if MPI_PARALLEL_ENABLED
  // by construction senders <= crossers (same predicate in both passes); a violation
  // means sendlist was overrun above -- make it fatal, never ship garbage entries
  if (nprtcl_send > ncross) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "sendlist overflow: " << nprtcl_send
              << " off-rank sends > " << ncross << " counted crossers" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  Kokkos::resize(sendlist, nprtcl_send);
  // sync sendlist device array with host
  sendlist.template modify<DevExeSpace>();
  sendlist.template sync<HostMemSpace>();

  // read the destroy counters back and publish the per-reason counts (the census that
  // CountSendsAndRecvs Allgathers and the count bookkeeping consumes)
  auto hdstc = Kokkos::create_mirror_view(dstc);
  Kokkos::deep_copy(hdstc, dstc);
  nprtcl_destroy = hdstc(0);
  for (int k=0; k<NPRTCL_DEATH_REASON; ++k) {
    pmy_part->ndestroy_thisrank[k] = hdstc(1+k);
  }
  // same overflow contract as the sendlist: both passes use the same predicate, so
  // appends > counted capacity means corruption -- fatal, never compact garbage
  if (nprtcl_destroy > ndest_ub) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "destroylist overflow: " << nprtcl_destroy
              << " destroyed > " << ndest_ub << " counted" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::resize(destroylist, nprtcl_destroy);
  destroylist.template modify<DevExeSpace>();
  destroylist.template sync<HostMemSpace>();
  // ascending index order is required by the merged hole compaction (atomic fill order
  // is arbitrary); only the host view is consumed downstream, so no device sync-back
  {
    namespace KE = Kokkos::Experimental;
    std::sort(KE::begin(destroylist.h_view), KE::end(destroylist.h_view));
  }
  // accumulate the destroyed-side conservation checksums (per-rank cumulative)
  if (dbg > 0 && nprtcl_destroy > 0) {
    auto hds = Kokkos::create_mirror_view(dsums);
    Kokkos::deep_copy(hds, dsums);
    pmy_part->ledger_dead[0] += hds(0);
    pmy_part->ledger_dead[1] += hds(1);
  }

  // Read the counters back. The face/edge/corner classification is published only under
  // <particles> debug, but the destination failures are acted on always: neither cause
  // (see the kernel above) may pass silently.
  {
    auto hcnt = Kokkos::create_mirror_view(dcnt);
    Kokkos::deep_copy(hcnt, dcnt);
    if (dbg > 0) {
      pmy_part->nmigr_face   = hcnt(0);
      pmy_part->nmigr_edge   = hcnt(1);
      pmy_part->nmigr_corner = hcnt(2);
      pmy_part->nsearch_fail = hcnt(3);
    }
    int nfail_tot = hcnt(3);
    int nfail_over = hcnt(4);
    int nfail_nghbr = nfail_tot - nfail_over;
    if (nfail_tot > 0) {
      // Kokkos::printf output lands in the C stdout buffer and MPI_Abort flushes
      // nothing, so fence and flush every stream first or the diagnostic can be lost.
      // Inside the failure branch, so the healthy path pays nothing.
      Kokkos::fence();
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "particle migration cannot place " << nfail_tot << " particle(s) on "
                << "rank " << global_variable::my_rank << " at cycle " << ncycle
                << " (time " << mtime << ", dt " << mdt << "):" << std::endl
                << "    " << nfail_over << " moved MORE THAN ONE MeshBlock width in one "
                << "update -- beyond the supported migration range." << std::endl
                << "    " << nfail_nghbr << " moved within the supported range but found "
                << "no neighbour -- the mesh's 2:1-balance / SetNeighbors contract is "
                << "broken." << std::endl
                << "    Per-particle detail was printed above for at most "
                << kMigrDetail << " of them. This is not repaired automatically: a hop "
                << "this long means an assumption broke (interpolated geometry, gauge/"
                << "shift excursion, or a timestep the particle CFL does not bound), and "
                << "re-homing the particle from a whole-mesh lookup would hide it."
                << std::endl;
      std::cout << std::flush;
      std::fflush(nullptr);       // drains the C stdout buffer the device printf uses
#if MPI_PARALLEL_ENABLED
      // one failing rank must kill the whole job: a plain exit here would leave the
      // other ranks blocked in the next cycle's collectives
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      std::exit(EXIT_FAILURE);
#endif
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::CountSendsAndRecvs()
//! \brief

TaskStatus ParticlesBoundaryValues::CountSendsAndRecvs() {
#if MPI_PARALLEL_ENABLED
  // Sort sendlist on host by destrank.
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(sendlist.h_view), KE::end(sendlist.h_view), SortByRank);
  // sync sendlist host array with device.  This results in sorted array on device
  sendlist.template modify<HostMemSpace>();
  sendlist.template sync<DevExeSpace>();

  // load STL::vector of ParticleMessageData with <sendrank, recvrank, nprtcls> for sends
  // from this rank. Length will be nsends; initially this length is unknown
  sends_thisrank.clear();
  if (nprtcl_send > 0) {
    int &myrank = global_variable::my_rank;
    int rank = sendlist.h_view(0).dest_rank;
    int nprtcl = 1;

    for (int n=1; n<nprtcl_send; ++n) {
      if (sendlist.h_view(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
        rank = sendlist.h_view(n).dest_rank;
        nprtcl = 1;
      }
    }
    sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
  }
  nsends = sends_thisrank.size();

  // Share the number of sends AND this cycle's destruction census among all ranks: one
  // Allgather of (1 + NPRTCL_DEATH_REASON) ints {nsends, ndestroy per reason} per rank
  // (widened from the legacy 1-int gather -- zero extra collectives). The global census
  // (i) keeps the count refresh in RecvAndUnpackPrtcls collective-free on send-quiet
  // cycles, (ii) feeds the cumulative destroyed ledger on Mesh, and (iii) makes the
  // death-log flush rank-consistent.
  const int ncnt = 1 + NPRTCL_DEATH_REASON;
  std::vector<int> cnt(ncnt);
  cnt[0] = nsends;
  for (int k=0; k<NPRTCL_DEATH_REASON; ++k) {cnt[1+k] = pmy_part->ndestroy_thisrank[k];}
  std::vector<int> cnts_all(ncnt*(global_variable::nranks));
  MPI_Allgather(cnt.data(), ncnt, MPI_INT, cnts_all.data(), ncnt, MPI_INT, mpi_comm_part);
  for (int k=0; k<NPRTCL_DEATH_REASON; ++k) {ndest_global[k] = 0;}
  for (int n=0; n<(global_variable::nranks); ++n) {
    nsends_eachrank[n] = cnts_all[ncnt*n];
    ndest_eachrank[n] = 0;
    for (int k=0; k<NPRTCL_DEATH_REASON; ++k) {
      ndest_eachrank[n] += cnts_all[ncnt*n+1+k];
      ndest_global[k] += cnts_all[ncnt*n+1+k];
    }
  }

  // Now share ParticleMessageData amongst all ranks
  // First create vector of starting indices in full vector
  std::vector<int> nsends_displ;
  nsends_displ.resize(global_variable::nranks);
  nsends_displ[0] = 0;
  for (int n=1; n<(global_variable::nranks); ++n) {
    nsends_displ[n] = nsends_displ[n-1] + nsends_eachrank[n-1];
  }
  int nsends_allranks = nsends_displ[global_variable::nranks - 1] +
                        nsends_eachrank[global_variable::nranks - 1];
  // Load ParticleMessageData on this rank into full vector
  sends_allranks.resize(nsends_allranks, ParticleMessageData(0,0,0));
  for (int n=0; n<nsends_eachrank[global_variable::my_rank]; ++n) {
    sends_allranks[n + nsends_displ[global_variable::my_rank]] = sends_thisrank[n];
  }

  // Share tuples using a temporary MPI derived datatype for a tuple of 3*int. Current
  // upstream frees this datatype after each collective, so no long-lived handle is
  // needed.
  MPI_Datatype mpi_ituple;
  MPI_Type_contiguous(3, MPI_INT, &mpi_ituple);
  MPI_Type_commit(&mpi_ituple);
  MPI_Allgatherv(MPI_IN_PLACE, nsends_eachrank[global_variable::my_rank],
                   mpi_ituple, sends_allranks.data(), nsends_eachrank.data(),
                   nsends_displ.data(), mpi_ituple, mpi_comm_part);
  MPI_Type_free(&mpi_ituple);

  // <particles> debug >= 2: rank 0 prints this cycle's global send matrix (every rank
  // already holds the full tuple list -- no extra communication)
  if (pmy_part->debug_lvl > 1 && global_variable::my_rank == 0 &&
      !(sends_allranks.empty())) {
    std::cout << "[prtcl-debug] cycle=" << pmy_part->pmy_pack->pmesh->ncycle << " sends:";
    for (auto &s : sends_allranks) {
      std::cout << " " << s.sendrank << "->" << s.recvrank << ":" << s.nprtcls;
    }
    std::cout << std::endl;
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::InitPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::InitPrtclRecv() {
#if MPI_PARALLEL_ENABLED
  // load STL::vector of ParticleMessageData with <sendrank,recvrank,nprtcl_recv> for
  // receives // on this rank. Length will be nrecvs, initially this length is unknown
  recvs_thisrank.clear();

  int nsends_allranks = sends_allranks.size();
  for (int n=0; n<nsends_allranks; ++n) {
    if (sends_allranks[n].recvrank == global_variable::my_rank) {
      recvs_thisrank.emplace_back(sends_allranks[n]);
    }
  }
  nrecvs = recvs_thisrank.size();

  // Figure out how many particles will be received from all ranks
  nprtcl_recv=0;
  for (int n=0; n<nrecvs; ++n) {
    nprtcl_recv += recvs_thisrank[n].nprtcls;
  }

  // Allocate receive buffer (skip the zero-extent reallocs on quiet cycles)
  if (nprtcl_recv > 0) {
    Kokkos::realloc(prtcl_rrecvbuf, (pmy_part->nrdata)*nprtcl_recv);
    Kokkos::realloc(prtcl_irecvbuf, (pmy_part->nidata)*nprtcl_recv);
  }

  // Post non-blocking receives
  bool no_errors=true;
  rrecv_req.clear();
  irecv_req.clear();
  for (int n=0; n<nrecvs; ++n) {
    rrecv_req.emplace_back(MPI_REQUEST_NULL);
    irecv_req.emplace_back(MPI_REQUEST_NULL);
  }

  // Init receives for Reals
  int data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = (pmy_part->nrdata)*(recvs_thisrank[n].nprtcls);
    int data_end = data_start + data_size;
    auto recv_ptr = Kokkos::subview(prtcl_rrecvbuf, std::make_pair(data_start, data_end));
    int drank = recvs_thisrank[n].sendrank;
    int tag = 0; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                         mpi_comm_part, &(rrecv_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
  }
  // Init receives for ints
  data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = (pmy_part->nidata)*(recvs_thisrank[n].nprtcls);
    int data_end = data_start + data_size;
    auto recv_ptr = Kokkos::subview(prtcl_irecvbuf, std::make_pair(data_start, data_end));
    int drank = recvs_thisrank[n].sendrank;
    int tag = 1; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_INT, drank, tag,
                         mpi_comm_part, &(irecv_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
  }

  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::PackAndSendPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::PackAndSendPrtcls() {
#if MPI_PARALLEL_ENABLED
  // Figure out how many particles will be sent from this ranks
  nprtcl_send=0;
  for (int n=0; n<nsends; ++n) {
    nprtcl_send += sends_thisrank[n].nprtcls;
  }

  bool no_errors=true;
  if (nprtcl_send > 0) {
    // Allocate send buffer
    Kokkos::realloc(prtcl_rsendbuf, (pmy_part->nrdata)*nprtcl_send);
    Kokkos::realloc(prtcl_isendbuf, (pmy_part->nidata)*nprtcl_send);

    // sendlist on device is already sorted by destrank in CountSendAndRecvs()
    // Use sendlist on device to load particles into send buffer ordered by dest_rank
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rsendbuf = prtcl_rsendbuf;
    auto &isendbuf = prtcl_isendbuf;
    // local ref so the device lambda does not capture (and dereference) host `this`
    auto &slist = sendlist;
    par_for("ppack",DevExeSpace(),0,(nprtcl_send-1), KOKKOS_LAMBDA(const int n) {
      int p = slist.d_view(n).prtcl_indx;
      for (int i=0; i<nidata; ++i) {
        isendbuf(nidata*n + i) = pi(i,p);
      }
      for (int i=0; i<nrdata; ++i) {
        rsendbuf(nrdata*n + i) = pr(i,p);
      }
    });

    // Post non-blocking sends
    Kokkos::fence();
    rsend_req.clear();
    isend_req.clear();
    for (int n=0; n<nsends; ++n) {
      rsend_req.emplace_back(MPI_REQUEST_NULL);
      isend_req.emplace_back(MPI_REQUEST_NULL);
    }

    // Send Reals
    int data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed, get pointer to variables
      int data_size = nrdata*(sends_thisrank[n].nprtcls);
      int data_end = data_start + data_size;
      auto send_ptr = Kokkos::subview(prtcl_rsendbuf,std::make_pair(data_start,data_end));
      int drank = sends_thisrank[n].recvrank;
      int tag = 0; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                           mpi_comm_part, &(rsend_req[n]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      data_start += data_size;
    }
    // Send ints
    data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed, get pointer to variables
      int data_size = nidata*(sends_thisrank[n].nprtcls);
      int data_end = data_start + data_size;
      auto send_ptr = Kokkos::subview(prtcl_isendbuf,std::make_pair(data_start,data_end));
      int drank = sends_thisrank[n].recvrank;
      int tag = 1; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_INT, drank, tag,
                           mpi_comm_part, &(isend_req[n]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      data_start += data_size;
    }
  }

  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::RecvAndUnpackPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::RecvAndUnpackPrtcls() {
  int npart = pmy_part->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
  // Sort sendlist on host by index in particle array (ascending hole order, required
  // by the merged hole list below)
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(sendlist.h_view), KE::end(sendlist.h_view), SortByIndex);
  // sync sendlist host array with device.  This results in sorted array on device
  sendlist.template modify<HostMemSpace>();
  sendlist.template sync<DevExeSpace>();

  // check that particle communications have all completed
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<nrecvs; ++n) {
    int test;
    int ierr = MPI_Test(&(rrecv_req[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
    ierr = MPI_Test(&(irecv_req[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if particle communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  // ---- ONE merged hole compaction (Stage 3c). Sent and destroyed particles both
  // leave holes; received particles fill holes first, surviving tail particles fill
  // the rest, then a single resize shrinks the arrays. This common path runs in
  // serial builds too (nprtcl_send = nprtcl_recv = 0 there -- destruction must work
  // without MPI; both are ctor-zeroed so the first cycle is well-defined).
  int nsend = nprtcl_send;
  int nrecv = nprtcl_recv;
  int ndest = nprtcl_destroy;
  int nholes = nsend + ndest;
  int new_npart = npart + nrecv - nholes;

  // merge the (index-sorted) sendlist and destroylist into one ascending hole list.
  // The two are disjoint by construction (marked-destroyed particles never enter the
  // search or the sendlist) -- a duplicate would corrupt the compaction, so assert.
  int ntarget = 0;   // UNFILLED holes below new_npart = the tail-fill destinations
  if (nholes > 0) {
    if (nholes > static_cast<int>(holelist.extent(0))) {
      Kokkos::realloc(holelist, nholes);
    }
    int a = 0, b = 0, k = 0;
    while (a < nsend && b < ndest) {
      int sv = sendlist.h_view(a).prtcl_indx;
      int dv = destroylist.h_view(b);
      holelist.h_view(k++) = (sv < dv) ? sendlist.h_view(a++).prtcl_indx
                                       : destroylist.h_view(b++);
    }
    while (a < nsend) {holelist.h_view(k++) = sendlist.h_view(a++).prtcl_indx;}
    while (b < ndest) {holelist.h_view(k++) = destroylist.h_view(b++);}
    for (int n=0; n<nholes; ++n) {
      if (n > 0 && holelist.h_view(n) == holelist.h_view(n-1)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "particle " << holelist.h_view(n)
                  << " is both sent and destroyed (disjointness violated)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // the first nrecv (smallest) holes are filled by received particles below; only
      // the unfilled remainder below new_npart needs a tail donor
      if (n >= nrecv && holelist.h_view(n) < new_npart) {ntarget++;}
    }
    holelist.template modify<HostMemSpace>();
    holelist.template sync<DevExeSpace>();
  }

  // increase size of particle arrays if needed (more receives than holes)
  if (nrecv > nholes) {
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_npart);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_npart);
  }

#if MPI_PARALLEL_ENABLED
  // unpack received particles into the holes (merged order); excess appends at the end
  if (nrecv > 0) {
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rrecvbuf = prtcl_rrecvbuf;
    auto &irecvbuf = prtcl_irecvbuf;
    // locals so the device lambda does not capture (and dereference) host `this`
    auto &hlist = holelist;
    int nh = nholes;
    int npart_old = npart;
    par_for("punpack",DevExeSpace(),0,(nprtcl_recv-1), KOKKOS_LAMBDA(const int n) {
      int p;
      if (n < nh) {
        p = hlist.d_view(n);          // fill holes left by sent/destroyed particles
      } else {
        p = npart_old + (n - nh);     // place particle at end of arrays
      }
      for (int i=0; i<nidata; ++i) {
        pi(i,p) = irecvbuf(nidata*n + i);
      }
      for (int i=0; i<nrdata; ++i) {
        pr(i,p) = rrecvbuf(nrdata*n + i);
      }
    });
  }
#endif

  // Fill the remaining holes by gathering surviving particles from the array tail:
  // one host pass builds the (dst,src) pair list, then ONE device kernel executes
  // every move. (The legacy host loop launched two deep_copies per moved particle and
  // depended on undocumented orderings; the prototype's copy of it had the B2 bug --
  // unsorted hole list + the smallest-vs-largest hole comparison -- which resurrected
  // dead particles and dropped live ones. The pair construction below asserts its own
  // invariants instead, and the debug=1 two-sided ledger is the end-to-end oracle.)
  int nremain = nholes - nrecv;
  if (nremain > 0) {
    if (2*nremain > static_cast<int>(cpairs.extent(0))) {
      Kokkos::realloc(cpairs, 2*nremain);
    }
    int npairs = 0;
    int i = nrecv;          // ascending: next unfilled hole (target while < new_npart)
    int hi = nholes - 1;    // descending: skips tail slots that are themselves holes
    for (int j=(npart-1); j>=new_npart; --j) {
      if (hi >= nrecv && holelist.h_view(hi) == j) {--hi; continue;}  // j is a hole
      // j holds a surviving particle: move it into the lowest unfilled hole
      int dst = holelist.h_view(i);
      if (i > hi || dst >= new_npart) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "compaction pairing imbalance (donor " << j
                  << ", next hole " << dst << ", new_npart " << new_npart << ")"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      cpairs.h_view(2*npairs) = dst;
      cpairs.h_view(2*npairs+1) = j;
      ++npairs; ++i;
    }
    // every hole below new_npart must have been paired with exactly one tail survivor
    // (holes at/above new_npart simply fall off with the resize)
    if (npairs != ntarget) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "compaction pairing incomplete: " << npairs
                << " pairs != " << ntarget << " holes below new_npart" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (npairs > 0) {
      cpairs.template modify<HostMemSpace>();
      cpairs.template sync<DevExeSpace>();
      int nrdata = pmy_part->nrdata;
      int nidata = pmy_part->nidata;
      auto &pr = pmy_part->prtcl_rdata;
      auto &pi = pmy_part->prtcl_idata;
      auto &cp = cpairs;
      // no races: every dst < new_npart <= every src, and dst/src are each unique
      par_for("pcompact",DevExeSpace(),0,(npairs-1), KOKKOS_LAMBDA(const int n) {
        int dst = cp.d_view(2*n);
        int src = cp.d_view(2*n+1);
        for (int i=0; i<nidata; ++i) {
          pi(i,dst) = pi(i,src);
        }
        for (int i=0; i<nrdata; ++i) {
          pr(i,dst) = pr(i,src);
        }
      });
    }
    // shrink particle arrays: the single resize of the whole compaction
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_npart);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_npart);
  }

  // ---- refresh the particle-count bookkeeping + the cumulative destroyed ledger ----
  pmy_part->nprtcl_thispack = new_npart;
  Mesh *pm = pmy_part->pmy_pack->pmesh;
  pm->nprtcl_thisrank = new_npart;
#if MPI_PARALLEL_ENABLED
  int gdest = 0;
  for (int k=0; k<NPRTCL_DEATH_REASON; ++k) {gdest += ndest_global[k];}
  if (!(sends_allranks.empty())) {
    // cross-rank traffic this cycle: refresh by Allgather on the particle communicator
    // (the authoritative path; destroys are already folded into new_npart)
    MPI_Allgather(&new_npart,1,MPI_INT,(pm->nprtcl_eachrank),1,MPI_INT,mpi_comm_part);
    pm->nprtcl_total = 0;
    for (int n=0; n<(global_variable::nranks); ++n) {
      pm->nprtcl_total += pm->nprtcl_eachrank[n];
    }
  } else if (gdest > 0) {
    // destruction only: every rank already knows every rank's destroy count from the
    // census -- update the bookkeeping locally, no collective needed
    for (int n=0; n<(global_variable::nranks); ++n) {
      pm->nprtcl_eachrank[n] -= ndest_eachrank[n];
    }
    pm->nprtcl_total -= gdest;
  }
  // else: quiet cycle (no sends and no destroys anywhere) -- counts unchanged
#else
  int gdest = 0;
  for (int k=0; k<NPRTCL_DEATH_REASON; ++k) {
    ndest_global[k] = pmy_part->ndestroy_thisrank[k];
    gdest += ndest_global[k];
  }
  if (gdest > 0) {
    pm->nprtcl_eachrank[0] = new_npart;
    pm->nprtcl_total = new_npart;
  }
#endif
  if (gdest > 0) {
    pm->TallyDestroyedPrtcls(ndest_global);
    // per-event death records -> <basename>.prtcl_destroy.csv. Collective in MPI
    // builds; safe because the census makes gdest identical on every rank.
    if (pmy_part->destroy_log) {pmy_part->FlushDeathLog();}
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearPrtclSend()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  // wait for all non-blocking sends for vars to finish before continuing
  for (int n=0; n<nsends; ++n) {
    int ierr = MPI_Wait(&(rsend_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    ierr = MPI_Wait(&(isend_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rsend_req.clear();
  isend_req.clear();
#endif
  nsends=0;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  // wait for all non-blocking receives to finish before continuing
  for (int n=0; n<nrecvs; ++n) {
    int ierr = MPI_Wait(&(rrecv_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    ierr = MPI_Wait(&(irecv_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rrecv_req.clear();
  irecv_req.clear();
#endif
  nrecvs=0;
  return TaskStatus::complete;
}

} // namespace particles
