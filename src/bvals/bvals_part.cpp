//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_part.cpp
//! \brief

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
  auto &psendl = sendlist;
  // GPU-safe device send counter (the legacy code atomically incremented a HOST stack
  // address from inside the device kernel, and the host read of it had no fence)
  DvceArray1D<int> scnt("psend_cnt",1);   // zero-initialized
  int *pcounter = scnt.data();
  bool &multi_d = pmy_part->pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_part->pmy_pack->pmesh->three_d;

  // migration debug instrumentation (<particles> debug >= 1): per-cycle crossing counters
  // {0: face, 1: edge, 2: corner, 3: destination-search failure}, accumulated on device
  // and copied back into the Particles members for CheckMigration. debug >= 2 adds a
  // per-event log.
  int dbg = pmy_part->debug_lvl;
  int ncycle = pmy_part->pmy_pack->pmesh->ncycle;
  DvceArray1D<int> dcnt("pdbg_cnt",4);   // zero-initialized

#if MPI_PARALLEL_ENABLED
  // Exact sendlist sizing, pass 1 of 2: count the particles that crossed a MeshBlock
  // boundary (cheap ownership comparisons only, no neighbor lookups). Crossers bound
  // off-rank senders from above, and both passes classify a crossing with the SAME
  // predicate (ComputeBlockOffsets), so the capacity grown here cannot be exceeded by
  // the pass-2 appends. The legacy guess of 0.1*npart overflowed the device atomic
  // appends (out-of-bounds writes) whenever more than 10% of a rank's particles left
  // in one cycle, and was zero for npart < 10. Serial builds never append (UpdateGID
  // is MPI-only), so the count and the growth are skipped entirely.
  int ncross = 0;
  Kokkos::parallel_reduce("part_count_cross",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &csum) {
      int m = pi(PGID,p) - gids;
      Real x3 = three_d ? pr(IPZ,p) : 0.0;
      int cix, ciy, ciz;
      ComputeBlockOffsets(mbsize.d_view(m), pr(IPX,p), pr(IPY,p), x3, three_d,
                          cix, ciy, ciz);
      if ((abs(cix) + abs(ciy) + abs(ciz)) != 0) {csum += 1;}
    }, Kokkos::Sum<int>(ncross));
  if (ncross > static_cast<int>(sendlist.extent(0))) {
    Kokkos::realloc(sendlist, ncross);
  }
#endif
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

    // sublock indices for faces and edges with S/AMR
    int fx = (x1 < 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max))? 0 : 1;
    int fy = (x2 < 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max))? 0 : 1;
    int fz = (x3 < 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max))? 0 : 1;
    fy = multi_d ? fy : 0;
    fz = three_d ? fz : 0;

    // only update particle GID if it has crossed MeshBlock boundary
    if ((abs(ix) + abs(iy) + abs(iz)) != 0) {
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
        // No destination: leave the particle on its current MeshBlock rather than write
        // a dangling GID (the legacy corner path wrote gid=-1 AND rank=-1, which aborted
        // MPI builds inside MPI_Isend). Reachable only if the particle moved more than
        // one block width in a step, exited a non-periodic physical boundary (handled by
        // the destruction machinery of a later Stage-3 session), or the 2:1-balance /
        // SetNeighbors contract broke. CheckMigration (debug >= 1) makes all of these
        // fatal via the search_fail counter and the containment check.
        Kokkos::atomic_add(&dcnt(3), 1);
        if (dbg > 0) {
          Kokkos::printf("[prtcl-debug] rank=%d cycle=%d tag=%d gid=%d SEARCH FAIL "
                         "off=(%d,%d,%d) pos=(%.6e,%.6e,%.6e)\n", myrank, ncycle,
                         pi(PTAG,p), oldgid, ix, iy, iz, x1, x2, x3);
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
      // [min,max) block-ownership convention.) NOTE: applied unconditionally; correct for
      // the strictly-periodic meshes Stage 3a tests. Per-direction BC gating and
      // destruction at physical boundaries is a later Stage-3 session.
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

  // store the per-cycle migration counters for CheckMigration (debug mode only)
  if (dbg > 0) {
    auto hcnt = Kokkos::create_mirror_view(dcnt);
    Kokkos::deep_copy(hcnt, dcnt);
    pmy_part->nmigr_face   = hcnt(0);
    pmy_part->nmigr_edge   = hcnt(1);
    pmy_part->nmigr_corner = hcnt(2);
    pmy_part->nsearch_fail = hcnt(3);
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

  // Share number of ranks to send to amongst all ranks
  nsends_eachrank[global_variable::my_rank] = nsends;
  MPI_Allgather(&nsends, 1, MPI_INT, nsends_eachrank.data(), 1, MPI_INT, mpi_comm_part);

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

  // Share tuples using the MPI derived data type for tuple of 3*int committed once in
  // the constructor (creating it here leaked one datatype handle per cycle)
  MPI_Allgatherv(MPI_IN_PLACE, nsends_eachrank[global_variable::my_rank],
                   mpi_ituple, sends_allranks.data(), nsends_eachrank.data(),
                   nsends_displ.data(), mpi_ituple, mpi_comm_part);

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
#if MPI_PARALLEL_ENABLED
  // Sort sendlist on host by index in particle array
  namespace KE = Kokkos::Experimental;
  std::sort(KE::begin(sendlist.h_view), KE::end(sendlist.h_view), SortByIndex);
  // sync sendlist host array with device.  This results in sorted array on device
  sendlist.template modify<HostMemSpace>();
  sendlist.template sync<DevExeSpace>();

  // increase size of particle arrays if needed
  int new_npart = pmy_part->nprtcl_thispack + (nprtcl_recv - nprtcl_send);
  if (nprtcl_recv > nprtcl_send) {
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_npart);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_npart);
  }

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

  // unpack particles into positions of sent particles
  if (nprtcl_recv > 0) {
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rrecvbuf = prtcl_rrecvbuf;
    auto &irecvbuf = prtcl_irecvbuf;
    int &npart = pmy_part->nprtcl_thispack;
    // locals so the device lambda does not capture (and dereference) host `this`
    auto &slist = sendlist;
    int nsend = nprtcl_send;
    par_for("punpack",DevExeSpace(),0,(nprtcl_recv-1), KOKKOS_LAMBDA(const int n) {
      int p;
      if (n < nsend) {
        p = slist.d_view(n).prtcl_indx;    // place particles in holes created by sends
      } else {
        p = npart + (n - nsend);           // place particle at end of arrays
      }
      for (int i=0; i<nidata; ++i) {
        pi(i,p) = irecvbuf(nidata*n + i);
      }
      for (int i=0; i<nrdata; ++i) {
        pr(i,p) = rrecvbuf(nrdata*n + i);
      }
    });
  }

  // At this point have filled npart_recv holes in particle arrays from sends
  // If (nprtcl_recv < nprtcl_send), have to move particles from end of arrays to fill
  // remaining holes
  int nremain = nprtcl_send - nprtcl_recv;
  if (nremain > 0) {
    int &npart = pmy_part->nprtcl_thispack;
    int i_last_hole = nprtcl_send-1;
    int i_next_hole = nprtcl_recv;
    for (int n=1; n<=nremain; ++n) {
      int nend = npart-n;
      if (nend > sendlist.h_view(i_last_hole).prtcl_indx) {
        // copy particle from end into hole
        int next_hole = sendlist.h_view(i_next_hole).prtcl_indx;
        auto rdest = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, next_hole);
        auto rsrc  = Kokkos::subview(pmy_part->prtcl_rdata, Kokkos::ALL, nend);
        Kokkos::deep_copy(rdest, rsrc);
        auto idest = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, next_hole);
        auto isrc  = Kokkos::subview(pmy_part->prtcl_idata, Kokkos::ALL, nend);
        Kokkos::deep_copy(idest, isrc);
        i_next_hole += 1;
      } else {
        // this index contains a hole, so do nothing except find new index of last hole
        i_last_hole -= 1;
      }
    }

    // shrink size of particle data arrays
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_npart);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_npart);
  }

  // Update nparticles_thisrank.  Update cost array (use npart_thismb[nmb]?)
  pmy_part->nprtcl_thispack = new_npart;
  Mesh *pm = pmy_part->pmy_pack->pmesh;
  pm->nprtcl_thisrank = new_npart;
  // refresh the global counts on the particle communicator (the legacy call was the lone
  // particle collective on MPI_COMM_WORLD), and keep nprtcl_total consistent with the
  // refreshed per-rank counts (a no-op invariant until destruction exists). If no rank
  // sent anything this cycle the counts cannot have changed: skip the collective
  // (sends_allranks is Allgather'd, so this branch is identical on every rank).
  if (!(sends_allranks.empty())) {
    MPI_Allgather(&new_npart,1,MPI_INT,(pm->nprtcl_eachrank),1,MPI_INT,mpi_comm_part);
    pm->nprtcl_total = 0;
    for (int n=0; n<(global_variable::nranks); ++n) {
      pm->nprtcl_total += pm->nprtcl_eachrank[n];
    }
  }
#endif
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
