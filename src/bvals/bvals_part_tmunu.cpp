//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_part_tmunu.cpp
//! \brief cross-rank transport of Tmunu ghost-image records (NRPIC Stage 4c).
//!
//! The deposit kernel (particles_tmunu.cpp) writes only its own MeshBlock's physical
//! cells; the share of a boundary-band particle's CIC cloud that falls in a neighbor is
//! delivered as a TmunuImage. When that neighbor is on another rank the image is staged
//! as a TmunuImageWire (carrying the GLOBAL target gid) and shipped here. The exchange is
//! SYNCHRONOUS and blocking -- the Tmunu deposit is the last task in after_timeintegrator
//! (nothing to overlap) and also runs as a driver-init seed OUTSIDE any task list, where
//! the migration-style "return incomplete and re-poll" split is unavailable. Received
//! images are appended to the SAME tmunu_images queue and deposited by the one canonical
//! (target_m, tag, off_code, lev) pass in set_prtcl_tmunu, so the per-cell accumulation
//! order is identical for every rank decomposition: cross-rank feedback is bitwise
//! rank-count invariant by construction (serial-host; GPU atomics are correct but not
//! bit-reproducible, as for the same-rank deposit).
//!
//! Mirrors the particle-migration census (bvals_part.cpp): a per-rank message Allgather
//! then an Allgatherv of {sendrank,recvrank,count} tuples on the SAME mpi_comm_part with
//! the SAME mpi_ituple datatype, but with distinct message tags (2 = Reals, 3 = ints) and
//! a SEPARATE census (the migration census runs in an earlier task, before images exist).
//! Order on the wire is irrelevant: the receiver re-sorts every image before depositing.

#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {

// buffer widths per image: 14 Reals {delta[3], x[3], mass, lorentz, u_d[3], sxmin[3]} and
// 8 ints {target_gid, tag, off_code, lev, idx[3], slev} -- match the pack/unpack kernels
// below (x[3]/lev/slev/sxmin[3] carry the cross-level (5b) payload: scheme B uses x[3];
// scheme A's restrict uses idx[3]+sxmin[3]+slev; same-level images ignore them)
namespace {
constexpr int kImgNR = 14;
constexpr int kImgNI = 8;
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ExchangeTmunuImages()
//! \brief ship cross-rank ghost images and append the received ones into tmunu_images.
//! No-op in serial builds (a single rank never stages a cross-rank image). Collective on
//! mpi_comm_part: must be called on every rank (it is -- the tmunu deposit runs on all
//! ranks whenever feedback is on, both in the task list and at the init seed).

void ParticlesBoundaryValues::ExchangeTmunuImages() {
#if MPI_PARALLEL_ENABLED
  Mesh *pm = pmy_part->pmy_pack->pmesh;
  int gids = pmy_part->pmy_pack->gids;
  int nmb  = pmy_part->pmy_pack->nmb_thispack;
  int myrank = global_variable::my_rank;
  int nranks = global_variable::nranks;
  int n_send = pmy_part->nimg_send_thispack;   // images staged by the deposit kernel
  int *rank_eachmb = pm->rank_eachmb;
  auto &imgs = pmy_part->tmunu_img_send;

  // ---- (a) group the staged images into one message per destination rank. The deposit
  // kernel filled the device view, so bring it to host, sort by rank_eachmb[target_gid]
  // (order within a message is irrelevant -- the receiver re-sorts), then push the sorted
  // order back to the device for the pack kernel below.
  if (n_send > 0) {
    namespace KE = Kokkos::Experimental;
    imgs.template modify<DevExeSpace>();
    imgs.template sync<HostMemSpace>();
    std::sort(KE::begin(imgs.h_view), KE::begin(imgs.h_view) + n_send,
              [rank_eachmb](const TmunuImageWire &a, const TmunuImageWire &b) {
                return rank_eachmb[a.target_gid] < rank_eachmb[b.target_gid];
              });
    imgs.template modify<HostMemSpace>();
    imgs.template sync<DevExeSpace>();
  }
  imgsends_thisrank.clear();
  if (n_send > 0) {
    int rank = rank_eachmb[imgs.h_view(0).target_gid];
    int cnt = 1;
    for (int n=1; n<n_send; ++n) {
      int r = rank_eachmb[imgs.h_view(n).target_gid];
      if (r == rank) {
        ++cnt;
      } else {
        imgsends_thisrank.emplace_back(ParticleMessageData(myrank, rank, cnt));
        rank = r;
        cnt = 1;
      }
    }
    imgsends_thisrank.emplace_back(ParticleMessageData(myrank, rank, cnt));
  }
  n_img_send_msgs = imgsends_thisrank.size();

  // ---- (b) census: Allgather the per-rank message counts, then Allgatherv the tuples
  // (reusing the migration's committed 3-int mpi_ituple). A separate exchange from the
  // migration census, which has already completed in an earlier task this cycle.
  MPI_Allgather(&n_img_send_msgs, 1, MPI_INT, n_imgsend_eachrank.data(), 1, MPI_INT,
                mpi_comm_part);
  std::vector<int> displ(nranks);
  displ[0] = 0;
  for (int n=1; n<nranks; ++n) {displ[n] = displ[n-1] + n_imgsend_eachrank[n-1];}
  int nmsg_all = displ[nranks-1] + n_imgsend_eachrank[nranks-1];
  imgsends_allranks.assign(nmsg_all, ParticleMessageData(0, 0, 0));
  for (int n=0; n<n_imgsend_eachrank[myrank]; ++n) {
    imgsends_allranks[n + displ[myrank]] = imgsends_thisrank[n];
  }
  MPI_Allgatherv(MPI_IN_PLACE, n_imgsend_eachrank[myrank], mpi_ituple,
                 imgsends_allranks.data(), n_imgsend_eachrank.data(), displ.data(),
                 mpi_ituple, mpi_comm_part);

  // ---- (c) my receives: scan the global send matrix for messages addressed to me
  imgrecvs_thisrank.clear();
  for (int n=0; n<nmsg_all; ++n) {
    if (imgsends_allranks[n].recvrank == myrank) {
      imgrecvs_thisrank.emplace_back(imgsends_allranks[n]);
    }
  }
  n_img_recv_msgs = imgrecvs_thisrank.size();
  n_img_recv = 0;
  for (int n=0; n<n_img_recv_msgs; ++n) {n_img_recv += imgrecvs_thisrank[n].nprtcls;}

  bool no_errors = true;

  // ---- (d) post non-blocking receives (one contiguous slice per sending rank), tags 2/3
  if (n_img_recv > 0) {
    Kokkos::realloc(img_rrecvbuf, kImgNR*n_img_recv);
    Kokkos::realloc(img_irecvbuf, kImgNI*n_img_recv);
  }
  img_rrecv_req.assign(n_img_recv_msgs, MPI_REQUEST_NULL);
  img_irecv_req.assign(n_img_recv_msgs, MPI_REQUEST_NULL);
  {
    int rstart = 0, istart = 0;
    for (int n=0; n<n_img_recv_msgs; ++n) {
      int np = imgrecvs_thisrank[n].nprtcls;
      int src = imgrecvs_thisrank[n].sendrank;
      auto rp = Kokkos::subview(img_rrecvbuf, std::make_pair(rstart, rstart + kImgNR*np));
      if (MPI_Irecv(rp.data(), kImgNR*np, MPI_ATHENA_REAL, src, 2, mpi_comm_part,
                    &(img_rrecv_req[n])) != MPI_SUCCESS) {no_errors = false;}
      rstart += kImgNR*np;
      auto ip = Kokkos::subview(img_irecvbuf, std::make_pair(istart, istart + kImgNI*np));
      if (MPI_Irecv(ip.data(), kImgNI*np, MPI_INT, src, 3, mpi_comm_part,
                    &(img_irecv_req[n])) != MPI_SUCCESS) {no_errors = false;}
      istart += kImgNI*np;
    }
  }

  // ---- (e) pack the staged images into flat buffers on device, then post the sends
  if (n_send > 0) {
    Kokkos::realloc(img_rsendbuf, kImgNR*n_send);
    Kokkos::realloc(img_isendbuf, kImgNI*n_send);
    auto &rbuf = img_rsendbuf;
    auto &ibuf = img_isendbuf;
    auto &simg = imgs;
    par_for("img_pack", DevExeSpace(), 0, (n_send-1), KOKKOS_LAMBDA(const int n) {
      TmunuImageWire w = simg.d_view(n);
      ibuf(kImgNI*n + 0) = w.target_gid;
      ibuf(kImgNI*n + 1) = w.tag;
      ibuf(kImgNI*n + 2) = w.off_code;
      ibuf(kImgNI*n + 3) = w.lev;
      ibuf(kImgNI*n + 4) = w.idx[0];
      ibuf(kImgNI*n + 5) = w.idx[1];
      ibuf(kImgNI*n + 6) = w.idx[2];
      ibuf(kImgNI*n + 7) = w.slev;
      rbuf(kImgNR*n + 0) = w.delta[0];
      rbuf(kImgNR*n + 1) = w.delta[1];
      rbuf(kImgNR*n + 2) = w.delta[2];
      rbuf(kImgNR*n + 3) = w.x[0];
      rbuf(kImgNR*n + 4) = w.x[1];
      rbuf(kImgNR*n + 5) = w.x[2];
      rbuf(kImgNR*n + 6) = w.mass;
      rbuf(kImgNR*n + 7) = w.lorentz;
      rbuf(kImgNR*n + 8) = w.u_d[0];
      rbuf(kImgNR*n + 9) = w.u_d[1];
      rbuf(kImgNR*n + 10) = w.u_d[2];
      rbuf(kImgNR*n + 11) = w.sxmin[0];
      rbuf(kImgNR*n + 12) = w.sxmin[1];
      rbuf(kImgNR*n + 13) = w.sxmin[2];
    });
    Kokkos::fence();
  }
  img_rsend_req.assign(n_img_send_msgs, MPI_REQUEST_NULL);
  img_isend_req.assign(n_img_send_msgs, MPI_REQUEST_NULL);
  {
    int rstart = 0, istart = 0;
    for (int n=0; n<n_img_send_msgs; ++n) {
      int np = imgsends_thisrank[n].nprtcls;
      int dst = imgsends_thisrank[n].recvrank;
      auto rp = Kokkos::subview(img_rsendbuf, std::make_pair(rstart, rstart + kImgNR*np));
      if (MPI_Isend(rp.data(), kImgNR*np, MPI_ATHENA_REAL, dst, 2, mpi_comm_part,
                    &(img_rsend_req[n])) != MPI_SUCCESS) {no_errors = false;}
      rstart += kImgNR*np;
      auto ip = Kokkos::subview(img_isendbuf, std::make_pair(istart, istart + kImgNI*np));
      if (MPI_Isend(ip.data(), kImgNI*np, MPI_INT, dst, 3, mpi_comm_part,
                    &(img_isend_req[n])) != MPI_SUCCESS) {no_errors = false;}
      istart += kImgNI*np;
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error posting ghost-image sends/recvs" << std::endl << std::flush;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // ---- (f) wait on receives, then append the received images into tmunu_images. The
  // append is a pure tail-write of the device view (slots [base, base+n_img_recv)); the
  // existing self+same-rank images in [0, base) are preserved by Kokkos::resize. The
  // canonical sort + deposit back in set_prtcl_tmunu then runs over the merged set.
  if (n_img_recv_msgs > 0) {
    MPI_Waitall(n_img_recv_msgs, img_rrecv_req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(n_img_recv_msgs, img_irecv_req.data(), MPI_STATUSES_IGNORE);
  }
  if (n_img_recv > 0) {
    int base = pmy_part->nimages_thispack;
    int need = base + n_img_recv;
    // the self/same-rank records in [0,base) were written on the DEVICE view by the
    // generation kernel; mark it modified so the DualView resize below grows from (and
    // preserves) the device copy unambiguously, not a stale host mirror.
    pmy_part->tmunu_images.template modify<DevExeSpace>();
    if (need > static_cast<int>(pmy_part->tmunu_images.extent(0))) {
      Kokkos::resize(pmy_part->tmunu_images, need);   // grows, preserving [0, base)
    }
    auto &img = pmy_part->tmunu_images;
    auto &rbuf = img_rrecvbuf;
    auto &ibuf = img_irecvbuf;
    DvceArray1D<int> rerr("img_recv_err", 1);   // zero-initialized: bad target_gid count
    par_for("img_unpack", DevExeSpace(), 0, (n_img_recv-1), KOKKOS_LAMBDA(const int n) {
      int target_gid = ibuf(kImgNI*n + 0);
      int tm = target_gid - gids;
      if (tm < 0 || tm >= nmb) {
        Kokkos::atomic_add(&rerr(0), 1);   // misrouted image: not a block on this rank
        return;
      }
      TmunuImage rec;
      rec.target_m = tm;
      rec.tag      = ibuf(kImgNI*n + 1);
      rec.off_code = ibuf(kImgNI*n + 2);
      rec.lev      = ibuf(kImgNI*n + 3);
      rec.idx[0]   = ibuf(kImgNI*n + 4);
      rec.idx[1]   = ibuf(kImgNI*n + 5);
      rec.idx[2]   = ibuf(kImgNI*n + 6);
      rec.slev     = ibuf(kImgNI*n + 7);
      rec.delta[0] = rbuf(kImgNR*n + 0);
      rec.delta[1] = rbuf(kImgNR*n + 1);
      rec.delta[2] = rbuf(kImgNR*n + 2);
      rec.x[0]     = rbuf(kImgNR*n + 3);
      rec.x[1]     = rbuf(kImgNR*n + 4);
      rec.x[2]     = rbuf(kImgNR*n + 5);
      rec.mass     = rbuf(kImgNR*n + 6);
      rec.lorentz  = rbuf(kImgNR*n + 7);
      rec.u_d[0]   = rbuf(kImgNR*n + 8);
      rec.u_d[1]   = rbuf(kImgNR*n + 9);
      rec.u_d[2]   = rbuf(kImgNR*n + 10);
      rec.sxmin[0] = rbuf(kImgNR*n + 11);
      rec.sxmin[1] = rbuf(kImgNR*n + 12);
      rec.sxmin[2] = rbuf(kImgNR*n + 13);
      img.d_view(base + n) = rec;
    });
    auto hrerr = Kokkos::create_mirror_view(rerr);
    Kokkos::deep_copy(hrerr, rerr);   // fences the unpack kernel
    if (hrerr(0) > 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "rank " << myrank << " received " << hrerr(0)
                << " ghost images whose target gid is not a local block (misrouted "
                << "transport or corrupt payload)" << std::endl << std::flush;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    pmy_part->nimages_thispack = need;   // received images joined the local queue
  }

  // wait on the sends so the staging buffers are free to be reused next cycle
  if (n_img_send_msgs > 0) {
    MPI_Waitall(n_img_send_msgs, img_rsend_req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(n_img_send_msgs, img_isend_req.data(), MPI_STATUSES_IGNORE);
  }

  // ---- (g) conservation validator (<particles> debug >= 1): every shipped image must be
  // received exactly once somewhere. Combined with the receive-side bounds check above
  // (which kills a misroute) and the global deposit identity in set_prtcl_tmunu (which
  // kills a lost/duplicated share), this closes the transport. Collective; debug_lvl is
  // input-uniform across ranks so every rank reaches it.
  if (pmy_part->debug_lvl >= 1) {
    int64_t sr[2] = {static_cast<int64_t>(n_send), static_cast<int64_t>(n_img_recv)};
    MPI_Allreduce(MPI_IN_PLACE, sr, 2, MPI_INT64_T, MPI_SUM, mpi_comm_part);
    if (sr[0] != sr[1]) {
      if (myrank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "ghost-image transport not conserved at cycle "
                  << pm->ncycle << ": global sent=" << sr[0] << " != received=" << sr[1]
                  << std::endl;
      }
      std::cout << std::flush;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (myrank == 0 && sr[0] > 0) {
      std::cout << "[tmunu-debug] cycle=" << pm->ncycle
                << " ghost images sent==received==" << sr[0] << " (global)" << std::endl;
    }
  }
#endif  // MPI_PARALLEL_ENABLED
  return;
}

}  // namespace particles
