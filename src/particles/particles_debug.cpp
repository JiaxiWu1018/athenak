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
//!   (4) the GLOBAL conservation ledger holds: {particle count, sum of tags, sum of
//!       tag^2} (Allreduced across ranks in MPI builds) equal the values captured at the
//!       first check -- no destruction exists yet, so all three are exact invariants.
//!       The tag checksums catch identity corruption that the count cannot: a lost
//!       particle replaced by a duplicate of another (the hole-compaction failure
//!       signature);
//!   (5) per-rank bookkeeping is consistent: this rank's count equals its published
//!       nprtcl_eachrank entry, and the published counts sum to nprtcl_total and to the
//!       Allreduced true total.
//! Any violation is FATAL: every offending particle is printed (tag, gid, position,
//! velocity, owning-block bbox), then the job is killed -- MPI_Abort in MPI builds (a
//! plain exit on one rank would hang the others at the next collective). A per-cycle
//! summary of face/edge/corner crossings (counted in SetNewPrtclGID) is printed when
//! nonzero, tagged with the rank.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "bvals/bvals.hpp"
#include "bvals/prtcl_search.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::CheckMigration
//! \brief validate the post-migration particle state (no-op unless debug >= 1)

TaskStatus Particles::CheckMigration(Driver *pdrive, int stage) {
  if (debug_lvl < 1) {return TaskStatus::complete;}

  int ncycle = pmy_pack->pmesh->ncycle;
  int npart = nprtcl_thispack;
  int myrank = global_variable::my_rank;

  // per-cycle migration/destruction summary (counters filled by SetNewPrtclGID)
  int ndest_cycle = ndestroy_thisrank[0] + ndestroy_thisrank[1] + ndestroy_thisrank[2];
  if ((nmigr_face + nmigr_edge + nmigr_corner + nsearch_fail + ndest_cycle) > 0) {
    std::cout << "[prtcl-debug] rank=" << myrank << " cycle=" << ncycle
              << " migrations: face=" << nmigr_face
              << " edge=" << nmigr_edge << " corner=" << nmigr_corner
              << " search_fail=" << nsearch_fail;
    if (ndest_cycle > 0) {
      std::cout << " destroyed={" << ndestroy_thisrank[0] << ","
                << ndestroy_thisrank[1] << "," << ndestroy_thisrank[2] << "}";
    }
    std::cout << " npart=" << npart << std::endl;
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

  // TWO-SIDED conservation ledger: GLOBAL alive {count, sum of tags, sum of tag^2}
  // plus the destroyed-side counterparts (count from the census-fed Mesh cums --
  // already global -- and checksums accumulated at marking time, Allreduced here).
  // Captured lazily at the first check as ledger0 = alive + dead (dead can already be
  // nonzero: the first check runs after cycle-1 destructions, and a restarted segment
  // re-captures against its own segment-local dead checksums); the invariant
  // alive + dead == ledger0 then holds component-wise every cycle. Cross-rank sends
  // move particles between ranks and destruction moves them to the dead side; nothing
  // may appear, vanish, or change identity -- which is exactly what the tag checksums
  // verify across compaction events.
  uint64_t tsum = 0, tsq = 0;
  Kokkos::parallel_reduce("part_ledger",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, uint64_t &s1, uint64_t &s2) {
      // cast BEFORE multiplying: int tag*tag overflows at tag >= 46341
      uint64_t t = static_cast<uint64_t>(pi(PTAG,p));
      s1 += t;
      s2 += t*t;
    }, Kokkos::Sum<uint64_t>(tsum), Kokkos::Sum<uint64_t>(tsq));
  // led = {alive count, alive tag-sum, alive tag-sq, dead tag-sum, dead tag-sq}
  uint64_t led[5] = {static_cast<uint64_t>(npart), tsum, tsq,
                     ledger_dead[0], ledger_dead[1]};
#if MPI_PARALLEL_ENABLED
  // collective is safe: this task runs on every rank each cycle and debug_lvl is
  // input-file-uniform across ranks
  MPI_Allreduce(MPI_IN_PLACE, led, 5, MPI_UINT64_T, MPI_SUM,
                pbval_part->mpi_comm_part);
#endif
  Mesh *pm = pmy_pack->pmesh;
  uint64_t dead_cnt = 0;
  for (int k=0; k<3; ++k) {
    dead_cnt += static_cast<uint64_t>(pm->nprtcl_destroyed_cum[k]);
  }
  if (!ledger_init) {
    ledger0[0] = led[0] + dead_cnt;
    ledger0[1] = led[1] + led[3];
    ledger0[2] = led[2] + led[4];
    ledger_init = true;
  }
  bool bad_ledger = (led[0] + dead_cnt != ledger0[0])
                 || (led[1] + led[3]  != ledger0[1])
                 || (led[2] + led[4]  != ledger0[2]);

  // per-rank bookkeeping consistency: the local count must match this rank's published
  // nprtcl_eachrank entry, and the published counts must sum to nprtcl_total and to the
  // Allreduced true total (validates both count-refresh paths: the Allgather on send
  // cycles and the census-based local decrement on destroy-only cycles)
  int64_t sum_each = 0;
  for (int n=0; n<(global_variable::nranks); ++n) {sum_each += pm->nprtcl_eachrank[n];}
  bool bad_counts = (npart != pm->nprtcl_eachrank[myrank]) ||
                    (sum_each != static_cast<int64_t>(pm->nprtcl_total)) ||
                    (static_cast<uint64_t>(sum_each) != led[0]);

  if ((nbad_gid + nbad_box + nsearch_fail) > 0 || bad_ledger || bad_counts) {
    // print every offending particle, then die
    par_for("part_check_dump",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
      int gid = pi(PGID,p);
      Real x3 = three_d ? pr(IPZ,p)  : 0.0;
      Real v3 = three_d ? pr(IPVZ,p) : 0.0;
      if (gid < gids || gid > gide) {
        Kokkos::printf("[prtcl-debug] rank=%d BAD GID: tag=%d gid=%d (pack range %d..%d)"
                       " pos=(%.16e,%.16e,%.16e) vel=(%.16e,%.16e,%.16e)\n",
                       myrank, pi(PTAG,p), gid, gids, gide,
                       pr(IPX,p), pr(IPY,p), x3, pr(IPVX,p), pr(IPVY,p), v3);
      } else {
        const RegionSize &sz = size.d_view(gid - gids);
        bool in = (pr(IPX,p) >= sz.x1min) && (pr(IPX,p) < sz.x1max)
               && (pr(IPY,p) >= sz.x2min) && (pr(IPY,p) < sz.x2max);
        if (three_d) {
          in = in && (pr(IPZ,p) >= sz.x3min) && (pr(IPZ,p) < sz.x3max);
        }
        if (!in) {
          Kokkos::printf("[prtcl-debug] rank=%d OUT OF BBOX: tag=%d gid=%d "
                         "pos=(%.16e,%.16e,%.16e) vel=(%.16e,%.16e,%.16e) "
                         "bbox x1=[%.16e,%.16e) x2=[%.16e,%.16e) x3=[%.16e,%.16e)\n",
                         myrank, pi(PTAG,p), gid,
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
          std::cout << "[prtcl-debug] rank=" << myrank << " tag=" << hi(PTAG,p)
                    << " has gid=" << gid
                    << " but should be gid=" << gids + mok << std::endl;
        } else {
          std::cout << "[prtcl-debug] rank=" << myrank << " tag=" << hi(PTAG,p)
                    << " has gid=" << gid
                    << " but no local MeshBlock contains its position" << std::endl;
        }
      }
    }

    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "particle migration check failed at cycle " << ncycle
              << " (rank " << myrank << "): bad_gid=" << nbad_gid
              << " out_of_bbox=" << nbad_box << " search_fail=" << nsearch_fail;
    if (bad_ledger) {
      std::cout << " ledger alive{count,tagsum,tagsq}={" << led[0] << "," << led[1]
                << "," << led[2] << "} + dead{" << dead_cnt << "," << led[3] << ","
                << led[4] << "} != initial {" << ledger0[0] << "," << ledger0[1] << ","
                << ledger0[2] << "}";
    }
    if (bad_counts) {
      std::cout << " counts: npart=" << npart << " eachrank[" << myrank << "]="
                << pm->nprtcl_eachrank[myrank] << " sum_eachrank=" << sum_each
                << " nprtcl_total=" << pm->nprtcl_total;
    }
    std::cout << std::endl;
#if MPI_PARALLEL_ENABLED
    // one failing rank must kill the whole job: a plain exit here would leave the other
    // ranks blocked in the next cycle's collectives
    std::cout << std::flush;
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    std::exit(EXIT_FAILURE);
#endif
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

                // ---- mirror the SetNewPrtclGID kernel pipeline via the SHARED offset
                // predicate (prtcl_search.hpp::ComputeBlockOffsets -- the audit must
                // never hand-copy the comparisons, or the two could drift apart)
                Real x3k = three_d ? pos[2] : bmin[2];
                int ix, iy, iz;
                ComputeBlockOffsets(sz, pos[0], pos[1], pos[2], three_d, ix, iy, iz);
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
