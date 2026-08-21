//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_destroy.cpp
//! \brief death-record ledger and end-of-run accounting for destroyed particles
//! (NRPIC Stage 3c). Every destroyed particle gets one CSV row with its exact state at
//! the marking step -- cycle, time, tag, reason (exit|sphere|lapse|horizon), position,
//! velocity, owning gid/rank, and the criterion value -- so destruction events are never
//! lost
//! between (possibly widely spaced) particle output dumps. The record is exact AT
//! marking: the first state that violates the criterion, i.e. within one dt of the true
//! crossing; the pre-violation state is reconstructible as x - v*dt (exact for the
//! drift pusher, first order for gr_boris).
//!
//! FlushDeathLog runs on every rank on every cycle with a nonzero GLOBAL destruction
//! census (rank-consistent by construction); records are gathered to rank 0, which
//! appends to <basename>.prtcl_destroy.csv. The file is opened in append mode (the
//! .hst convention), so restarted runs continue the same file; tags are globally unique
//! and die at most once, so cross-segment merging/dedup is trivially by tag.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "bvals/bvals.hpp"

namespace particles {

namespace {
// one death record, POD for the byte-wise MPI_Gatherv (3 ints + pad + 7 Reals)
struct DeathRec {
  int tag, gid, reason;
  Real r[7];   // {x, y, z, vx, vy, vz, crit}
};
// indexed by ParticlesDeathReason. Adding a name here EXTENDS the set of values the
// `reason` column of <basename>.prtcl_destroy.csv can take; the column set, the header
// line, and the meaning of every existing value are unchanged, so readers that switch on
// the known names keep working and only need a new case for the new one.
const char *reason_name[NPRTCL_DEATH_REASON] = {"exit", "sphere", "lapse", "horizon"};
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void Particles::FlushDeathLog()
//! \brief gather this cycle's death records to rank 0 and append them to the CSV

void Particles::FlushDeathLog() {
  int nloc = pbval_part->nprtcl_destroy;

  // pack this rank's records from the device record arrays.
  // The record arrays are (7, cap) / (3, cap) LayoutRight and cap is grow-only, so a
  // column subview (ALL, 0:nloc) is NON-CONTIGUOUS whenever nloc < cap. Mirroring such a
  // strided device view to the host has no available copy mechanism on a separate-memory
  // -space backend (HIP/CUDA) and Kokkos throws
  //   "deep_copy with no available copy mechanism ... must be contiguous".
  // Copy the FULL (contiguous) views instead and read only the nloc active columns; the
  // capacity tail is stale padding and is never consumed, so the CSV is unchanged.
  std::vector<DeathRec> loc(nloc);
  if (nloc > 0) {
    auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  pbval_part->destroy_rec_r);
    auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  pbval_part->destroy_rec_i);
    for (int n=0; n<nloc; ++n) {
      loc[n].tag    = hi(0,n);
      loc[n].gid    = hi(1,n);
      loc[n].reason = hi(2,n);
      for (int k=0; k<7; ++k) {loc[n].r[k] = hr(k,n);}
    }
  }

  // the particle tasks run before the driver advances time, so this cycle's deaths
  // happened at time + dt (the post-push instant the criterion was evaluated at)
  Mesh *pm = pmy_pack->pmesh;
  Real tdeath = pm->time + pm->dt;
  int cycle = pm->ncycle;

  int nranks = global_variable::nranks;
  int myrank = global_variable::my_rank;
  std::vector<DeathRec> all;
  std::vector<int> nrec_eachrank(nranks, nloc);
#if MPI_PARALLEL_ENABLED
  // per-rank record counts are already known from the destruction census -- size the
  // gather without any extra communication
  int ntot = 0;
  std::vector<int> rcnt(nranks), rdis(nranks);
  for (int r=0; r<nranks; ++r) {
    nrec_eachrank[r] = pbval_part->ndest_eachrank[r];
    rcnt[r] = nrec_eachrank[r]*static_cast<int>(sizeof(DeathRec));
    rdis[r] = ntot*static_cast<int>(sizeof(DeathRec));
    ntot += nrec_eachrank[r];
  }
  if (nloc != nrec_eachrank[myrank]) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "death-log census mismatch: " << nloc << " records vs "
              << nrec_eachrank[myrank] << " in census" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (myrank == 0) {all.resize(ntot);}
  MPI_Gatherv(loc.data(), nloc*static_cast<int>(sizeof(DeathRec)), MPI_BYTE,
              all.data(), rcnt.data(), rdis.data(), MPI_BYTE, 0,
              pbval_part->mpi_comm_part);
  if (myrank != 0) {return;}
#else
  all.swap(loc);
#endif

  // rank 0: append one row per record (source rank recovered from the gather order)
  FILE *pfile = std::fopen(destroy_log_fname.c_str(), "a");
  if (pfile == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "could not open '" << destroy_log_fname << "'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::fseek(pfile, 0, SEEK_END);
  if (std::ftell(pfile) == 0) {
    std::fprintf(pfile, "# cycle,time,tag,reason,x1,x2,x3,v1,v2,v3,gid,rank,crit\n");
  }
  int n = 0;
  for (int r=0; r<nranks; ++r) {
    for (int q=0; q<nrec_eachrank[r]; ++q, ++n) {
      const DeathRec &d = all[n];
      std::fprintf(pfile,
          "%d,%.17g,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g\n",
          cycle, tdeath, d.tag,
          (d.reason >= 0 && d.reason < NPRTCL_DEATH_REASON) ? reason_name[d.reason]
                                                            : "unknown",
          d.r[0], d.r[1], d.r[2], d.r[3], d.r[4], d.r[5], d.gid, r, d.r[6]);
    }
  }
  std::fclose(pfile);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::PrintFinalSummary()
//! \brief end-of-run particle accounting: initial/final counts, destroyed per reason,
//! and the count-conservation verdict. Uses only rank-identical Mesh bookkeeping (the
//! census-fed cumulative ledger), so it is exact with or without <particles> debug.
//! Called on rank 0 from Driver::Finalize.

void Particles::PrintFinalSummary() {
  Mesh *pm = pmy_pack->pmesh;
  int64_t dtot = 0;
  for (int k=0; k<NPRTCL_DEATH_REASON; ++k) {dtot += pm->nprtcl_destroyed_cum[k];}
  bool ok = (static_cast<int64_t>(pm->nprtcl_total) + dtot ==
             static_cast<int64_t>(pm->nprtcl_initial));
  // The horizon field is appended AFTER lapse and BEFORE the verdict, so the existing
  // "exit=/sphere=/lapse=" fields keep their names, order and meaning.
  std::cout << std::endl << "particles: initial=" << pm->nprtcl_initial
            << " final=" << pm->nprtcl_total
            << " destroyed: exit=" << pm->nprtcl_destroyed_cum[PrtclDeathExit]
            << " sphere=" << pm->nprtcl_destroyed_cum[PrtclDeathSphere]
            << " lapse=" << pm->nprtcl_destroyed_cum[PrtclDeathLapse]
            << " horizon=" << pm->nprtcl_destroyed_cum[PrtclDeathHorizon]
            << (ok ? "  [conservation OK]" : "  [conservation VIOLATED]") << std::endl;
  return;
}

} // namespace particles
