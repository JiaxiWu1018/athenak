//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rk3_predictor.cpp
//! \brief Stage-consistent SSPRK3 coupling to implicit-midpoint dust particles.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "bvals/bvals.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {

void Particles::RKCheckDomain() {
  auto pr = prtcl_rdata;
  auto box = pmy_pack->pmesh->mesh_size;
  int bad = 0;
  Kokkos::parallel_reduce("rk_particle_domain", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, nprtcl_thispack), KOKKOS_LAMBDA(int p, int &count) {
    Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
    if (!(x >= box.x1min && x < box.x1max && y >= box.x2min && y < box.x2max &&
          z >= box.x3min && z < box.x3max)) {++count;}
  }, Kokkos::Sum<int>(bad));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (bad != 0) {
    std::cerr << "### FATAL ERROR rk3_predictor: " << bad
              << " particle states leave the domain; removal/wrapping during "
              << "temporary stages is unsupported." << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::exit(EXIT_FAILURE);
  }
}

TaskStatus Particles::RKParticleStage(Driver *pdriver, int stage) {
  if (stage == 1) {
    rk_stage_active = true;
    if (rk_prediction.extent_int(0) != nrdata ||
        rk_prediction.extent_int(1) != nprtcl_thispack) {
      Kokkos::realloc(rk_prediction, nrdata, nprtcl_thispack);
      Kokkos::realloc(rk_endpoint, nrdata, nprtcl_thispack);
    }
    Kokkos::deep_copy(rk_prediction, prtcl_rdata);
  } else if (stage == 3) {
    Kokkos::deep_copy(rk_endpoint, prtcl_rdata);
  }
  if (stage == 1 || stage == 3) {
    switch (pmy_pack->pmesh->mb_indcs.ng) {
      case 2: RKParticleSolve<2>(stage == 1); break;
      case 3: RKParticleSolve<3>(stage == 1); break;
      case 4: RKParticleSolve<4>(stage == 1); break;
      default: std::exit(EXIT_FAILURE);
    }
  }
  // Stage 1 reuses the endpoint deposition from initialization / the preceding step.
  // Stages 2 and 3 use U_A and U_B respectively, after their boundary/ADM refreshes.
  if (stage == 2 || stage == 3) {RKStageDeposit(pdriver, stage);}
  return TaskStatus::complete;
}

void Particles::RKStageDeposit(Driver *pdriver, int stage) {
  Mesh *mesh = pmy_pack->pmesh;
  const int committed_count = nprtcl_thispack;
  const int saved_cross[4] = {nmigr_face, nmigr_edge, nmigr_corner, nsearch_fail};
  const int saved_rank_count = mesh->nprtcl_thisrank;
  const auto saved_total = mesh->nprtcl_total;
  std::vector<int> saved_counts(mesh->nprtcl_eachrank,
                               mesh->nprtcl_eachrank + global_variable::nranks);
  // Shallow handles keep the committed arrays alive while scratch routing reallocates.
  auto committed_r = prtcl_rdata;
  auto committed_i = prtcl_idata;
  if (rk_scratch_r.extent_int(0) != nrdata ||
      rk_scratch_r.extent_int(1) != committed_count) {
    Kokkos::realloc(rk_scratch_r, nrdata, committed_count);
    Kokkos::realloc(rk_scratch_i, nidata, committed_count);
  }
  Kokkos::deep_copy(rk_scratch_i, committed_i);
  if (stage == 2) {
    Kokkos::deep_copy(rk_scratch_r, rk_prediction);
  } else {
    Kokkos::deep_copy(rk_scratch_r, committed_r);
    auto sr = rk_scratch_r;
    auto endpoint = rk_endpoint;
    par_for("rk_midpoint_particles", DevExeSpace(), 0, committed_count-1,
      KOKKOS_LAMBDA(int p) {
        for (int a = IPX; a <= IPVZ; ++a) {
          sr(a,p) = 0.5*(committed_r(a,p) + endpoint(a,p));
        }
      });
  }
  prtcl_rdata = rk_scratch_r;
  prtcl_idata = rk_scratch_i;
  RKCheckDomain();  // Never let temporary routing touch the destruction ledger.
  pbval_part->SetNewPrtclGID();
  ShipAfterAMR();   // Established collective transport; no regrid is performed here.
  SetPrtclTmunu(pdriver, stage);
  Kokkos::fence();
  rk_scratch_r = prtcl_rdata;
  rk_scratch_i = prtcl_idata;
  prtcl_rdata = committed_r;
  prtcl_idata = committed_i;
  nprtcl_thispack = committed_count;
  mesh->nprtcl_thisrank = saved_rank_count;
  mesh->nprtcl_total = saved_total;
  for (int n = 0; n < global_variable::nranks; ++n) {
    mesh->nprtcl_eachrank[n] = saved_counts[n];
  }
  nmigr_face = saved_cross[0];
  nmigr_edge = saved_cross[1];
  nmigr_corner = saved_cross[2];
  nsearch_fail = saved_cross[3];
}

} // namespace particles
