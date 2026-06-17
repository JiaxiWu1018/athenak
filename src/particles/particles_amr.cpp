//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_amr.cpp
//! \brief NRPIC Stage 5a entry points for particle redistribution through a dynamic-AMR
//! regrid. RedistAndRefineMeshBlocks calls these synchronously: RelabelForAMR rewrites
//! PGIDs + builds the sendlist while the OLD block geometry is still live (the half-tests
//! need the old parent center); ShipAfterAMR runs the existing migration chain once the
//! NEW MeshBlockPack gids/ranks are installed. Feedback is OFF in 5a (cross-level Tmunu
//! deposition is Stage 5b), so these never run with feedback=true.

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::RelabelForAMR()
//! \brief Thin wrapper: rewrite every PGID from its old block's fate and build the
//! cross-rank sendlist (the device pass lives in bvals_part_amr.cpp). Called at the
//! regrid hook where the OLD mb_size and the (oldtonew / refine_flag / newrank) maps are
//! all still live.

TaskStatus Particles::RelabelForAMR(const DualArray1D<int> &oldtonew,
                                    const DualArray1D<int> &newrank,
                                    const DualArray1D<int> &refflag, int old_gids) {
  return pbval_part->SetPrtclGIDForAMR(oldtonew, newrank, refflag, old_gids);
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::ShipAfterAMR()
//! \brief Ship the cross-rank movers built by RelabelForAMR through the EXISTING
//! migration chain, then validate. Runs on every rank (CountSendsAndRecvs is collective
//! -- an Allgather/Allgatherv on mpi_comm_part -- so never guard it by nprtcl_send).
//! The chain is grid-agnostic given a sendlist; it is reused verbatim. CheckMigration is
//! run here (not deferred to the next cycle's task list) so a remap bug trips the
//! containment + two-sided ledger oracle BEFORE the next Push reads the new PGIDs.

void Particles::ShipAfterAMR() {
  pbval_part->CountSendsAndRecvs();
  pbval_part->InitPrtclRecv();
  pbval_part->PackAndSendPrtcls();
  // RecvAndUnpackPrtcls returns incomplete until the non-blocking receives land; spin (a
  // synchronous analogue of the task list re-invoking it). Serial builds complete on
  // the first call (no MPI receives). The merged hole compaction + count refresh happen
  // inside it; with nprtcl_destroy == 0 the only holes are the shipped movers.
  while (pbval_part->RecvAndUnpackPrtcls() == TaskStatus::incomplete) {}
  pbval_part->ClearPrtclSend();
  pbval_part->ClearPrtclRecv();
  CheckMigration(nullptr, 0);
  return;
}

} // namespace particles
