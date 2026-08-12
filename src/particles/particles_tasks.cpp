//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_tasks.cpp
//! \brief functions that control Particles tasks stored in tasklists in MeshBlockPack

#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <cstdlib>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::AssembleTasks
//! \brief Adds hydro tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after Hydro constructor.

void Particles::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // Particle integration runs in the "after_timeintegrator" task list, which executes
  // once per cycle with the full dt (after the fluid/Z4c stages). The full-dt Boris/GR
  // pushers therefore need no per-stage gating, and for a dynamical metric the push reads
  // the updated (n+1) metric while the *_last snapshots hold step n.
  auto &atl = tl["after_timeintegrator"];
  id.push   = atl->AddTask(&Particles::Push, this, none);
  // excision marking (Stage 3c(b)): scheduled only when a criterion is enabled; writes
  // the excise_flag/crit arrays that NewGID's destruction marking consumes
  TaskID newgid_dep = id.push;
  if (excise_any) {
    id.excise = atl->AddTask(&Particles::MarkExcised, this, id.push);
    newgid_dep = id.excise;
  }
  id.newgid = atl->AddTask(&Particles::NewGID, this, newgid_dep);
  id.count  = atl->AddTask(&Particles::SendCnt, this, id.newgid);
  id.irecv  = atl->AddTask(&Particles::InitRecv, this, id.count);
  id.sendp  = atl->AddTask(&Particles::SendP, this, id.irecv);
  id.recvp  = atl->AddTask(&Particles::RecvP, this, id.sendp);
  id.crecv  = atl->AddTask(&Particles::ClearRecv, this, id.recvp);
  id.csend  = atl->AddTask(&Particles::ClearSend, this, id.crecv);
  // post-migration validation (no-op unless <particles> debug >= 1; particles_debug.cpp)
  id.check  = atl->AddTask(&Particles::CheckMigration, this, id.csend);
  // particle timestep + conserved energy, after migration so positions/gids/counts are
  // final
  id.newdt  = atl->AddTask(&Particles::NewTimeStep, this, id.check);
  id.energy = atl->AddTask(&Particles::EnergyCalculation, this, id.newdt);
  // stress-energy deposition (Stage 4): last in the chain so positions, velocities and
  // the t^{n+1} metric (Z4cToADM ran at the final RK stage) are all final; the deposited
  // Tmunu is then frozen across every RK substage of the next cycle (first-order matter
  // coupling). Queued only when feedback is on.
  if (feedback) {
    id.tmunu = atl->AddTask(&Particles::SetPrtclTmunu, this, id.energy);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::NewGID
//! \brief Wrapper task list function to set new GID for particles that move between
//! MeshBlocks.

TaskStatus Particles::NewGID(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->SetNewPrtclGID();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::SendCnt
//! \brief Wrapper task list function to set share number of particles communicated with
//! MPI between all ranks

TaskStatus Particles::SendCnt(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->CountSendsAndRecvs();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI).

TaskStatus Particles::InitRecv(Driver *pdrive, int stage) {
  // post receives for particles
  TaskStatus tstat = pbval_part->InitPrtclRecv();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::SendP()
//! \brief Wrapper task list function to pack/send particles

TaskStatus Particles::SendP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->PackAndSendPrtcls();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::RecvP
//! \brief Wrapper task list function to receive/unpack particles

TaskStatus Particles::RecvP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->RecvAndUnpackPrtcls();
  return tstat;
}


//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.

TaskStatus Particles::ClearSend(Driver *pdrive, int stage) {
  // check sends of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclSend();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed.

TaskStatus Particles::ClearRecv(Driver *pdrive, int stage) {
  // check receives of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclRecv();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::EnergyCalculation
//! \brief Wrapper task that dispatches calc_prtcl_energy<NGHOST> on the active ghost
//! count.

TaskStatus Particles::EnergyCalculation(Driver *pdrive, int stage) {
  int ng = pmy_pack->pmesh->mb_indcs.ng;
  switch (ng) {
    case 2: calc_prtcl_energy<2>(); break;
    case 3: calc_prtcl_energy<3>(); break;
    case 4: calc_prtcl_energy<4>(); break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "particle energy supports NGHOST=2,3,4 only (got " << ng << ")"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  if (pusher == ParticlesPusher::gr_boris && gr_boris_diagnostics) {
    GRBorisDiagnostics();
  }
  return TaskStatus::complete;
}

} // namespace particles
