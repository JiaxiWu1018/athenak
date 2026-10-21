#ifndef Z4C_FROZEN_AH_CHECKS_HPP_
#define Z4C_FROZEN_AH_CHECKS_HPP_

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include "fastflow.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c.hpp"
#include "particles/particles.hpp"
#include "particles/gi_diagnostics.hpp"
#include "particles/gi_math_checks.hpp"

// Opt-in non-evolving validation: no integrator, pusher, or AMR step is executed.
// Used for the analytic fixture and subsequently for saved production states.
inline void FrozenAhChecks(Mesh *mesh, ParameterInput *pin, int repetitions) {
  int device=-1;
#if defined(KOKKOS_ENABLE_HIP)
  device=Kokkos::HIP().hip_device();
#elif defined(KOKKOS_ENABLE_CUDA)
  device=Kokkos::Cuda().cuda_device();
#endif
  std::vector<int> devices(global_variable::nranks,device);
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&device,1,MPI_INT,devices.data(),1,MPI_INT,MPI_COMM_WORLD);
#endif
  if(global_variable::my_rank==0)
    for(int r=0;r<global_variable::nranks;++r)
      std::cout<<"FROZEN_GPU_BINDING rank="<<r<<" device="<<devices[r]<<std::endl;
  particles::GiMathChecks();
  auto pack = mesh->pmb_pack;
  if (!pack->pz4c) throw std::runtime_error("Frozen AH checks require Z4c");
  if (!pack->ppart) throw std::runtime_error("Frozen particle/AH checks require particles");
  auto u = pack->pz4c->u0;
  auto before = Kokkos::create_mirror(u);
  Kokkos::deep_copy(before, u);
  const Real old_dt = mesh->dt;
  auto pr = pack->ppart->prtcl_rdata;
  auto particle_before = Kokkos::create_mirror(pr);
  Kokkos::deep_copy(particle_before, pr);
  mesh->dt = 0.0;  // event timestamps are the unchanged saved-state time
  const bool require_accept = pin->GetOrAddBoolean("diagnostic_checks", "require_accept", true);
  const int selected = pin->GetOrAddInteger("diagnostic_checks", "horizon", -1);
  for (size_t h=0; h<pack->pz4c->pfastflow.size(); ++h) {
    if (selected >= 0 && static_cast<int>(h) != selected) continue;
    auto *finder = pack->pz4c->pfastflow[h].get();
    switch (mesh->mb_indcs.ng) {
      case 2: finder->MetricDerivatives<2>(mesh->time); break;
      case 3: finder->MetricDerivatives<3>(mesh->time); break;
      case 4: finder->MetricDerivatives<4>(mesh->time); break;
    }
    bool accepted = false, tested_pending = false;
    for (int n=0; n<repetitions; ++n) {
      finder->Find(n, mesh->time);
      accepted |= finder->published_this_search;
      if (finder->search_reason == "pending_persistence" && !tested_pending) {
        if (!finder->search_surface_valid)
          throw std::runtime_error("quality-passing provisional seed discarded");
        std::stringstream serialized;
        pin->ParameterDump(serialized);
        ParameterInput loaded;
        loaded.LoadFromStream(serialized);
        FastFlow restored(pack, &loaded, static_cast<int>(h));
        if (!restored.search_surface_valid || restored.ah_surf_valid)
          throw std::runtime_error("restored search/publication state conflated");
        for (int a=0; a<3; ++a)
          if (restored.search_center[a] != finder->search_center[a])
            throw std::runtime_error("provisional center serialization differs");
        for (int i=0; i<finder->GetLmax()+1; ++i)
          if (restored.a0_search.h_view(i) != finder->a0_search.h_view(i))
            throw std::runtime_error("provisional coefficient serialization differs");
        for (int i=0; i<finder->GetLmpoints(); ++i)
          if (restored.ac_search.h_view(i) != finder->ac_search.h_view(i) ||
              restored.as_search.h_view(i) != finder->as_search.h_view(i))
            throw std::runtime_error("provisional shape serialization differs");
        switch (mesh->mb_indcs.ng) {
          case 2: restored.MetricDerivatives<2>(mesh->time); break;
          case 3: restored.MetricDerivatives<3>(mesh->time); break;
          case 4: restored.MetricDerivatives<4>(mesh->time); break;
        }
        for (int j=0; j<restored.ah_surf_persist; ++j) {
          restored.Find(j, mesh->time);
          if (j+1 < restored.ah_surf_persist && restored.published_this_search)
            throw std::runtime_error("restart bypassed persistence requirement");
        }
        if (!restored.published_this_search && require_accept)
          throw std::runtime_error("restored provisional seed failed to requalify");
        tested_pending = true;
      }
    }
    if (require_accept && !accepted)
      throw std::runtime_error("frozen analytic/saved horizon did not qualify");
    if (accepted) {
      const std::string key = "search_state_a0_" + std::to_string(h) + "_0";
      const Real saved = pin->GetReal("fastflow", key);
      pin->SetReal("fastflow", key, -1.0);
      {
        FastFlow invalid(pack, pin, static_cast<int>(h));
        if (invalid.search_surface_valid || invalid.ah_surf_valid)
          throw std::runtime_error("invalid serialized seed was accepted");
      }
      pin->SetReal("fastflow", key, saved);
    }
  }
  if (pack->ppart->gi_diagnostics) {
    pack->ppart->gi_diagnostics->BeforeStep();
    pack->ppart->gi_diagnostics->AfterPush();
    pack->ppart->gi_diagnostics->Sample(true);
  }
  auto particle_after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pr);
  for (int p=0; p<pack->ppart->nprtcl_thispack; ++p)
    for (int q=0; q<pack->ppart->nrdata; ++q)
      if (particle_before(q,p) != particle_after(q,p))
        throw std::runtime_error("frozen diagnostics changed particle state");
  auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
  if (std::memcmp(before.data(),after.data(),u.size()*sizeof(Real)) != 0)
    throw std::runtime_error("non-evolving AH check changed evolution fields");
  mesh->dt = old_dt;
  if (global_variable::my_rank == 0)
    std::cout << "FROZEN_AH_CHECKS_PASS time=" << std::setprecision(17) << mesh->time
              << " cycle=" << mesh->ncycle << " evolution_advances=0" << std::endl;
}
#endif
