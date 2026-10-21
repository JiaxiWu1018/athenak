#ifndef PGEN_PARTICLES_ST_PRODUCTION_HPP_
#define PGEN_PARTICLES_ST_PRODUCTION_HPP_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <sys/stat.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "z4c/fastflow.hpp"

inline void STFrozenHorizonCheck(ParameterInput *pin, Mesh *mesh) {
  auto *pack=mesh->pmb_pack;
  auto adm=pack->padm->adm;
  auto size=pack->pmb->mb_size;
  const auto ix=mesh->mb_indcs;
  Kokkos::deep_copy(pack->padm->u_adm,0.0);
  par_for("STFrozenSchwarzschild",DevExeSpace(),0,pack->nmb_thispack-1,
    0,ix.nx3+2*ix.ng-1,0,ix.nx2+2*ix.ng-1,0,ix.nx1+2*ix.ng-1,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      const auto sz=size.d_view(m);
      Real x=CellCenterX(i-ix.is,ix.nx1,sz.x1min,sz.x1max);
      Real y=CellCenterX(j-ix.js,ix.nx2,sz.x2min,sz.x2max);
      Real z=CellCenterX(k-ix.ks,ix.nx3,sz.x3min,sz.x3max);
      Real rad=Kokkos::sqrt(x*x+y*y+z*z);
      Real psi=1+0.5/rad, psi4=psi*psi*psi*psi;
      adm.g_dd(m,0,0,k,j,i)=psi4;
      adm.g_dd(m,1,1,k,j,i)=psi4;
      adm.g_dd(m,2,2,k,j,i)=psi4;
      adm.psi4(m,k,j,i)=psi4;
      adm.alpha(m,k,j,i)=1.0;
    });
  auto &ah=*pack->pz4c->pfastflow[0];
  for (int k=0;k<3;++k) {
    ah.MetricDerivatives<4>(k*0.5); ah.Find(0,k*0.5); ah.Write(0,k*0.5);
    if ((k<2 && ah.ah_found) || (k==2 && !ah.ah_found)) {
      Kokkos::abort("Frozen Schwarzschild persistence/acceptance failed");
    }
  }
  if (std::abs(ah.GetIrreducibleMass()-1)>0.05) {
    Kokkos::abort("Frozen Schwarzschild mass differs by over five percent");
  }
  ah.center[0]=1e6;
  ah.MetricDerivatives<4>(2.0); ah.Find(0,2.0);
  if (ah.ah_found || ah.ah_surf_valid) {Kokkos::abort("Off-grid AH was accepted");}
  if (global_variable::my_rank==0) {
    std::ofstream f("FROZEN_AH_OK"); f<<"known_mass=1 tolerance=.05 persistence=3 offgrid=rejected zero_steps=true\n";
  }
}

// Passive full-population mass audit and fixed-ID movie data. All device counters
// live in device allocations. Missing/dead cohort members have an explicit mask.
inline void STProductionStep(ParameterInput *pin, Mesh *mesh, bool final) {
  if (pin->GetOrAddBoolean("problem","production_frozen_ah_check",false) &&
      !pin->GetOrAddBoolean("problem","production_frozen_ah_done",false)) {
    STFrozenHorizonCheck(pin,mesh);
    pin->SetBoolean("problem","production_frozen_ah_done",true);
  }
  auto *pack=mesh->pmb_pack;
  auto pr=pack->ppart->prtcl_rdata;
  auto pi=pack->ppart->prtcl_idata;
  const int np=mesh->nprtcl_thisrank;
  const int initial=pin->GetInteger("problem","cluster_n");
  const Real cadence=pin->GetOrAddReal("problem","production_history_dt",0.25);
  const Real last=pin->GetOrAddReal("problem","production_history_last",-1.0);
  const std::string base=pin->GetString("job","basename");
  if (!(cadence>0)) {Kokkos::abort("Invalid production history cadence");}
  const bool due=last<0 || std::floor((mesh->time+1e-10)/cadence)>
      std::floor((last+1e-10)/cadence) || (final && mesh->time>last+1e-12);
  if (due) {
    Real mass=0.0;
    Kokkos::parallel_reduce("STProductionMass", Kokkos::RangePolicy<>(DevExeSpace(),0,np),
      KOKKOS_LAMBDA(int p, Real &sum) {sum+=pr(IPM,p);},Kokkos::Sum<Real>(mass));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&mass,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
    Real m0=pin->GetOrAddReal("problem","production_initial_rest_mass",-1.0);
    if (m0<0) {
      if (mesh->ncycle!=0 || mesh->nprtcl_total!=initial) {
        Kokkos::abort("Initial rest mass missing from production restart");
      }
      m0=mass; pin->SetReal("problem","production_initial_rest_mass",m0);
    }
    const Real weight=m0/initial;
    Real error=0.0;
    Kokkos::parallel_reduce("STProductionWeights",Kokkos::RangePolicy<>(DevExeSpace(),0,np),
      KOKKOS_LAMBDA(int p, Real &v) {
        const Real d=Kokkos::fabs(pr(IPM,p)-weight)/weight;
        v=Kokkos::fmax(v,Kokkos::isfinite(d) ? d : 1e100);
      },Kokkos::Max<Real>(error));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
    // Kokkos::Max uses its lowest-value identity for an empty population.
    // With no surviving weights, the equal-weight violation is zero.
    error=std::max(error,static_cast<Real>(0.0));
    int dead=0;
    for (int k=0;k<NPRTCL_DEATH_REASON;++k) {dead+=mesh->nprtcl_destroyed_cum[k];}
    const Real closure=(mass+dead*weight-m0)/m0;
    if (mesh->nprtcl_total+dead!=initial || !std::isfinite(mass) ||
        std::abs(closure)>1e-10 || error>1e-10) {
      if (global_variable::my_rank==0) {
        std::cerr<<"ST ledger: alive="<<mesh->nprtcl_total<<" dead="<<dead
                 <<" initial="<<initial<<" mass="<<std::setprecision(17)<<mass
                 <<" m0="<<m0<<" closure="<<closure<<" weight_error="<<error<<std::endl;
      }
      Kokkos::abort("ST production rest-mass/particle ledger failed");
    }
    Real dx=std::numeric_limits<Real>::max();
    for (int m=0;m<pack->nmb_thispack;++m) {
      dx=std::min(dx,pack->pmb->mb_size.h_view(m).dx1);
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&dx,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank==0) {
      const auto name=base+".mass.csv";
      FILE *out=std::fopen(name.c_str(),"a");
      if (!out) {Kokkos::abort("Cannot write mass ledger");}
      if (std::ftell(out)==0) {
        std::fprintf(out,"time,cycle,n_alive,n_exit,n_sphere,n_lapse,n_horizon,m_alive,m_initial,closure,weight_error,nblocks,dx_min,dt\n");
      }
      std::fprintf(out,"%.17g,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g\n",
        mesh->time,mesh->ncycle,mesh->nprtcl_total,
        mesh->nprtcl_destroyed_cum[0],mesh->nprtcl_destroyed_cum[1],
        mesh->nprtcl_destroyed_cum[2],mesh->nprtcl_destroyed_cum[3],
        mass,m0,closure,error,mesh->nmb_total,dx,mesh->dt_last_completed);
      std::fclose(out);
    }
    pin->SetReal("problem","production_history_last",mesh->time);
  }

  const Real movie_dt=pin->GetOrAddReal("problem","production_cohort_dt",0.5);
  const Real movie_last=pin->GetOrAddReal("problem","production_cohort_last",-1.0);
  const int stride=pin->GetOrAddInteger("problem","production_cohort_stride",100);
  if (!(movie_dt>0) || stride<1) {Kokkos::abort("Invalid cohort controls");}
  if (movie_last<0 || std::floor((mesh->time+1e-10)/movie_dt)>
      std::floor((movie_last+1e-10)/movie_dt) || (final && mesh->time>movie_last+1e-12)) {
    static_assert(sizeof(Real)==8,"S8 cohort format requires double precision");
    const int nc=(initial+stride-1)/stride;
    DvceArray1D<Real> values("s8_cohort",8*nc);
    DvceArray1D<int> mask("s8_cohort_mask",nc);
    Kokkos::deep_copy(values,0.0); Kokkos::deep_copy(mask,0);
    par_for("STProductionCohort",DevExeSpace(),0,np-1,KOKKOS_LAMBDA(int p) {
      const int tag=pi(PTAG,p);
      if (tag>=0 && tag<initial && tag%stride==0) {
        const int slot=tag/stride;
        Kokkos::atomic_increment(&mask(slot));
        values(8*slot)=tag;
        values(8*slot+1)=pr(IPX,p); values(8*slot+2)=pr(IPY,p);
        values(8*slot+3)=pr(IPZ,p); values(8*slot+4)=pr(IPVX,p);
        values(8*slot+5)=pr(IPVY,p); values(8*slot+6)=pr(IPVZ,p);
        values(8*slot+7)=pr(IPM,p);
      }
    });
    auto v=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),values);
    auto m=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mask);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,v.data(),8*nc,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,m.data(),nc,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
    for (int j=0;j<nc;++j) {
      if (m(j)>1) {Kokkos::abort("Duplicate fixed cohort ID");}
      if (m(j)) {for (int k=0;k<8;++k) {
        if (!std::isfinite(v(8*j+k))) {Kokkos::abort("Nonfinite cohort record");}
      }}
    }
    if (global_variable::my_rank==0) {
      mkdir("cohort",0775);
      std::ostringstream name; name<<"cohort/"<<base<<"."<<std::setw(10)
          <<std::setfill('0')<<mesh->ncycle<<".s8c";
      FILE *out=std::fopen(name.str().c_str(),"wb");
      if (!out) {Kokkos::abort("Cannot write cohort");}
      std::fprintf(out,"S8COHORT1 %.17g %d %d %d\n",mesh->time,mesh->ncycle,nc,stride);
      if (std::fwrite(m.data(),sizeof(int),nc,out)!=static_cast<size_t>(nc) ||
          std::fwrite(v.data(),sizeof(Real),8*nc,out)!=static_cast<size_t>(8*nc)) {
        Kokkos::abort("Short cohort write");
      }
      std::fclose(out);
    }
    pin->SetReal("problem","production_cohort_last",mesh->time);
  }

  if (!pack->pz4c->pfastflow.empty()) {
    const Real first=pack->pz4c->pfastflow[0]->time_first_found;
    if (first>=0 && !pin->GetOrAddBoolean("problem","production_ah_checkpoint",false)) {
      pin->SetBoolean("problem","production_ah_checkpoint",true);
      pin->SetBoolean("time","checkpoint_requested",true);
      pin->SetReal("problem","production_ah_onset",first);
      if (global_variable::my_rank==0) {
        std::ofstream f(base+".ah_onset.txt"); f<<std::setprecision(17)<<first<<"\n";
      }
    }
    if (first>=0 && mesh->time>=first+20.0 && global_variable::my_rank==0) {
      const auto stop=pin->GetOrAddString("time","stop_file","REQUEST_STOP");
      std::ofstream f(stop); f<<"ah_tail_complete\n";
    }
  }
}
#endif  // PGEN_PARTICLES_ST_PRODUCTION_HPP_
