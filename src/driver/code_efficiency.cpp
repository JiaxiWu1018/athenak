// Default-off performance instrumentation; scientific output is never required.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "driver/code_efficiency.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
namespace efficiency {
State &Get() {static State state; return state;}
void Configure(ParameterInput *pin) {
  auto &s=Get();
  if (!pin->DoesBlockExist("benchmark")) return;
  s.enabled=pin->GetOrAddBoolean("benchmark","enabled",false);
  s.profile=pin->GetOrAddBoolean("benchmark","profile",false);
  s.validation_payload=s.enabled&&pin->GetOrAddBoolean("benchmark","validation_payload",false);
  s.mesh_only=s.enabled&&pin->GetOrAddBoolean("benchmark","mesh_only",false);
  s.warmup=pin->GetOrAddReal("benchmark","warmup",0.5);
  if (s.enabled && s.profile) Kokkos::realloc(s.device_cycles,5,1);
}
KOKKOS_INLINE_FUNCTION std::uint64_t Mix(std::uint64_t x) {
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;
  return x^(x>>31);
}
KOKKOS_INLINE_FUNCTION std::uint64_t Bits(Real x) {
  union {Real value; std::uint64_t bits;} v;
  v.bits=0;v.value=x; return v.bits;
}
std::uint64_t Hash(DvceArray5D<Real> a) {
  std::uint64_t h=0;
  auto n=a.size(); auto ptr=a.data();
  Kokkos::parallel_reduce("benchmark_geometry_hash",Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<std::size_t>>(0,n),
    KOKKOS_LAMBDA(std::size_t i,std::uint64_t &v) {v+=Mix(Bits(ptr[i])^Mix(i));},Kokkos::Sum<std::uint64_t>(h));
  return h;
}
int Nonfinite(DvceArray5D<Real> a) {
  int bad=0;auto n=a.size();auto ptr=a.data();
  Kokkos::parallel_reduce("benchmark_geometry_finite",Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<std::size_t>>(0,n),KOKKOS_LAMBDA(std::size_t i,int &v){v+=!Kokkos::isfinite(ptr[i]);},Kokkos::Sum<int>(bad));
  return bad;
}
void Audit(Mesh *pm,bool initial) {
  auto &s=Get(); if (!s.enabled) return;
  Kokkos::fence(); auto *pk=pm->pmb_pack; auto *p=pk->ppart;
  auto pr=p->prtcl_rdata; auto pi=p->prtcl_idata; auto sz=pk->pmb->mb_size;
  int gids=pk->gids,nmb=pk->nmb_thispack,np=p->nprtcl_thispack;
  auto ng=pm->mb_indcs;
  std::uint64_t sum=0,square=0,payload=0; int bad=0;
  Kokkos::parallel_reduce("benchmark_tag_sum",Kokkos::RangePolicy<>(0,np),KOKKOS_LAMBDA(int i,std::uint64_t &v){v+=static_cast<std::uint64_t>(pi(PTAG,i));},Kokkos::Sum<std::uint64_t>(sum));
  Kokkos::parallel_reduce("benchmark_tag_square",Kokkos::RangePolicy<>(0,np),KOKKOS_LAMBDA(int i,std::uint64_t &v){auto t=static_cast<std::uint64_t>(pi(PTAG,i));v+=t*t;},Kokkos::Sum<std::uint64_t>(square));
  Kokkos::parallel_reduce("benchmark_payload_hash",Kokkos::RangePolicy<>(0,np),KOKKOS_LAMBDA(int i,std::uint64_t &v){
    auto t=static_cast<std::uint64_t>(pi(PTAG,i));
    std::uint64_t h=Mix(t);
    for(int c=0;c<8;++c) if(c!=IPEN) h+=Mix(Bits(pr(c,i))^Mix(8*t+c));
    v+=h;
  },Kokkos::Sum<std::uint64_t>(payload));
  Kokkos::parallel_reduce("benchmark_finite_ownership",Kokkos::RangePolicy<>(0,np),KOKKOS_LAMBDA(int i,int &v){
    int m=pi(PGID,i)-gids; bool b=m<0||m>=nmb;
    for(int c=0;c<8;++c) b=b||!Kokkos::isfinite(pr(c,i));
    if(!b) b=pr(IPX,i)<sz.d_view(m).x1min||pr(IPX,i)>=sz.d_view(m).x1max||pr(IPY,i)<sz.d_view(m).x2min||pr(IPY,i)>=sz.d_view(m).x2max||pr(IPZ,i)<sz.d_view(m).x3min||pr(IPZ,i)>=sz.d_view(m).x3max;
    v+=b;
  },Kokkos::Sum<int>(bad));
  std::uint64_t ledger[4]={static_cast<std::uint64_t>(np),sum,square,payload};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,ledger,4,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  std::uint64_t geometry[4]={Hash(pk->pz4c->u0),Hash(pk->padm->u_adm),Hash(p->z4c_last),Hash(p->adm_last)};
  int geometry_bad=Nonfinite(pk->pz4c->u0)+Nonfinite(pk->padm->u_adm)+Nonfinite(p->z4c_last)+Nonfinite(p->adm_last);
  bool pass=bad==0&&geometry_bad==0;
  if(initial) {
    s.initial_count=ledger[0];s.initial_sum=ledger[1];s.initial_square=ledger[2];
    std::copy(geometry,geometry+4,s.geometry);
  } else {
    pass=pass&&ledger[0]==s.initial_count&&ledger[1]==s.initial_sum&&ledger[2]==s.initial_square;
    for(int k=0;k<NPRTCL_DEATH_REASON;++k) pass=pass&&pm->nprtcl_destroyed_cum[k]==0;
    if(pk->pz4c->opt.frozen_background) for(int k=0;k<4;++k) pass=pass&&geometry[k]==s.geometry[k];
  }
  DvceArray1D<int> blocks("benchmark_blocks",nmb);
  Kokkos::parallel_for("benchmark_block_counts",Kokkos::RangePolicy<>(0,np),KOKKOS_LAMBDA(int i){int m=pi(PGID,i)-gids;if(m>=0&&m<nmb)Kokkos::atomic_add(&blocks(m),1);});
  auto hb=Kokkos::create_mirror_view_and_copy(HostMemSpace(),blocks);
  std::ofstream out(std::string(initial?"initial":"final")+".rank"+std::to_string(global_variable::my_rank)+".json");
  out<<std::setprecision(17)<<"{\"rank\":"<<global_variable::my_rank<<",\"local_count\":"<<np<<",\"count\":"<<ledger[0]<<",\"tag_sum\":"<<ledger[1]<<",\"tag_square\":"<<ledger[2]<<",\"payload_hash\":"<<ledger[3]<<",\"valid\":"<<(pass?"true":"false")<<",\"block_counts\":[";
  for(int m=0;m<nmb;++m){if(m)out<<',';out<<hb(m);} out<<"],\"geometry_hash\":[";
  for(int k=0;k<4;++k){if(k)out<<',';out<<geometry[k];}out<<"]";
  if(initial&&p->feedback) {
    auto a=pk->ptmunu->u_tmunu;
    out<<",\"source_moments\":[";
    for(int c=0;c<Tmunu::N_Tmunu;++c) {
      double v=0,w=0;int nv=ng.nx1*ng.nx2*ng.nx3;
      Kokkos::parallel_reduce("benchmark_source_sum",Kokkos::RangePolicy<>(0,nmb*nv),KOKKOS_LAMBDA(int idx,double &b){int m=idx/nv,t=idx%nv,i=t%ng.nx1+ng.is,j=(t/ng.nx1)%ng.nx2+ng.js,k=t/(ng.nx1*ng.nx2)+ng.ks;b+=a(m,c,k,j,i);},Kokkos::Sum<double>(v));
      Kokkos::parallel_reduce("benchmark_source_square",Kokkos::RangePolicy<>(0,nmb*nv),KOKKOS_LAMBDA(int idx,double &b){int m=idx/nv,t=idx%nv,i=t%ng.nx1+ng.is,j=(t/ng.nx1)%ng.nx2+ng.js,k=t/(ng.nx1*ng.nx2)+ng.ks;double x=a(m,c,k,j,i);b+=x*x;},Kokkos::Sum<double>(w));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE,&v,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);MPI_Allreduce(MPI_IN_PLACE,&w,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif
      if(c)out<<',';out<<v<<','<<w;
    }out<<"]";
  }
  if(initial&&p->feedback&&s.source_mass.extent(0)>0) {
    auto src=pk->ptmunu->u_tmunu;auto metric=pk->padm->u_adm;
    auto mass=s.source_mass,u=s.source_u;auto domain=pm->mesh_size;auto grid=pm->mesh_indcs;
    int nv=ng.nx1*ng.nx2*ng.nx3;double error=0;
    Kokkos::parallel_reduce("benchmark_source_reference",Kokkos::RangePolicy<>(0,nmb*nv),KOKKOS_LAMBDA(int idx,double &maximum){
      int m=idx/nv,t=idx%nv,i=t%ng.nx1+ng.is,j=(t/ng.nx1)%ng.nx2+ng.js,k=t/(ng.nx1*ng.nx2)+ng.ks;
      auto region=sz.d_view(m);
      int gi=static_cast<int>(llround((region.x1min-domain.x1min)/region.dx1))+i-ng.is;
      int gj=static_cast<int>(llround((region.x2min-domain.x2min)/region.dx2))+j-ng.js;
      int gk=static_cast<int>(llround((region.x3min-domain.x3min)/region.dx3))+k-ng.ks;
      int cell=(gk*grid.nx2+gj)*grid.nx1+gi;
      double gx=metric(m,adm::ADM::I_ADM_GXX,k,j,i),gy=metric(m,adm::ADM::I_ADM_GYY,k,j,i),gz=metric(m,adm::ADM::I_ADM_GZZ,k,j,i);
      double U=u(cell),W[3]={sqrt(1+U*U/gx),sqrt(1+U*U/gy),sqrt(1+U*U/gz)};
      double factor=mass(cell)/(sqrt(gx*gy*gz)*region.dx1*region.dx2*region.dx3);
      double expected[10]={factor*U*U/(3*W[0]),0,0,factor*U*U/(3*W[1]),0,factor*U*U/(3*W[2]),factor*(W[0]+W[1]+W[2])/3,0,0,0};
      for(int c=0;c<10;++c)maximum=fmax(maximum,fabs(src(m,c,k,j,i)-expected[c])/fmax(expected[6],1.e-20));
    },Kokkos::Max<double>(error));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
    out<<",\"source_reference_max_relative_error\":"<<error;
    pass=pass&&error<1.e-8;
  }
  out<<"}\n";out.close();
  if(s.validation_payload) {
    auto real=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pr);
    auto integer=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pi);
    std::ofstream values(std::string(initial?"initial":"final")+".payload.rank"+std::to_string(global_variable::my_rank)+".csv");
    values<<"tag,mass,energy,x,ux,y,uy,z,uz\n"<<std::setprecision(17);
    for(int i=0;i<np;++i){values<<integer(PTAG,i);for(int c=0;c<8;++c)values<<','<<real(c,i);values<<'\n';}
  }
  int ok=pass?1:0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
#endif
  if(!ok) {std::cerr<<"Benchmark liveness/freeze gate failed\n";std::exit(1);}
}
void StartWindow(Mesh *pm) {
  auto &s=Get();if(!s.enabled||s.measuring||pm->time+1.e-10<s.warmup) return;
  Kokkos::fence();
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  s.start_time=pm->time;s.start_cycle=pm->ncycle;s.phases.clear();
  if(s.profile)Kokkos::deep_copy(s.device_cycles,0ULL);
  s.sent_particles=0;s.source_images=0;s.measuring=true;
#ifdef KOKKOS_ENABLE_CUDA
  if(s.profile)nvtxRangePushA("benchmark_window");
#endif
  s.timer.reset();
}
Phase::Phase(const char *name):name_(name),active_(Get().enabled&&Get().profile&&Get().measuring) {
  if(active_) {Kokkos::fence();Kokkos::Profiling::pushRegion(name_);
#ifdef KOKKOS_ENABLE_CUDA
  nvtxRangePushA(name_.c_str());
#endif
  timer_.reset();}
}
void Phase::Stop(){if(active_){Kokkos::fence();Get().phases[name_]+=timer_.seconds();Kokkos::Profiling::popRegion();
#ifdef KOKKOS_ENABLE_CUDA
  nvtxRangePop();
#endif
  active_=false;}}
Phase::~Phase(){Stop();}
void Finish(Mesh *pm) {
  auto &s=Get();if(!s.enabled){s.source_mass=DvceArray1D<Real>();s.source_u=DvceArray1D<Real>();s.device_cycles=DvceArray2D<unsigned long long>();return;}Kokkos::fence();s.elapsed=s.measuring?s.timer.seconds():0;
#ifdef KOKKOS_ENABLE_CUDA
  if(s.profile&&s.measuring)nvtxRangePop();
#endif
  double maximum=s.elapsed;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&maximum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
  int cycles=pm->ncycle-s.start_cycle;double dt=pm->time-s.start_time;
  double cells=static_cast<double>(pm->nmb_total)*pm->NumberOfMeshBlockCells();
  std::ofstream o("timing.rank"+std::to_string(global_variable::my_rank)+".json");
  o<<std::setprecision(17)<<"{\"rank\":"<<global_variable::my_rank<<",\"device_timing_compiled\":"<<(DeviceTimingCompiled()?"true":"false")<<",\"profile\":"<<(s.profile?"true":"false")<<",\"mesh_only\":"<<(s.mesh_only?"true":"false")<<",\"frozen\":"<<(pm->pmb_pack->pz4c->opt.frozen_background?"true":"false")<<",\"elapsed_local\":"<<s.elapsed<<",\"elapsed_max\":"<<maximum<<",\"start_time\":"<<s.start_time<<",\"final_time\":"<<pm->time<<",\"cycles\":"<<cycles<<",\"dt\":"<<pm->dt_last_completed<<",\"physical_cells\":"<<cells<<",\"particles_per_second\":"<<(maximum>0&&!s.mesh_only?s.initial_count*static_cast<double>(cycles)/maximum:0)<<",\"cells_per_second\":"<<(maximum>0?cells*cycles/maximum:0)<<",\"seconds_per_M\":"<<(dt>0?maximum/dt:0)<<",\"sent_particles\":"<<s.sent_particles<<",\"source_images\":"<<s.source_images<<",\"boris_fallbacks\":"<<pm->pmb_pack->ppart->boris_nfail_cum<<",\"boris_rejections\":"<<pm->pmb_pack->ppart->boris_nreject_cum<<",\"phases\":{";
  bool first=true;for(auto &x:s.phases){if(!first)o<<',';first=false;o<<'"'<<x.first<<"\":"<<x.second;}o<<"},\"device_cycles\":[";
  if(s.profile){auto h=Kokkos::create_mirror_view_and_copy(HostMemSpace(),s.device_cycles);for(int c=0;c<5;++c){if(c)o<<',';o<<h(c,0);}}
  o<<"]}\n";o.close();s.measuring=false;Audit(pm,false);
  s.source_mass=DvceArray1D<Real>();s.source_u=DvceArray1D<Real>();s.device_cycles=DvceArray2D<unsigned long long>();
  if(global_variable::my_rank==0)std::cout<<"[benchmark] measured "<<cycles<<" cycles, "<<maximum<<" wall seconds, "<<dt<<" M\n";
}
} // namespace efficiency
