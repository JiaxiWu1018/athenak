// Session011 observational diagnostics: disabled by default, never mutate particles.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles.hpp"
#include "z4c/z4c.hpp"
#include "z4c/compact_object_tracker.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {
void Particles::OrbitBeginPush() {
  if (orbit_dt <= 0.0) return;
  if (orbit_old.extent(1) < static_cast<size_t>(nprtcl_thispack)) {
    Kokkos::realloc(orbit_old, 6, nprtcl_thispack);
  }
  auto old = orbit_old;
  auto pr = prtcl_rdata;
  par_for("s11_observe_before_push",DevExeSpace(),0,nprtcl_thispack-1,
    KOKKOS_LAMBDA(int p) {
      old(0,p)=pr(IPX,p); old(1,p)=pr(IPY,p); old(2,p)=pr(IPZ,p);
      old(3,p)=pr(IPVX,p); old(4,p)=pr(IPVY,p); old(5,p)=pr(IPVZ,p);
    });
}

void Particles::OrbitEndPush() {
  if (orbit_dt <= 0.0) return;
  auto old = orbit_old;
  auto pr = prtcl_rdata;
  auto pi = prtcl_idata;
  auto changes = orbit_changes;
  auto cohort = orbit_cohort_changes;
  const Real fallback_mass = mass;
  const bool individual_mass = pr.extent(0) > IPM;
  par_for("s11_observe_after_push",DevExeSpace(),0,nprtcl_thispack-1,
    KOKKOS_LAMBDA(int p) {
      const int tag=pi(PTAG,p);
      const int comp = tag<3000000 ? 0 : (tag<4000000 ? 1 : 2);
      const Real mu=individual_mass ? pr(IPM,p) : fallback_mass;
      // Exact discrete product rule with midpoint averages, before any excision:
      // delta(x*u_y-y*u_x) = xbar*delta u_y-ybar*delta u_x
      //                       + u_ybar*delta x-u_xbar*delta y.
      const Real momentum=mu*(0.5*(old(0,p)+pr(IPX,p))*(pr(IPVY,p)-old(4,p))
                            -0.5*(old(1,p)+pr(IPY,p))*(pr(IPVX,p)-old(3,p)));
      const Real position=mu*(0.5*(old(4,p)+pr(IPVY,p))*(pr(IPX,p)-old(0,p))
                            -0.5*(old(3,p)+pr(IPVX,p))*(pr(IPY,p)-old(1,p)));
      Kokkos::atomic_add(&changes(2*comp),momentum);
      Kokkos::atomic_add(&changes(2*comp+1),position);
      const int local=tag-(comp==0?0:(comp==1?3000000:4000000));
      const int stride=comp==0?750:250;
      if(local>=0 && local%stride==0 && local/stride<4000) {
        int s=(comp*4000+local/stride)*6;
        Kokkos::atomic_add(&cohort(s),pr(IPX,p)-old(0,p));
        Kokkos::atomic_add(&cohort(s+1),pr(IPY,p)-old(1,p));
        Kokkos::atomic_add(&cohort(s+2),pr(IPZ,p)-old(2,p));
        Kokkos::atomic_add(&cohort(s+3),pr(IPVX,p)-old(3,p));
        Kokkos::atomic_add(&cohort(s+4),pr(IPVY,p)-old(4,p));
        Kokkos::atomic_add(&cohort(s+5),pr(IPVZ,p)-old(5,p));
      }
    });
}

void Particles::OrbitWrite(bool force) {
  if (orbit_dt <= 0.0) return;
  auto mesh=pmy_pack->pmesh;
  const Real tm=mesh->time;
  if (tm == orbit_last || (!force && orbit_last>=0.0 &&
      tm+1.e-10 < (std::floor((orbit_last+1.e-10)/orbit_dt)+1)*orbit_dt)) return;
  auto pr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),prtcl_rdata);
  auto pi=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),prtcl_idata);
  auto inc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),orbit_changes);
  auto cohort=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),orbit_cohort_changes);
  std::vector<Real> updates(12000*6);
  for (int k=0;k<12000*6;++k) updates[k]=cohort(k);
  // 3 components x (N,mass,Jz,weighted xyz,weighted uxyz,weighted r,weighted r^2).
  constexpr int width=11, bins=500, fields=6;
  std::vector<Real> sum(3*width+6+3*bins*fields,0.0);
  for (int c=0;c<6;++c) sum[3*width+c]=inc(c);
  const bool fresh = orbit_last < 0.0;
  std::string sample=orbit_basename+".cohort.rank"+
      std::to_string(global_variable::my_rank)+".csv";
  bool sample_new=access(sample.c_str(),F_OK)!=0;
  FILE *fp=fopen(sample.c_str(),"a");
  if (!fp) throw std::runtime_error("Cannot write cohort diagnostics");
  if (sample_new) fprintf(fp,"cycle,time,tag,component,mass,x,y,z,u1_cov,u2_cov,u3_cov,E_specific,gid\n");
  for (int p=0;p<nprtcl_thispack;++p) {
    const int tag=pi(PTAG,p), c=tag<3000000?0:(tag<4000000?1:2);
    const Real mu=pr(IPM,p),x=pr(IPX,p),y=pr(IPY,p),z=pr(IPZ,p);
    const Real ux=pr(IPVX,p),uy=pr(IPVY,p),uz=pr(IPVZ,p);
    const Real r=std::sqrt(x*x+y*y+z*z), j=mu*(x*uy-y*ux);
    Real vals[width]={1.0,mu,j,mu*x,mu*y,mu*z,mu*ux,mu*uy,mu*uz,mu*r,mu*r*r};
    for(int k=0;k<width;++k) sum[c*width+k]+=vals[k];
    if (r<50.0) {
      int b=static_cast<int>(r/0.1);
      if (b>=bins) b=bins-1;
      int off=3*width+6+(c*bins+b)*fields;
      Real radial=r>0.0?(x*ux+y*uy+z*uz)/r:0.0;
      sum[off]+=1.0; sum[off+1]+=mu; sum[off+2]+=j;
      sum[off+3]+=mu*radial; sum[off+4]+=mu*radial*radial;
      sum[off+5]+=mu*(ux*ux+uy*uy+uz*uz);
    }
    // Deterministic t0 tag cohort, never replenished: 4000 tags/component.
    const int local=tag-(c==0?0:(c==1?3000000:4000000));
    const int stride=c==0?750:250;
    if (local>=0 && local%stride==0 && local/stride<4000) {
      fprintf(fp,"%d,%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n",
        mesh->ncycle,tm,tag,c,mu,x,y,z,ux,uy,uz,pr(IPEN,p),pi(PGID,p));
    }
  }
  fclose(fp);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,sum.data(),sum.size(),MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,updates.data(),updates.size(),MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  Kokkos::deep_copy(orbit_changes,0.0);
  Kokkos::deep_copy(orbit_cohort_changes,0.0);
  if (global_variable::my_rank==0) {
    if (pmy_pack->pz4c != nullptr) {
      std::string track=orbit_basename+".trackers.csv";
      bool is_new=access(track.c_str(),F_OK)!=0;
      FILE *tr=fopen(track.c_str(),"a");
      if(!tr) throw std::runtime_error("Cannot write tracker state-time mapping");
      if(is_new) fprintf(tr,"cycle,time,tracker,x,y,z\n");
      int i=0;
      for(auto &pt:pmy_pack->pz4c->ptracker) {
        fprintf(tr,"%d,%.17g,%d,%.17g,%.17g,%.17g\n",mesh->ncycle,tm,i++,
          pt->GetPos(0),pt->GetPos(1),pt->GetPos(2));
      }
      fclose(tr);
    }
    std::string path=orbit_basename+".components.csv";
    bool nf=access(path.c_str(),F_OK)!=0;
    FILE *f=fopen(path.c_str(),"a");
    if (!f) throw std::runtime_error("Cannot write component diagnostics");
    if(nf) fprintf(f,"cycle,time,dt,leaf_blocks,component,count,rest_mass,Jz,mu_x,mu_y,mu_z,mu_u1,mu_u2,mu_u3,mu_r,mu_r2,deltaJ_momentum_before_removal,deltaJ_position_before_removal,interval_start\n");
    for(int c=0;c<3;++c) {
      fprintf(f,"%d,%.17g,%.17g,%d,%d",mesh->ncycle,tm,mesh->dt_last_completed,mesh->nmb_total,c);
      for(int k=0;k<width;++k) fprintf(f,",%.17g",sum[c*width+k]);
      fprintf(f,",%.17g,%.17g,%.17g\n",sum[3*width+2*c],sum[3*width+2*c+1],fresh?tm:orbit_last);
    }
    fclose(f);
    path=orbit_basename+".cohort_updates.csv";
    nf=access(path.c_str(),F_OK)!=0; f=fopen(path.c_str(),"a");
    if(!f) throw std::runtime_error("Cannot write cohort updates");
    if(nf) fprintf(f,"cycle,time,interval_start,tag,component,delta_x,delta_y,delta_z,delta_u1_cov,delta_u2_cov,delta_u3_cov\n");
    for(int s=0;s<12000;++s) {
      bool any=false;
      for(int k=0;k<6;++k) any=any || updates[s*6+k]!=0.0;
      if(!any && !fresh) continue;
      int c=s/4000,tag=(s%4000)*(c==0?750:250)+(c==0?0:(c==1?3000000:4000000));
      fprintf(f,"%d,%.17g,%.17g,%d,%d",mesh->ncycle,tm,fresh?tm:orbit_last,tag,c);
      for(int k=0;k<6;++k) fprintf(f,",%.17g",updates[s*6+k]);
      fprintf(f,"\n");
    }
    fclose(f);
    path=orbit_basename+".radial_profiles.csv";
    nf=access(path.c_str(),F_OK)!=0; f=fopen(path.c_str(),"a");
    if(!f) throw std::runtime_error("Cannot write radial profiles");
    if(nf) fprintf(f,"cycle,time,component,r_lo,r_hi,count,rest_mass,Jz,mu_u_rad,mu_u_rad2,mu_u_squared\n");
    for(int c=0;c<3;++c) for(int b=0;b<bins;++b) {
      int off=3*width+6+(c*bins+b)*fields;
      fprintf(f,"%d,%.17g,%d,%.17g,%.17g",mesh->ncycle,tm,c,0.1*b,0.1*(b+1));
      for(int k=0;k<fields;++k) fprintf(f,",%.17g",sum[off+k]);
      fprintf(f,"\n");
    }
    fclose(f);
  }
  orbit_last=tm;
}
}  // namespace particles
