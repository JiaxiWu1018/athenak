// Performance-only ST/flat sampler. All initialization runs inside the Slurm allocation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "particles/particles.hpp"
#include "driver/code_efficiency.hpp"
#include "pgen/particles/efficiency_st_profile.hpp"
#include "pgen/pgen.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
namespace {
KOKKOS_INLINE_FUNCTION std::uint64_t RandomBits(std::uint64_t x) {
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
KOKKOS_INLINE_FUNCTION Real Uniform(std::uint64_t tag,int field,int seed) {
  return static_cast<Real>(RandomBits(tag*16+field+static_cast<std::uint64_t>(seed)*0x9e3779b97f4a7c15ULL)>>11)*0x1.0p-53;
}
KOKKOS_INLINE_FUNCTION void Metric(Real r,DvceArray1D<Real> rr,DvceArray1D<Real> aa,
  DvceArray1D<Real> al,DvceArray1D<Real> yy,Real &A,Real &alpha,Real &y) {
  int hi=rr.extent_int(0)-1,lo=0;
  if(r>=rr(hi)){Real q=0.5/r;A=(1+q)*(1+q);alpha=(1-q)/(1+q);y=1;return;}
  while(lo+1<hi){int mid=(lo+hi)/2;if(rr(mid)<=r)lo=mid;else hi=mid;}
  Real f=(r-rr(lo))/(rr(hi)-rr(lo));A=aa(lo)+f*(aa(hi)-aa(lo));
  alpha=al(lo)+f*(al(hi)-al(lo));y=yy(lo)+f*(yy(hi)-yy(lo));
}
Real HostInterp(Real x,const std::vector<Real> &rr,const std::vector<Real> &vv) {
  auto it=std::upper_bound(rr.begin(),rr.end(),x);
  if(it==rr.end())return vv.back();if(it==rr.begin())return vv.front();
  int j=it-rr.begin();Real f=(x-rr[j-1])/(rr[j]-rr[j-1]);return vv[j-1]+f*(vv[j]-vv[j-1]);
}
}
void ProblemGenerator::UserProblem(ParameterInput *pin,const bool restart) {
  auto *pk=pmy_mesh_->pmb_pack;auto *p=pk->ppart;auto ind=pmy_mesh_->mb_indcs;
  if(restart||!p||!pk->pz4c||!pk->padm||!pmy_mesh_->three_d||pmy_mesh_->adaptive){std::cerr<<"Efficiency benchmark requires a fresh uniform 3D particle+Z4c mesh\n";std::exit(1);}
  std::string metric=pin->GetOrAddString("benchmark","metric","st");
  std::string sampling=pin->GetOrAddString("benchmark","sampling","random");
  std::string distribution=pin->GetOrAddString("benchmark","distribution","uniform");
  bool flat=metric=="flat";
  int seed=pin->GetOrAddInteger("benchmark","seed",1);
  int ppc=pin->GetOrAddInteger("benchmark","ppc",6);
  int ntotal=pin->GetOrAddInteger("benchmark","n",1000000);
  Real yc=pin->GetOrAddReal("benchmark","yc",0.819);
  if((metric!="st"&&!flat)||(sampling!="random"&&sampling!="replicas")||yc<=0||yc>=1||ppc<6||ppc%6){std::cerr<<"Invalid efficiency sampler options\n";std::exit(1);}
  auto profile=efficiency_st::ConstructProfile(yc,1.e-4);
  int nprof=profile.x.size();
  DvceArray1D<Real> rr("st_radius",nprof),aa("st_A",nprof),al("st_alpha",nprof),yy("st_y",nprof);
  auto hr=Kokkos::create_mirror_view(rr),ha=Kokkos::create_mirror_view(aa),hl=Kokkos::create_mirror_view(al),hy=Kokkos::create_mirror_view(yy);
  for(int i=0;i<nprof;++i){hr(i)=profile.riso_over_m[i];ha(i)=profile.conformal_a[i];hl(i)=profile.lapse[i];hy(i)=profile.y[i];}
  Kokkos::deep_copy(rr,hr);Kokkos::deep_copy(aa,ha);Kokkos::deep_copy(al,hl);Kokkos::deep_copy(yy,hy);
  auto sz=pk->pmb->mb_size;auto adm=pk->padm->adm;int nmb=pk->nmb_thispack;
  par_for("benchmark_metric",DevExeSpace(),0,nmb-1,ind.ks-ind.ng,ind.ke+ind.ng,ind.js-ind.ng,ind.je+ind.ng,ind.is-ind.ng,ind.ie+ind.ng,
    KOKKOS_LAMBDA(int m,int k,int j,int i){
      Real x=CellCenterX(i-ind.is,ind.nx1,sz.d_view(m).x1min,sz.d_view(m).x1max),y=CellCenterX(j-ind.js,ind.nx2,sz.d_view(m).x2min,sz.d_view(m).x2max),z=CellCenterX(k-ind.ks,ind.nx3,sz.d_view(m).x3min,sz.d_view(m).x3max);
      Real A=1,alpha=1,yp=1;if(!flat)Metric(sqrt(x*x+y*y+z*z),rr,aa,al,yy,A,alpha,yp);
      adm.psi4(m,k,j,i)=A*A;adm.alpha(m,k,j,i)=alpha;
      for(int a=0;a<3;++a){adm.beta_u(m,a,k,j,i)=0;for(int b=a;b<3;++b){adm.g_dd(m,a,b,k,j,i)=a==b?A*A:0;adm.vK_dd(m,a,b,k,j,i)=0;}}
    });
  switch(ind.ng){case 4:pk->pz4c->ADMToZ4c<4>(pk,pin);break;default:std::cerr<<"Benchmark requires four ghost zones\n";std::exit(1);}
  pk->pz4c->Z4cToADM(pk);
  auto domain=pmy_mesh_->mesh_size;auto grid=pmy_mesh_->mesh_indcs;
  int gids=pk->gids,nlocal=0;
  if(sampling=="random") {
    Kokkos::parallel_reduce("benchmark_random_ownership",Kokkos::RangePolicy<>(0,ntotal),KOKKOS_LAMBDA(int tag,int &v){
      Real x=domain.x1min+(domain.x1max-domain.x1min)*Uniform(tag,0,seed),y=domain.x2min+(domain.x2max-domain.x2min)*Uniform(tag,1,seed),z=domain.x3min+(domain.x3max-domain.x3min)*Uniform(tag,2,seed);
      for(int m=0;m<nmb;++m)if(x>=sz.d_view(m).x1min&&x<sz.d_view(m).x1max&&y>=sz.d_view(m).x2min&&y<sz.d_view(m).x2max&&z>=sz.d_view(m).x3min&&z<sz.d_view(m).x3max){++v;break;}
    },Kokkos::Sum<int>(nlocal));
    Kokkos::realloc(p->prtcl_rdata,8,nlocal);Kokkos::realloc(p->prtcl_idata,2,nlocal);auto pr=p->prtcl_rdata;auto pi=p->prtcl_idata;Real M0=profile.rest_mass_over_m;
    Kokkos::parallel_scan("benchmark_random_fill",Kokkos::RangePolicy<>(0,ntotal),KOKKOS_LAMBDA(int tag,int &offset,bool final){
      Real x=domain.x1min+(domain.x1max-domain.x1min)*Uniform(tag,0,seed),y=domain.x2min+(domain.x2max-domain.x2min)*Uniform(tag,1,seed),z=domain.x3min+(domain.x3max-domain.x3min)*Uniform(tag,2,seed);
      int owner=-1;for(int m=0;m<nmb;++m)if(x>=sz.d_view(m).x1min&&x<sz.d_view(m).x1max&&y>=sz.d_view(m).x2min&&y<sz.d_view(m).x2max&&z>=sz.d_view(m).x3min&&z<sz.d_view(m).x3max){owner=m;break;}
      if(owner>=0){int q=offset++;if(final){Real A=1,alpha=1,yp=1;if(!flat)Metric(sqrt(x*x+y*y+z*z),rr,aa,al,yy,A,alpha,yp);
        Real c=2*Uniform(tag,3,seed)-1,phi=2*M_PI*Uniform(tag,4,seed),u=0.5*Uniform(tag,5,seed)*A,t=sqrt(1-c*c);
        pr(IPX,q)=x;pr(IPY,q)=y;pr(IPZ,q)=z;pr(IPVX,q)=u*t*cos(phi);pr(IPVY,q)=u*t*sin(phi);pr(IPVZ,q)=u*c;pr(IPEN,q)=0;pr(IPM,q)=M0*(0.5+Uniform(tag,6,seed))/ntotal;pi(PGID,q)=gids+owner;pi(PTAG,q)=tag;}}
    });
  } else {
    int nx=grid.nx1,ny=grid.nx2,nz=grid.nx3,ncells=nx*ny*nz;
    if(static_cast<std::int64_t>(ppc)*ncells>std::numeric_limits<int>::max()) {std::cerr<<"Particle count exceeds supported integer indices\n";std::exit(1);}
    ntotal=ppc*ncells;Real dx=(domain.x1max-domain.x1min)/nx,dy=(domain.x2max-domain.x2min)/ny,dz=(domain.x3max-domain.x3min)/nz;
    std::vector<Real> mass(ncells),Acell(ncells),ucell(ncells),weight(ncells),fraction(ncells);
    std::vector<int> replicas(ncells,1),order(ncells);std::vector<std::int64_t> prefix(ncells+1,0);
    Real sum_mass=0,sum_weight=0;int target=-1;
    for(int cell=0;cell<ncells;++cell){int i=cell%nx,j=(cell/nx)%ny,k=cell/(nx*ny);Real x=domain.x1min+(i+0.5)*dx,y=domain.x2min+(j+0.5)*dy,z=domain.x3min+(k+0.5)*dz,r=sqrt(x*x+y*y+z*z);
      Real A=HostInterp(r,profile.riso_over_m,profile.conformal_a),yp=HostInterp(r,profile.riso_over_m,profile.y);Acell[cell]=A;ucell[cell]=r<profile.riso_over_m_surface?A*sqrt(std::max(Real(0),1/yp-1)):0;
      Real ratio=std::max(Real(0),(1/yp-1)/(1/yc-1));mass[cell]=r<profile.riso_over_m_surface?sqrt(yc)*pow(yp/yc,-1)*sqrt(ratio)*A*A*A:0;sum_mass+=mass[cell];
      Real w=1;if(distribution=="central4"||distribution=="central1"){Real sigma=distribution=="central4"?4:1;w=exp(-0.5*r*r/(sigma*sigma));}
      else if(distribution=="block")w=x>=0&&y>=0&&z>=0?1:0;
      else if(distribution=="point")w=0;
      else if(distribution!="uniform"){std::cerr<<"Unknown replica distribution\n";std::exit(1);}
      if(std::abs(x-1)<dx/2&&std::abs(y-1)<dy/2&&std::abs(z-1)<dz/2)target=cell;
      weight[cell]=w;sum_weight+=w;order[cell]=cell;
    }
    if(distribution=="point"){if(target<0){std::cerr<<"No hotspot site\n";std::exit(1);}weight[target]=1;sum_weight=1;}
    int extras=(ppc/6-1)*ncells,used=0;
    for(int c=0;c<ncells;++c){double exact=extras*weight[c]/sum_weight;int q=static_cast<int>(floor(exact));replicas[c]+=q;used+=q;fraction[c]=exact-q;mass[c]*=profile.rest_mass_over_m/sum_mass;}
    std::sort(order.begin(),order.end(),[&](int a,int b){return fraction[a]!=fraction[b]?fraction[a]>fraction[b]:a<b;});
    for(int q=0;q<extras-used;++q)++replicas[order[q]];
    for(int c=0;c<ncells;++c)prefix[c+1]=prefix[c]+6*replicas[c];
    std::vector<int> cells,owners;std::vector<std::int64_t> local_prefix(1,0);
    for(int c=0;c<ncells;++c){int i=c%nx,j=(c/nx)%ny,k=c/(nx*ny);Real x=domain.x1min+(i+0.5)*dx,y=domain.x2min+(j+0.5)*dy,z=domain.x3min+(k+0.5)*dz;int m=p->FindContainingMeshBlock(x,y,z);if(m>=0){cells.push_back(c);owners.push_back(m);local_prefix.push_back(local_prefix.back()+6*replicas[c]);}}
    nlocal=local_prefix.back();int nowned=cells.size();
    DvceArray1D<int> dc("replica_cells",nowned),dm("replica_owners",nowned),dk("replica_k",ncells);
    DvceArray1D<std::int64_t> dp("replica_prefix",nowned+1),dt("replica_tags",ncells+1);
    DvceArray1D<Real> massd("replica_mass",ncells),ud("replica_u",ncells);
    auto hc=Kokkos::create_mirror_view(dc),hm=Kokkos::create_mirror_view(dm),hk=Kokkos::create_mirror_view(dk);auto hp=Kokkos::create_mirror_view(dp),ht=Kokkos::create_mirror_view(dt);auto hmass=Kokkos::create_mirror_view(massd),hu=Kokkos::create_mirror_view(ud);
    for(int i=0;i<nowned;++i){hc(i)=cells[i];hm(i)=owners[i];}for(int i=0;i<=nowned;++i)hp(i)=local_prefix[i];
    for(int i=0;i<ncells;++i){hk(i)=replicas[i];hmass(i)=mass[i];hu(i)=ucell[i];}for(int i=0;i<=ncells;++i)ht(i)=prefix[i];
    Kokkos::deep_copy(dc,hc);Kokkos::deep_copy(dm,hm);Kokkos::deep_copy(dk,hk);Kokkos::deep_copy(dp,hp);Kokkos::deep_copy(dt,ht);Kokkos::deep_copy(massd,hmass);Kokkos::deep_copy(ud,hu);
    efficiency::Get().source_mass=massd;efficiency::Get().source_u=ud;
    Kokkos::realloc(p->prtcl_rdata,8,nlocal);Kokkos::realloc(p->prtcl_idata,2,nlocal);auto pr=p->prtcl_rdata;auto pi=p->prtcl_idata;
    Kokkos::parallel_for("benchmark_replica_fill",Kokkos::RangePolicy<>(0,nlocal),KOKKOS_LAMBDA(int q){
      int lo=0,hi=nowned;while(lo+1<hi){int mid=(lo+hi)/2;if(dp(mid)<=q)lo=mid;else hi=mid;}int c=dc(lo),direction=(q-dp(lo))/dk(c),i=c%nx,j=(c/nx)%ny,k=c/(nx*ny);
      pr(IPX,q)=domain.x1min+(i+0.5)*dx;pr(IPY,q)=domain.x2min+(j+0.5)*dy;pr(IPZ,q)=domain.x3min+(k+0.5)*dz;
      pr(IPVX,q)=direction/2==0?(direction%2?-ud(c):ud(c)):0;pr(IPVY,q)=direction/2==1?(direction%2?-ud(c):ud(c)):0;pr(IPVZ,q)=direction/2==2?(direction%2?-ud(c):ud(c)):0;
      pr(IPM,q)=massd(c)/(6*dk(c));pr(IPEN,q)=0;pi(PGID,q)=gids+dm(lo);pi(PTAG,q)=dt(c)+q-dp(lo);
    });
    if(global_variable::my_rank==0){std::ofstream counts("initial_cell_counts.csv");counts<<"cell,particles,mass\n"<<std::setprecision(17);for(int c=0;c<ncells;++c)counts<<c<<','<<6*replicas[c]<<','<<mass[c]<<'\n';}
  }
  p->nprtcl_thispack=nlocal;pmy_mesh_->nprtcl_thisrank=nlocal;
  pmy_mesh_->nprtcl_eachrank[global_variable::my_rank]=nlocal;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&nlocal,1,MPI_INT,pmy_mesh_->nprtcl_eachrank,1,MPI_INT,MPI_COMM_WORLD);
#endif
  pmy_mesh_->nprtcl_total=std::accumulate(pmy_mesh_->nprtcl_eachrank,pmy_mesh_->nprtcl_eachrank+global_variable::nranks,0);
  if(pmy_mesh_->nprtcl_total!=ntotal){std::cerr<<"Incomplete benchmark initialization\n";std::exit(1);}
  Kokkos::deep_copy(p->adm_last,pk->padm->u_adm);Kokkos::deep_copy(p->z4c_last,pk->pz4c->u0);
  if(global_variable::my_rank==0)std::cout<<std::setprecision(12)<<"Efficiency sampler: metric="<<metric<<" sampling="<<sampling<<" distribution="<<distribution<<" N="<<ntotal<<" yc="<<yc<<" Riso/M="<<profile.riso_over_m_surface<<" M0/M="<<profile.rest_mass_over_m<<'\n';
}
