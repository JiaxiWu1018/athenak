#include "gi_diagnostics.hpp"
#include "z4c/tmunu.hpp"
#include "gi_diagnostic_math.hpp"
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "particles.hpp"
#include "outputs/outputs.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/compact_object_tracker.hpp"

namespace particles {
namespace {
void Open(std::ofstream &f,const std::string &name,const std::string &header) {
  std::ifstream old(name); bool empty=old.peek()==std::ifstream::traits_type::eof();
  f.open(name,std::ios::app); if(!f) throw std::runtime_error("cannot open "+name);
  f<<std::setprecision(17); if(empty) f<<header<<'\n';
}
void Sum(std::vector<Real> &a) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,a.data(),a.size(),MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
}
struct Crossing { int tag,h,old_in,adv_in,new_in,kind; Real q[18]; };
}

GiDiagnostics::GiDiagnostics(MeshBlockPack *p,ParameterInput *in):pack(p),mesh(p->pmesh),pin(in){
  if(!pack->pz4c || !pack->padm || !pack->ppart)
    throw std::runtime_error("GI accounting requires particles and live Z4c/ADM");
  nh=pack->pz4c->pfastflow.size(); if(nh<1 || nh>3)
    throw std::runtime_error("GI accounting supports one to three horizons");
  lm=pack->pz4c->pfastflow[0]->GetLmax(); lmp=(lm+1)*(lm+1);
  for(auto &f:pack->pz4c->pfastflow) if(f->GetLmax()!=lm)
    throw std::runtime_error("GI accounting requires equal harmonic order");
  Kokkos::realloc(old_par,nh,NAH_PAR); Kokkos::realloc(new_par,nh,NAH_PAR);
  Kokkos::realloc(old_coef,nh,lm+1+2*lmp); Kokkos::realloc(new_coef,nh,lm+1+2*lmp);
  Kokkos::realloc(shell_ledger,3,GI_NSHELL,8); Kokkos::deep_copy(shell_ledger,0.);
  common=pin->GetOrAddBoolean("gi_diagnostics","strict_common_seen",false);
  first_pair=pin->GetOrAddBoolean("gi_diagnostics","individual_pair_seen",false);
  close_pair=pin->GetOrAddBoolean("gi_diagnostics","close_pair_seen",false);
  next_sample=mesh->time;
  std::string base=pin->GetString("job","basename");
  Open(events,base+".gi_crossings.rank"+std::to_string(global_variable::my_rank)+".csv",
    "cycle,time,tag,component,horizon,kind,old_inside,advected_inside,new_inside,mass,x,y,z,u1_cov,u2_cov,u3_cov,En_trilinear,Ecoord_trilinear,metric_valid,Jz_origin,old_Jz_origin,old_x,old_y,old_z,old_u1_cov,old_u2_cov,old_u3_cov");
  if(global_variable::my_rank==0){
    Open(profiles,base+".gi_particles.csv","cycle,time,component,shell,mask,count,rest_mass,En_trilinear,Ecoord_trilinear,Ecoord_native,Px,Py,Pz,Jx,Jy,Jz,mx,my,mz,mxx,mxy,mxz,myy,myz,mzz,m_ur,invalid_metric_mass");
    Open(grid,base+".gi_grid.csv","cycle,time,shell,mask,proper_volume,H2_volume,M2_volume,alpha_volume,g_rr_volume,beta_r_volume,E_volume,invalid_coordinate_volume");
    Open(status,base+".gi_status.csv","cycle,time,dt,blocks,max_rank_blocks,capacity,valid_bits,ever_bits,strict_common,tracker0_live,tracker1_live,separation,x0,y0,z0,x1,y1,z1,proper_exterior_line,line_samples,line_valid_samples");
    Open(transport,base+".gi_shell_transfer.csv","cycle,time,component,shell,mass_in,mass_out,Jz_adv_in,Jz_adv_out,Jz_force_delta,Jz_AH_advection,Jz_AH_sweep,unknown_AH_mass_steps");
    Open(milestones,base+".gi_milestones.csv","cycle,time,event,restart_number");
  }
  Stage(new_par,new_coef,new_valid,new_ever,mesh->ncycle-1);
}

void GiDiagnostics::Stage(DvceArray2D<Real> &par,DvceArray2D<Real> &coef,
                          int &valid,int &ever,int expected_cycle){
  auto hp=Kokkos::create_mirror(par), hc=Kokkos::create_mirror(coef);
  Kokkos::deep_copy(hp,0.); Kokkos::deep_copy(hc,0.); valid=ever=0;
  for(int h=0;h<nh;++h){
    auto &f=*pack->pz4c->pfastflow[h];
    if(f.ever_published) ever|=1<<h;
    if(!f.published_this_search || f.last_search_cycle!=expected_cycle) continue;
    valid|=1<<h; hp(h,IAHVALID)=1;
    for(int a=0;a<3;++a) hp(h,a)=f.ah_surf_center[a];
    Real bound=0;
    for(int l=0;l<=lm;++l){
      hc(h,l)=f.a0_surf.h_view(l);
      Real ynorm=std::sqrt((2*l+1)/(4*M_PI));
      bound+=std::abs(hc(h,l))*ynorm;
      for(int m=1;m<=l;++m){int q=lmindex(l,m,lm);
        bound+=std::sqrt(2.)*ynorm*(std::abs(f.ac_surf.h_view(q))+std::abs(f.as_surf.h_view(q)));
      }
    }
    for(int q=0;q<lmp;++q){hc(h,lm+1+q)=f.ac_surf.h_view(q); hc(h,lm+1+lmp+q)=f.as_surf.h_view(q);}
    hp(h,IAHRMAX)=bound*(1+32*std::numeric_limits<Real>::epsilon());
  }
  Kokkos::deep_copy(par,hp); Kokkos::deep_copy(coef,hc);
}

void GiDiagnostics::BeforeStep(){
  Stage(old_par,old_coef,old_valid,old_ever,mesh->ncycle-1);
  auto pr=pack->ppart->prtcl_rdata; int n=pack->ppart->nprtcl_thispack;
  Kokkos::realloc(before,n,6); auto b=before;
  Kokkos::parallel_for("GI_save_particle_start",Kokkos::RangePolicy<DevExeSpace>(0,n),
    KOKKOS_LAMBDA(int p){b(p,0)=pr(IPX,p);b(p,1)=pr(IPY,p);b(p,2)=pr(IPZ,p);
      b(p,3)=pr(IPVX,p);b(p,4)=pr(IPVY,p);b(p,5)=pr(IPVZ,p);});
}

void GiDiagnostics::AfterPush(){
  Stage(new_par,new_coef,new_valid,new_ever,mesh->ncycle);
  auto pr=pack->ppart->prtcl_rdata; auto pi=pack->ppart->prtcl_idata;
  int n=pack->ppart->nprtcl_thispack, H=nh,L=lm,ov=old_valid,nv=new_valid,oe=old_ever;
  auto b=before,op=old_par,np=new_par,oc=old_coef,nc=new_coef;
  auto ledger=shell_ledger;
  if(before.extent(0)!=static_cast<size_t>(n)) throw std::runtime_error("GI pre-migration particle count changed");
  DvceArray2D<int> codes("GI_crossing_codes",n,H);
  Kokkos::parallel_for("GI_population_transfer",Kokkos::RangePolicy<DevExeSpace>(0,n),
    KOKKOS_LAMBDA(int p){
      Real x=pr(IPX,p),y=pr(IPY,p),z=pr(IPZ,p),m=pr(IPM,p);
      int c=GiComponent(pi(PTAG,p));
      int s0=GiShell(Kokkos::sqrt(b(p,0)*b(p,0)+b(p,1)*b(p,1)+b(p,2)*b(p,2)));
      int s1=GiShell(Kokkos::sqrt(x*x+y*y+z*z));
      Real j0=m*(b(p,0)*b(p,4)-b(p,1)*b(p,3));
      Real j1=m*(x*pr(IPVY,p)-y*pr(IPVX,p));
      if(s0!=s1){Kokkos::atomic_add(&ledger(c,s0,1),m);Kokkos::atomic_add(&ledger(c,s1,0),m);
        Kokkos::atomic_add(&ledger(c,s0,3),j0);Kokkos::atomic_add(&ledger(c,s1,2),j0);}
      Kokkos::atomic_add(&ledger(c,s1,4),j1-j0);
      bool union0=false,uniona=false,union1=false;
      for(int h=0;h<H;++h){
        bool i0=(ov&(1<<h)) && GiInside(op,oc,h,L,b(p,0),b(p,1),b(p,2));
        bool ia=(ov&(1<<h)) && GiInside(op,oc,h,L,x,y,z);
        bool i1=(nv&(1<<h)) && GiInside(np,nc,h,L,x,y,z);
        union0|=i0;uniona|=ia;union1|=i1;
        // Bits: old/adv/new membership, old/current validity; kind assigned on host.
        int code=int(i0)+2*int(ia)+4*int(i1)+8*int(bool(ov&(1<<h)))+16*int(bool(nv&(1<<h)));
        bool event=((ov&nv&(1<<h)) && (i0!=ia || ia!=i1)) ||
                   (!(ov&(1<<h)) && (nv&(1<<h)) && i1);
        codes(p,h)=event?code:-1;
      }
      // Union ledger avoids counting individual + common overlaps twice.
      if((ov==nv) && !(oe&~nv)){
        Kokkos::atomic_add(&ledger(c,s1,5),j0*(int(uniona)-int(union0)));
        Kokkos::atomic_add(&ledger(c,s1,6),j0*(int(union1)-int(uniona)));
      }else Kokkos::atomic_add(&ledger(c,s1,7),m);
    });
  int count=0;
  Kokkos::parallel_reduce("GI_count_crossings",Kokkos::RangePolicy<DevExeSpace>(0,n*H),
    KOKKOS_LAMBDA(int k,int &sum){if(codes(k/H,k%H)>=0)++sum;},count);
  if(count){
    DualArray1D<Crossing> out("GI_crossings",count);
    auto adm=pack->padm->u_adm,zu=pack->pz4c->u0;
    auto size=pack->pmb->mb_size.d_view;int gids=pack->gids,ng=mesh->mb_indcs.ng;
    Kokkos::parallel_scan("GI_pack_crossings",Kokkos::RangePolicy<DevExeSpace>(0,n*H),
      KOKKOS_LAMBDA(int k,int &offset,bool final){int p=k/H,h=k%H,code=codes(p,h);
        if(code<0)return;
        if(final){auto &e=out.d_view(offset);e.tag=pi(PTAG,p);e.h=h;
          e.old_in=(code&8)?int(bool(code&1)):-1;e.adv_in=(code&8)?int(bool(code&2)):-1;
          e.new_in=(code&16)?int(bool(code&4)):-1;
          // 1 transport/sweep; 2 first enclosure; 3 requalification; 4 common reassignment.
          e.kind=(code&8)?1:((oe&(1<<h))?3:(h==2?4:2));
          e.q[0]=pr(IPM,p);e.q[1]=pr(IPX,p);e.q[2]=pr(IPY,p);e.q[3]=pr(IPZ,p);
          e.q[4]=pr(IPVX,p);e.q[5]=pr(IPVY,p);e.q[6]=pr(IPVZ,p);
          auto metric=GiEnergy(adm,zu,size,pi(PGID,p)-gids,ng,e.q[1],e.q[2],e.q[3],e.q[4],e.q[5],e.q[6],e.q[0]);
          e.q[7]=metric.en;e.q[8]=metric.ec;e.q[9]=metric.valid;
          e.q[10]=e.q[0]*(e.q[1]*e.q[5]-e.q[2]*e.q[4]);
          e.q[11]=e.q[0]*(b(p,0)*b(p,4)-b(p,1)*b(p,3));
          for(int a=0;a<6;++a)e.q[12+a]=b(p,a);
        } ++offset;
      });
    out.template modify<DevExeSpace>();out.template sync<HostMemSpace>();
    for(int k=0;k<count;++k){auto &e=out.h_view(k);
      events<<mesh->ncycle+1<<','<<mesh->time+mesh->dt<<','<<e.tag<<','<<GiComponent(e.tag)<<','<<e.h<<','<<e.kind<<','<<e.old_in<<','<<e.adv_in<<','<<e.new_in;
      for(auto q:e.q)events<<','<<q;events<<'\n';
    } events.flush(); if(!events)throw std::runtime_error("GI crossing write failed");
  }
}

void GiDiagnostics::Sample(bool force){
  if(!force && mesh->time+1.e-12<next_sample)return;
  next_sample=mesh->time+(common?.05:.025);
  constexpr int Q=24, MASKS=5;
  const int H=nh,L=lm,nv=new_valid,ne=new_ever;
  const bool coverage=(ne&~nv)==0;
  auto par=new_par,coef=new_coef,pr=pack->ppart->prtcl_rdata;
  auto pi=pack->ppart->prtcl_idata;
  auto sz=pack->pmb->mb_size.d_view;
  auto adm=pack->padm->u_adm,zu=pack->pz4c->u0;
  int gids=pack->gids,ng=mesh->mb_indcs.ng,n=pack->ppart->nprtcl_thispack;
  // Per-block histogram limits contention and retains a small memory footprint.
  int nb=pack->nmb_thispack;
  DvceArray5D<Real> sums("GI_particle_profiles",nb,3,GI_NSHELL,MASKS,Q);
  Kokkos::deep_copy(sums,0.);
  Kokkos::parallel_for("GI_reduce_population",Kokkos::RangePolicy<DevExeSpace>(0,n),
    KOKKOS_LAMBDA(int p){
      int block=pi(PGID,p)-gids,c=GiComponent(pi(PTAG,p));
      Real x=pr(IPX,p),y=pr(IPY,p),z=pr(IPZ,p),m=pr(IPM,p);
      Real ux=pr(IPVX,p),uy=pr(IPVY,p),uz=pr(IPVZ,p),r=Kokkos::sqrt(x*x+y*y+z*z);
      int s=GiShell(r);bool inside[3]={};bool any=false;
      for(int h=0;h<H;++h){inside[h]=GiInside(par,coef,h,L,x,y,z);any|=inside[h];}
      auto metric=GiEnergy(adm,zu,sz,block,ng,x,y,z,ux,uy,uz,m);
      Real v[Q]={1,m,metric.en,metric.ec,m*pr(IPEN,p),m*ux,m*uy,m*uz,
        m*(y*uz-z*uy),m*(z*ux-x*uz),m*(x*uy-y*ux),m*x,m*y,m*z,
        m*x*x,m*x*y,m*x*z,m*y*y,m*y*z,m*z*z,
        r>0?m*(x*ux+y*uy+z*uz)/r:0,metric.valid?0:m,0,0};
      // mask 0 all; 1 known AH exterior; 2/3/4 individual/common interiors.
      for(int mask=0;mask<MASKS;++mask){
        bool use=mask==0 || (mask==1 && coverage && !any) || (mask>=2 && inside[mask-2]);
        if(use)for(int q=0;q<Q;++q)Kokkos::atomic_add(&sums(block,c,s,mask,q),v[q]);
      }
    });
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),sums);
  std::vector<Real> total(3*GI_NSHELL*MASKS*Q,0.);
  for(int b=0;b<nb;++b)for(int c=0;c<3;++c)for(int s=0;s<GI_NSHELL;++s)
    for(int mask=0;mask<MASKS;++mask)for(int q=0;q<Q;++q)
      total[((c*GI_NSHELL+s)*MASKS+mask)*Q+q]+=host(b,c,s,mask,q);
  Sum(total);
  if(global_variable::my_rank==0){
    for(int c=0;c<3;++c)for(int s=0;s<GI_NSHELL;++s)for(int mask=0;mask<MASKS;++mask){
      int idx=((c*GI_NSHELL+s)*MASKS+mask)*Q;
      if(total[idx]==0 && !(mask==1 && !coverage)) continue;
      profiles<<mesh->ncycle<<','<<mesh->time<<','<<c<<','<<s<<','<<mask;
      for(int q=0;q<22;++q)profiles<<','<<((mask==1&&!coverage)?std::numeric_limits<Real>::quiet_NaN():total[idx+q]);profiles<<'\n';
    }profiles.flush();
  }
  auto lh=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),shell_ledger);
  std::vector<Real> lt(3*GI_NSHELL*8);
  for(int c=0;c<3;++c)for(int s=0;s<GI_NSHELL;++s)for(int q=0;q<8;++q)lt[(c*GI_NSHELL+s)*8+q]=lh(c,s,q);
  Sum(lt);Kokkos::deep_copy(shell_ledger,0.);
  if(global_variable::my_rank==0){
    for(int c=0;c<3;++c)for(int s=0;s<GI_NSHELL;++s){
      transport<<mesh->ncycle<<','<<mesh->time<<','<<c<<','<<s;
      for(int q=0;q<8;++q)transport<<','<<lt[(c*GI_NSHELL+s)*8+q];transport<<'\n';
    }transport.flush();
  }

  switch(ng){case 2:pack->pz4c->ADMConstraints<2>(pack);break;
    case 3:pack->pz4c->ADMConstraints<3>(pack);break;case 4:pack->pz4c->ADMConstraints<4>(pack);break;}
  auto con=pack->pz4c->u_con;
  DvceArray5D<Real> matter;
  bool have_matter=pack->ptmunu!=nullptr;if(have_matter)matter=pack->ptmunu->u_tmunu;
  Real chi_cut=pack->pz4c->opt.excise_chi;
  constexpr int GQ=8,GM=3;
  DvceArray4D<Real> gs("GI_grid_profiles",nb,GI_NSHELL,GM,GQ);Kokkos::deep_copy(gs,0.);
  auto ind=mesh->mb_indcs;int nx=ind.nx1,ny=ind.nx2,nz=ind.nx3;
  Kokkos::parallel_for("GI_reduce_grid",Kokkos::RangePolicy<DevExeSpace>(0,nb*nx*ny*nz),
    KOKKOS_LAMBDA(int t){int i=t%nx,j=(t/nx)%ny,k=(t/(nx*ny))%nz,b=t/(nx*ny*nz);
      auto box=sz(b);Real x=box.x1min+(i+.5)*box.dx1,y=box.x2min+(j+.5)*box.dx2,z=box.x3min+(k+.5)*box.dx3;
      i+=ng;j+=ng;k+=ng;Real r=Kokkos::sqrt(x*x+y*y+z*z);int s=GiShell(r);
      bool any=false;for(int h=0;h<H;++h)any|=GiInside(par,coef,h,L,x,y,z);
      Real g[6];for(int a=0;a<6;++a)g[a]=adm(b,a,k,j,i);
      Real dv=box.dx1*box.dx2*box.dx3,det=Primitive::GetDeterminant(g);
      bool good=det>0 && Kokkos::isfinite(det);Real v=good?dv*Kokkos::sqrt(det):0;
      Real grr=r>0?(g[0]*x*x+g[3]*y*y+g[5]*z*z+2*(g[1]*x*y+g[2]*x*z+g[4]*y*z))/(r*r):0;
      Real br=r>0?(zu(b,z4c::Z4c::I_Z4C_BETAX,k,j,i)*x+zu(b,z4c::Z4c::I_Z4C_BETAY,k,j,i)*y+zu(b,z4c::Z4c::I_Z4C_BETAZ,k,j,i)*z)/r:0;
      Real q[GQ]={v,v*con(b,1,k,j,i)*con(b,1,k,j,i),v*con(b,2,k,j,i),
        v*zu(b,z4c::Z4c::I_Z4C_ALPHA,k,j,i),v*grr,v*br,
        have_matter?v*matter(b,6,k,j,i):0,good?0:dv};
      for(int mask=0;mask<GM;++mask){bool use=mask==0 || (mask==1 && coverage&&!any) ||
        (mask==2 && zu(b,z4c::Z4c::I_Z4C_CHI,k,j,i)>=chi_cut);
        if(use)for(int a=0;a<GQ;++a)Kokkos::atomic_add(&gs(b,s,mask,a),q[a]);}
    });
  auto gh=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),gs);
  std::vector<Real> gt(GI_NSHELL*GM*GQ,0.);
  for(int b=0;b<nb;++b)for(int s=0;s<GI_NSHELL;++s)for(int mask=0;mask<GM;++mask)for(int q=0;q<GQ;++q)gt[(s*GM+mask)*GQ+q]+=gh(b,s,mask,q);
  Sum(gt);
  auto &tr=pack->pz4c->ptracker;
  bool live0=tr.size()>0&&tr[0]->DiagnosticLive(),live1=tr.size()>1&&tr[1]->DiagnosticLive();
  Real pos[6];for(auto &x:pos)x=std::numeric_limits<Real>::quiet_NaN();
  for(int h=0;h<std::min(2,static_cast<int>(tr.size()));++h)for(int a=0;a<3;++a)pos[h*3+a]=tr[h]->GetPos(a);
  Real d=std::sqrt(SQR(pos[0]-pos[3])+SQR(pos[1]-pos[4])+SQR(pos[2]-pos[5]));
  constexpr int NLINE=2048;
  std::vector<Real> line_total(NLINE*2,0.);
  bool binary_live=live0&&live1&&!common&&coverage&&std::isfinite(d)&&d>0;
  if(binary_live){
    DvceArray2D<Real> line("GI_exterior_line",NLINE,2);Kokkos::deep_copy(line,0.);
    Kokkos::parallel_for("GI_proper_line",Kokkos::RangePolicy<DevExeSpace>(0,NLINE),
      KOKKOS_LAMBDA(int p){
        Real f=(p+.5)/NLINE,dx=pos[3]-pos[0],dy=pos[4]-pos[1],dz=pos[5]-pos[2];
        Real x=pos[0]+f*dx,y=pos[1]+f*dy,z=pos[2]+f*dz;
        for(int b=0;b<nb;++b){auto box=sz(b);
          if(x<box.x1min||x>=box.x1max||y<box.x2min||y>=box.x2max||z<box.x3min||z>=box.x3max)continue;
          bool interior=false;for(int h=0;h<H;++h)interior|=GiInside(par,coef,h,L,x,y,z);
          if(interior){line(p,1)=1;break;}
          Real q[3]={(x-box.x1min)/box.dx1+ng-.5,(y-box.x2min)/box.dx2+ng-.5,(z-box.x3min)/box.dx3+ng-.5};
          int ix[3];Real w[3];for(int a=0;a<3;++a){ix[a]=static_cast<int>(Kokkos::floor(q[a]));w[a]=q[a]-ix[a];}
          Real g[6]={};
          for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i){
            Real wt=(i?w[0]:1-w[0])*(j?w[1]:1-w[1])*(k?w[2]:1-w[2]);
            for(int a=0;a<6;++a)g[a]+=wt*adm(b,a,ix[2]+k,ix[1]+j,ix[0]+i);
          }
          Real ds2=g[0]*dx*dx+g[3]*dy*dy+g[5]*dz*dz+2*(g[1]*dx*dy+g[2]*dx*dz+g[4]*dy*dz);
          if(ds2>0&&Kokkos::isfinite(ds2)){line(p,0)=Kokkos::sqrt(ds2)/NLINE;line(p,1)=1;}
          break;
        }
      });
    auto hl=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),line);
    for(int p=0;p<NLINE;++p){line_total[p*2]=hl(p,0);line_total[p*2+1]=hl(p,1);}Sum(line_total);
  }
  Real proper=0;int line_valid=0;
  for(int p=0;p<NLINE;++p){proper+=line_total[p*2];if(line_total[p*2+1]==1)++line_valid;}
  if(!binary_live||line_valid!=NLINE)proper=std::numeric_limits<Real>::quiet_NaN();
  if(global_variable::my_rank==0){
    for(int s=0;s<GI_NSHELL;++s)for(int mask=0;mask<GM;++mask){
      grid<<mesh->ncycle<<','<<mesh->time<<','<<s<<','<<mask;
      for(int q=0;q<GQ;++q)grid<<','<<((mask==1&&!coverage)?std::numeric_limits<Real>::quiet_NaN():gt[(s*GM+mask)*GQ+q]);grid<<'\n';
    }grid.flush();
    int maxb=0;for(int r=0;r<global_variable::nranks;++r)maxb=std::max(maxb,mesh->nmb_eachrank[r]);
    status<<mesh->ncycle<<','<<mesh->time<<','<<mesh->dt<<','<<mesh->nmb_total<<','<<maxb<<','<<mesh->nmb_maxperrank<<','<<nv<<','<<ne<<','<<common<<','<<live0<<','<<live1<<','<<d;
    for(auto x:pos)status<<','<<x;status<<','<<proper<<','<<NLINE<<','<<line_valid<<'\n';status.flush();
    if(!profiles||!grid||!status||!transport)throw std::runtime_error("GI compact output failed");
  }
}

void GiDiagnostics::Milestones(Outputs *outputs){
  auto &tr=pack->pz4c->ptracker;
  bool live=tr.size()>=2 && tr[0]->DiagnosticLive() && tr[1]->DiagnosticLive();
  Real d=std::numeric_limits<Real>::infinity();
  if(live){d=0;for(int a=0;a<3;++a)d+=SQR(tr[0]->GetPos(a)-tr[1]->GetPos(a));d=std::sqrt(d);}
  auto par=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),new_par);
  auto coef=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),new_coef);
  bool enclosed=nh==3 && (new_valid&4) && live &&
    GiInside(par,coef,2,lm,tr[0]->GetPos(0),tr[0]->GetPos(1),tr[0]->GetPos(2)) &&
    GiInside(par,coef,2,lm,tr[1]->GetPos(0),tr[1]->GetPos(1),tr[1]->GetPos(2));
  bool pair=(new_valid&3)==3 && live && !common && !enclosed;
  std::string event;
  if(pair&&!first_pair){first_pair=true;pin->SetBoolean("gi_diagnostics","individual_pair_seen",true);event="first_individual_pair";}
  bool distinct=pair && !GiInside(par,coef,0,lm,tr[1]->GetPos(0),tr[1]->GetPos(1),tr[1]->GetPos(2)) &&
    !GiInside(par,coef,1,lm,tr[0]->GetPos(0),tr[0]->GetPos(1),tr[0]->GetPos(2));
  if(distinct&&!close_pair&&d<1&&previous_separation>d){close_pair=true;pin->SetBoolean("gi_diagnostics","close_pair_seen",true);event+=(event.empty()?"":"+")+std::string("first_inward_d_lt_1");}
  if(enclosed&&!common){common=true;pin->SetBoolean("gi_diagnostics","strict_common_seen",true);event+=(event.empty()?"":"+")+std::string("strict_common");}
  previous_separation=d;
  for(auto &out:outputs->pout_list){
    if(out->out_params.file_type=="trk")out->out_params.dt=(!common && ((mesh->time>=6&&mesh->time<=12)||d<2))?.05:.25;
    if(event.empty())continue;
    if(out->out_params.file_type=="rst" || out->out_params.file_type=="trk"){
      int number=out->out_params.file_number;
      out->LoadOutputData(mesh);out->WriteOutputFile(mesh,pin);
      out->out_params.last_time=mesh->time;pin->SetReal(out->out_params.block_name,"last_time",mesh->time);
      if(out->out_params.file_type=="rst"&&global_variable::my_rank==0){milestones<<mesh->ncycle<<','<<mesh->time<<','<<event<<','<<number<<'\n';milestones.flush();}
    }
  }
}
}  // namespace particles
