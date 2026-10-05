// Included inside nr_pic_plummer.cpp's anonymous namespace, after its shared helpers.
void InitializeIsotropicPlummer(Mesh *pm, ParameterInput *pin, bool restart) {
  auto pmbp=pm->pmb_pack;
  const bool live=pmbp->pz4c!=nullptr;
  const Real M=pin->GetOrAddReal("problem","plummer_mass",1.0);
  const Real a=pin->GetOrAddReal("problem","plummer_isotropic_a",10.0);
  const Real cut=pin->GetOrAddReal("problem","plummer_sampling_radius",1000.0);
  const int npair=pin->GetOrAddInteger("problem","plummer_npair",1056768);
  const int seed=pin->GetOrAddInteger("problem","plummer_seed",1985);
  if (npair<=0 || 2LL*npair>INT_MAX) Fatal("invalid isotropic particle-pair count");
  const int ntotal=2*npair;
  for(int d=0;d<3;++d) {
    plummer_center[d]=pin->GetOrAddReal("problem","plummer_center_x"+std::to_string(d+1),0.0);
  }
  plummer_shell_nbin=pin->GetOrAddInteger("problem","plummer_shell_nbin",64);
  plummer_shell_rmin=pin->GetOrAddReal("problem","plummer_shell_rmin",0.3125);
  plummer_shell_rmax=pin->GetOrAddReal("problem","plummer_shell_rmax",4000.0);
  plummer_field_nbin=pin->GetOrAddInteger("problem","plummer_field_nbin",80);
  plummer_field_rmin=pin->GetOrAddReal("problem","plummer_field_rmin",0.3125);
  plummer_field_rmax=pin->GetOrAddReal("problem","plummer_field_rmax",30000.0);
  plummer_cohort_nbin=pin->GetOrAddInteger("problem","plummer_cohort_nbin",32);
  plummer_lmax=pin->GetOrAddInteger("problem","plummer_lmax",4);
  if(plummer_shell_nbin<1 || plummer_field_nbin<1 || plummer_cohort_nbin<1 ||
     plummer_lmax<1 || plummer_lmax>4 || !(plummer_shell_rmax>plummer_shell_rmin &&
     plummer_shell_rmin>0 && plummer_field_rmax>plummer_field_rmin && plummer_field_rmin>0))
    Fatal("invalid isotropic diagnostic bins");
  const std::string base=pin->GetString("job","basename");
  plummer_shell_fname=base+".plummer_shells.csv";
  plummer_cohort_fname=base+".plummer_cohorts.csv";
  plummer_field_fname=base+".plummer_fields.csv";
  plummer_physical_fname=base+".plummer_physical.csv";
  plummer_health_fname=base+".plummer_health.csv";
  plummer_constraint_reference=pin->GetOrAddReal("problem","plummer_constraint_reference",0.0);
  plummer_constraint_strikes=pin->GetOrAddInteger("problem","plummer_constraint_initial_strikes",0);
  plummer::IsotropicProfile prof(M,a,cut);
  const Real mu=prof.M0/ntotal;
  plummer_particle_mass=mu;plummer_M0=prof.M0;plummer_MADM=M;
  plummer_bscale=a;plummer_rt=cut;plummer_Rt=cut;
  plummer_Phalf=prof.P_ref;plummer_rhalf=prof.areal(prof.rhalf);
  plummer_npair=npair;plummer_ntotal=ntotal;
  PlummerSetupADMMomentum(pm,pin,base);
  auto &ix=pm->mb_indcs;
  if(!restart || !live) {
    auto adm=pmbp->padm->adm;
    auto size=pmbp->pmb->mb_size;
    const int is=ix.is,js=ix.js,ks=ix.ks,nx=ix.nx1,ny=ix.nx2,nz=ix.nx3;
    const Real cx=plummer_center[0],cy=plummer_center[1],cz=plummer_center[2];
    par_for("isotropic Plummer analytic metric",DevExeSpace(),0,pmbp->nmb_thispack-1,
      ks-ix.ng,ix.ke+ix.ng,js-ix.ng,ix.je+ix.ng,is-ix.ng,ix.ie+ix.ng,
      KOKKOS_LAMBDA(int mb,int k,int j,int i) {
        const Real x=CellCenterX(i-is,nx,size.d_view(mb).x1min,size.d_view(mb).x1max)-cx;
        const Real y=CellCenterX(j-js,ny,size.d_view(mb).x2min,size.d_view(mb).x2max)-cy;
        const Real z=CellCenterX(k-ks,nz,size.d_view(mb).x3min,size.d_view(mb).x3max)-cz;
        const Real u=M/(2*Kokkos::sqrt(x*x+y*y+z*z+a*a)),p=1+u,p2=p*p,p4=p2*p2;
        adm.psi4(mb,k,j,i)=p4;adm.alpha(mb,k,j,i)=(1-u)/p;
        for(int d=0;d<3;++d) {
          adm.beta_u(mb,d,k,j,i)=0;
          for(int e=d;e<3;++e) {
            adm.g_dd(mb,d,e,k,j,i)=d==e?p4:0;
            adm.vK_dd(mb,d,e,k,j,i)=0;
          }
        }
      });
    Kokkos::fence();
    if(live && !restart) {
      switch(ix.ng) {
        case 2:pmbp->pz4c->ADMToZ4c<2>(pmbp,pin);break;
        case 3:pmbp->pz4c->ADMToZ4c<3>(pmbp,pin);break;
        case 4:pmbp->pz4c->ADMToZ4c<4>(pmbp,pin);break;
        default:Fatal("isotropic Plummer requires nghost=2,3,4");
      }
      pmbp->pz4c->Z4cToADM(pmbp);
    }
  }
  auto snapshots=[&]() {
    Kokkos::deep_copy(DevExeSpace(),pmbp->ppart->adm_last,pmbp->padm->u_adm);
    if(live) Kokkos::deep_copy(DevExeSpace(),pmbp->ppart->z4c_last,pmbp->pz4c->u0);
  };
  if(pin->GetOrAddString("particles","init","ppc")!="pgen") Fatal("isotropic Plummer requires init=pgen");
  PrtclStage stage;
  auto part=pmbp->ppart;
  const auto key=static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed));
  if(!live) Kokkos::realloc(plummer_orbit_reference,4,npair);
  Real pair_error=0;
  for(int k=0;k<npair;++k) {
    const Real r=prof.InvertF0((k+HashUnitId(key,k,0))/npair);
    const Real zp=2*HashUnitId(key,k,1)-1,pp=2*M_PI*HashUnitId(key,k,2);
    const Real zm=2*HashUnitId(key,k,4)-1,pmom=2*M_PI*HashUnitId(key,k,5);
    const Real sp=std::sqrt(std::max(0.,1-zp*zp)),sm=std::sqrt(std::max(0.,1-zm*zm));
    const Real x[3]={plummer_center[0]+r*sp*std::cos(pp),
                     plummer_center[1]+r*sp*std::sin(pp),plummer_center[2]+r*zp};
    const Real q=prof.SampleQ(r,HashUnitId(key,k,3));
    const Real um=prof.psi(r)*prof.psi(r)*q;
    const Real v[3]={um*sm*std::cos(pmom),um*sm*std::sin(pmom),um*zm};
    if(!live) {
      plummer_orbit_reference.h_view(0,k)=prof.alpha(r)*std::sqrt(1+q*q);
      const Real xr=x[0]-plummer_center[0],yr=x[1]-plummer_center[1],zr=x[2]-plummer_center[2];
      plummer_orbit_reference.h_view(1,k)=yr*v[2]-zr*v[1];
      plummer_orbit_reference.h_view(2,k)=zr*v[0]-xr*v[2];
      plummer_orbit_reference.h_view(3,k)=xr*v[1]-yr*v[0];
    }
    if(restart) continue;
    const int mb=part->FindContainingMeshBlock(x[0],x[1],x[2]);
    if(mb>=0) {
      stage.Add(x[0],x[1],x[2],v[0],v[1],v[2],pmbp->gids+mb,2*k);
      stage.Add(x[0],x[1],x[2],-v[0],-v[1],-v[2],pmbp->gids+mb,2*k+1);
    }
    for(int d=0;d<3;++d) pair_error=std::max(pair_error,std::abs(v[d]+(-v[d])));
  }
  if(!live) {
    plummer_orbit_reference.template modify<HostMemSpace>();
    plummer_orbit_reference.template sync<DevExeSpace>();
  }
  if(restart) {snapshots();return;}
  const int nl=stage.x.size();
  Kokkos::realloc(part->prtcl_rdata,part->nrdata,nl);
  Kokkos::realloc(part->prtcl_idata,part->nidata,nl);
  auto hr=Kokkos::create_mirror_view(part->prtcl_rdata);
  auto hi=Kokkos::create_mirror_view(part->prtcl_idata);
  for(int p=0;p<nl;++p) {
    hi(PGID,p)=stage.gid[p];hi(PTAG,p)=stage.tag[p];hr(IPM,p)=mu;hr(IPEN,p)=0;
    hr(IPX,p)=stage.x[p];hr(IPY,p)=stage.y[p];hr(IPZ,p)=stage.z[p];
    hr(IPVX,p)=stage.ux[p];hr(IPVY,p)=stage.uy[p];hr(IPVZ,p)=stage.uz[p];
  }
  Kokkos::deep_copy(part->prtcl_rdata,hr);Kokkos::deep_copy(part->prtcl_idata,hi);
  part->nprtcl_thispack=nl;part->mass=mu;pm->nprtcl_thisrank=nl;
  pm->nprtcl_eachrank[global_variable::my_rank]=nl;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&nl,1,MPI_INT,pm->nprtcl_eachrank,1,MPI_INT,MPI_COMM_WORLD);
#endif
  pm->nprtcl_total=0;
  for(int rank=0;rank<global_variable::nranks;++rank) pm->nprtcl_total+=pm->nprtcl_eachrank[rank];
  if(pm->nprtcl_total!=ntotal || pair_error!=0) Fatal("isotropic initial accounting/pair cancellation failed");
  snapshots();
  if(global_variable::my_rank==0) {
    std::ofstream out(base+".isotropic_initial.csv");
    out<<std::setprecision(17)<<"M,a,eta,sampling_radius,M0,M0_inf,omitted_M0,mu,N,Npair,seed,rmedian_iso,rmedian_areal,P_ref,pair_momentum_error\n"
       <<M<<','<<a<<','<<M/a<<','<<cut<<','<<prof.M0<<','<<prof.M0_inf<<','
       <<prof.M0_inf-prof.M0<<','<<mu<<','<<ntotal<<','<<npair<<','<<seed<<','
       <<prof.rhalf<<','<<prof.areal(prof.rhalf)<<','<<prof.P_ref<<','<<pair_error<<'\n';
    std::cout<<std::setprecision(17)<<"ISOTROPIC_PLUMMER M0="<<prof.M0<<" M0_inf="<<prof.M0_inf
             <<" P_ref="<<prof.P_ref<<" N="<<ntotal<<" sampling_radius="<<cut
             <<" omitted_M0="<<prof.M0_inf-prof.M0<<std::endl;
  }
}
