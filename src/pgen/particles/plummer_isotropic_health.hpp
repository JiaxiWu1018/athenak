Real plummer_core_h_l2=0,plummer_core_m_l2=0;
Real plummer_char_bound=0,plummer_psi_max=0;
// Cartesian ADM mass: differentiate the same tensor-product Lagrange polynomial
// used for the extraction. No conformal-flatness assumption is made in evolution.
void IsotropicADMMass(Mesh *pm) {
  static Real last=-1.e100;
  if(pm->time-last<plummer_Phalf/10-0.5*pm->dt && !pm->user_stop_requested) return;
  last=pm->time;
  auto pack=pm->pmb_pack;auto ua=pack->padm->u_adm;auto size=pack->pmb->mb_size;
  auto ix=pm->mb_indcs;
  const int is=ix.is,js=ix.js,ks=ix.ks,ng=ix.ng,nx=ix.nx1,ny=ix.nx2,nz=ix.nx3;
  for(std::size_t sphere=0;sphere<plummer_pmom_grids.size();++sphere) {
    auto grid=plummer_pmom_grids[sphere];auto ids=grid->interp_indcs.d_view;
    auto wg=grid->interp_wghts.d_view;auto xp=grid->cart_pos.d_view;
    const Real R=grid->radius;
    DvceArray1D<Real> integrand("ADM mass sphere",grid->nangles);
    Kokkos::parallel_for("Cartesian ADM mass",Kokkos::RangePolicy<>(DevExeSpace(),0,grid->nangles),
      KOKKOS_LAMBDA(int n) {
        const int mb=ids(n,0);if(mb<0) {integrand(n)=0;return;}
        Real dw[3][8];
        const Real lo[3]={size.d_view(mb).x1min,size.d_view(mb).x2min,size.d_view(mb).x3min};
        const Real hi[3]={size.d_view(mb).x1max,size.d_view(mb).x2max,size.d_view(mb).x3max};
        const int nn[3]={nx,ny,nz};
        for(int d=0;d<3;++d) for(int v=0;v<2*ng;++v) {
          const Real xv=CellCenterX(ids(n,d+1)-ng+v+1,nn[d],lo[d],hi[d]);
          Real deriv=0;
          for(int excluded=0;excluded<2*ng;++excluded) if(excluded!=v) {
            const Real xe=CellCenterX(ids(n,d+1)-ng+excluded+1,nn[d],lo[d],hi[d]);
            Real term=1/(xv-xe);
            for(int k=0;k<2*ng;++k) if(k!=v && k!=excluded) {
              const Real xk=CellCenterX(ids(n,d+1)-ng+k+1,nn[d],lo[d],hi[d]);
              term*=(xp(n,d)-xk)/(xv-xk);
            }
            deriv+=term;
          }
          dw[d][v]=deriv;
        }
        Real result=0;
        const int map[3][3]={{0,1,2},{1,3,4},{2,4,5}};
        for(int i=0;i<2*ng;++i) for(int j=0;j<2*ng;++j) for(int k=0;k<2*ng;++k) {
          const int ii=ids(n,1)-ng+i+is+1,jj=ids(n,2)-ng+j+js+1,kk=ids(n,3)-ng+k+ks+1;
          const Real w[3]={wg(n,i,0),wg(n,j,1),wg(n,k,2)};
          const Real deriv[3]={dw[0][i]*w[1]*w[2],w[0]*dw[1][j]*w[2],w[0]*w[1]*dw[2][k]};
          Real g[6];for(int v=0;v<6;++v) g[v]=ua(mb,adm::ADM::I_ADM_GXX+v,kk,jj,ii);
          const Real trace=g[0]+g[3]+g[5];
          for(int d=0;d<3;++d) {
            Real f=-trace*deriv[d];
            for(int e=0;e<3;++e) f+=g[map[d][e]]*deriv[e];
            result+=xp(n,d)/R*f;
          }
        }
        integrand(n)=result;
      });
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),integrand);
    Real sum=0;for(int n=0;n<grid->nangles;++n) sum+=host(n)*plummer_pmom_wt[sphere][n];
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
    const Real measured=sum*R*R/(16*M_PI);
    const Real a=plummer_bscale,M=plummer_MADM,p=1+M/(2*std::hypot(R,a));
    const Real initial=M*std::pow(R/std::hypot(R,a),3)*p*p*p;
    if(!std::isfinite(measured)) Fatal("ISOTROPIC_HEALTH_FAIL non-finite ADM mass extraction");
    if(global_variable::my_rank==0) {
      const std::string name=plummer_health_fname.substr(0,plummer_health_fname.find(".plummer_health"))+".plummer_admmass.csv";
      const bool header=FileIsEmpty(name);std::ofstream out(name,std::ios::app);
      if(header) out<<"time,cycle,R,M_adm,analytic_initial_M_adm,M_asymptotic\n";
      out<<std::setprecision(17)<<pm->time<<','<<pm->ncycle<<','<<R<<','<<measured<<','<<initial<<','<<M<<'\n';
    }
    if(pm->time==0 && std::abs(measured-initial)>1.e-7*std::abs(initial))
      Fatal("ADM mass positive analytic initial-metric check failed");
  }
}
// Validate fields before interpolating particles or generating scientific products.
void CheckIsotropicFields(Mesh *pm) {
  auto pack=pm->pmb_pack;
  auto ua=pack->padm->u_adm;
  DvceArray5D<Real> u0,uc,ut;
  const bool live=pack->pz4c!=nullptr,source=pack->ptmunu!=nullptr;
  if(live) {
    // In particular at t=0, derived constraints must include the freshly seeded
    // particle source rather than an uninitialized or vacuum-only diagnostic.
    switch(pm->mb_indcs.ng) {
      case 2:pack->pz4c->ADMConstraints<2>(pack);break;
      case 3:pack->pz4c->ADMConstraints<3>(pack);break;
      case 4:pack->pz4c->ADMConstraints<4>(pack);break;
    }
    u0=pack->pz4c->u0;uc=pack->pz4c->u_con;
  }
  if(source) ut=pack->ptmunu->u_tmunu;
  const int na=ua.extent(1),nv=live?u0.extent(1):0,nc=live?uc.extent(1):0,
            nt=source?ut.extent(1):0;
  auto ix=pm->mb_indcs;auto sz=pack->pmb->mb_size;
  const int nx=ix.nx1,ny=ix.nx2,nz=ix.nx3,is=ix.is,js=ix.js,ks=ix.ks;
  const int cells=pack->nmb_thispack*nx*ny*nz;
  const Real cx=plummer_center[0],cy=plummer_center[1],cz=plummer_center[2],core=10*plummer_bscale;
  const int bins=plummer_shell_nbin;
  const Real lr=std::log(plummer_shell_rmin),ilr=bins/std::log(plummer_shell_rmax/plummer_shell_rmin);
  DvceArray1D<Real> volumes("physical shell volume",bins*3);
  DvceArray1D<Real> peaks("field propagation bounds",2);
  Kokkos::deep_copy(volumes,0.0);
  Kokkos::deep_copy(peaks,0.0);
  Real invalid=0,h2=0,m2=0,vol=0;
  Kokkos::parallel_reduce("isotropic field health",Kokkos::RangePolicy<>(DevExeSpace(),0,cells),
    KOKKOS_LAMBDA(int idx,Real &bad,Real &hs,Real &ms,Real &vs) {
      const int i=idx%nx+is,j=(idx/nx)%ny+js,k=(idx/(nx*ny))%nz+ks,m=idx/(nx*ny*nz);
      bool finite=true;
      for(int v=0;v<na;++v) finite=finite && Kokkos::isfinite(ua(m,v,k,j,i));
      for(int v=0;v<nv;++v) finite=finite && Kokkos::isfinite(u0(m,v,k,j,i));
      for(int v=0;v<nc;++v) finite=finite && Kokkos::isfinite(uc(m,v,k,j,i));
      for(int v=0;v<nt;++v) finite=finite && Kokkos::isfinite(ut(m,v,k,j,i));
      Real g[6];for(int d=0;d<6;++d) g[d]=ua(m,adm::ADM::I_ADM_GXX+d,k,j,i);
      const Real determinant=Primitive::GetDeterminant(g);
      const bool spd=g[0]>0 && g[0]*g[3]-g[1]*g[1]>0 && determinant>0 && Kokkos::isfinite(determinant);
      if(!finite || !spd || (live && !(u0(m,z4c::Z4c::I_Z4C_CHI,k,j,i)>0))) {bad+=1;return;}
      Real inverse[6];Primitive::InvertMatrix(inverse,g,Primitive::GetDeterminant(g));
      const Real eigen_bound=Kokkos::fmax(inverse[0]+Kokkos::fabs(inverse[1])+Kokkos::fabs(inverse[2]),
        Kokkos::fmax(inverse[3]+Kokkos::fabs(inverse[1])+Kokkos::fabs(inverse[4]),
                     inverse[5]+Kokkos::fabs(inverse[2])+Kokkos::fabs(inverse[4])));
      // Live ADM storage excludes lapse and shift; their shallow tensor views
      // refer to Z4c.u0. Access the owning array explicitly in both modes.
      const Real alpha=live?u0(m,z4c::Z4c::I_Z4C_ALPHA,k,j,i):ua(m,adm::ADM::I_ADM_ALPHA,k,j,i);
      const Real bx=live?u0(m,z4c::Z4c::I_Z4C_BETAX,k,j,i):ua(m,adm::ADM::I_ADM_BETAX,k,j,i);
      const Real by=live?u0(m,z4c::Z4c::I_Z4C_BETAY,k,j,i):ua(m,adm::ADM::I_ADM_BETAY,k,j,i);
      const Real bz=live?u0(m,z4c::Z4c::I_Z4C_BETAZ,k,j,i):ua(m,adm::ADM::I_ADM_BETAZ,k,j,i);
      // Fixed benchmark gauge: 1+log coefficient 2 and legacy shift_Gamma=1.
      // The longitudinal shift speed uses the inverse conformal metric, so its
      // bound includes psi^2, not lapse, before multiplying sqrt(gamma^{-1}).
      const Real psi2=Kokkos::pow(determinant,1.0/6);
      const Real speed=Kokkos::sqrt(bx*bx+by*by+bz*bz)+Kokkos::sqrt(eigen_bound)*
        Kokkos::fmax(Kokkos::sqrt(2*Kokkos::fabs(alpha)),psi2*Kokkos::sqrt(4.0/3));
      Kokkos::atomic_max(&peaks(0),speed);
      Kokkos::atomic_max(&peaks(1),Kokkos::pow(Primitive::GetDeterminant(g),1.0/12));
      const Real x=CellCenterX(i-is,nx,sz.d_view(m).x1min,sz.d_view(m).x1max)-cx;
      const Real y=CellCenterX(j-js,ny,sz.d_view(m).x2min,sz.d_view(m).x2max)-cy;
      const Real z=CellCenterX(k-ks,nz,sz.d_view(m).x3min,sz.d_view(m).x3max)-cz;
      const Real r2=x*x+y*y+z*z,r=Kokkos::sqrt(r2);
      Real grr=0;
      if(r2>0) grr=(g[0]*x*x+g[3]*y*y+g[5]*z*z+2*(g[1]*x*y+g[2]*x*z+g[4]*y*z))/r2;
      const Real rareal=r*Kokkos::sqrt((g[0]+g[3]+g[5]-grr)/2);
      int bin=static_cast<int>((Kokkos::log(Kokkos::fmax(rareal,1.e-12))-lr)*ilr);
      bin=bin<0?0:bin>=bins?bins-1:bin;
      const Real proper=Kokkos::sqrt(Primitive::GetDeterminant(g))*sz.d_view(m).dx1*sz.d_view(m).dx2*sz.d_view(m).dx3;
      Kokkos::atomic_add(&volumes(bin*3),proper);
      Kokkos::atomic_add(&volumes(bin*3+1),1.0);
      Kokkos::atomic_add(&volumes(bin*3+2),proper*sz.d_view(m).dx1);
      if(live && x*x+y*y+z*z<=core*core) {
        const Real dv=sz.d_view(m).dx1*sz.d_view(m).dx2*sz.d_view(m).dx3;
        const Real H=uc(m,z4c::Z4c::I_CON_H,k,j,i);
        hs+=H*H*dv;ms+=uc(m,z4c::Z4c::I_CON_M,k,j,i)*dv;vs+=dv;
      }
    },invalid,h2,m2,vol);
  Real red[4]={invalid,h2,m2,vol};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,red,4,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  if(red[0]!=0) Fatal("ISOTROPIC_HEALTH_FAIL non-finite field/source or invalid spatial metric at t="+std::to_string(pm->time));
  auto hv=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),volumes);
  plummer_proper_volumes.assign(hv.data(),hv.data()+hv.extent(0));
  auto peak_host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),peaks);
  Real peak_values[2]={peak_host(0),peak_host(1)};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,plummer_proper_volumes.data(),plummer_proper_volumes.size(),MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,peak_values,2,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  plummer_char_bound=peak_values[0];plummer_psi_max=peak_values[1];
  plummer_core_h_l2=red[3]>0?std::sqrt(red[1]/red[3]):0;
  plummer_core_m_l2=red[3]>0?std::sqrt(red[2]/red[3]):0;
}

void RecordIsotropicHealth(Mesh *pm,const PlummerParticleHealth &H,const PlummerFieldHealth &F) {
  int count=pm->pmb_pack->ppart->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&count,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  const Real error=std::abs(H.mass_total-plummer_M0)/plummer_M0;
  const bool hard=(count!=plummer_ntotal || H.nonfinite!=0 || !std::isfinite(error) || error>1.e-10);
  const bool runaway=plummer_constraint_reference>0 &&
      std::max(plummer_core_h_l2,plummer_core_m_l2)>10*plummer_constraint_reference;
  // Finalization/restart can repeat a history time. It must not add a strike.
  static Real previous=-1;
  if(pm->time>previous) {plummer_constraint_strikes=runaway?plummer_constraint_strikes+1:0;previous=pm->time;}
  const bool collapse=F.alpha_min<.2;
  const bool stopped=collapse || plummer_constraint_strikes>=3;
  if(global_variable::my_rank==0) {
    const bool header=FileIsEmpty(plummer_health_fname);
    std::ofstream out(plummer_health_fname,std::ios::app);
    if(header) out<<"time,cycle,N,N_expected,M0,M0_expected,mass_error,particle_nonfinite,alpha_min,H_core_L2,M_core_L2,constraint_reference,constraint_strikes,healthy,physical_stop,constraints_available,coordinate_characteristic_bound,psi_max\n";
    out<<std::setprecision(17)<<pm->time<<','<<pm->ncycle<<','<<count<<','<<plummer_ntotal<<','
       <<H.mass_total<<','<<plummer_M0<<','<<error<<','<<H.nonfinite<<','<<F.alpha_min<<','
       <<plummer_core_h_l2<<','<<plummer_core_m_l2<<','<<plummer_constraint_reference<<','
       <<plummer_constraint_strikes<<','<<(!hard)<<','<<stopped<<','<<(pm->pmb_pack->pz4c!=nullptr)<<','
       <<plummer_char_bound<<','<<plummer_psi_max<<'\n';
  }
  if(hard) Fatal("ISOTROPIC_HEALTH_FAIL particle count, finite state or rest-mass accounting at t="+std::to_string(pm->time));
  if(stopped) {
    pm->user_stop_requested=true;
    if(global_variable::my_rank==0) std::cout<<"ISOTROPIC_PHYSICAL_STOP t="<<std::setprecision(17)<<pm->time
      <<" alpha_min="<<F.alpha_min<<" constraint_strikes="<<plummer_constraint_strikes<<std::endl;
  }
  IsotropicADMMass(pm);
}
