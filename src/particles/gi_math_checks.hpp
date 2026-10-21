#ifndef PARTICLES_GI_MATH_CHECKS_HPP_
#define PARTICLES_GI_MATH_CHECKS_HPP_
#include <stdexcept>
#include "gi_diagnostic_math.hpp"
namespace particles {
// Independent analytic GPU fixtures. No simulation data is accessed or changed.
inline void GiMathChecks() {
  DvceArray2D<Real> par("test_surface",1,NAH_PAR),coef("test_coeff",1,3);
  auto hp=Kokkos::create_mirror(par),hc=Kokkos::create_mirror(coef);
  Kokkos::deep_copy(hp,0.);Kokkos::deep_copy(hc,0.);
  hp(0,0)=.25;hp(0,IAHVALID)=1;hp(0,IAHRMAX)=1.00000000001;
  hc(0,0)=std::sqrt(4*M_PI);
  Kokkos::deep_copy(par,hp);Kokkos::deep_copy(coef,hc);
  DvceArray5D<Real> adm("test_flat_adm",1,6,4,4,4),zu("test_flat_gauge",1,z4c::Z4c::nz4c,4,4,4);
  Kokkos::deep_copy(adm,0.);Kokkos::deep_copy(zu,0.);
  DvceArray1D<RegionSize> size("test_flat_size",1);
  auto hs=Kokkos::create_mirror(size);
  hs(0).x1min=hs(0).x2min=hs(0).x3min=-1;
  hs(0).x1max=hs(0).x2max=hs(0).x3max=1;
  hs(0).dx1=hs(0).dx2=hs(0).dx3=1;Kokkos::deep_copy(size,hs);
  Kokkos::parallel_for("test_flat_metric",Kokkos::RangePolicy<DevExeSpace>(0,64),
    KOKKOS_LAMBDA(int p){int i=p%4,j=(p/4)%4,k=p/16;
      adm(0,0,k,j,i)=adm(0,3,k,j,i)=adm(0,5,k,j,i)=1;
      zu(0,z4c::Z4c::I_Z4C_ALPHA,k,j,i)=1;});
  int failures=0;
  Kokkos::parallel_reduce("test_GI_math",Kokkos::RangePolicy<DevExeSpace>(0,1),
    KOKKOS_LAMBDA(int,int &bad){
      if(!GiInside(par,coef,0,0,.25,0,0))++bad;
      if(!GiInside(par,coef,0,0,1.249,0,0))++bad;
      if(GiInside(par,coef,0,0,1.251,0,0))++bad;
      if(GiInside(par,coef,0,0,.25,0,1.001))++bad;
      auto e=GiEnergy(adm,zu,size,0,1,0,0,0,3,4,0,2);
      if(!e.valid || Kokkos::fabs(e.en-2*Kokkos::sqrt(26.))>1.e-12 || e.ec!=e.en)++bad;
      if(GiEnergy(adm,zu,size,0,1,100,0,0,0,0,0,1).valid)++bad;
      if(GiShell(0)!=0 || GiShell(7.99)!=63 || GiShell(8)!=64 || GiShell(71.99)!=127 || GiShell(72)!=128)++bad;
      if(GiComponent(2999999)!=0 || GiComponent(3000000)!=1 || GiComponent(3999999)!=1 || GiComponent(4000000)!=2)++bad;
    },failures);
  if(failures)throw std::runtime_error("GI analytic diagnostic math failed");
}
}
#endif
