#ifndef PARTICLES_GI_DIAGNOSTIC_MATH_HPP_
#define PARTICLES_GI_DIAGNOSTIC_MATH_HPP_
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/horizon_query.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {
constexpr int GI_NSHELL=129;
KOKKOS_INLINE_FUNCTION int GiShell(Real r) {
  return r < 8 ? static_cast<int>(r*8) :
    (r < 72 ? 64+static_cast<int>(r-8) : 128);
}
KOKKOS_INLINE_FUNCTION int GiComponent(int tag) {
  return tag < 3000000 ? 0 : (tag < 4000000 ? 1 : 2);
}
template<class P, class C>
KOKKOS_INLINE_FUNCTION bool GiInside(P par, C coef, int h, int lm,
                                   Real x, Real y, Real z) {
  if (par(h,IAHVALID) == 0) return false;
  x-=par(h,0); y-=par(h,1); z-=par(h,2);
  Real r=Kokkos::sqrt(x*x+y*y+z*z);
  // Upper bound is from harmonic coefficient norms, not collocation extrema.
  if (r>=par(h,IAHRMAX)) return false;
  if (r==0) return true;
  Real theta=Kokkos::acos(Kokkos::fmax(-1.,Kokkos::fmin(1.,z/r)));
  Real R=AHSurfaceRadius(coef,h,lm,(lm+1)*(lm+1),theta,Kokkos::atan2(y,x));
  return Kokkos::isfinite(R) && r<R;
}

// Trilinear diagnostic interpolation, explicitly separate from the pusher's
// high-order interpolation. Reports validity instead of clipping bad geometry.
struct GiMetric { Real en, ec, alpha; bool valid; };
template<class A, class Z, class S>
KOKKOS_INLINE_FUNCTION GiMetric GiEnergy(A adm, Z z4c, S size, int m, int ng,
                                      Real x,Real y,Real z,Real ux,Real uy,Real uz,
                                      Real mass) {
  GiMetric out{0,0,0,false};
  if (m<0 || m>=static_cast<int>(size.extent(0))) return out;
  auto b=size(m);
  Real q[3]={(x-b.x1min)/b.dx1+ng-.5,(y-b.x2min)/b.dx2+ng-.5,
             (z-b.x3min)/b.dx3+ng-.5};
  int ijk[3]; Real f[3];
  for(int a=0;a<3;++a){ijk[a]=static_cast<int>(Kokkos::floor(q[a])); f[a]=q[a]-ijk[a];}
  if(ijk[0]<0 || ijk[1]<0 || ijk[2]<0 ||
     ijk[0]+1>=static_cast<int>(adm.extent(4)) ||
     ijk[1]+1>=static_cast<int>(adm.extent(3)) ||
     ijk[2]+1>=static_cast<int>(adm.extent(2))) return out;
  Real g[6]={}, gauge[4]={};
  for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i){
    Real w=(i?f[0]:1-f[0])*(j?f[1]:1-f[1])*(k?f[2]:1-f[2]);
    for(int a=0;a<6;++a) g[a]+=w*adm(m,a,ijk[2]+k,ijk[1]+j,ijk[0]+i);
    gauge[0]+=w*z4c(m,z4c::Z4c::I_Z4C_ALPHA,ijk[2]+k,ijk[1]+j,ijk[0]+i);
    gauge[1]+=w*z4c(m,z4c::Z4c::I_Z4C_BETAX,ijk[2]+k,ijk[1]+j,ijk[0]+i);
    gauge[2]+=w*z4c(m,z4c::Z4c::I_Z4C_BETAY,ijk[2]+k,ijk[1]+j,ijk[0]+i);
    gauge[3]+=w*z4c(m,z4c::Z4c::I_Z4C_BETAZ,ijk[2]+k,ijk[1]+j,ijk[0]+i);
  }
  Real det=Primitive::GetDeterminant(g), inv[6];
  if (!(det>0) || !(g[0]>0) || !(g[0]*g[3]-g[1]*g[1]>0) ||
      !Kokkos::isfinite(det) || !(gauge[0]>0)) return out;
  Primitive::InvertMatrix(inv,g,det);
  Real w2=1+inv[0]*ux*ux+inv[3]*uy*uy+inv[5]*uz*uz+
    2*(inv[1]*ux*uy+inv[2]*ux*uz+inv[4]*uy*uz);
  if (!(w2>=1) || !Kokkos::isfinite(w2)) return out;
  out.en=mass*Kokkos::sqrt(w2);
  out.ec=gauge[0]*out.en-mass*(gauge[1]*ux+gauge[2]*uy+gauge[3]*uz);
  out.alpha=gauge[0]; out.valid=Kokkos::isfinite(out.ec); return out;
}
}  // namespace particles
#endif
