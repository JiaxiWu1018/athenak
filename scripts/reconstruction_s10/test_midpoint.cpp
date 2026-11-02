#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#define KOKKOS_INLINE_FUNCTION inline
using Real = double;
#include "../../src/particles/rk_midpoint.hpp"

// Exact coupled nonlinear solution U=exp(t), Z=exp(-t).
// F(U,Z)=U+UZ-1; G(U,Z)=-Z+UZ-1. Neither component can be integrated
// independently by replacing the other component with the exact reference.
Real F(Real U, Real Z) {return U+U*Z-1.0;}
Real G(Real U, Real Z) {return -Z+U*Z-1.0;}
struct Update {
  Real geometry, zn, h;
  bool operator()(const Real *x, const Real *u, Real *xx, Real *uu, bool euler) const {
    xx[0]=zn+h*G(geometry, euler ? zn : .5*(zn+x[0]));
    for (int a=0; a<3; ++a) {uu[a]=0; if(a>0) xx[a]=0;}
    return std::isfinite(xx[0]);
  }
};
int main() {
  Real old_u=0,old_z=0;
  std::cout<<std::setprecision(16);
  for(int n : {20,40,80,160,320}) {
    Real U=1,Z=1,h=1.0/n;
    for(int k=0;k<n;++k) {
      Real pred=Z+h*G(U,Z), A=U+h*F(U,Z);
      Real B=.75*U+.25*(A+h*F(A,pred));
      Real x0[3]={Z,0,0},u0[3]={0,0,0},dx[3]={1,1,1};
      Real x[3]={pred,0,0},u[3]={0,0,0},res;int it;
      if(!particles::RKMidpointSolve(Update{B,Z,h},x0,u0,dx,x,u,1.e-14,100,it,res))
        return 1;
      Real mid=.5*(Z+x[0]);
      U=U/3.0+2.0/3.0*(B+h*F(B,mid));Z=x[0];
    }
    Real eu=fabs(U-std::exp(1.0)),ez=fabs(Z-std::exp(-1.0));
    Real pu=old_u>0?std::log2(old_u/eu):0,pz=old_z>0?std::log2(old_z/ez):0;
    std::cout<<n<<" "<<eu<<" "<<ez<<" "<<pu<<" "<<pz<<"\n";
    if(n>=80 && (!(pu>1.8&&pu<2.2)||!(pz>1.8&&pz<2.2)))return 2;
    old_u=eu;old_z=ez;
  }
  Real x0[3]={1,0,0},u0[3]={0,0,0},dx[3]={1,1,1};
  Real x[3]={1,0,0},u[3]={0,0,0},res;int it;
  if(particles::RKMidpointSolve(Update{3,1,1},x0,u0,dx,x,u,1.e-14,1,it,res))return 3;
  x[0]=NAN;
  if(particles::RKMidpointSolve(Update{1,1,.1},x0,u0,dx,x,u,1.e-14,10,it,res))return 4;
  std::cout<<"PASS coupled order, failed-solve rejection, nonfinite rejection\n";
}
