#ifndef PGEN_PARTICLES_PLUMMER_ISOTROPIC_PROFILE_HPP_
#define PGEN_PARTICLES_PLUMMER_ISOTROPIC_PROFILE_HPP_
// Part I of Relativistic_Plummer_Step_by_Step.pdf. G=c=1. Radius is ISOTROPIC.
// Fhat = mu F/epsilon_star. A spatial sampling cut does not truncate the metric
// or the energy distribution, and is not an orbital or particle-removal boundary.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "plummer_profile.hpp"

namespace plummer {
class IsotropicProfile {
 public:
  double M, a, rcut, eta, bmax, epsilon_star, M0, M0_inf, rhalf, P_ref;
  static constexpr double eta_max = 0.61947210322903373047;
  static constexpr int NF = 4096, NA = 1024, NQ = 1024;
  std::vector<double> scaled_f, radius, mass_cdf, radial_pdf, qcdf, qpdf;

  IsotropicProfile(double mass, double scale, double cut, int panels = 8192)
      : M(mass), a(scale), rcut(cut), eta(mass/scale) {
    if (!(M > 0 && a > 0 && rcut > 0 && eta <= eta_max && panels >= 128))
      throw std::invalid_argument("isotropic Plummer: require M,a,rcut>0, eta<=0.619472103229, panels>=128");
    bmax = 1.0-alpha(0.0);
    const double k = 0.5*M;
    epsilon_star = 3*M*a*a/(128*M_PI*std::pow(k, 5));
    scaled_f.resize(NF+1);
    for (int i=0; i<=NF; ++i) scaled_f[i] = ScaledFExact(bmax*i/NF);
    radius.resize(panels+1); mass_cdf.resize(panels+1); radial_pdf.resize(panels+1);
    const double top = std::log1p(rcut/a);
    for (int i=0; i<=panels; ++i) {
      radius[i] = a*std::expm1(top*i/panels);
      radial_pdf[i] = dM0dr(radius[i]);
      if (i) mass_cdf[i] = mass_cdf[i-1] + GaussLegendre16(
          [&](double r){return dM0dr(r);}, radius[i-1], radius[i]);
    }
    M0 = mass_cdf.back();
    // r=a tan(theta) removes the infinite-domain endpoint and its tail singularity.
    M0_inf = 0;
    for (int i=0; i<512; ++i) M0_inf += GaussLegendre16([&](double t) {
      const double co=std::cos(t); return dM0dr(a*std::tan(t))*a/(co*co);
    }, (M_PI/2)*i/512, (M_PI/2)*(i+1)/512);
    if (!(M0 > 0 && M0 >= M0_inf/2))
      throw std::invalid_argument("sampling cut must enclose the untruncated rest-mass median");
    rhalf = InvertMass(M0_inf/2);
    const double p=psi(rhalf), W=1-2*u(rhalf)*rhalf*rhalf/
        ((1+u(rhalf))*(rhalf*rhalf+a*a));
    const double Ap=M*rhalf/(std::pow(rhalf*rhalf+a*a, 1.5)*p*p);
    const double omega2=alpha(rhalf)*Ap/(std::pow(p,4)*rhalf*W);
    P_ref=2*M_PI/std::sqrt(omega2);
    BuildMomentumCDF();
  }
  double u(double r) const {return M/(2*std::hypot(r,a));}
  double psi(double r) const {return 1+u(r);}
  double alpha(double r) const {const double z=u(r);return (1-z)/(1+z);}
  double binding(double r) const {const double z=u(r);return 2*z/(1+z);}
  double eps(double r) const {return 3*M*a*a/(4*M_PI*std::pow(std::hypot(r,a),5)*std::pow(psi(r),5));}
  double pressure(double r) const {return eps(r)*u(r)/(3*(1-u(r)));}
  double areal(double r) const {return r*psi(r)*psi(r);}
  double gravmass(double r) const {
    const double W=1-2*u(r)*r*r/((1+u(r))*(r*r+a*a));
    return areal(r)*(1-W*W)/2;
  }
  // Evaluate Fhat/(1-e)^(7/2). Algebraic cancellation of the endpoint powers
  // keeps this well-conditioned even when e rounds close to one (PDF Eq. 9.5).
  static double ScaledFExact(double b) {
    const double e=1-b, k=b*(2-b);
    const auto f=[&](double t) {
      const double z=std::sqrt(e*e+k*t*t), z2=z*z;
      const double poly=105*z*z2-69*z2+3*z+1;
      return std::pow(1-t*t,3)*poly/(z*z2*std::pow(1+z,3));
    };
    return e*std::pow(2-b,3.5)/(4*M_PI*M_PI)*
        (GaussLegendre16(f,0,0.5)+GaussLegendre16(f,0.5,1));
  }
  double FhatBinding(double b) const {
    if (b<=0) return 0;
    if (b>bmax*(1+1e-12)) throw std::domain_error("inaccessible particle energy");
    const double x=std::min(b/bmax*NF, static_cast<double>(NF));
    const int i=std::min(static_cast<int>(x),NF-1);
    const double t=x-i, h=bmax/NF;
    const double d0=(i==0) ? (-3*scaled_f[0]+4*scaled_f[1]-scaled_f[2])/(2*h)
                          : (scaled_f[i+1]-scaled_f[i-1])/(2*h);
    const double d1=(i+1==NF) ? (3*scaled_f[NF]-4*scaled_f[NF-1]+scaled_f[NF-2])/(2*h)
                             : (scaled_f[i+2]-scaled_f[i])/(2*h);
    const double v=Hermite(scaled_f[i],scaled_f[i+1],h*d0,h*d1,t);
    if (v < -1e-12) throw std::domain_error("negative distribution on accessible energies");
    return std::max(0.0,v)*std::pow(b,3.5);
  }
  // Integral mu F d^3q, and the energy and isotropic pressure moments.
  double Moment(double r, int which) const {
    const double b=binding(r), A=1-b, qm=std::sqrt(b*(2-b))/A;
    const auto f=[&](double s) {
      const double q=qm*s, gamma=std::sqrt(1+q*q);
      const double be=b-A*q*q/(gamma+1);
      double v=s*s*FhatBinding(be);
      if (which==1) v*=gamma;
      if (which==2) v*=q*q/(3*gamma);
      return v;
    };
    return 4*M_PI*epsilon_star*std::pow(qm,3)*
        (GaussLegendre16(f,0,0.5)+GaussLegendre16(f,0.5,1));
  }
  double rho0(double r) const {return Moment(r,0);}
  double dM0dr(double r) const {return 4*M_PI*r*r*std::pow(psi(r),6)*rho0(r);}
  double InvertF0(double z) const {return InvertMass(std::clamp(z,0.0,1.0)*M0);}
  double F0(double r) const {
    if (r<=0) return 0;
    if (r>=rcut) return 1;
    const int i=static_cast<int>(std::upper_bound(radius.begin(),radius.end(),r)-radius.begin())-1;
    const double h=radius[i+1]-radius[i], t=(r-radius[i])/h;
    return Hermite(mass_cdf[i],mass_cdf[i+1],h*radial_pdf[i],h*radial_pdf[i+1],t)/M0;
  }
  double SampleQ(double r,double quantile) const {
    const double b=binding(r), A=1-b, row=std::clamp(b/bmax*NA,0.0,static_cast<double>(NA));
    const int j=std::min(static_cast<int>(row),NA-1);
    const double s0=InvertQRow(j,quantile), s1=InvertQRow(j+1,quantile);
    return std::sqrt(b*(2-b))/A*(s0+(row-j)*(s1-s0));
  }
  static double Hermite(double y0,double y1,double d0,double d1,double t) {
    return (2*t*t*t-3*t*t+1)*y0+(t*t*t-2*t*t+t)*d0+
        (-2*t*t*t+3*t*t)*y1+(t*t*t-t*t)*d1;
  }
 private:
  double InvertMass(double target) const {
    if (target<=0) return 0;
    if (target>=M0) return rcut;
    const int i=static_cast<int>(std::upper_bound(mass_cdf.begin(),mass_cdf.end(),target)-mass_cdf.begin())-1;
    const double h=radius[i+1]-radius[i];
    double lo=0,hi=1,t=(target-mass_cdf[i])/(mass_cdf[i+1]-mass_cdf[i]);
    for (int it=0;it<40;++it) {
      const double f=Hermite(mass_cdf[i],mass_cdf[i+1],h*radial_pdf[i],h*radial_pdf[i+1],t)-target;
      if (std::abs(f)<2e-16*M0) break;
      if (f>0) hi=t; else lo=t;
      const double d=(6*t*t-6*t)*mass_cdf[i]+(3*t*t-4*t+1)*h*radial_pdf[i]+
          (-6*t*t+6*t)*mass_cdf[i+1]+(3*t*t-2*t)*h*radial_pdf[i+1];
      double next=t-f/d;
      if (!(next>lo && next<hi)) next=(lo+hi)/2;
      t=next;
    }
    return radius[i]+h*t;
  }
  void BuildMomentumCDF() {
    qcdf.resize((NA+1)*(NQ+1));qpdf.resize(qcdf.size());
    for (int j=0;j<=NA;++j) {
      const double b=bmax*j/NA,A=1-b,qm2=b*(2-b)/(A*A);
      // Remove the common b^(7/2) normalization before the weak-field limit.
      const auto density=[&](double s) {
        if (b==0) return s*s*std::pow(std::max(0.0,1-s*s),3.5);
        const double q2=qm2*s*s,be=b-A*q2/(std::sqrt(1+q2)+1);
        return s*s*FhatBinding(be)/std::pow(b,3.5);
      };
      const int off=j*(NQ+1);
      for (int i=0;i<=NQ;++i) {
        qpdf[off+i]=density(static_cast<double>(i)/NQ);
        if (i) qcdf[off+i]=qcdf[off+i-1]+GaussLegendre16(density,
            static_cast<double>(i-1)/NQ,static_cast<double>(i)/NQ);
      }
      const double norm=qcdf[off+NQ];
      for (int i=0;i<=NQ;++i) {qcdf[off+i]/=norm;qpdf[off+i]/=norm;}
    }
  }
  double InvertQRow(int row,double z) const {
    if (z<=0) return 0;if (z>=1) return 1;
    const int off=row*(NQ+1);
    const auto beg=qcdf.begin()+off;
    const int i=static_cast<int>(std::upper_bound(beg,beg+NQ+1,z)-beg)-1;
    const double y0=qcdf[off+i],y1=qcdf[off+i+1],d0=qpdf[off+i]/NQ,d1=qpdf[off+i+1]/NQ;
    double lo=0,hi=1,t=(z-y0)/(y1-y0);
    for (int it=0;it<24;++it) {
      const double f=Hermite(y0,y1,d0,d1,t)-z;
      if (std::abs(f)<2e-15) break;
      if (f>0) hi=t;else lo=t;
      const double d=(6*t*t-6*t)*y0+(3*t*t-4*t+1)*d0+(-6*t*t+6*t)*y1+(3*t*t-2*t)*d1;
      double next=t-f/d;if (!(next>lo && next<hi)) next=(lo+hi)/2;t=next;
    }
    return (i+t)/NQ;
  }
};
} // namespace plummer
#endif
