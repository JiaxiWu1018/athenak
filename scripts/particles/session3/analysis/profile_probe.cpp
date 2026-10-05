#include <iomanip>
#include <iostream>
#include "plummer_isotropic_profile.hpp"
int main() {
  std::cout << std::setprecision(17);
  for (double cut : {1000.,2000.}) {
    plummer::IsotropicProfile p(1.,10.,cut);
    std::cout << "norm," << cut << ',' << p.M0 << ',' << p.M0_inf << ','
              << p.rhalf << ',' << p.P_ref << '\n';
    for(double r : {0.,.1,1.,5.,10.,13.,20.,50.,100.,500.,999.}) {
      std::cout << "moment," << cut << ',' << r << ',' << p.psi(r) << ','
                << p.alpha(r) << ',' << p.rho0(r) << ',' << p.Moment(r,1)
                << ',' << p.Moment(r,2) << '\n';
      for(double z : {.0001,.01,.1,.5,.9,.99,.9999})
        std::cout << "q," << cut << ',' << r << ',' << z << ',' << p.SampleQ(r,z) << '\n';
    }
    for(double z : {.000001,.0001,.01,.1,.5,.9,.99,.9999,.999999})
      std::cout << "radius," << cut << ',' << z << ',' << p.InvertF0(z) << '\n';
    for(double b : {1.e-12,1.e-8,1.e-5,.001,.01,.025,.05,.075,.09,p.bmax})
      std::cout << "F," << cut << ',' << b << ',' << p.FhatBinding(b) << '\n';
  }
}
