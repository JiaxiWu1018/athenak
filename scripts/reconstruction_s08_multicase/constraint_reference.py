"""Independent angular quadrature and radial Green-integral validation."""
import numpy as np
from common import *
from extension_resources import admission
from st_migration_lib import cluster_profile

def cumulative(y,x):return np.r_[0,np.cumsum((y[1:]+y[:-1])*np.diff(x)/2)]

def verify(c):
    p=cluster_profile(c['yc'],dx=1e-4);yc=c['yc']
    x=p['x'];y=p['y'];r0=p['riso_over_m'];zeta=(1-yc)/3
    compact=np.ones_like(x);compact[1:]=1-2*zeta*p['v'][1:]/x[1:]
    rho0=np.sqrt(yc)*(y/yc)**-1*np.sqrt(np.maximum(1/y-1,0)/(1/yc-1))
    pdf=x*x*rho0/np.sqrt(compact)/p['v0s']
    normalization=float(cumulative(pdf,x)[-1]);assert abs(normalization-1)<2e-6
    D=np.zeros_like(x);D[1:]=p['M0_over_M']*rho0[1:]*x[1:]**3/(4*np.pi*p['v0s']*r0[1:]**3)
    mu,weights=np.polynomial.legendre.leggauss(192);weights/=2
    results=[]
    for U in [c['U']]:
        path=EXT/'evidence/constraints'/f"{c['key']}.txt";tab=np.loadtxt(path);r=tab[:,0];psi=np.sqrt(tab[:,1]);a=tab[:,2]
        inside=r<=r0[-1];d=np.zeros_like(r);yy=np.ones_like(r);frac=np.ones_like(r)
        d[inside]=np.interp(r[inside],r0,D);yy[inside]=np.interp(r[inside],r0,y);frac[inside]=np.interp(r[inside],r0,x/x[-1])
        psi0=1+.5/np.maximum(r,1e-30);psi0[inside]=np.sqrt(np.interp(r[inside],r0,p['conformal_a']))
        q=U*(3*frac-frac**3)/2;speed=np.sqrt(np.maximum(1-yy,0))
        w=1/np.sqrt(yy[:,None]-q[:,None]**2-2*speed[:,None]*q[:,None]*mu)
        meanw=w@weights;meanur=((speed[:,None]*mu+q[:,None])*w)@weights
        # Outside the matter the angular moments are irrelevant: D=0.
        F=6*np.pi*d*psi**2*meanur
        moment=cumulative(r**3*F,r);expected_a=np.zeros_like(r);expected_a[1:]=moment[1:]/r[1:]**3
        a_error=float(np.max(abs(expected_a-a))/np.max(abs(a)))
        dq=2*np.pi*d*(meanw/psi-1/np.sqrt(yy)/psi0)+a*a/(3*psi**7)
        inner=cumulative(r*r*dq,r);outer=-cumulative((r*dq)[::-1],r[::-1])[::-1]
        expected_delta=outer+moment[-1]**2/(12*r[-1]**4);expected_delta[1:]+=inner[1:]/r[1:]
        delta=psi-psi0
        psi_error=float(np.max(abs(expected_delta-delta))/np.max(abs(delta)))
        assert a_error<.002 and psi_error<.002,(U,a_error,psi_error)
        delta_adm=float(2*inner[-1]+2*moment[-1]**2/(9*r[-1]**3))
        results.append(dict(case=c['key'],yc=yc,proper_pdf_integral=normalization,U=U,source_profile=receipt(path),momentum_relative_max_error=a_error,hamiltonian_green_relative_max_error=psi_error,independent_delta_ADM_mass=delta_adm,angular_rule='192-point Gauss-Legendre, independent of native64-point midpoint'))
    return results[0]

def main():
    job();admission();out=EXT/'evidence/constraints';out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[2]/'src/pgen/particles/relativistic_cluster.cpp'
    # When the workflow is frozen under a job directory, use the pinned checkout.
    if not source.exists():source=LOCAL/'src/athenak/src/pgen/particles/relativistic_cluster.cpp'
    text=source.read_text()
    parts=text[text.index('struct ClusterState {'):text.index('std::size_t CDFIndex(')]
    parts+=text[text.index('struct CTTMomentumProfile {'):text.index('struct BiasStats {')]
    header='#include <algorithm>\n#include <cmath>\n#include <cstdlib>\n#include <iostream>\n#include <iomanip>\n#include <limits>\n#include <vector>\nusing Real=double;\n'
    native=r"""
int main(int argc,char**argv){
  double yc=std::atof(argv[1]),U=std::atof(argv[2]);auto p=ConstructProfile(yc,1e-4);
  auto c=ConstructIncrementalConstraintProfile(p,1.,U,true);
  std::cerr<<"yc="<<yc<<" U="<<U<<" iterations="<<c.iterations<<" rel_update="<<c.final_relative_update<<" delta_adm="<<c.delta_adm_mass<<"\n";
  std::cout<<std::setprecision(17);
  for(std::size_t i=0;i<c.radius.size();++i)std::cout<<c.radius[i]<<" "<<c.conformal_a[i]<<" "<<c.a[i]<<"\n";
}
"""
    f=out/'reference.cpp';f.write_text(header+parts+native)
    cmd(['g++','-O3','-std=c++17',f,'-o',out/'reference'])
    results=[]
    for raw in ALL_CASES:
        c=case_config(raw['id'])
        if not c['U']:continue
        path=out/f"{c['key']}.txt";assert not path.exists(),'Do not overwrite a reference'
        with path.open('w') as data,(out/f"{c['key']}.log").open('w') as log:
            cmd([out/'reference',str(c['yc']),str(c['U'])],stdout=data,stderr=log)
        results.append(verify(c))
    record=dict(utc=now(),job=job(),results=results,relative_tolerance=.002,source=receipt(source),extracted_cpp=receipt(f),executable=receipt(out/'reference'),scope='Independent continuum initialization reference; not discrete evolution or convergence')
    atomic(EXT/'evidence/INDEPENDENT_CONSTRAINT_REFERENCE_OK.json',record)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
