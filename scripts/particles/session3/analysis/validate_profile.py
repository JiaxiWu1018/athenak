"""Independent reference: PDF (9.7) at 70 digits, adaptive SciPy quadrature.

Unlike the C++ regularized Abel quadrature, this uses the elementary closed form.
Run only inside Slurm. JSON records every check and refuses numerical promotion.
"""
import csv, json, math, sys
from functools import lru_cache
from pathlib import Path
import mpmath as mp
from scipy.integrate import quad
mp.mp.dps = 70
M, a = 1., 10.
estar = 3*M*a*a/(128*math.pi*(M/2)**5)

def fhat(b):
    if b <= 0: return 0.
    e = 1-mp.mpf(float(b))
    value = ((1733*e**4+1274*e**2+8)/e*mp.sqrt(1-e**2)
             -15*e*(21*e**4+140*e**2+40)*mp.acosh(1/e))/(32*mp.pi**2)
    return float(value)

def geometry(r):
    u = M/(2*math.hypot(r,a)); p=1+u
    return p,(1-u)/p,2*u/p

@lru_cache(maxsize=30000)
def moments(r):
    p,A,b = geometry(r); qm=math.sqrt(b*(2-b))/A
    def term(s, kind):
        q=qm*s; W=math.sqrt(1+q*q)
        f=s*s*fhat(b-A*q*q/(W+1))
        return f * (1 if kind==0 else W if kind==1 else q*q/(3*W))
    return tuple(4*math.pi*estar*qm**3*quad(lambda s:term(s,k),0,1,
                 epsabs=1.e-24,epsrel=2.e-11,limit=100)[0] for k in range(3))

def dmass(r):
    p,_,_=geometry(r)
    return 4*math.pi*r*r*p**6*moments(float(r))[0]

@lru_cache(maxsize=500)
def mass(r):
    # A different domain transform from the C++ log-grid and tan-grid integrals.
    top=math.log1p(r/a)
    return quad(lambda x:dmass(a*math.expm1(x))*a*math.exp(x),0,top,
                epsabs=2.e-11,epsrel=2.e-11,limit=150)[0]

def qcdf(r,q):
    _,A,b=geometry(r);qm=math.sqrt(b*(2-b))/A
    def term(s):
        qs=qm*s;W=math.sqrt(1+qs*qs)
        return s*s*fhat(b-A*qs*qs/(W+1))
    return quad(term,0,q/qm,epsabs=1.e-24,epsrel=2.e-11)[0]/quad(
           term,0,1,epsabs=1.e-24,epsrel=2.e-11)[0]

def main():
    source,target=map(Path,sys.argv[1:]); checks=[]; norms={}
    def check(name,actual,expected,relative=True):
        error=abs(actual-expected)/(max(abs(expected),1.e-300) if relative else 1)
        checks.append(dict(name=name,actual=actual,reference=expected,error=error,
                           relative=relative,passed=bool(error<=1.e-8)))
    with source.open() as handle: rows=list(csv.reader(handle))
    for row in rows:
        kind=row[0]; v=list(map(float,row[1:]));cut=v[0]
        if kind=='norm':
            norms[cut]=v[1:]
            check(f'M0_{cut}',v[1],mass(cut))
            # Infinite mass: integrate s=r/(r+a), with endpoint values avoided.
            mi=quad(lambda s:dmass(a*s/(1-s))*a/(1-s)**2,0,1,
                    epsabs=2.e-11,epsrel=2.e-11,limit=100)[0]
            check(f'M0_inf_{cut}',v[2],mi)
            check(f'median_{cut}',mass(v[3]),mi/2)
            p,A,b=geometry(v[3]);r=v[3];u=p-1
            dA=M*r/(math.hypot(r,a)**3*p*p)
            dC=2*r*p**4-4*r**3*p**3*M/(2*math.hypot(r,a)**3)
            pref=2*math.pi/math.sqrt(2*A*dA/dC)
            check(f'period_{cut}',v[4],pref)
        elif kind=='F':check(f'F_{cut}_{v[1]}',v[2],fhat(v[1]))
        elif kind=='moment':
            r=v[1];p,A,_=geometry(r);rho,eps,pr=moments(r)
            u=p-1;analytic=3*M*a*a/(4*math.pi*math.hypot(r,a)**5*p**5)
            for label,got,ref in zip(['psi','alpha','rho0','eps','pressure'],v[2:],
                                    [p,A,rho,analytic,analytic*u/(3*(1-u))]):
                check(f'{label}_{cut}_{r}',got,ref)
            check(f'python_energy_identity_{r}',eps,analytic)
            check(f'python_pressure_identity_{r}',pr,analytic*u/(3*(1-u)))
        elif kind=='radius':check(f'radius_cdf_{cut}_{v[1]}',mass(v[2])/mass(cut),v[1])
        elif kind=='q':check(f'q_cdf_{cut}_{v[1]}_{v[2]}',qcdf(v[1],v[3]),v[2])
    for cut,(m0,mi,rh,pr) in norms.items():
        norms[cut]=dict(M0=m0,M0_inf=mi,rest_mass_median=rh,P_ref=pr,
                       omitted_rest_mass=mi-m0,omitted_fraction=(mi-m0)/mi)
    report=dict(reference='PDF Eq 9.7, mpmath 70 digits, adaptive quadrature',
                tolerance=1.e-8,passed=all(c['passed'] for c in checks),
                max_error=max(c['error'] for c in checks),normalizations=norms,checks=checks)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
    for c in checks:
        if not c['passed']:print('FAIL',c['name'],c['error'])
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
