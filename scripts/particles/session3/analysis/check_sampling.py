"""Validate actual initialized particle payload and all metric-aware shell moments."""
import csv,json,sys
from pathlib import Path
import numpy as np
from pvtk_reader import read_pvtk
from health import require_window,read_rows

def main(run,reference,target):
    run=Path(run);window=require_window(run)
    payload=read_pvtk(sorted((run/'out/pvtk').glob('*.part.vtk'))[0],
                      {'pos','ptag','prtcl_vel','prtcl_mass'})
    initial_files=list((run/'out').glob('*.isotropic_initial.csv'))+list(run.glob('*.isotropic_initial.csv'))
    initial=read_rows(initial_files[0])[0]
    n=int(initial['N']);cut=float(initial['sampling_radius']);mu=float(initial['mu'])
    tag=payload['ptag'].astype(np.int64);order=np.argsort(tag)
    exact_tags=np.array_equal(tag[order],np.arange(n))
    pos=payload['pos'][order].astype(np.float64);u=payload['prtcl_vel'][order].astype(np.float64)
    pairpos=bool(np.array_equal(pos[::2],pos[1::2]));pairmomentum=bool(np.array_equal(u[::2],-u[1::2]))
    r=np.linalg.norm(pos,axis=1);psi=1+1/(2*np.sqrt(r*r+100));alpha=(2-psi)/psi
    q2=np.sum(u*u,axis=1)/psi**4;W=np.sqrt(1+q2)
    radial=np.sum(pos*u,axis=1)/np.maximum(r,1.e-100)/psi**2
    energy=np.sum(mu*W);pr=np.sum(mu*radial**2/W)
    tangential=np.sum(mu*(q2-radial**2)/W)
    rows=read_rows(next((run/'out').glob('*.plummer_physical.csv')))
    rows=list({int(row['bin']):row for row in rows if float(row['time'])==0}.values())
    theory={}
    with Path(reference).open() as f:
        for row in csv.reader(f):
            if row[0]=='shell' and float(row[1])==cut:theory[int(row[2])]=list(map(float,row[5:]))
    checks=[]
    for row in rows:
        b=int(row['bin']);m,e,p,v=theory[b];N=float(row['N'])
        # Stratified radius makes count uncertainty an edge-quantization effect.
        checks.append(dict(bin=b,quantity='M0',actual=float(row['M0']),expected=m,
                           uncertainty=4*mu,passed=abs(float(row['M0'])-m)<=4*mu+1.e-8*m))
        for name,expected in [('E',e),('Sr',p),('Stheta',p),('Sphi',p)]:
            actual=float(row[name]);squared=float(row[name+'_square'])
            # Pair members are perfectly correlated in even moments at initialization.
            variance=max(0,squared-actual*actual/max(N,1))
            sem=np.sqrt(2*variance)
            lower_iso=float(row['rlo'])/1.05**2
            psi_lo=1+1/(2*np.sqrt(lower_iso**2+100))
            alpha_lo=(2-psi_lo)/psi_lo
            qmax2=alpha_lo**-2-1
            edge=4*mu*(np.sqrt(1+qmax2) if name=='E' else qmax2)
            checks.append(dict(bin=b,quantity=name,actual=actual,expected=expected,
              uncertainty=float(sem),zscore=float((actual-expected)/max(sem,1.e-300)),
              edge_quantization_bound=float(edge),
              passed=bool(abs(actual-expected)<=5*sem+edge+1.e-8*expected)))
    E=sum(float(row['E']) for row in rows);Sr=sum(float(row['Sr']) for row in rows)
    St=sum(float(row['Stheta'])+float(row['Sphi']) for row in rows)
    agreements={'energy':abs(E-energy)/max(energy,1.e-300),
                'radial_stress':abs(Sr-pr)/max(pr,1.e-300),
                'tangential_stress_sum':abs(St-tangential)/max(tangential,1.e-300)}
    report=dict(passed=bool(exact_tags and pairpos and pairmomentum and
      all(c['passed'] for c in checks) and max(agreements.values())<1.e-5),
      N=n,exact_contiguous_tags=exact_tags,co_located_pairs=pairpos,
      exact_opposite_stored_momenta=pairmomentum,shell_checks=checks,
      dump_vs_in_code_agreement=agreements,dump_precision='float32; 1e-5 comparison tolerance',
      statistical_policy='5 sigma familywise screen, even moments use N/2 correlated pairs; radial-count bound 4 mu',
      window=window)
    Path(target).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='shell_checks'},indent=2))
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
