"""Allocated arithmetic checks against the preserved baseline and native data."""
import csv
import sys
import numpy as np
from cfg import *
sys.path.insert(0,str(HERE.parent/'reconstruction_s06'))
import analyze as A
import reduce_run as R
from analyze_extension import window_metrics

def main():
    require_job()
    with (BASE/'reduced/shells.csv').open()as stream:rows=list(csv.DictReader(stream))
    with (BASE/'reduced/windows.csv').open()as stream:old=list(csv.DictReader(stream))
    t=np.array([float(r['time'])for r in rows])
    q=np.array([[float(r[f'r{j}'])for j in range(10,100,10)]for r in rows])
    maxdiff=0.;checked=0
    for width,family in ((100.,'100M'),(P,'P'),(2*P,'2P')):
        bounds=[(k*width,(k+1)*width)for k in range(int(1000/width))]
        current=window_metrics(t,q,q[0],bounds,family)
        for row in current:
            prior=next(v for v in old if v['family']==family and int(v['quantile'])==row['quantile']
                       and abs(float(v['start'])-row['start'])<1e-8 and abs(float(v['end'])-row['end'])<1e-8)
            for key in ('half_range_fraction','mean_shift_fraction','detrended_half_range_fraction','slope_fraction_per_M'):
                diff=abs(float(prior[key])-row[key]);maxdiff=max(maxdiff,diff);assert diff<1e-12
            # Independently assemble exact boundary trapezoids for the mean.
            j=row['quantile']//10-1;lo,hi=row['start'],row['end']
            tt=[lo]+t[(t>lo)&(t<hi)].tolist()+[hi]
            vv=[float(np.interp(v,t,q[:,j]))for v in tt]
            area=sum((tt[k+1]-tt[k])*(vv[k+1]+vv[k])/2 for k in range(len(tt)-1))
            mean_fraction=(area/(hi-lo)-q[0,j])/q[0,j]
            assert abs(mean_fraction-row['mean_shift_fraction'])<1e-12
            checked+=1
    path=next((BASE/'runs/production/out/pvtk').glob('*.00250.part.vtk'))
    time,cycle,n,data=R.read_pvtk(str(path));mass=data['prtcl_mass']
    pos=np.column_stack([data[a]for a in 'xyz'])
    model=dict(np.load(BASE/'samples/model.npz'));native,_,_,r=A.sample_state(pos,mass,model,'proper')
    saved=min(rows,key=lambda h:abs(float(h['time'])-time));assert abs(float(saved['time'])-time)<1e-8
    differences={k:abs(float(saved[k])-native[k])for k in native}
    assert max(differences.values())<1e-12
    # All archived equal-mass deciles must agree with a separately sorted scalar
    # radius array's adjacent order statistics, allowing finite-CDF rank rounding.
    rr=np.sort(r)
    brackets={}
    for j in range(10,100,10):
        index=int(n*j/100);neighbors=rr[index-2:index+3]
        assert neighbors.min()-1e-10<=native[f'r{j}']<=neighbors.max()+1e-10
        brackets[str(j)]=[float(neighbors.min()),float(neighbors.max())]
    atomic(ROOT/'evidence/ANALYSIS_ARITHMETIC_QA.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],
        source=source(),passed=True,baseline_window_rows=checked,maximum_baseline_window_difference=maxdiff,
        independent_trapezoid_means=True,native_time=time,native_count=n,native_sha256=sha(path),
        native_shell_max_abs_difference=max(differences.values()),independent_order_statistic_brackets=brackets,
        scope='Baseline arithmetic, native schema and unchanged definitions; not new-case evolution QA.'))
    print('ANALYSIS_ARITHMETIC_QA_OK',flush=True)

if __name__=='__main__':main()
