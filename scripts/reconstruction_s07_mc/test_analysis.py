"""Allocated independent synthetic checks for offsets, drift, cadence and zero ratios."""
import math
import numpy as np
from cfg import *
from compare_cases import residual,window,ratio

def main():
    require_job();a=np.array([0.,1.,3.,5.]);b=np.array([0.,2.,4.,5.])
    t,d,s=residual(a,2*a,b,2*b+3,0,5)
    assert np.allclose(d,3)and abs(s['RMS']-3)<1e-12
    _,d,s=residual(a,2*a,b,2*b+3,0,5,changes=True)
    assert np.max(abs(d))<1e-12 and s['RMS']<1e-12
    _,d,s=residual(a,np.zeros(4),b,2*b,0,5)
    independent=math.sqrt(sum((t1-t0)*(x*x+y*y)/2 for t0,t1,x,y in zip(t[:-1],t[1:],2*t[:-1],2*t[1:]))/5)
    assert abs(s['RMS']-independent)<1e-12
    w=window(a,2*a+3,3,0,5,1)
    assert abs(w['slope_per_M']-2)<1e-12 and w['detrended_half_range']<1e-12
    assert ratio(0,0)is None and ratio(1,0)is None and ratio(1,2)==.5
    from analyze_extension import window_metrics
    q=np.column_stack([10+j+2*a for j in range(9)])
    old=window_metrics(a,q,q[0],[(0,5)],'synthetic')
    for j,row in enumerate(old):
        v=window(a,q[:,j],q[0,j],0,5,q[0,j]);assert abs(v['mean_shift']-row['mean_shift_fraction'])<1e-12
    atomic(ROOT/'evidence/ANALYSIS_TESTS_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],passed=True,
        tests=['constant offset','initial-offset removal','different native cadences','independent RMS quadrature','linear detrend','zero ratio','inherited window equivalence']))
    print('ANALYSIS_TESTS_OK',flush=True)
if __name__=='__main__':main()
