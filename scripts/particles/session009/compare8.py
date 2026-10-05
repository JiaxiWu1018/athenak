"""Compare retained reduced Session008 data over the actual common time interval."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from runtime import atomic,digest
BASE=Path('/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331')
def compare(out):
 old=BASE/'orbit.csv';new=out/'orbit.csv'
 if not old.exists() or not new.exists():
  atomic(out/'session8_comparison.json',dict(available=False,reason='Retained reduced baseline or current tracker table unavailable.'));return
 a=np.loadtxt(old,delimiter=',',ndmin=2);b=np.loadtxt(new,delimiter=',',ndmin=2)
 end=min(a[-1,0],b[-1,0]);a=a[a[:,0]<=end];b=b[b[:,0]<=end]
 fig,ax=plt.subplots(figsize=(8,4));ax.plot(a[:,0],a[:,1],label='Session008');ax.plot(b[:,0],b[:,1],label='Session009')
 ax.set(xlabel='t/M_ref',ylabel='live tracker coordinate separation/M_ref',title='Shared time range; changed propagation mesh/domain');ax.legend();fig.tight_layout();fig.savefig(out/'session8_separation_comparison.png',dpi=170);plt.close(fig)
 atomic(out/'session8_comparison.json',dict(available=True,last_common_time=float(end),inputs={str(old):digest(old),str(new):digest(new)},limitations='Same boost/physical model/M_ref/gauge; different domain/mesh/GPU decomposition. Coordinate diagnostics, not a clean convergence test. Session008 reaches onlyt12 and has no merger waveform at distant radii.'))
