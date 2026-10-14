"""Allocated mathematical checks of continuum quadrature and measurement semantics."""
import time
import numpy as np
from common import ROOT, atomic, require_job
from generate import model, cells, cell_positions
from reference import cloud, coordinates
from analyze import angular_residual, window_measurements
from plot_report import aligned, ratio

def main():
    require_job();(ROOT/'samples').mkdir(parents=True,exist_ok=True)
    t=np.r_[0,np.arange(2,302,2)+.017]
    q0=np.arange(1,10,dtype=float)
    q=q0[None,:]*(1+.001*t[:,None])
    windows=window_measurements(t,q,300)
    for row in windows:
        if not np.isclose(row['half_range_fraction'],.05,atol=1e-14):
            raise ValueError('Common-window amplitude is biased by cadence skew')
        expected=.001*(row['start']+row['end'])/2
        if not np.isclose(row['mean_shift_fraction'],expected,atol=1e-14):
            raise ValueError('Nonuniform-time mean is incorrect')
        if abs(row['detrended_half_range_fraction'])>1e-12 or not np.isclose(row['slope_fraction_per_M'],.001,atol=1e-14):
            raise ValueError('Linear drift was mistaken for oscillation')
    rows=[{'time':t,'q':1+.001*t},{'time':t+.004,'q':1+.001*(t+.004)}]
    grid=np.arange(2,301,2)
    if not np.allclose(aligned(rows,'q',grid),1+.001*grid):raise ValueError('Seed alignment')
    if np.isfinite(ratio(np.array([.1]),np.array([0.]))).any():raise ValueError('Undefined drift ratio was hidden')
    xyz=coordinates();rr=np.linalg.norm(xyz,axis=1)
    radial=1+.03*rr**2
    if np.max(np.abs(angular_residual(radial,rr)))>1e-10:raise ValueError('Radial gradients leak into angular scatter')
    if not np.allclose(angular_residual(xyz[:,0],rr),xyz[:,0],atol=1e-12):raise ValueError('Angular dipole was removed')
    pos=np.zeros((2,3));mass=np.array([1.,3.]);w=cloud(pos,mass);w2=cloud(pos,mass,True)
    if not np.isclose(w.sum(),4) or not np.allclose(w[w>0],.5) or not np.allclose(w2[w2>0],10/64):
        raise ValueError('CIC weight moments do not match the analytic cloud')
    if not np.allclose(w[w>0]**2/w2[w2>0],1.6):raise ValueError('Effective-count definition')
    results={'window_and_alignment':'passed','radial_vs_angular':'passed','CIC_weight_moments':'passed'}
    p=model()
    for radial in ('legacy','proper'):
        begin=time.monotonic();table=cells(p,radial)
        pos,mass,info=cell_positions(p,radial,1,table)
        if not np.all(mass>0) or not np.isclose(mass.sum(),p['M0_over_M'],rtol=1e-12):raise ValueError('Cell sample mass')
        if np.max(np.linalg.norm(pos,axis=1))>=p['Riso_over_M']:raise ValueError('Sample outside cluster')
        # Independent coordinate-cell histogram checks actual positions, not bookkeeping.
        grid_cell=np.floor(pos/.5).astype(int);table_cell=np.rint(table['lo']/.5).astype(int)
        extent=int(np.max(np.abs(table_cell)))+2
        code=lambda a:((a[:,0]+extent)*(2*extent)+(a[:,1]+extent))*(2*extent)+(a[:,2]+extent)
        lookup=np.bincount(code(grid_cell));count=lookup[code(table_cell)]
        if count.max()-count.min()>1 or count.sum()!=1500000:raise ValueError('Actual cell occupancy differs')
        results[radial]=dict(info,actual_occupancy_min=int(count.min()),actual_occupancy_max=int(count.max()),seconds=time.monotonic()-begin)
        print(radial,results[radial],flush=True)
    atomic(ROOT/'evidence/QA_MEASUREMENTS_OK.json',results)

if __name__=='__main__':main()
