"""Independent full-population transformation checks inside Slurm."""
import sys
import numpy as np
from common import *

def initial_check(work,c):
    job()
    with (ROOT/'samples/yc0500.bin').open('rb') as f:
        f.readline();base=np.fromfile(f,dtype='<f8').reshape(-1,8)
    with (work/'initial_particles.bin').open('rb') as f:
        h=f.readline().decode();native=np.fromfile(f,dtype='<f8').reshape(-1,9)
    assert 'STMIGP05' in h and base.shape==(2000000,8)
    assert np.array_equal(native[:,0],base[:,0]) and np.all(native[:,1]==0)
    assert np.array_equal(native[:,2:5],base[:,1:4]) and np.array_equal(native[:,8],base[:,7])
    U=float(os.environ.get('ST8X_U','0'))
    if U==0:
        assert np.array_equal(native[:,5:8],base[:,4:7])
        result=dict(U=U,exact_zero_mode=True)
    else:
        sys.path.insert(0,str(Path(os.environ['ST8X_BASE'])/'src/athenak/scripts/reconstruction_s05'))
        from st_migration_lib import cluster_profile
        p=cluster_profile(.5,dx=1e-4)
        pos=base[:,1:4];r=np.linalg.norm(pos,axis=1);n=pos/r[:,None]
        A=np.interp(r,p['riso_over_m'],p['conformal_a'])
        w0=np.sqrt(1+np.sum(base[:,4:7]**2,axis=1)/A**2)
        v0=base[:,4:7]/(A*w0)[:,None]
        x=A*r/p['R_over_M'];dv=U*(3*x-x**3)/2
        v=v0+dv[:,None]*n;v2=np.sum(v*v,axis=1)
        assert np.max(v2)<1 and np.max(dv)<0
        # Extract the coupled solver's radial metric profile from its reproducible
        # standalone C++ reference dump, not from transformed particle momenta.
        corr=np.loadtxt(Path(os.environ['ST8X_ROOT'])/'evidence'/f'constraint_{abs(U):.3f}.txt')
        Ac=np.interp(r,corr[:,0],corr[:,1])
        expected=Ac[:,None]*v/np.sqrt(1-v2)[:,None]
        error=float(np.max(np.abs(native[:,5:8]-expected)))
        assert error<2e-7,('momentum_transform',error)
        actual=native[:,5:8]/Ac[:,None]
        wa=np.sqrt(1+np.sum(actual**2,axis=1));va=actual/wa[:,None]
        tang=(va-v)-np.sum((va-v)*n,axis=1)[:,None]*n
        assert np.max(np.linalg.norm(tang,axis=1))<1e-7
        result=dict(U=U,max_momentum_error=error,max_speed=float(np.sqrt(v2.max())),mean_delta_vr=float(dv.mean()),source_sample=receipt(ROOT/'samples/yc0500.bin'))
        alpha=np.interp(r,p['riso_over_m'],p['lapse']);mass=base[:,7]
        L0=np.linalg.norm(np.cross(pos,base[:,4:7]),axis=1);L1=np.linalg.norm(np.cross(pos,native[:,5:8]),axis=1)
        result.update(particle_Eulerian_energy_initial=float(np.sum(mass*w0)),particle_Eulerian_energy_perturbed=float(np.sum(mass*wa)),particle_Killing_energy_initial=float(np.sum(mass*alpha*w0)),particle_Killing_energy_perturbed=float(np.sum(mass*alpha*wa)),relative_mean_abs_L_change=float(np.sum(mass*L1)/np.sum(mass*L0)-1),mean_base_vr=float(np.mean(np.sum(v0*n,axis=1))),mean_perturbed_vr=float(np.mean(np.sum(va*n,axis=1))),energy_caveat='particle energies at initialization; not total ADM energy or conserved live-metric Killing energy')
    vtk=sorted((work/'pvtk').glob('*.vtk'))
    if vtk:
        from reduce_run import read_pvtk
        time,cycle,count,fields=read_pvtk(str(vtk[0]));order=np.argsort(fields['ptag'])
        assert time==0 and cycle==0 and count==len(base)
        assert np.array_equal(fields['ptag'][order],native[:,0])
        for key,col in [('x',2),('y',3),('z',4),('ux',5),('uy',6),('uz',7),('prtcl_mass',8)]:
            assert np.array_equal(fields[key][order],native[:,col].astype(np.float32)),key
        from run_segment import cohort
        time,cycle,mask,values=cohort(sorted((work/'cohort').glob('*.s8c'))[0])
        assert time==0 and cycle==0 and mask.sum()==20000
        expected=np.column_stack((native[:,0],native[:,2:]))
        assert np.array_equal(values,expected[::100])
        result.update(actual_full_particle_output_checked=True,actual_fixed_ID_cohort_checked=True)
    else:
        assert os.environ.get('ST8X_ZERO_PREFLIGHT')=='1', 'Production initialization must verify its actual full-particle output'
        result.update(actual_full_particle_output_checked=False,scope='Zero-step preflight generation dump plus actual deposited-field constraints; production actual-particle output gate still required')
    result.update(utc=now(),job=job(),ranks=int(os.environ['SLURM_NTASKS']),count=len(base),positions_ids_weights_exact=True,executable=receipt(ROOT/'protected/athena'))
    atomic(work/'INITIAL_OK.json',result)
    return result
