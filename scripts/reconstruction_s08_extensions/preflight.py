"""Zero-step fixed-mesh preflight while full production nodes are queued."""
import sys
import numpy as np
from common import *
from extension_resources import EXT,admission
from campaign import setup,section

def main():
    job();admission();records=[]
    assert int(os.environ['SLURM_NTASKS'])==1
    for name,U in [('control',0.),('inward0020',-.02)]:
        root=ROOT/name;setup(root,U,EXT/'protected/athena');work=root/'runs/yc0500'
        text=(root/'inputs/yc0500.athinput').read_text()
        text=section(text,'mesh_refinement',dict(refinement='static',max_nmb_per_rank=344))
        text=section(text,'time',dict(nlim=0,tlim=0))
        for k in range(2,9):text=section(text,f'output{k}',dict(dt=0))
        inp=root/'inputs/preflight.athinput';inp.write_text(text)
        atomic(ROOT/'evidence/active.json',dict(job=job(),work=str(work),phase='non_evolving_static_preflight'))
        args=native_launcher()+[str(root/'protected/athena'),'--kokkos-map-device-id-by=mpi_rank','-i',str(inp),'-d',str(work)]
        atomic(work/'launch.json',dict(utc=now(),job=job(),command=args,input=receipt(inp),executable=receipt(root/'protected/athena'),no_evolution=True))
        with (work/'native.log').open('w') as out:cmd(args,stdout=out,stderr=subprocess.STDOUT)
        env=os.environ.copy();env.update(ROOT=str(root),ST8X_U=str(U))
        script="from common import *;from initial_validation import initial_check;initial_check(ROOT/'runs/yc0500',CASES[0])"
        cmd(['python3','-c',script],env=env)
        records.append(dict(U=U,initial=load(work/'INITIAL_OK.json'),history=receipt(work/'yc0500.user.hst'),global_history=receipt(work/'yc0500.z4c.user.hst')))
    a=np.atleast_2d(np.loadtxt(ROOT/'control/runs/yc0500/yc0500.user.hst'))[0];b=np.atleast_2d(np.loadtxt(ROOT/'inward0020/runs/yc0500/yc0500.user.hst'))[0]
    ga=np.atleast_2d(np.loadtxt(ROOT/'control/runs/yc0500/yc0500.z4c.user.hst'))[0];gb=np.atleast_2d(np.loadtxt(ROOT/'inward0020/runs/yc0500/yc0500.z4c.user.hst'))[0]
    assert a[0]==b[0]==ga[0]==gb[0]==0
    ratios={name:float(np.sqrt(b[k]/a[k])) for name,k in [('H_r10',3),('M_r10',4),('H_r16',9),('M_r16',10)]}
    ratios.update(H_global=float(np.sqrt(gb[3]/ga[3])),M_global=float(np.sqrt(gb[4]/ga[4])))
    result=dict(utc=now(),job=job(),records=records,ratios=ratios,threshold=1.005,passed=max(ratios.values())<=1.005,scope='Full2M particle transformation and discrete initial constraints on identical344-block fixed mesh. Zero evolution steps; does not replace matched adaptive production-layout admission or authorize singleGPU evolution.')
    atomic(EXT/'evidence/STATIC_PREFLIGHT.json',result)
    assert result['passed'],ratios
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
