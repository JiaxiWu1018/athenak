"""Evolving one-particle stage-crossing fixtures, including empty MPI ranks."""
import argparse
import os
from pathlib import Path
import subprocess
import re
import numpy as np
from state_io import require_job, read_rst, final_rst, compare_states, write_json, sha

require_job()
ap=argparse.ArgumentParser();ap.add_argument('--exe',required=True);ap.add_argument('--root',required=True)
a=ap.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=False)
repo=Path(__file__).resolve().parents[2]
base=(repo/'tst/inputs/part_tmunu_smr_pytest.athinput').read_text()
base=base.replace('pusher = drift','pusher = gr_boris\ncoupling = rk3_predictor\nmidpoint_tolerance=1e-11\nmidpoint_max_iterations=100')
base=base.replace('midpoint_max_iterations=100','midpoint_max_iterations=100\nmidpoint_stencil='+os.environ.get('ST10_MIDPOINT_STENCIL','moving'))
base=base.replace('integrator = rk2','integrator = rk3').replace('nlim = 0','nlim = 100000')
base=base.replace('tlim = 1.0','tlim = .05').replace('cfl_number = 0.25','cfl_number = .2')
base=base.replace('pmass = 1.0','pmass = 1e-8')
base=base.replace('dt = 0.001','dt = .05')
base+='\n<output2>\nfile_type=rst\ndt=.05\n<output3>\nfile_type=pvtk\nvariable=prtcl_all\ndt=.05\n'
cases={'same_level':((.499,.25,.25),(.2,0,0)),
       'coarse_to_fine':((-.001,.25,.25),(.2,0,0)),
       'fine_to_coarse':((.001,.25,.25),(-.2,0,0)),
       'edge':((-.001,-.001,.25),(.2,.2,0)),
       'corner':((-.001,-.001,-.001),(.2,.2,.2))}
results={}
for name,(pos,vel) in cases.items():
    states=[]
    for ranks in (1,4):
        work=root/f'{name}_np{ranks}';work.mkdir()
        text=base
        for axis,x,u in zip('xyz',pos,vel):
            text=re.sub(rf'^p{axis}\s*=.*$',f'p{axis}={x}',text,flags=re.M)
            text=re.sub(rf'^pu{axis}\s*=.*$',f'pu{axis}={u}',text,flags=re.M)
        inp=work/'input.athinput';inp.write_text(text)
        with (work/'run.log').open('w') as f:
            code=subprocess.run(['mpirun','-np',str(ranks),a.exe,'-i',str(inp),'-d',str(work)],stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode
        assert code==0,(name,ranks,(work/'run.log').read_text()[-3000:])
        state=read_rst(final_rst(work));assert state['count']==1 and abs(state['time']-.05)<1e-10
        p=state['particles'];initial=np.array(pos);final=p[[2,4,6],0]
        assert np.dot(final-initial,np.array(vel))>0
        for axis in range(3):
            if vel[axis]:
                boundary=.5 if name=='same_level' else 0
                assert (pos[axis]-boundary)*(final[axis]-boundary)<0,(name,final)
        states.append(state)
    comparison=compare_states(*states)
    assert comparison['grid_max_abs']<2e-12 and comparison['particle_max_abs']<2e-12,comparison
    results[name]=comparison
write_json(root/'GATES_OK.json',dict(cases=results,exe_sha256=sha(a.exe)))
