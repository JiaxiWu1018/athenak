"""Native small-ST regression, restart, guards, and solver tolerance gates."""
import argparse
from pathlib import Path
import os
import re
import subprocess
import numpy as np
from state_io import require_job, read_rst, final_rst, compare_states, write_json, sha


def deck(n=1000, coupling='rk3_predictor', integrator='rk3', tol='1e-11', limit=100):
    text = '<job>\nbasename = gate\n<mesh>\nnghost = 4\n'
    for a in (1, 2, 3):
        text += f'nx{a}=32\nx{a}min=-16\nx{a}max=16\nix{a}_bc=outflow\nox{a}_bc=outflow\n'
    text += '<meshblock>\nnx1=8\nnx2=8\nnx3=8\n<mesh_refinement>\nrefinement=static\n'
    text += '<refined_region1>\nlevel=1\n'
    for a in (1, 2, 3):
        text += f'x{a}min=-8\nx{a}max=8\n'
    text += f'''<time>
evaluate_initial_constraints=true
evolution=dynamic
integrator={integrator}
cfl_number=.2
nlim=100000
tlim=.4
ndiag=1
<z4c>
use_z4c=true
lapse_harmonic=0
lapse_oplog=2
shift_driver=legacy
shift_Gamma=1
shift_H=0
shift_alpha2Gamma=0
shift_eta=2
diss=.1
chi_div_floor=.00001
damp_kappa1=.02
damp_kappa2=0
<particles>
particle_type=dust
pusher=gr_boris
init=pgen
feedback=true
mass=1
debug=1
excise_lapse=0
excise_ah=false
destroy_log=true
cross_level_deposit=conservative
coupling={coupling}
synchronized_constraints=true
midpoint_tolerance={tol}
midpoint_max_iterations={limit}
midpoint_stencil={os.environ.get("ST10_MIDPOINT_STENCIL", "moving") if coupling == "rk3_predictor" else "moving"}
<fastflow>
num_horizons=0
<problem>
cluster_n={n}
cluster_yc=.819
cluster_mass=1
cluster_seed=1
cluster_profile_dx=.0001
cluster_sampling=random
cluster_radial_measure=legacy
cluster_velocity_sampling=random
user_hist=true
<output1>
file_type=hst
dt=.1
data_format=%23.15e
<output2>
file_type=rst
dt=.2
<output3>
file_type=pvtk
dt=.1
variable=prtcl_all
'''
    return text


def run(exe, root, name, text, ranks=4, args=(), fail=None):
    work=root/name
    work.mkdir(parents=True, exist_ok=False)
    inp=work/'input.athinput';inp.write_text(text)
    cmd=['mpirun','-np',str(ranks),str(exe),'-i',str(inp),'-d',str(work)]+list(args)
    with (work/'run.log').open('w') as f:
        proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
    log=(work/'run.log').read_text(errors='replace')
    write_json(work/'LAUNCH.json',dict(command=cmd,exit_code=proc.returncode,exe_sha256=sha(exe)))
    if fail:
        assert proc.returncode and fail in log, (name,log[-2000:])
        return work
    assert proc.returncode == 0 and 'FATAL' not in log, (name,log[-3000:])
    state=read_rst(final_rst(work))
    assert abs(state['time']-.4)<1.e-10 and np.all(np.isfinite(state['grid']))
    if 'coupling=rk3_predictor' in text:
        matches=re.findall(r'\[rk3-particles\].*stage=3.*max_residual=(\S+).*failed=(\d+)',log)
        assert matches and all(int(f)==0 and float(r)<=float(state['params']['particles']['midpoint_tolerance'])*1.01 for r,f in matches)
    return work


def main():
    require_job()
    ap=argparse.ArgumentParser();ap.add_argument('--exe',required=True);ap.add_argument('--root',required=True)
    opts=ap.parse_args();exe=Path(opts.exe);root=Path(opts.root);root.mkdir(parents=True,exist_ok=False)
    cases={}
    for ranks in (1,4):
        cases[f'np{ranks}']=run(exe,root,f'np{ranks}',deck(),ranks)
    a=read_rst(final_rst(cases['np1']));b=read_rst(final_rst(cases['np4']))
    rank=compare_states(a,b)
    assert rank['grid_max_abs']<2.e-12 and rank['particle_max_abs']<2.e-12,rank
    tight=run(exe,root,'tight',deck(tol='1e-13'))
    tolerance=compare_states(b,read_rst(final_rst(tight)))
    assert tolerance['grid_max_abs']<1.e-8 and tolerance['particle_max_abs']<1.e-8,tolerance
    run(exe,root,'empty_ranks',deck(n=1))
    # Restart from the existing interior .2M checkpoint, preserving all originals.
    checkpoint=next(p for p in (cases['np4']/'rst').glob('*.rst') if abs(read_rst(p)['time']-.2)<1.e-10)
    resumed=run(exe,root,'restart',deck(),args=('-r',str(checkpoint),'time/tlim=.4'))
    restart=compare_states(b,read_rst(final_rst(resumed)))
    assert restart['grid_max_abs']<1.e-10 and restart['particle_max_abs']<1.e-10,restart
    # An observation-only diagnostic refresh must not change the legacy trajectory.
    legacy=run(exe,root,'legacy_sync',deck(coupling='legacy',integrator='rk4'))
    default=run(exe,root,'legacy_default',deck(coupling='legacy',integrator='rk4').replace('coupling=legacy\n','').replace('synchronized_constraints=true','synchronized_constraints=false'))
    unchanged=compare_states(read_rst(final_rst(legacy)),read_rst(final_rst(default)))
    assert unchanged['grid_max_abs']==0 and unchanged['particle_max_abs']==0,unchanged
    run(exe,root,'bad_stencil',deck().replace('midpoint_stencil='+os.environ.get('ST10_MIDPOINT_STENCIL','moving'),'midpoint_stencil=invalid'),fail='invalid particle coupling configuration')
    run(exe,root,'bad_integrator',deck(integrator='rk4'),fail='invalid particle coupling configuration')
    run(exe,root,'bad_excision',deck().replace('excise_lapse=0','excise_lapse=.1'),fail='invalid particle coupling configuration')
    run(exe,root,'bad_solver',deck(limit=1),fail='particle solve failed')
    run(exe,root,'bad_amr',deck().replace('refinement=static','refinement=adaptive\nnum_levels=2\nmax_nmb_per_rank=128'),fail='invalid particle coupling configuration')
    write_json(root/'GATES_OK.json',dict(rank=rank,tolerance=tolerance,restart=restart,legacy_unchanged=unchanged,exe_sha256=sha(exe)))

if __name__=='__main__':main()
