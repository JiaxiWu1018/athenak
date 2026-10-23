"""Actual full-sample initialization and matched discrete constraint admission."""
import sys
import numpy as np
from common import *
from setup_case import setup
from run_segment import run

def initialize(c,control=False):
    work=setup(c,U=0 if control else c['U'],control=control)
    if not (work/'state.json').exists():
        from extension_resources import storage
        disk=storage();assert disk['own']+8*GIB<disk['limit'],'Full initial outputs storage forecast'
        run(c,initialize=True,target=c['target'],wall_seconds=900)
    state=load(work/'state.json');initial=load(work/'INITIAL_OK.json')
    assert state['latest']['time']==0 or not control
    assert initial['actual_full_particle_output_checked'] and initial['actual_fixed_ID_cohort_checked']
    assert initial['executable']['sha256']==sha(ROOT/'protected/athena')
    return work

def admit(c,work):
    prior=ROOT/'evidence/INITIAL_CONSTRAINT_ADMISSION.json'
    if prior.exists():
        p=load(prior)
        assert p['executable_sha256']==sha(ROOT/'protected/athena')
        assert p['ranks']==int(os.environ['SLURM_NTASKS']), 'Initialize-layout changed; matched control review required'
        return p
    if MACHINE=='perseus':
        env=os.environ.copy();env['ROOT']=str(EXT/'backend_checks')
        cmd(['python3',Path(__file__).with_name('backend_checks.py')],env=env)
    reference=load(EXT/'evidence/INDEPENDENT_CONSTRAINT_REFERENCE_OK.json')
    ref=next(v for v in reference['results'] if v['case']==c['key'])
    assert ref['U']==c['U'] and ref['yc']==c['yc']
    assert ref['source_profile']['sha256']==sha(EXT/'evidence/constraints'/f"{c['key']}.txt")
    exe=sha(ROOT/'protected/athena');ranks=int(os.environ['SLURM_NTASKS'])
    control=EXT/'controls'/f"{c['id']}_{MACHINE}_n{ranks}_{exe[:12]}"
    env=os.environ.copy();env.update(ROOT=str(control),ST8X_U='0')
    cmd(['python3',Path(__file__).resolve(),'--control'],env=env)
    initialize(c)
    a=np.atleast_2d(np.loadtxt(control/'runs'/c['id']/f"{c['id']}.user.hst"))[0]
    b=np.atleast_2d(np.loadtxt(work/f"{c['id']}.user.hst"))[0]
    ga=np.atleast_2d(np.loadtxt(control/'runs'/c['id']/f"{c['id']}.z4c.user.hst"))[0]
    gb=np.atleast_2d(np.loadtxt(work/f"{c['id']}.z4c.user.hst"))[0]
    assert a[0]==b[0]==ga[0]==gb[0]==0
    ratios={name:float(np.sqrt(b[k]/a[k])) for name,k in [('H_r10',3),('M_r10',4),('H_r16',9),('M_r16',10)]}
    ratios.update(H_global=float(np.sqrt(gb[3]/ga[3])),M_global=float(np.sqrt(gb[4]/ga[4])))
    assert np.isfinite(list(ratios.values())).all() and max(ratios.values())<=1.005,ratios
    ca=load(control/'runs'/c['id']/'state.json')['latest'];cb=load(work/'state.json')['latest']
    assert ca['nblocks']==cb['nblocks'] and ca['particle_count']==cb['particle_count']==2000000
    record=dict(utc=now(),job=job(),case=c,executable_sha256=exe,ranks=ranks,
      constraint_norm_ratios=ratios,tolerance=1.005,
      initial_rows=dict(control_local=a.tolist(),perturbed_local=b.tolist(),control_global=ga.tolist(),perturbed_global=gb.tolist()),
      control_root=str(control),control_checkpoint=ca,perturbed_checkpoint=cb,continuum_reference=ref)
    atomic(prior,record);atomic(ROOT/'evidence/PRODUCTION_ADMISSION.json',record)
    return record

if __name__=='__main__':
    assert sys.argv[1:]==['--control']
    initialize(CASE,control=True)
