"""Allocated, immutable segmented ST runs with provenance and resource guards."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import numpy as np
from state_io import require_job, sha, write_json, read_rst, final_rst, parse_input

ROOT=Path(os.environ.get('ST10_ROOT','/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102'))
USER_ROOT=ROOT.parent
CASES={}
for n in (1,2,4,8):
    CASES[f'predictor_{n}m_h020']=dict(n=n*1000000,dt=.2,integrator='rk3',coupling='rk3_predictor',end=500)
for label,coupling,integrator in [('predictor','rk3_predictor','rk3'),('legacy','legacy','rk4')]:
    for dt in (.2,.1,.05):
        CASES.setdefault(f'{label}_1m_h{round(dt*100):03d}',dict(n=1000000,dt=dt,integrator=integrator,coupling=coupling,end=100))
CASES['frozen_rk3_1m_h020']=dict(n=1000000,dt=.2,integrator='rk3',coupling='legacy',end=100)
# Versioned strict fixed-node cases never resume original moving-stencil trajectories.
for old,cfg in list(CASES.items()):
    if old.startswith('predictor_'):
        CASES[old.replace('predictor_', 'fixed_v1_')]=dict(cfg,revision='fixed_v1',stencil='predicted_fixed')
# Same strict fixed-node scheme with guarded reuse of identical-stage derivatives.
# h=.2 is retained only as the failed stability experiment; long-run candidates use h=.1.
for n in (1,2,4,8):
    CASES[f'fixed_fast_v1_{n}m_h010']=dict(n=n*1000000,dt=.1,integrator='rk3',
        coupling='rk3_predictor',end=500,revision='fixed_fast_v1',stencil='predicted_fixed')
for label,dt in (('h005',.05),('h0025',.025)):
    CASES[f'fixed_fast_v1_1m_{label}']=dict(n=1000000,dt=dt,integrator='rk3',
        coupling='rk3_predictor',end=100,revision='fixed_fast_v1',stencil='predicted_fixed')




def input_text(case,target,output_dir=None):
    cfg=CASES[case]
    if output_dir is None:
        output_dir=ROOT/'initial_dumps'/('job_'+os.environ['SLURM_JOB_ID'])
        output_dir.mkdir(parents=True,exist_ok=True)
    params=parse_input((Path(__file__).parent/'baseline.athinput').read_text())
    params={k:v for k,v in params.items() if not k.startswith('output')}
    params['comment']={'problem':'ST Session10 stage-consistent RK3 predictor experiment','note':'Exact Session7 seed1 samples; all endpoint constraints synchronized'}
    params['job']['basename']=case
    params['time'].update(integrator=cfg['integrator'],cfl_number=str(2*cfg['dt']),tlim=str(target),ndiag='10')
    params['particles'].update(coupling=cfg['coupling'],synchronized_constraints='true',midpoint_tolerance='1e-11',midpoint_max_iterations='100')
    if 'stencil' in cfg:params['particles']['midpoint_stencil']=cfg['stencil']
    params['problem'].update(cluster_n=str(cfg['n']),cluster_sample_file=str(ROOT/f"reference/session7/samples/mc_{cfg['n']//1000000}m_s1.bin"),cluster_t0_dump=str(Path(output_dir)/'initial_particles.bin'))
    params['output1']={'file_type':'hst','dt':'2.0','data_format':'%23.15e'}
    params['output2']={'file_type':'rst','dt':'100.0'}
    # Sparse particles preserve independent census/trajectory evidence without movies.
    params['output3']={'file_type':'pvtk','dt':'50.0','variable':'prtcl_all'}
    return '\n\n'.join('<'+k+'>\n'+'\n'.join(f'{a} = {b}' for a,b in v.items()) for k,v in params.items())+'\n'


def resource_snapshot():
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    # du is intentionally allocated, serialized, and failure-tolerant.
    with (ROOT/'evidence/storage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for key,path in [('whole_user_bytes',USER_ROOT),('campaign_bytes',ROOT)]:
            try:
                out=subprocess.check_output(['du','-s','-B1',str(path)],text=True,timeout=240,stderr=subprocess.STDOUT)
                result[key]=int(out.split()[0])
            except Exception as exc:result[key+'_error']=repr(exc)
    try:
        jobs=json.loads((ROOT/'evidence/job_registry.json').read_text())['jobs']
        ids={str(j['job_id']) for j in jobs if j['host']=='AMD'}|{os.environ['SLURM_JOB_ID']}
        out=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(sorted(ids)),
            '--format=JobIDRaw,ElapsedRaw,AllocNodes'],text=True,timeout=30)
        rows=[line.split('|') for line in out.splitlines() if line.strip()]
        result['node_hours']=(sum(int(r[1])*int(r[2]) for r in rows)+sum(j.get('elapsed_seconds',0)*j.get('nodes',1) for j in jobs if j['host']!='AMD'))/3600
    except Exception as exc:result['accounting_error']=repr(exc)
    result['warning']=result.get('whole_user_bytes',0)>=1.5*2**40
    result['stop']=(result.get('whole_user_bytes',0)>=1.7*2**40 or
                    result.get('campaign_bytes',0)>=298*2**30 or result.get('node_hours',0)>=39.5)
    return result


class Watchdog:
    def __init__(self,work):
        self.work=work;self.done=threading.Event();self.proc=None;self.stopped=False
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        next_storage=0.0
        while not self.done.is_set():
            try:
                snap={'stop':False}
                if time.monotonic()>=next_storage:
                    snap=resource_snapshot();next_storage=time.monotonic()+300
                    with (self.work/'resources.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
                try:
                    out=subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json'],text=True,timeout=20,stderr=subprocess.STDOUT)
                    with (self.work/'gpu_memory.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'output':out})+'\n')
                except Exception as exc:
                    with (self.work/'gpu_memory.jsonl').open('a') as f:f.write(json.dumps({'error':repr(exc)})+'\n')
                if snap['stop']:
                    self.stopped=True
                    write_json(self.work/'RESOURCE_STOP.json',snap)
                    if self.proc and self.proc.poll() is None:os.killpg(self.proc.pid,signal.SIGTERM)
                    return
            except Exception as exc:
                with (self.work/'resources.jsonl').open('a') as f:f.write(json.dumps({'watchdog_error':repr(exc)})+'\n')
            self.done.wait(30)


def sample_array(path):
    with path.open('rb') as f:
        header=f.readline().decode().split();offset=f.tell()
    assert header[0]=='STSAMP05'
    return np.memmap(path,mode='r',dtype='<f8',offset=offset,shape=(int(header[1]),8))


def validate_initial(work,sample):
    with (work/'initial_particles.bin').open('rb') as f:
        header=f.readline().decode();offset=f.tell()
    assert header.startswith('STMIGP05')
    data=np.memmap(work/'initial_particles.bin',mode='r',dtype='<f8',offset=offset,shape=(len(sample),9))
    for i in range(0,len(sample),100000):
        a=sample[i:i+100000];b=data[i:i+100000]
        assert np.array_equal(a[:,0],b[:,0]) and np.all(b[:,1]==0)
        assert np.array_equal(a[:,1:7],b[:,2:8]) and np.array_equal(a[:,7],b[:,8])
    return dict(exact_sample_match=True,count=len(sample),sha256=sha(work/'initial_particles.bin'))


def validate_endpoint(path,sample,target):
    state=read_rst(path)
    assert abs(state['time']-target)<2e-10,(state['time'],target)
    assert state['count']==len(sample)
    p=state['particles'];order=state['order']
    assert np.array_equal(p[9,order],sample[:,0])
    assert np.array_equal(p[0,order],sample[:,7])
    for block in state['grid']:assert np.all(np.isfinite(block))
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,time=state['time'],
                count=state['count'],cycle=state['cycle'],finite=True,original_ids_masses=True,removals=0)


def run_segment(case,target):
    cfg=CASES[case];base=ROOT/'runs'/case;base.mkdir(parents=True,exist_ok=True)
    completed=sorted((json.loads(p.read_text()) for p in base.glob('*/COMPLETE.json')),key=lambda d:d['target'])
    previous=completed[-1] if completed else None
    assert target>(previous['target'] if previous else 0) and target<=cfg['end']
    work=base/f'to_{target:03d}_job{os.environ["SLURM_JOB_ID"]}';work.mkdir(exist_ok=False)
    snap=resource_snapshot();write_json(work/'resource_admission.json',snap);assert not snap['stop'],snap
    input_path=work/'input.athinput';input_path.write_text(input_text(case,target,work))
    sample_path=ROOT/f"reference/session7/samples/mc_{cfg['n']//1000000}m_s1.bin"
    revision=cfg.get('revision')
    assert revision in (None,'fixed_v1','fixed_fast_v1')
    exe=ROOT/('protected/athena.'+revision if revision else 'protected/athena.rk3')
    expected=json.loads((ROOT/('evidence/'+revision.upper()+'_OK.json' if revision else 'evidence/INPUTS_OK.json')).read_text())
    if revision:assert expected['passed'] and expected['stencil']=='predicted_fixed'
    assert sha(exe)==expected['hip_sha256']
    sample_hash=sha(sample_path)
    manifest=json.loads((ROOT/'evidence/TRANSFER_MANIFEST.json').read_text())
    expected_sample=next(row['sha256'] for row in manifest['files'] if row['path']==str(sample_path.relative_to(ROOT/'reference/session7')))
    assert sample_hash==expected_sample
    sample=sample_array(sample_path);assert len(sample)==cfg['n']
    cmd=['prun',str(exe),'--kokkos-map-device-id-by=mpi_rank','-i',str(input_path),'-d',str(work),'-t','10:00:00']
    if previous:
        old_launch=json.loads((Path(previous['work'])/'LAUNCH.json').read_text())
        assert old_launch['exe_sha256']==sha(exe) and old_launch['configuration']==cfg
        checkpoint=Path(previous['checkpoint']['path'])
        assert sha(checkpoint)==previous['checkpoint']['sha256']
        cmd+=['-r',str(checkpoint)]
    launch=dict(case=case,configuration=cfg,target=target,start=previous['target'] if previous else 0,
        host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],command=cmd,
        sample_sha256=sha(sample_path),input_sha256=sha(input_path),exe_sha256=sha(exe),
        source=expected,workflow_commit=expected.get('source_commit',(ROOT/'evidence/WORKFLOW_COMMIT.txt').read_text().strip()),workflow_sha256=sha(Path(__file__)),previous=previous['checkpoint'] if previous else None)
    write_json(work/'LAUNCH.json',launch)
    watchdog=Watchdog(work);watchdog.thread.start();started=time.monotonic()
    try:
        with (work/'run.log').open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            watchdog.proc=proc
            if watchdog.stopped:os.killpg(proc.pid,signal.SIGTERM)
            code=proc.wait()
    finally:
        watchdog.done.set();watchdog.thread.join(timeout=510)
    elapsed=time.monotonic()-started
    write_json(work/'EXIT.json',dict(exit_code=code,elapsed_seconds=elapsed,resource_stop=watchdog.stopped))
    assert code==0 and not watchdog.stopped,(case,code)
    log=(work/'run.log').read_text(errors='replace')
    assert 'FATAL' not in log and 'Terminating on time limit' in log
    solver={}
    if cfg['coupling']=='rk3_predictor':
        matches=re.findall(r'\[rk3-particles\].*stage=3.*max_iter=(\d+).*max_residual=(\S+).*failed=(\d+)',log)
        assert matches and all(int(b)==0 and float(r)<=1.01e-11 for _,r,b in matches)
        solver=dict(steps=len(matches),max_iterations=max(int(i) for i,_,_ in matches),
                    max_residual=max(float(r) for _,r,_ in matches),failures=0)
    for hp in work.glob('*.hst'):
        assert np.all(np.isfinite(np.loadtxt(hp))),str(hp)
    initial=validate_initial(work,sample) if previous is None else None
    checkpoint=validate_endpoint(final_rst(work),sample,target)
    # Hash every immutable output after the native process exits.
    manifest=[dict(path=str(p.relative_to(work)),bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(work.rglob('*')) if p.is_file()]
    write_json(work/'OUTPUT_MANIFEST.json',manifest)
    result=dict(case=case,target=target,start=launch['start'],elapsed_seconds=elapsed,
        seconds_per_M=elapsed/(target-launch['start']),solver=solver,initial=initial,
        checkpoint=checkpoint,job_id=os.environ['SLURM_JOB_ID'],work=str(work))
    write_json(work/'COMPLETE.json',result)
    print(json.dumps(result),flush=True)
    return result


def main():
    require_job()
    ap=argparse.ArgumentParser();ap.add_argument('case',choices=sorted(CASES));ap.add_argument('--targets',type=int,nargs='+',required=True)
    args=ap.parse_args()
    gates=json.loads((ROOT/'evidence/NATIVE_GATES_OK.json').read_text())
    assert gates['passed'] is True
    for target in args.targets:run_segment(args.case,target)

if __name__=='__main__':main()
