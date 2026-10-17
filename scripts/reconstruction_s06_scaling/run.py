"""Fresh full-N preflight or 1000M evolution with case-aware native gates."""
import argparse
import math
import os
import re
import socket
import subprocess
import time
import numpy as np
from cfg import *
import run_case as native
import reduce_run as R
from os_gate import queue

def run(case,role,end,walltime,restart=None):
    require_job()
    assert not queue(local=True), 'New OS work has priority; do not launch ST'
    assert load(ROOT/'evidence/OS_PRIORITY_RELEASE.json')['verified_complete']
    initial=ROOT/'samples'/(case['id']+'.bin');evidence=load(ROOT/'evidence/samples'/(case['id']+'.json'))
    assert sha(initial)==evidence['sha256']
    exe=ROOT/'protected/athena';assert sha(exe)==EXE_SHA
    whole=int(subprocess.check_output(['du','-s','-B1','/work1/eliasmost/jiaxiwu'],text=True).split()[0])
    forecast=case['raw_forecast'] if role=='production' else 11000000000
    # Include the other admitted case: concurrent production must fit together.
    reserved=sum(c['raw_forecast'] for c in CASES) if role=='production' else 35000000000
    assert whole+reserved<1869169767219,'Projected full-user storage exceeds1.7TiB; preserve data and review'
    work=ROOT/'runs'/case['id']/role;work.mkdir(parents=True,exist_ok=False)
    out=work/'out';out.mkdir()
    deck=ROOT/'inputs'/(case['id']+'.athinput')
    command=['prun',str(exe),'--kokkos-map-device-id-by=mpi_rank','-t',walltime,
             '-r' if restart else '-i',str(restart or deck),'-d',str(out),f'time/tlim={end:g}']
    atomic(work/'launch.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),
        case=case,role=role,command=command,input_sha256=sha(deck),sample_sha256=evidence['sha256'],
        executable_sha256=EXE_SHA,compiled_source=PHYSICS,kokkos=KOKKOS,scripts_source=source(),
        whole_user_bytes=whole,concurrent_reserved_growth_bytes=reserved,case_growth_forecast=forecast))
    begin=time.monotonic()
    with (work/'solver.log').open('w') as stream:
        process=subprocess.Popen(command,cwd=work,stdout=stream,stderr=subprocess.STDOUT)
        while process.poll() is None:
            with (work/'memory_samples.jsonl').open('a') as mem:
                mem.write(json.dumps(native.memory_sample(exe))+'\n')
            time.sleep(5)
    text=(work/'solver.log').read_text()
    if process.returncode or 'Terminating on time limit' not in text or '[conservation OK]' not in text:
        raise RuntimeError('Evolution did not reach its requested endpoint cleanly; preserve failure')
    assert not re.search(r'search_fail=[1-9]|rejected=[1-9]|nonfinite=[1-9]',text)
    states=[native.checkpoint(p,initial=initial,n=case['n']) for p in sorted((out/'rst').glob('*.rst'))]
    assert states and abs(max(s['time'] for s in states)-end)<1e-7
    parsed=R.parse_log(str(work/'solver.log'))
    elapsed=time.monotonic()-begin
    atomic(work/'completion.json',dict(utc=now(),case=case,role=role,job=os.environ['SLURM_JOB_ID'],
        endpoint_M=end,solver_wall_seconds=elapsed,checkpoints=states,
        boris_fallbacks=parsed['boris_fallbacks'],boris_fallback_cycles=parsed['boris_fallback_cycles']))
    # Hash on the allocated node, after the solver has closed every output.
    # This source manifest lets later transfers verify native bytes independently.
    files=[dict(path=str(p.relative_to(work)),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(out.rglob('*')) if p.is_file()]
    atomic(work/'OUTPUT_MANIFEST.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],
        case=case['id'],source=source(),endpoint_M=end,files=files,
        logical_bytes=sum(f['bytes'] for f in files)))
    return work,elapsed,states

def check_particles(case,work):
    _,initial=sample(ROOT/'samples'/(case['id']+'.bin'))
    summaries=[]
    for path in sorted((work/'out/pvtk').glob('*.part.vtk')):
        t,cycle,n,fields=R.read_pvtk(str(path));tags=fields['ptag'].astype(np.int64)
        assert n==case['n'] and np.array_equal(np.sort(tags),np.arange(n))
        assert all(np.isfinite(v).all() for v in fields.values())
        assert np.array_equal(fields['prtcl_mass'],initial[tags,7].astype(np.float32).astype(float))
        pos=np.column_stack([fields[c] for c in 'xyz'])
        if t==0:
            assert np.array_equal(pos,initial[tags,1:4].astype(np.float32).astype(float))
        summaries.append(dict(time_M=t,cycle=cycle,n=n,sha256=sha(path),bytes=path.stat().st_size))
    assert len(summaries)==11 and summaries[0]['time_M']==0 and abs(summaries[-1]['time_M']-20)<1e-6
    atomic(work/'PARTICLE_QA.json',dict(frames=summaries,full_count_unique_IDs_finite_fixed_mass=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('role',choices=['preflight','production']);parser.add_argument('index',type=int)
    args=parser.parse_args();case=CASES[args.index];load(ROOT/'evidence/SAMPLES_OK.json')
    if args.role=='preflight':
        work,elapsed,states=run(case,'preflight',20,'00:45:00')
        check_particles(case,work)
        if case['n']==6000000:
            final=max(states,key=lambda r:r['time'])
            probe,_,probe_states=run(case,'restart_probe',22,'00:15:00',Path(final['path']))
            assert abs(max(r['time'] for r in probe_states)-22)<1e-7
        forecast=elapsed*1000/20
        atomic(ROOT/'evidence/preflight'/(case['id']+'.json'),dict(utc=now(),case=case,job=os.environ['SLURM_JOB_ID'],
            passed=True,elapsed_seconds=elapsed,forecast_1000M_node_hours=forecast/3600,
            recommended_wall_hours=max(2,2*math.ceil(1.25*forecast/7200)),
            all_states_full_payload_IDs_masses_verified=True,restart_probe_passed=case['n']==6000000))
        print('PREFLIGHT_OK',case['id'],'forecast_nodeh',forecast/3600,flush=True)
    else:
        gates=[load(ROOT/'evidence/preflight'/(c['id']+'.json')) for c in CASES]
        assert all(g['passed'] for g in gates)
        hours=int(os.environ['ST_WALL_HOURS']);seconds=hours*3600-600
        wall=f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
        work,_,_=run(case,'production',1000,wall)
        atomic(work/'RUN_OK',now()+'\n')
        print('PRODUCTION_OK',case['id'],flush=True)

if __name__=='__main__':
    main()
