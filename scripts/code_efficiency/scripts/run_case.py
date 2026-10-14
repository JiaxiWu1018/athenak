#!/usr/bin/env python3
"""Run one case inside Slurm; retain compact provenance, timing, and liveness evidence."""
import argparse,json,os,subprocess,time,threading,hashlib,resource,shutil
from pathlib import Path
ROOT=Path(os.environ.get('EFFICIENCY_SESSION',Path(__file__).resolve().parents[1]))
def command(args):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False).stdout

def run(c,kind='throughput',trace=False,gate=False,target_override=None):
    from generate_matrix import deck
    profile=kind in ['profile','device']
    target=target_override if target_override is not None else (1.5 if profile else (1 if gate else 5))
    warmup=.5 if profile or not gate else .25
    base=ROOT/'results'/kind/c['name'];base.mkdir(parents=True,exist_ok=True)
    folder=base
    if (folder/'receipt.json').exists():
        old=json.loads((folder/'receipt.json').read_text())
        if old.get('valid'):return old
        number=len(list(base.glob('attempt*')))+2
        folder=base/f'attempt{number:02d}';folder.mkdir()
        previous=base/f'previous_receipt_{number-1:02d}.json'
        shutil.copy2(base/'receipt.json',previous)
        assert previous.read_bytes()==(base/'receipt.json').read_bytes()
        (folder/'previous_attempt.json').write_text(json.dumps(dict(previous_receipt=str(previous),preserved_files=[dict(path=str(p),bytes=p.stat().st_size) for p in base.iterdir() if p.is_file()]),indent=2)+'\n')
    inp=folder/'input.athinput';text=deck(c,profile=profile,tlim=target,warmup=warmup)
    if gate:text=text.replace('debug = 0','debug = 1')
    inp.write_text(text)
    exe=ROOT/'binaries'/('profile' if kind=='device' else 'release')/'athena'
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise RuntimeError('Runs require a Slurm compute allocation')
    launch=['/home/emost/codes/openmpi/bin/mpirun','-np',str(c['ranks']),'--bind-to','none',str(exe),'--kokkos-map-device-id-by=mpi_rank','-i',str(inp),'-d',str(folder)]
    if trace:launch=['/usr/local/cuda-12.9/bin/nsys','profile','--trace=cuda,nvtx,mpi','--sample=none','--cpuctxsw=none','--capture-range=nvtx','--nvtx-capture=benchmark_window','--capture-range-end=stop','--force-overwrite=false','--output',str(folder/'timeline')]+launch
    provenance=dict(case=c,kind=kind,job_id=job,host=command(['hostname']).strip(),start_utc=command(['date','-u','+%FT%TZ']).strip(),launch=launch,exe_sha256=hashlib.file_digest(exe.open('rb'),'sha256').hexdigest(),input_sha256=hashlib.file_digest(inp.open('rb'),'sha256').hexdigest(),commit=command(['git','-C',str(ROOT/'src/athenak'),'rev-parse','HEAD']).strip(),submodule=command(['git','-C',str(ROOT/'src/athenak'),'submodule','status']).strip(),source_status=command(['git','-C',str(ROOT/'src/athenak'),'status','--porcelain']),cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_gpus=os.environ.get('SLURM_JOB_GPUS'),slurm=command(['scontrol','show','job',job]),gpu_metadata=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv']),mpi=command(['/home/emost/codes/openmpi/bin/mpirun','--version']))
    (folder/'source.patch').write_text(command(['git','-C',str(ROOT/'src/athenak'),'diff','HEAD']))
    (folder/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    done=threading.Event(); samples=[]
    def monitor():
        while not done.is_set():
            q=['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm','--format=csv,noheader,nounits']
            ids=os.environ.get('SLURM_JOB_GPUS')
            if ids:q+=['-i',ids]
            samples.append(dict(time=time.time(),gpu=command(q)))
            done.wait(5)
    t=threading.Thread(target=monitor,daemon=True);t.start();start=time.time()
    with (folder/'solver.log').open('w') as out:
        process=subprocess.run(['/usr/bin/time','-v','-o',str(folder/'resource.txt')]+launch,cwd=folder,stdout=out,stderr=subprocess.STDOUT,check=False)
    done.set();t.join(10)
    (folder/'memory_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    timings=[json.loads(p.read_text()) for p in sorted(folder.glob('timing.rank*.json'))]
    finals=[json.loads(p.read_text()) for p in sorted(folder.glob('final.rank*.json'))]
    initials=[json.loads(p.read_text()) for p in sorted(folder.glob('initial.rank*.json'))]
    valid=process.returncode==0 and len(timings)==c['ranks'] and len(finals)==c['ranks'] and all(x['valid'] for x in finals) and all(abs(x['final_time']-target)<1e-8 for x in timings) and all(x['cycles']>0 for x in timings)
    rec=dict(case=c,kind=kind,job_id=job,valid=valid,exit_code=process.returncode,wall_seconds=time.time()-start,endpoint=target,timings=timings,initial=initials,final=finals,trace_requested=trace,end_utc=command(['date','-u','+%FT%TZ']).strip())
    rec['attempt_path']=str(folder.relative_to(ROOT))
    (folder/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
    if folder!=base:(base/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
    if trace and list(folder.glob('timeline.nsys-rep')):
        with (folder/'nsys_stats.log').open('w') as out:
            subprocess.run(['/usr/local/cuda-12.9/bin/nsys','stats','--report','cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,mpi_event_sum','--format','csv','--output',str(folder/'nsys'),str(folder/'timeline.nsys-rep')],stdout=out,stderr=subprocess.STDOUT,check=False)
    print(c['name'],kind,'PASS' if valid else 'FAIL',f'{rec["wall_seconds"]:.2f}s',flush=True)
    return rec
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--kind',choices=['throughput','profile','device','gate'],default='throughput');p.add_argument('--trace',action='store_true');a=p.parse_args()
    source='gates.json' if a.kind=='gate' else 'matrix.json'
    c=next(x for x in json.loads((ROOT/source).read_text()) if x['name']==a.name)
    result=run(c,a.kind,a.trace,a.kind=='gate')
    raise SystemExit(0 if result['valid'] else 1)
