"""Extension-only allocated-node admission and robust whole-user watchdog."""
import time
from common import *
EXT=Path(os.environ['ST8X_ROOT'])
USERROOT=Path('/work1/eliasmost/jiaxiwu')

def storage():
    def du(p):return int(subprocess.check_output(['du','-s','-B1',str(p)],text=True,timeout=180).split()[0])
    whole=du(USERROOT); own=du(EXT)
    return dict(whole=whole,extension=own,forecast=whole+max(0,200*GIB-own))

def accounting():
    records=load(EXT/'evidence/jobs.json') if (EXT/'evidence/jobs.json').exists() else []
    ids={str(x['id']) for x in records if x.get('host')=='amd'};ids.add(job())
    lines=subprocess.check_output(['sacct','-X','-j',','.join(sorted(ids)),'-n','-P','--format=JobIDRaw,State,ElapsedRaw,AllocNodes'],text=True,timeout=40)
    rows=[r.split('|') for r in lines.splitlines() if r.strip()]
    assert job() in {r[0] for r in rows}
    return sum(int(r[2] or 0)*int(r[3] or 0) for r in rows)/3600

def admission():
    job();d=storage();d.update(utc=now(),job=job(),node_hours=accounting())
    atomic(ROOT/'evidence/jobs'/job()/'admission.json',d)
    assert d['whole']<1.7*TIB and d['extension']<199*GIB and d['forecast']<1.9*TIB,d
    assert d['node_hours']<39,d
    return d

def stop(reason):
    atomic(ROOT/'evidence/RESOURCE_STOP.json',dict(utc=now(),job=job(),reason=reason))
    a=ROOT/'evidence/active.json'
    if a.exists():
        w=Path(load(a)['work'])
        assert ROOT in w.parents
        (w/'REQUEST_STOP').write_text(reason+'\n')

def watchdog():
    jid=job();out=ROOT/'evidence/jobs'/jid;out.mkdir(parents=True,exist_ok=True)
    last_disk=last_mem=last_acct=time.monotonic(); next_disk=next_acct=0
    while True:
        t=time.monotonic();d=dict(utc=now(),job=jid)
        if t>=next_disk:
            try:
                q=storage();d.update(q);last_disk=t
                d['storage_warning']=q['whole']>=1.5*TIB
                if q['whole']>=1.7*TIB or q['extension']>=199*GIB or q['forecast']>=1.9*TIB:stop('extension_storage_review')
            except Exception as e:d['storage_error']=repr(e)
            next_disk=t+300
        if t-last_disk>900:stop('storage_telemetry_stale')
        try:
            nodes=int(os.environ.get('SLURM_NNODES','1'))
            if nodes==1:
                gpu=json.loads(subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json'],text=True,timeout=30))
            else:
                probe="import json,socket,subprocess;d=json.loads(subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json']));print(json.dumps({socket.gethostname()+':'+k:v for k,v in d.items()}))"
                raw=subprocess.check_output(['srun','--overlap','-N',str(nodes),'-n',str(nodes),'--ntasks-per-node=1','python3','-c',probe],text=True,timeout=45)
                gpu={}
                for line in raw.splitlines():gpu.update(json.loads(line))
                assert len(gpu)>=nodes
            ratios=[]
            for f in gpu.values():
                if not isinstance(f,dict):continue
                totals=[int(v) for k,v in f.items() if 'Total Memory' in k]
                used=[int(v) for k,v in f.items() if 'Used Memory' in k]
                if totals and used:ratios.append(used[0]/totals[0])
            assert ratios
            d['gpu_memory']=gpu;d['vram_max_fraction']=max(ratios);last_mem=t
            if max(ratios)>=.85:stop('gpu_memory_85_percent')
        except Exception as e:d['memory_error']=repr(e)
        if t-last_mem>300:stop('memory_telemetry_stale')
        if t>=next_acct:
            try:
                d['node_hours']=accounting();last_acct=t
                if d['node_hours']>=39:stop('extension_node_hour_review')
            except Exception as e:d['accounting_error']=repr(e)
            next_acct=t+60
        if t-last_acct>300:stop('accounting_telemetry_stale')
        atomic(out/'LATEST.json',d)
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        time.sleep(10)
if __name__=='__main__':watchdog()
