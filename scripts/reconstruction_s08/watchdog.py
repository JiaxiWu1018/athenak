"""Allocated-node storage/memory/accounting telemetry; stop only Session 8 work."""
import time
from common import *

def stop(reason):
    atomic(ROOT/'evidence/RESOURCE_STOP.json',dict(utc=now(),job=job(),reason=reason))
    active=ROOT/'evidence/active.json'
    if active.exists():
        work=Path(load(active)['work'])
        if ROOT in work.parents:
            (work/'REQUEST_STOP').write_text(reason+'\n')

def main():
    jid=job();out=ROOT/'evidence/jobs'/jid;out.mkdir(parents=True,exist_ok=True)
    last_storage=time.monotonic();next_storage=0;last_memory=time.monotonic()
    while True:
        sample=dict(utc=now(),job=jid)
        if time.monotonic()>=next_storage:
            try:
                def du(p):return int(subprocess.check_output(['du','-s','-B1',str(p)],text=True,timeout=180).split()[0])
                whole=du(ROOT.parent);own=du(ROOT)
                sample.update(whole_user_bytes=whole,session_bytes=own)
                last_storage=time.monotonic()
                if whole>=1.7*TIB or own>=350*GIB:stop('storage_limit')
                if whole>=1.5*TIB:sample['storage_warning']=True
            except Exception as e:sample['storage_error']=repr(e)
            next_storage=time.monotonic()+300
        if time.monotonic()-last_storage>900:stop('storage_telemetry_stale')
        try:
            result=subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json'],text=True,timeout=30)
            gpu=json.loads(result);sample['gpu_memory']=gpu
            ratios=[]
            for card,fields in gpu.items():
                if isinstance(fields,dict):
                    totals=[int(v) for k,v in fields.items() if 'Total Memory' in k]
                    used=[int(v) for k,v in fields.items() if 'Used Memory' in k]
                    if totals and used:ratios.append(used[0]/totals[0])
            if not ratios:raise RuntimeError('VRAM telemetry not parsed')
            last_memory=time.monotonic();sample['vram_max_fraction']=max(ratios)
            if max(ratios)>=.80:sample['memory_warning']=True
            if max(ratios)>=.85:stop('gpu_memory_85_percent')
        except Exception as e:sample['memory_error']=repr(e)
        if time.monotonic()-last_memory>300:stop('memory_telemetry_stale')
        try:
            jobs=load(ROOT/'evidence/jobs.json') if (ROOT/'evidence/jobs.json').exists() else []
            ids=set(str(x['id']) for x in jobs);ids.add(jid)
            result=subprocess.check_output(['sacct','-X','-j',','.join(sorted(ids)),'-n','-P','--format=JobIDRaw,State,ElapsedRaw,AllocNodes'],text=True,timeout=30)
            rows=[x.split('|') for x in result.splitlines() if x.strip()]
            hours=sum(int(x[2] or 0)*int(x[3] or 0) for x in rows)/3600
            sample['node_hours']=hours
            if hours>=60:sample['cost_review_due']=True
            # Finalization reserve: stop with at least half a node-hour left.
            if hours>=99.5:stop('node_hour_cap_reserve')
        except Exception as e:sample['accounting_error']=repr(e)
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
        atomic(out/'LATEST.json',sample)
        time.sleep(30)
if __name__=='__main__':main()
