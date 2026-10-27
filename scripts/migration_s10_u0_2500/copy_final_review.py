"""One-way thin review copy; successful copying never grants visual acceptance."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

LOCAL = Path('/data/jiaxiwu/NRPIC/ST_migration/session_10_yc0500_outward_20261024/extensions/u0_1500_to2500_20261026')
MAX_FILE = 64*1024**2
MAX_TOTAL = 128*1024**2

PROGRAM = r'''
from pathlib import Path
import json,hashlib,subprocess,sys,tarfile
r=Path('/work1/eliasmost/jiaxiwu/st_migration_s10_u0_2500_20261026')
def load(p):return json.loads(p.read_text())
def receipt(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
ready=load(r/'evidence/FINAL_PRODUCTS_READY.json');done=load(r/'evidence/PIPELINE_DONE.json')
assert done['products']==ready and receipt(Path(ready['generation']['path']))==ready['generation']
g=load(Path(ready['generation']['path']));delivery=Path(ready['delivery']);assert delivery.parent==r/'delivery'
state=load(r/'yc0500_control/runs/yc0500/state.json');cp=state['latest'];assert state['complete'] and cp==g['checkpoint']
t=cp['time'];assert t<=2500+1e-7 and (t>=2500-1e-7 or (cp['ah_onset']>=0 and t>=cp['ah_onset']+20-1e-7))
acceptance=load(Path(g['numerical_acceptance']['path']));assert acceptance['numerical_checks_passed'] and acceptance['checkpoint']==cp
a=r/'evidence/allocations'/ready['job'];assert load(a/'ALLOCATION_END.json')['exit_status']==0 and (a/'WORKER_OK.json').exists()
terminal=a/'scheduler_terminal.json'
if terminal.exists():scheduler=load(terminal)['scheduler']
else:scheduler=subprocess.check_output(['scontrol','show','job',ready['job'],'-o'],text=True)
assert 'JobState=COMPLETED' in scheduler and 'ExitCode=0:0' in scheduler
selected=[*g['plots'],*g['tables'],g['central_lapse_sources'],g['report'],g['numerical_acceptance'],ready['generation']]
for key in ['report_pdf','pdf_layout']:
 if key in g:selected.append(g[key])
selected+=g.get('pdf_previews',[])
for p in [r/'evidence/FINAL_PRODUCTS_READY.json',r/'evidence/PIPELINE_DONE.json',r/'evidence/SOURCE_PIN.json',r/'evidence/job_registry.json',r/'APPROVED_PLAN.md',r/'yc0500_control/inputs/yc0500.athinput',r/'yc0500_control/runs/yc0500/FINAL_FIELDS_OK.json',a/'STARTED.json',a/'ALLOCATION_END.json',a/'WORKER_OK.json']:
 assert p.is_file() and p.stat().st_size<1024**2;selected.append(receipt(p))
entries={}
for item in selected:
 p=Path(item['path']);assert r in p.parents and p.resolve().is_relative_to(r.resolve())
 assert p.suffix in ['.png','.pdf','.csv','.md','.json','.athinput'] and p.is_file() and p.stat().st_size==item['bytes'] and item['bytes']<=64*1024**2
 name=p.name if p.parent==delivery else 'provenance/'+str(p.relative_to(r))
 assert name not in entries or entries[name]==item;entries[name]=item
assert len(entries)<=100 and sum(x['bytes'] for x in entries.values())<=128*1024**2
manifest=dict(job=ready['job'],delivery=str(delivery),endpoint_M=t,checkpoint=cp,files=entries,actual_visual_review_pending=True)
print(json.dumps(manifest),flush=True)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,item in entries.items():archive.add(item['path'],arcname=name,recursive=False)
'''


def main():
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               'hpcfund.amd.com', 'python3 -c '+shlex.quote(PROGRAM)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        header = proc.stdout.readline(1024*1024)
        assert header, 'No completed endpoint delivery is available'
        manifest = json.loads(header)
        files = manifest['files']
        assert 1 <= len(files) <= 100 and sum(x['bytes'] for x in files.values()) <= MAX_TOTAL
        target = LOCAL/'review_mirror'/Path(manifest['delivery']).name
        target.mkdir(parents=True, exist_ok=True)
        seen = set()
        copied = []
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                relative = Path(member.name)
                assert not relative.is_absolute() and '..' not in relative.parts
                assert member.isfile() and member.name in files and member.name not in seen
                item = files[member.name]
                assert 0 <= member.size == item['bytes'] <= MAX_FILE
                stream = archive.extractfile(member)
                destination = target/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix+'.copying')
                digest = hashlib.sha256()
                count = 0
                with temporary.open('wb') as out:
                    while True:
                        block = stream.read(1024**2)
                        if not block:break
                        count += len(block)
                        digest.update(block)
                        out.write(block)
                assert count == item['bytes'] and digest.hexdigest() == item['sha256']
                if destination.exists():
                    assert hashlib.sha256(destination.read_bytes()).hexdigest() == item['sha256'], 'Historical review file differs'
                    temporary.unlink()  # Only the just-written redundant temporary review copy.
                else:
                    temporary.replace(destination)
                copied.append(dict(remote=item, local=str(destination)))
                seen.add(member.name)
        assert seen == set(files)
        assert proc.wait(timeout=30) == 0, proc.stderr.read().decode()
        record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            endpoint_M=manifest['endpoint_M'],checkpoint=manifest['checkpoint'],
            source_delivery=manifest['delivery'],files=copied,
            SHA_verified=True,one_way=True,actual_visual_review_pending=True)
        path = target/'COPY_VERIFIED.json'
        if path.exists():
            previous=json.loads(path.read_text())
            assert previous['files']==record['files'] and previous['checkpoint']==record['checkpoint']
        else:path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(review_copy=str(target),files_SHA_verified=len(copied),
            bytes=sum(v['bytes'] for v in files.values()),actual_visual_review_pending=True)))
    finally:
        if proc.poll() is None:proc.terminate()


if __name__ == '__main__':
    main()
