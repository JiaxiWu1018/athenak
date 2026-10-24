"""Preserve interrupted output bytes and resume only a fully verified prefix."""
import re
import shutil
import time
from common import *
from checkpoint import validate

def output_time(path):
    with path.open('rb') as f:
        if path.suffix=='.vtk':
            f.readline();return float(re.search(rb'time=\s*([-+0-9.eE]+)',f.readline())[1])
        if path.suffix=='.s8c':return float(f.readline().split()[1])
        assert path.suffix=='.bin' and f.readline().strip()==b'Athena binary output version=1.1'
        count=int(f.readline().split(b'=')[-1]);header={}
        for _ in range(count-1):
            key,value=f.readline().decode().split('=',1);header[key.strip()]=value.strip()
        return float(header['time'])

def recover(c,work):
    """Caller holds the case-wide allocation lock. No original bytes are deleted."""
    state=load(work/'state.json')
    accepted_logs={(str(r.get('job')),Path(r['log']).name) for r in state['segments'] if r.get('log')}
    launches=[p for p in work.glob('*.launch.json') if (str(load(p).get('job')),Path(load(p)['log']).name) not in accepted_logs]
    valid=[];invalid=[]
    for path in sorted((work/'rst').glob('*.rst')):
        try:
            d=validate(path);d.pop('parameters');valid.append(d)
        except Exception as error:
            invalid.append(dict(file=receipt(path),error=repr(error)))
    assert valid,'No complete restart remains'
    latest=max(valid,key=lambda r:(r['cycle'],r['path']));cut=latest['time'];cycle=latest['cycle']
    assert cut>=state['latest']['time']-1e-8
    ledger=work/f"{c['id']}.mass.csv";tail=False
    if ledger.exists():
        from collections import deque
        last=deque(ledger.open(),maxlen=1)[0]
        if not last.startswith('time'):tail=float(last.split(',')[0])>cut+1e-8
    if not launches and not invalid and not tail and latest['sha256']==state['latest']['sha256']:return state
    archive=ROOT/'interrupted'/f'{job()}_{time.time_ns()}';archive.mkdir(parents=True)
    selected=[];texts=[];attempts=[]
    for name in ['pvtk','cohort','bin']:
        selected += [p for p in (work/name).glob('*') if p.is_file() and output_time(p)>cut+1e-8]
    for p in work.iterdir():
        if p.is_file() and (p.suffix=='.hst' or p.suffix=='.csv'):texts.append(p)
    for launch in launches:
        r=load(launch);assert r['job']!=job(),'Current native child still owns this attempt'
        log=Path(r['log']);text=log.read_text(errors='replace') if log.exists() else ''
        assert '### FATAL' not in text and not re.search(r'search_fail=[1-9]',text), 'Native failure requires diagnosis before recovery'
        record=subprocess.run(['scontrol','show','job',str(r['job'])],capture_output=True,text=True)
        if record.returncode==0:
            assert not re.search(r'JobState=(RUNNING|COMPLETING|PENDING)',record.stdout),record.stdout
        attempts.append(dict(launch=r,scheduler_stdout=record.stdout,scheduler_stderr=record.stderr,
          scheduler_record_available=record.returncode==0,log=receipt(log) if log.exists() else None))
        selected.append(launch)
        result=launch.with_name(launch.name.replace('.launch.','.result.'))
        if result.exists():selected.append(result)
        if log.exists():selected.append(log)
    selected += [Path(d['file']['path']) for d in invalid]
    files=[receipt(p) for p in sorted(set(selected+texts))]
    atomic(archive/'BEFORE_MUTATION.json',dict(utc=now(),job=job(),checkpoint=latest,
      original_state=state,attempts=attempts,invalid_checkpoints=invalid,files=files,
      policy='Every moved original byte remains in this archive. Canonical text files are explicitly filtered derivatives.'))
    for p in set(selected):
        relative=p.relative_to(work);dest=archive/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        p.rename(dest)
    derivatives=[]
    for p in texts:
        original=archive/p.name;p.rename(original);keep=[]
        for line in original.read_text().splitlines(keepends=True):
            if not line.strip() or line.startswith(('#','time,','cycle,')):keep.append(line);continue
            values=line.strip().split(',') if ',' in line else line.split()
            accept=(int(values[0])<cycle and float(values[1])<=cut+1e-8) if p.name.endswith('prtcl_destroy.csv') else float(values[0])<=cut+1e-8
            if accept:keep.append(line)
        p.write_text(''.join(keep));derivatives.append(dict(original=str(original),derived=receipt(p)))
    # Accepted logs contain only records through the verified cycle. The complete
    # original remains outside the scientific work tree and all later audits.
    for attempt in attempts:
        r=attempt['launch'];p=Path(r['log']);original=archive/p.name
        if not original.exists():continue
        lines=[];past=False
        for line in original.read_text(errors='replace').splitlines(keepends=True):
            match=re.search(r'cycle(?:=|\s+)(\d+)',line)
            if match:
                n=int(match[1]);past=past or n>cycle
                if n<=cycle:lines.append(line)
            elif not past:lines.append(line)
        accepted=work/(p.stem+'_recovered_prefix.log');accepted.write_text(''.join(lines))
        r['log']=str(accepted)
        derivatives.append(dict(original=str(original),derived=receipt(accepted)))
    advanced=cut-state['latest']['time']
    if advanced>1e-8:
        r=attempts[-1]['launch'] if attempts else dict(job='unregistered_interrupted_checkpoint')
        r.update(checkpoint=latest,complete=False,accepted_partial_prefix=True,
          simulation_advanced=advanced,recovery_archive=str(archive),native_node_hours_per_M=None)
        state['segments'].append(r)
    state.update(latest=latest,complete=cut>=c['target']-1e-8 or (latest['ah_onset']>=0 and cut>=latest['ah_onset']+20-1e-8))
    atomic(work/'state.json',state)
    # Keep prior receipts, including authorized rotations, and register each valid
    # restart by its actual bytes. No invalid partial file can be promoted later.
    index=load(work/'checkpoints.json') if (work/'checkpoints.json').exists() else []
    bypath={v['path']:v for v in index};bypath.update({v['path']:v for v in valid})
    atomic(work/'checkpoints.json',sorted(bypath.values(),key=lambda r:(r['cycle'],r['path'])))
    for entry in files:
        original=archive/Path(entry['path']).relative_to(work)
        assert original.exists() and sha(original)==entry['sha256'],'Archive verification failed'
    atomic(archive/'RECOVERY_OK.json',dict(utc=now(),job=job(),checkpoint=latest,
      original_bytes_preserved=True,derivatives=derivatives,canonical_endpoint_M=cut))
    return state
