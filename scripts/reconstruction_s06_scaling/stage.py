"""Allocated, OS-gated staging and unique AMD preparation/preflight submission."""
import os
import re
import shutil
import subprocess
from cfg import *
from os_gate import require_release

def submit(role, extra=()):
    intent=ROOT/'evidence'/('AMD_'+role.upper()+'_SUBMIT_INTENT.json')
    assert not intent.exists(), 'Unresolved submission intent: inspect scheduler before retrying'
    command=['sbatch','--parsable','--job-name=st_s06ps_'+role,*extra,
             str(AMD/'src/athenak/scripts/reconstruction_s06_scaling/amd.sbatch'),role]
    atomic(intent,dict(utc=now(),command=command,source=source()))
    reply=subprocess.check_output(['ssh','hpcfund.amd.com',*command],text=True,stderr=subprocess.STDOUT,timeout=90)
    ids=re.findall(r'^(\d+)(?:;[^\n]*)?$',reply,re.M)
    assert len(ids)==1,reply
    job=dict(id=ids[0],host='amd',role=role,command=command,reply=reply)
    atomic(ROOT/'evidence'/('AMD_'+role.upper()+'_SUBMITTED.json'),dict(utc=now(),**job))
    state=load(ROOT/'evidence/state.json');state['jobs'].append(job);state.update(updated_utc=now(),phase=role+'_submitted')
    atomic(ROOT/'evidence/state.json',state)
    return ids[0]

def main():
    require_job();gate=require_release(ROOT)
    pin=load(ROOT/'evidence/SOURCE_PIN.json')
    repo=ROOT/'src/athenak'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==pin['head']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    subprocess.run(['git','-C',str(repo),'diff','--exit-code',PHYSICS,'HEAD','--','src'],check=True)
    for folder in ['samples','inputs','protected','reference','baseline','evidence/stage']:
        (ROOT/folder).mkdir(exist_ok=True,parents=True)
    originals={ROOT/'samples/parent_1p5m.bin':BASE/'samples/proper_stratified_s1.bin',
        ROOT/'samples/model.npz':BASE/'samples/model.npz',
        ROOT/'reference/proper.npz':BASE/'reference/proper.npz',
        ROOT/'protected/athena':S05/'archive/amd_closeout_20261016/protected/athena'}
    for dest,src in originals.items():
        if not dest.exists():shutil.copy2(src,dest)
        assert sha(dest)==sha(src)
    assert sha(ROOT/'samples/parent_1p5m.bin')==PARENT_SHA and sha(ROOT/'protected/athena')==EXE_SHA
    for dest,src in [(ROOT/'baseline/reduced',BASE/'reduced'),
                     (ROOT/'baseline/movie_cache',BASE/'particle_movie_20261017')]:
        if not dest.exists():dest.symlink_to(src,target_is_directory=True)
    template=(S05/'inputs/proper_stratified_s1.athinput').read_text()
    for case in CASES:
        text=template.replace('basename = proper_stratified_s1','basename = '+case['id'])
        text=text.replace('cluster_n = 1500000','cluster_n = '+str(case['n'])).replace('tlim = 300','tlim = 1000')
        text=re.sub(r'^cluster_sample_file = .*$',f"cluster_sample_file = {AMD}/samples/{case['id']}.bin",text,flags=re.M)
        deck=ROOT/'inputs'/(case['id']+'.athinput')
        if deck.exists():assert deck.read_text()==text
        else:deck.write_text(text)
    # The new remote root is never a historical output destination.
    remote=('from pathlib import Path;import json;'
            f'r=Path({str(AMD)!r});p=r/"evidence/SOURCE_PIN.json";'
            f'assert not p.exists() or json.loads(p.read_text())["head"]=={pin["head"]!r};'
            'r.mkdir(exist_ok=True);'
            '[(r/n).mkdir(exist_ok=True,parents=True)for n in ["src/athenak","samples","inputs","protected","reference","logs","evidence/jobs","evidence/preflight","evidence/samples","runs"]]')
    subprocess.run(['ssh','hpcfund.amd.com','python3','-'],input=remote,text=True,check=True,timeout=60)
    bundle=ROOT/'evidence/stage/source.tar'
    with bundle.open('wb') as stream:
        subprocess.run(['git','-C',str(repo),'archive','HEAD'],stdout=stream,check=True)
    subprocess.run(['rsync','-a',str(bundle),'hpcfund.amd.com:'+str(AMD/'source.tar')],check=True)
    subprocess.run(['ssh','hpcfund.amd.com','tar','-xf',str(AMD/'source.tar'),'-C',str(AMD/'src/athenak')],check=True)
    for name in ['samples','inputs','protected','reference']:
        subprocess.run(['rsync','-a',str(ROOT/name)+'/',f'hpcfund.amd.com:{AMD}/{name}/'],check=True)
    for name in ['SOURCE_PIN.json','OS_PRIORITY_RELEASE.json']:
        subprocess.run(['rsync','-a',str(ROOT/'evidence'/name),'hpcfund.amd.com:'+str(AMD/'evidence'/name)],check=True)
    files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
           for name in ['samples','inputs','protected','reference'] for p in sorted((ROOT/name).iterdir()) if p.is_file()]
    remote_verify='import json,hashlib;from pathlib import Path;r=Path('+repr(str(AMD))+');rows=json.loads('+repr(json.dumps(files))+');assert all((r/x["path"]).stat().st_size==x["bytes"] and hashlib.sha256((r/x["path"]).read_bytes()).hexdigest()==x["sha256"]for x in rows);print("REMOTE_STAGE_HASHES_OK")'
    subprocess.run(['ssh','hpcfund.amd.com','python3','-'],input=remote_verify,text=True,check=True,timeout=180)
    atomic(ROOT/'evidence/STAGE_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],os_gate=gate,
           source=pin['head'],source_tar_sha256=sha(bundle),files=files,remote_hashes_verified=True,
           source_snapshot='Independent Perseus Git clone; pinned runtime source archive on AMD; no Mac transfer'))
    prepare=submit('prepare',['--time=01:00:00'])
    submit('preflight',['--array=0-2%1','--dependency=afterok:'+prepare,'--time=01:30:00'])
    print('STAGE_AND_SUBMISSION_OK',flush=True)

if __name__=='__main__':
    main()
