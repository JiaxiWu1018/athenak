"""Allocated staging of source, immutable inputs and executable, never evolved raw data."""
import os
import re
import shutil
import subprocess
from cfg import *

def ssh_python(code):
    subprocess.run(['ssh','-o','BatchMode=yes','hpcfund.amd.com','python3','-'],input=code,text=True,check=True,timeout=180)

def main():
    require_job();repo=ROOT/'src/athenak';pin=load(ROOT/'evidence/SOURCE_PIN.json')
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==pin['head']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    subprocess.run(['git','-C',str(repo),'diff','--exit-code',PHYSICS,'HEAD','--','src'],check=True)
    for name in ['samples','inputs','protected','reference','dependencies','evidence/stage']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    files={ROOT/'samples/model.npz':PREVIOUS/'samples/model.npz',
        ROOT/'evidence/model.json':PREVIOUS/'evidence/model.json',
        ROOT/'reference/proper.npz':PREVIOUS/'reference/proper.npz',
        ROOT/'reference/proper.json':PREVIOUS/'reference/proper.json',
        ROOT/'protected/athena':PREVIOUS/'followups/paired_scaling_20261017/protected/athena',
        ROOT/'evidence/ORIGINAL_SOURCE_PROVENANCE.json':PREVIOUS/'evidence/SOURCE_PROVENANCE.json'}
    for dest,src in files.items():
        if not dest.exists():shutil.copy2(src,dest)
        assert sha(dest)==sha(src)
    assert sha(ROOT/'protected/athena')==EXE_SHA
    template=(S05/'inputs/proper_mc_s1.athinput').read_text()
    for case in CASES:
        text=template.replace('basename = proper_mc_s1','basename = '+case['id'])
        text=text.replace('cluster_n = 1500000','cluster_n = '+str(case['n'])).replace('tlim = 300','tlim = 1000')
        text=re.sub(r'^cluster_sample_file = .*$',f"cluster_sample_file = {AMD}/samples/{case['id']}.bin",text,flags=re.M)
        target=ROOT/'inputs'/(case['id']+'.athinput')
        if target.exists():assert target.read_text()==text
        else:target.write_text(text)
    # Binary wheels for AMD Python3.9; install only in the allocated AMD job.
    subprocess.run(['python3','-m','pip','download','--only-binary=:all:','--no-deps',
        '--platform','manylinux2014_x86_64','--python-version','39','--implementation','cp','--abi','cp39',
        '--dest',str(ROOT/'dependencies'),'imageio-ffmpeg==0.6.0','pypdfium2==4.30.0'],check=True)
    ssh_python('from pathlib import Path;import json;r=Path('+repr(str(AMD))+');assert not r.exists();r.mkdir();[(r/n).mkdir(parents=True)for n in ["src/athenak","samples","inputs","protected","reference","dependencies","logs","evidence/jobs","evidence/preflight","evidence/samples","runs"]]')
    bundle=ROOT/'evidence/stage/source.tar'
    with bundle.open('wb')as f:subprocess.run(['git','-C',str(repo),'archive','HEAD'],stdout=f,check=True)
    subprocess.run(['rsync','-a',str(bundle),f'hpcfund.amd.com:{AMD}/source.tar'],check=True)
    subprocess.run(['ssh','hpcfund.amd.com','tar','-xf',str(AMD/'source.tar'),'-C',str(AMD/'src/athenak')],check=True)
    for folder in ['samples','inputs','protected','reference','dependencies']:
        subprocess.run(['rsync','-a',str(ROOT/folder)+'/',f'hpcfund.amd.com:{AMD}/{folder}/'],check=True)
    for name in ['SOURCE_PIN.json','model.json','ORIGINAL_SOURCE_PROVENANCE.json']:
        subprocess.run(['rsync','-a',str(ROOT/'evidence'/name),f'hpcfund.amd.com:{AMD}/evidence/{name}'],check=True)
    rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))for folder in ['samples','inputs','protected','reference','dependencies']for p in sorted((ROOT/folder).iterdir())if p.is_file()]
    ssh_python('import hashlib,json;from pathlib import Path;r=Path('+repr(str(AMD))+');rows=json.loads('+repr(json.dumps(rows))+');assert all((r/x["path"]).stat().st_size==x["bytes"] and hashlib.sha256((r/x["path"]).read_bytes()).hexdigest()==x["sha256"]for x in rows);print("AMD_STAGE_HASHES_OK")')
    ssh_python('import hashlib;from pathlib import Path;p=Path('+repr(str(AMD/'source.tar'))+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(sha(bundle)))
    atomic(ROOT/'evidence/STAGE_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=pin['head'],
        files=rows,source_tar_sha256=sha(bundle),all_destination_checksums_verified=True,native_data_policy='AMD only'))
    command=['sbatch','--parsable','--job-name=st_s07_prepare','--time=00:45:00',str(AMD/'src/athenak/scripts/reconstruction_s07_mc/amd.sbatch'),'prepare']
    atomic(ROOT/'evidence/PREPARE_SUBMIT_INTENT.json',dict(utc=now(),command=command))
    reply=subprocess.check_output(['ssh','hpcfund.amd.com',*command],text=True,timeout=90)
    job=reply.strip().split(';')[0];assert job.isdecimal()
    atomic(ROOT/'evidence/PREPARE_SUBMITTED.json',dict(id=job,host='amd',role='prepare',reply=reply,utc=now()))
    print('STAGE_OK PREPARE',job,flush=True)
if __name__=='__main__':main()
