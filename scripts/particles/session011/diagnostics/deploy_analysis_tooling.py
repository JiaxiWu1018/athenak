"""Install validated endpoint-only tooling between launches under submit.lock."""
import fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['SESSION_ROOT'])
sys.path.insert(0,str(root/'scripts'))
from runtime import active_job_state,atomic,digest,require_compute
require_compute()
qa_path=root/'evidence'/f'analysis_saved_{os.environ["ANALYSIS_JOB"]}/RESULT.json'
qa=json.loads(qa_path.read_text())
assert qa['status']=='pass' and qa['evolution_advances']==0 and qa['bitwise_particle_check']
assert qa['unknown_native_energy']>0 and qa['accepted']['published']=='1'
assert digest(qa['executable'])==qa['executable_sha256']
files={}
for name,expected in qa['tooling_sha256'].items():
    assert name in ('endpoint.py','products.py','test_runtime.py','test_products.py')
    data=subprocess.check_output(['git','-C',str(root/'code'),'show',qa['source']+':scripts/particles/session011/'+name])
    assert hashlib.sha256(data).hexdigest()==expected
    files[name]=data
assert len(files)==4
out=root/'evidence'/f'analysis_deploy_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
def replace(path,data):
    tmp=path.with_suffix(path.suffix+'.analysis_tmp');tmp.write_bytes(data);os.replace(tmp,path)
with (root/'control/submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    submission=json.loads((root/'control/submission.json').read_text())
    state=active_job_state(submission['job_id'])
    run=root/'runs'/f'segment_{submission["segment"]:04d}'
    if state!='RUNNING' or not (run/'START_EPOCH').exists() or (run/'END_EPOCH').exists():
        atomic(out/'RESULT.json',dict(status='deferred',reason='need an active initialized production interval',job_id=os.environ['SLURM_JOB_ID'],submission=submission))
        sys.exit(75)
    with (run/'run.log').open('rb') as f:
        f.seek(max(0,(run/'run.log').stat().st_size-65536));tail=f.read()
    if b'cycle=' not in tail:
        atomic(out/'RESULT.json',dict(status='deferred',reason='production initialization still in progress',job_id=os.environ['SLURM_JOB_ID']))
        sys.exit(75)
    gate_path=root/'control/VALIDATION_PASS.json';gate=json.loads(gate_path.read_text())
    assert gate['executable_sha256']==qa['production_executable_sha256']==digest(gate['executable'])
    for relative,expected in gate['runtime_scripts'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
    saved=root/'control/SAVED_AH_TOOLING.json'
    before={gate_path:gate_path.read_bytes(),saved:saved.read_bytes() if saved.exists() else None}
    before.update({root/'scripts'/name:(root/'scripts'/name).read_bytes() for name in files})
    for path,data in before.items():
        if data is not None:(out/('previous_'+path.name)).write_bytes(data)
    try:
        atomic(saved,dict(qa,validation_receipt=str(qa_path)))
        for name,data in files.items():
            replace(root/'scripts'/name,data)
            (out/name).write_bytes(data)
            gate['runtime_scripts']['scripts/'+name]=hashlib.sha256(data).hexdigest()
        gate['endpoint_analysis_tooling']=dict(source=qa['source'],receipt=str(saved),
             validation_receipt=str(qa_path),deployment_job=os.environ['SLURM_JOB_ID'])
        atomic(gate_path,gate)
    except Exception:
        for path,data in before.items():
            if data is None:
                if path.exists():path.unlink()
            else:replace(path,data)
        raise
    atomic(out/'RESULT.json',dict(status='pass',job_id=os.environ['SLURM_JOB_ID'],
          analysis_validation=str(qa_path),production_job=submission['job_id'],
          production_executable_unchanged=True,source=qa['source'],tooling_sha256=qa['tooling_sha256'],
          installer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print((out/'RESULT.json').read_text())
