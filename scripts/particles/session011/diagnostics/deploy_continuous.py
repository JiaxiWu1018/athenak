"""Release verified long-allocation scripts without touching a live application."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
root=Path(os.environ['SESSION_ROOT'])
sys.path.insert(0,str(root/'scripts'))
from runtime import active_job_state,atomic,digest,require_compute
require_compute()
qa_path=root/'evidence'/f'continuous_qa_{os.environ["CONTINUOUS_QA_JOB"]}/RESULT.json'
qa=json.loads(qa_path.read_text())
assert qa['status']=='pass' and qa['native_parameter_check'] and qa['evolution_advances']==0
out=root/'evidence'/f'continuous_deploy_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
files={}
for name,expected in qa['tooling_sha256'].items():
    assert name in ('runtime.py','amd_segment.sbatch','continuous.py','test_continuous.py')
    data=subprocess.check_output(['git','-C',str(root/'code'),'show',qa['source']+':scripts/particles/session011/'+name])
    assert hashlib.sha256(data).hexdigest()==expected
    files[name]=data
assert len(files)==4
# Exercise the new live audit against the current, fully audited native state.
# No retention/deletion is performed during this software-validation check.
source=root/'evidence'/f'continuous_{qa["source"]}/source/scripts/particles/session011'
sys.path.insert(0,str(source))
import importlib.util
spec=importlib.util.spec_from_file_location('continuous_runtime_validation',source/'runtime.py')
new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
checkpoint=json.loads((root/'control/latest_checkpoint.json').read_text())
submission=json.loads((root/'control/submission.json').read_text())
run=root/'runs'/f'segment_{submission["segment"]:04d}'
before=digest(root/'control/dead_tags.npy')
audit=new.audit_population(root,run,[checkpoint],checkpoint,publish=False,require_online=False)
assert audit['exact_partition'] and digest(root/'control/dead_tags.npy')==before
assert not (run/'REQUEST_STOP').exists()
atomic(out/'live_audit_validation.json',dict(status='pass',checkpoint=checkpoint,
      population=audit,boundary_ledger_unchanged=True,evolution_interrupted=False))
def replace(path,data):
    temp=path.with_suffix(path.suffix+'.continuous_tmp');temp.write_bytes(data);os.replace(temp,path)
with (root/'control/submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    current=json.loads((root/'control/submission.json').read_text())
    assert current==submission,'submission changed during validation; defer'
    assert active_job_state(current['job_id'])=='RUNNING'
    assert (run/'START_EPOCH').exists() and not (run/'END_EPOCH').exists()
    gate_path=root/'control/VALIDATION_PASS.json';gate=json.loads(gate_path.read_text())
    assert digest(gate['executable'])==qa['production_executable_sha256']==gate['executable_sha256']
    for relative,expected in gate['runtime_scripts'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
    paths=[gate_path]+[root/'scripts'/name for name in files]
    before_files={p:p.read_bytes() if p.exists() else None for p in paths}
    for p,data in before_files.items():
        if data is not None:(out/('previous_'+p.name)).write_bytes(data)
    try:
        for name,data in files.items():
            replace(root/'scripts'/name,data)
            gate['runtime_scripts']['scripts/'+name]=hashlib.sha256(data).hexdigest()
        gate['allocation_policy']=dict(reservation_hours=24,application_chunk_hours=23+20/60,
             checkpoint_interval_Mref=1,finalization_reserve_minutes=40,
             maximum_enforced_by='AMD sbatch submit filter; partition advertises96h',
             source=qa['source'],qa=str(qa_path),deployment=str(out),
             live_checkpoint_analysis_does_not_stop_evolution=True)
        atomic(gate_path,gate)
    except Exception:
        for p,data in before_files.items():
            if data is None:
                if p.exists():p.unlink()
            else:replace(p,data)
        raise
    atomic(out/'RESULT.json',dict(status='pass',source=qa['source'],qa=str(qa_path),
          deployment_job=os.environ['SLURM_JOB_ID'],production_job=current['job_id'],
          current_application_unchanged=True,production_executable_unchanged=True,
          tooling_sha256=qa['tooling_sha256'],allocation_policy=gate['allocation_policy'],
          live_saved_audit='pass',live_boundary_ledger_unchanged=True))
print((out/'RESULT.json').read_text())
