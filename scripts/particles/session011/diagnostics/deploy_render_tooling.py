"""Install only a validated renderer during an initialized production interval."""
import fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['SESSION_ROOT'])
sys.path.insert(0,str(root/'scripts'))
from runtime import active_job_state,atomic,digest,require_compute
require_compute()
qa_path=root/'evidence'/f'render_qa_{os.environ["RENDER_JOB"]}/RESULT.json'
qa=json.loads(qa_path.read_text());assert qa['status']=='pass'
visual_path=Path(qa['real_render_receipt'])
assert digest(visual_path)==qa['real_render_sha256']
visual=json.loads(visual_path.read_text())
assert visual['status']=='pass' and visual['evolution_advances']==0
assert visual['exact_initial_ID_subset'] and visual['exact_rest_weights'] and visual['render_did_not_mutate_records']
assert visual['source']==qa['source']
assert visual['renderer_sha256']==qa['tooling_sha256']['products.py']
close=next(v for v in visual['views'] if v['view']=='close')
assert close['accepted_horizons']==[0,1]
assert {v['horizon'] for v in close['surface_visibility']}=={0,1}
assert all(v['near_black_pixels']>0 for v in close['surface_visibility'])
data=subprocess.check_output(['git','-C',str(root/'code'),'show',qa['source']+':scripts/particles/session011/products.py'])
assert hashlib.sha256(data).hexdigest()==qa['tooling_sha256']['products.py']
out=root/'evidence'/f'render_deploy_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
def replace(path,content):
    temporary=path.with_suffix(path.suffix+'.render_tmp');temporary.write_bytes(content);os.replace(temporary,path)
with (root/'control/submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    submission=json.loads((root/'control/submission.json').read_text())
    run=root/'runs'/f'segment_{submission["segment"]:04d}'
    state=active_job_state(submission['job_id'])
    if state!='RUNNING' or not (run/'START_EPOCH').exists() or (run/'END_EPOCH').exists():
        atomic(out/'RESULT.json',dict(status='deferred',reason='need an active initialized production interval',submission=submission))
        sys.exit(75)
    with (run/'run.log').open('rb') as f:
        f.seek(max(0,(run/'run.log').stat().st_size-65536));tail=f.read()
    if b'cycle=' not in tail:
        atomic(out/'RESULT.json',dict(status='deferred',reason='production initialization still in progress'))
        sys.exit(75)
    gate_path=root/'control/VALIDATION_PASS.json';gate=json.loads(gate_path.read_text())
    saved=json.loads((root/'control/SAVED_AH_TOOLING.json').read_text())
    assert saved['status']=='pass' and saved['bitwise_particle_check'] and saved['evolution_advances']==0
    assert saved['production_executable_sha256']==gate['executable_sha256']==digest(gate['executable'])
    assert digest(saved['executable'])==saved['executable_sha256']
    for relative,expected in gate['runtime_scripts'].items():
        assert digest(root/relative)==expected,relative
    for name in ('endpoint.py','test_runtime.py','test_products.py'):
        assert digest(root/'scripts'/name)==qa['tooling_sha256'][name],name
    product_path=root/'scripts/products.py'
    before={gate_path:gate_path.read_bytes(),product_path:product_path.read_bytes()}
    for path,content in before.items():(out/('previous_'+path.name)).write_bytes(content)
    try:
        replace(product_path,data)
        gate['runtime_scripts']['scripts/products.py']=qa['tooling_sha256']['products.py']
        gate['render_tooling']=dict(source=qa['source'],validation_receipt=str(qa_path),
            visual_receipt=str(visual_path),deployment_job=os.environ['SLURM_JOB_ID'])
        atomic(gate_path,gate)
    except Exception:
        for path,content in before.items():replace(path,content)
        raise
    (out/'products.py').write_bytes(data)
    atomic(out/'RESULT.json',dict(status='pass',job_id=os.environ['SLURM_JOB_ID'],source=qa['source'],
        validation_receipt=str(qa_path),production_job=submission['job_id'],
        production_executable_unchanged=True,renderer_sha256=qa['tooling_sha256']['products.py'],
        installer_sha256=digest(Path(__file__))))
print((out/'RESULT.json').read_text())
