"""Evidence-gated audit repair and continuation; never edits a saved physical state."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path('/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021')
fix=root/'evidence/nonfinite_20261021'
assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
source=os.environ['TOOLING_SOURCE']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
metric=json.loads((fix/'metric_473909.json').read_text())
assert metric['evolution_steps']==0 and not any(metric['grid_nonfinite_by_variable'])
a=metric['summary']['invalid'];assert a['high']['nonpositive_metric']==330
assert a['high']['negative_discriminant']==330 and a['high']['nonfinite_nodes']==0
assert a['linear']['nonpositive_metric']==a['linear']['negative_discriminant']==a['linear']['nonfinite_nodes']==0
assert a['linear']['min_alpha']>0
qa_path=fix/f'qa_{os.environ["QA_JOB"]}/RESULT.json'
qa=json.loads(qa_path.read_text())
assert qa['status']=='pass' and qa['evolution_steps']==0
assert qa['audit']['exact_partition'] and qa['audit']['cycle']==1902
assert qa['audit']['native_energy_quality']['unknown_count']==330
changed=('runtime.py','products.py','test_runtime.py','test_products.py')
for name in changed:
    assert sha(fix/'tooling'/name)==qa['tooling_sha256'][name]
    assert (root/'code/scripts/particles/session011'/name).read_bytes()==(fix/'tooling'/name).read_bytes()
assert subprocess.check_output(['git','-C',str(root/'code'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(root/'code'),'status','--porcelain'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(root/'code'),'diff','df6ab41d','HEAD','--','src','CMakeLists.txt','cmake'],text=True).strip()
assert not subprocess.check_output(['squeue','-h','-j','473355','-o','%T'],text=True).strip()
assert json.loads((root/'control/submission.json').read_text())['segment']==3
assert json.loads((root/'control/latest_checkpoint.json').read_text())['cycle']==1230
assert not (root/'control/PAUSED.json').exists()
run=root/'runs/segment_0003';assert (run/'EXIT_CODE').read_text().strip()=='0'
assert not (run/'population_audit.json').exists()
gate=json.loads((root/'control/VALIDATION_PASS.json').read_text())
assert sha(Path(gate['executable']))==gate['executable_sha256']
assert sha(root/'inputs/production.athinput')==gate['input_sha256']
for relative,digest in gate['runtime_scripts'].items():assert sha(root/relative)==digest,relative
out=fix/f'release_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
for name in ('VALIDATION_PASS.json','latest_checkpoint.json','submission.json','dead_tags.npy','checkpoint_inventory.json'):
    shutil.copy2(root/'control'/name,out/name)
for name in changed:
    shutil.copy2(root/'scripts'/name,out/('previous_'+name))
# Preserve original seal, engine, exact input and state; replace only tested Python tooling.
for name in changed:
    tmp=root/'scripts'/(name+'.repair_tmp');shutil.copy2(fix/'tooling'/name,tmp);os.replace(tmp,root/'scripts'/name)
    gate['runtime_scripts']['scripts/'+name]=sha(root/'scripts'/name)
gate['tooling_source']=source
gate['diagnostic_energy_repair']=dict(job=os.environ['SLURM_JOB_ID'],epoch=time.time(),
    previous_gate=str(out/'VALIDATION_PASS.json'),metric_evidence=str(fix/'metric_473909.json'),
    quality_validation=str(qa_path),engine_unchanged=True,
    checkpoint_mutation=False,policy='unknown native IPEN retained; all physical fields, partition, rest weights and Jz remain required')
sys.path.insert(0,str(root/'scripts'))
import runtime
runtime.atomic(root/'control/VALIDATION_PASS.json',gate)
try:
    assert runtime.finish(root,run,0)
    latest=json.loads((root/'control/latest_checkpoint.json').read_text())
    assert latest['cycle']==1902 and abs(latest['time']-8.803125)<1.e-9
    runtime.atomic(out/'RESULT.json',dict(status='audit_pass',job_id=os.environ['SLURM_JOB_ID'],
                   source=source,checkpoint=latest,evolution_steps=0))
    runtime.submit(root)
    runtime.atomic(out/'submission_after_release.json',json.loads((root/'control/submission.json').read_text()))
except Exception as exc:
    runtime.atomic(root/'control/PAUSED.json',dict(kind='review',reason=f'energy repair: {exc}',job=os.environ['SLURM_JOB_ID']))
    raise
print((root/'control/submission.json').read_text())
