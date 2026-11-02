"""Audit and accept an already completed segment after an output-path validator defect."""
from pathlib import Path
import json
import os
import shutil
import numpy as np
from campaign import ROOT,sample_array,validate_endpoint,resource_snapshot
from state_io import require_job,read_rst,final_rst,write_json,sha
require_job()
case='legacy_1m_h020';work=ROOT/'runs'/case/'to_020_job495241'
assert not (work/'COMPLETE.json').exists()
launch=json.loads((work/'LAUNCH.json').read_text());status=json.loads((work/'EXIT.json').read_text())
assert status['exit_code']==0 and not status['resource_stop']
log=(work/'run.log').read_text();assert 'Terminating on time limit' in log and 'FATAL' not in log
sample_path=ROOT/'reference/session7/samples/mc_1m_s1.bin'
assert sha(sample_path)==launch['sample_sha256']
sample=sample_array(sample_path)
initial_file=next(p for p in (work/'rst').glob('*.rst') if abs(read_rst(p,validate=False)['time'])<1e-14)
s=read_rst(initial_file);p=s['particles'][:,s['order']]
assert len(sample)==s['count'] and np.array_equal(p[9],sample[:,0])
assert np.array_equal(p[0],sample[:,7])
assert np.array_equal(p[[2,4,6,3,5,7]].T,sample[:,1:7])
initial=dict(exact_sample_match=True,count=len(sample),method='native t=0 double-precision checkpoint',
             path=str(initial_file),sha256=sha(initial_file),note='Original relative pgen dump escaped run directory; checkpoint validates actual initialized state independently')
endpoint=validate_endpoint(final_rst(work),sample,20)
# Preserve the extant accidental shared file; do not delete or alter it.
shared=Path('/home1/jiaxiwu/initial_particles.bin')
retained=ROOT/f'evidence/initial_dump_path_incident_{os.environ["SLURM_JOB_ID"]}'
retained.mkdir()
if shared.exists():
    record=dict(path=str(shared),bytes=shared.stat().st_size,mtime_ns=shared.stat().st_mtime_ns,sha256=sha(shared))
    copy=retained/'shared_initial_particles.bin';shutil.copy2(shared,copy)
    assert sha(copy)==record['sha256']
    record.update(preserved_copy=str(copy),original_retained=True,provenance='Shared pathname used by initial pilots/diagnostics; do not assign it to one run')
    write_json(retained/'INVENTORY.json',record)
manifest=[dict(path=str(p.relative_to(work)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(work.rglob('*')) if p.is_file()]
write_json(work/'OUTPUT_MANIFEST.json',manifest)
result=dict(case=case,target=20,start=0,elapsed_seconds=status['elapsed_seconds'],seconds_per_M=status['elapsed_seconds']/20,
    solver={},initial=initial,checkpoint=endpoint,job_id='495241',work=str(work),
    validator_recovery_job=os.environ['SLURM_JOB_ID'],recovery_reason='Absolute initialization-dump path missing; actual initialized checkpoint and completed endpoint independently audited')
write_json(work/'COMPLETE.json',result)
write_json(retained/'RECOVERY.json',dict(result=result,resources=resource_snapshot(),script_sha256=sha(__file__)))
print(json.dumps(result,indent=2))
