"""Validate analysis-only AH checks on the retained NaN-energy production state."""
import csv,json,os,re,subprocess,sys
from pathlib import Path
root=Path(os.environ['SESSION_ROOT']);scripts=root/'code/scripts/particles/session011'
sys.path.insert(0,str(scripts))
from runtime import atomic,checkpoint,digest,require_compute
from make_input import parse
require_compute()
build=root/'evidence'/f'analysis_build_{os.environ["BUILD_JOB"]}'
receipt=json.loads((build/'RESULT.json').read_text())
assert receipt['status']=='build_and_python_tests_pass'
assert digest(receipt['executable'])==receipt['executable_sha256']
gate=json.loads((root/'control/VALIDATION_PASS.json').read_text())
assert digest(gate['executable'])==receipt['production_executable_sha256']==gate['executable_sha256']
source=root/'runs/segment_0003/rst/s11_amd.00005.rst'
state=checkpoint(source)
assert state['sha256']=='1fd79b708bbb8ba5547d3c0390b914e63e90d802a377f4989d62ea4a9842d3f2'
before=source.stat()
out=root/'evidence'/f'analysis_saved_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
run=out/'run';run.mkdir();os.environ['RUN']=str(run)
blocks=parse((root/'inputs/production.athinput').read_text())
disable=[f'{b}/{key}={value}' for b in blocks if b.startswith('output') for key,value in [('dt',-1),('dcycle',0)]]
command=['prun',str(root/'scripts/amd_rank_wrapper.sh'),receipt['executable'],
 '--kokkos-map-device-id-by=mpi_rank','-r',str(source),'-d',str(run),
 'time/nlim=0','time/tlim=75','diagnostic_checks/frozen_ah_searches=6',
 'diagnostic_checks/require_accept=true','diagnostic_checks/horizon=0',
 'gi_diagnostics/enabled=false','mesh_refinement/max_nmb_per_rank=256']+disable
atomic(out/'command.json',command)
with (out/'run.log').open('w') as log:
    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,f'analysis returncode {result.returncode}'
log=(out/'run.log').read_text()
match=re.search(r'FROZEN_AH_CHECKS_PASS time=(\S+) cycle=(\d+) evolution_advances=0 bitwise_particle_check=pass unknown_native_energy=(\d+)',log)
assert match,'missing nonmutation verdict'
assert int(match[2])==1902 and abs(float(match[1])-state['time'])<1.e-10
assert int(match[3])>0,'regression state did not exercise unknown energy'
accepted=[]
for path in run.glob('*.ah_events_*.csv'):
    accepted += [row for row in csv.DictReader(path.open()) if row['published']=='1' and row['horizon']=='0']
assert accepted,'saved individual horizon did not requalify'
assert source.stat().st_size==before.st_size and source.stat().st_mtime_ns==before.st_mtime_ns
assert digest(gate['executable'])==receipt['production_executable_sha256']
atomic(out/'RESULT.json',dict(receipt,status='pass',job_id=os.environ['SLURM_JOB_ID'],
 build_job=os.environ['BUILD_JOB'],evolution_advances=0,bitwise_particle_check=True,
 unknown_native_energy=int(match[3]),checkpoint=state,accepted=accepted[-1],run=str(out)))
print((out/'RESULT.json').read_text())
