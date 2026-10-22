"""Validate analysis-only AH checks on the retained NaN-energy production state."""
import csv,json,os,re,subprocess,sys
from pathlib import Path
root=Path(os.environ['SESSION_ROOT']);scripts=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(scripts))
from runtime import atomic,checkpoint,digest,require_compute
from make_input import parse
from endpoint import saved_ah_controls,saved_output_overrides
require_compute()
build=root/'evidence'/f'analysis_build_{os.environ["BUILD_JOB"]}'
receipt=json.loads((build/'RESULT.json').read_text())
assert receipt['status']=='build_and_python_tests_pass'
qa_path=root/'evidence'/f'render_qa_{os.environ["LAUNCH_QA_JOB"]}/RESULT.json'
qa=json.loads(qa_path.read_text());assert qa['status']=='pass'
for name,expected in qa['tooling_sha256'].items():
    assert digest(scripts/name)==expected,name
assert digest(receipt['executable'])==receipt['executable_sha256']
gate=json.loads((root/'control/VALIDATION_PASS.json').read_text())
assert digest(gate['executable'])==receipt['production_executable_sha256']==gate['executable_sha256']
source=root/'runs/segment_0003/rst/s11_amd.00005.rst'
state=checkpoint(source)
assert state['sha256']=='1fd79b708bbb8ba5547d3c0390b914e63e90d802a377f4989d62ea4a9842d3f2'
before=source.stat()
out=root/'evidence'/f'analysis_saved_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
run=out/'run';run.mkdir();os.environ['RUN']=str(run)
overlay=saved_ah_controls(run/'diagnostic_controls.athinput',0,True)
blocks=parse((root/'inputs/production.athinput').read_text())
disable=saved_output_overrides(blocks)
command=['prun',str(root/'scripts/amd_rank_wrapper.sh'),receipt['executable'],
 '--kokkos-map-device-id-by=mpi_rank','-r',str(source),'-i',str(overlay),'-d',str(run),
 'time/nlim=0','time/tlim=75',
 'gi_diagnostics/enabled=false','mesh_refinement/max_nmb_per_rank=256']+disable
atomic(out/'command.json',command)
# Exercise the native parser before allocating the full saved mesh. Compare every
# effective parameter to the checkpoint plus the explicit analysis overrides.
with source.open('rb') as f:
    header=[]
    for _ in range(100000):
        line=f.readline()
        if line.strip()==b'<par_end>':break
        if not line:raise RuntimeError('missing restart parameter terminator')
        header.append(line.decode())
    else:raise RuntimeError('unbounded restart parameter header')
expected=parse(''.join(header))
for block,values in parse(overlay.read_text()).items():expected.setdefault(block,{}).update(values)
for argument in command:
    if '/' in argument and '=' in argument:
        key,value=argument.split('=',1);block,name=key.split('/',1)
        assert block in expected and name in expected[block],argument
        expected[block][name]=value
parse_command=command+['-n'];atomic(out/'parse_command.json',parse_command)
with (out/'parameter_dump.log').open('w') as log:
    parsed=subprocess.run(parse_command,stdout=log,stderr=subprocess.STDOUT)
assert parsed.returncode==0,'native restart/overlay parameter parsing failed'
actual=parse((out/'parameter_dump.log').read_text());actual.pop('par_end',None)
assert actual==expected,'native parameter dump changed checkpoint parameters beyond explicit analysis controls'
atomic(out/'parameter_overlay_check.json',dict(status='pass',all_parameters_match=True,
    physical_and_serialized_parameters_preserved=True,overlay_sha256=digest(overlay),
    parse_command=parse_command))
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
 source=qa['source'],tooling_sha256=qa['tooling_sha256'],build_source=receipt['source'],
 launch_qa=str(qa_path),parameter_overlay_check=True,diagnostic_overlay_sha256=digest(overlay),
 build_job=os.environ['BUILD_JOB'],evolution_advances=0,bitwise_particle_check=True,
 unknown_native_energy=int(match[3]),checkpoint=state,accepted=accepted[-1],run=str(out)))
print((out/'RESULT.json').read_text())
