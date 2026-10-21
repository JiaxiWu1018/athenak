"""Metadata-only reservation adjustment; safe for the explicitly pending job."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021')
job='473967';script=root/'scripts/amd_segment.sbatch';gate_path=root/'control/VALIDATION_PASS.json'
sha=lambda b:hashlib.sha256(b).hexdigest()
def atomic_bytes(path,data):
    tmp=path.with_suffix(path.suffix+'.reservation_tmp');tmp.write_bytes(data);os.replace(tmp,path)
with (root/'control/submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert json.loads((root/'control/submission.json').read_text())['job_id']==job
    before_job=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    assert 'JobState=PENDING' in before_job and 'UserId=jiaxiwu(' in before_job
    assert 'JobName=jeans11-production' in before_job and 'Reason=JobHeldUser' not in before_job
    times=subprocess.check_output(['sacct','-X','-n','-P','-j','473069,473181,473209,473355','-o','JobID,ElapsedRaw,AllocNodes'],text=True)
    rows=[line.split('|') for line in times.splitlines() if line.strip()]
    assert len(rows)==4 and all(int(r[2])==6 for r in rows)
    maximum=max(int(r[1]) for r in rows);assert maximum<3*3600
    old_script=script.read_bytes();old_gate=gate_path.read_bytes();gate=json.loads(old_gate)
    assert sha(old_script)==gate['runtime_scripts']['scripts/amd_segment.sbatch']
    assert old_script.count(b'#SBATCH -t 12:00:00\n')==1
    new_script=old_script.replace(b'#SBATCH -t 12:00:00\n',b'#SBATCH -t 04:00:00\n')
    assert new_script.replace(b'#SBATCH -t 04:00:00\n',b'#SBATCH -t 12:00:00\n')==old_script
    out=root/'evidence/reservation_4h_20261021';out.mkdir()
    (out/'previous_amd_segment.sbatch').write_bytes(old_script)
    (out/'previous_VALIDATION_PASS.json').write_bytes(old_gate)
    (out/'job_before.txt').write_text(before_job);(out/'measured_allocations.txt').write_text(times)
    subprocess.run(['scontrol','hold',job],check=True)
    try:
        held=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'JobState=PENDING' in held and 'Reason=JobHeldUser' in held
        atomic_bytes(script,new_script)
        gate['runtime_scripts']['scripts/amd_segment.sbatch']=sha(new_script)
        gate['allocation_policy']=dict(reservation_hours=4,application_chunk_hours=2,
            maximum_observed_allocation_seconds=maximum,evidence=str(out),epoch=time.time(),
            executable_script_body_unchanged=True,reason='shorter backfill window with nearly two hours startup/finalization reserve')
        atomic_bytes(gate_path,(json.dumps(gate,indent=2)+'\n').encode())
        subprocess.run(['scontrol','update','JobId='+job,'TimeLimit=04:00:00'],check=True)
    except Exception:
        atomic_bytes(script,old_script);atomic_bytes(gate_path,old_gate)
        raise
    finally:
        subprocess.run(['scontrol','release',job],check=True)
    (out/'job_after.txt').write_text(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True))
    (out/'RESULT.json').write_text(json.dumps(dict(status='pass',production_job=job,
        reservation_hours=4,application_chunk_hours=2,maximum_observed_allocation_seconds=maximum,
        script_sha256=sha(new_script),no_evolution_change=True),indent=2)+'\n')
    print((out/'RESULT.json').read_text())
