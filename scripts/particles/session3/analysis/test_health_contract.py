"""Adversarial safety checks: duplicate failures, physical stops and endpoints."""
import csv,hashlib,json,struct,subprocess,sys,tempfile
from pathlib import Path
from health import evaluate
def record(t,**changes):
    r=dict(time=t,cycle=int(t),N=4,N_expected=4,M0=1.,M0_expected=1.,mass_error=0.,
           particle_nonfinite=0,alpha_min=.9,H_core_L2=1.e-4,M_core_L2=0.,healthy=1,
           physical_stop=0,constraint_strikes=0)
    r.update(changes);return r
with tempfile.TemporaryDirectory() as directory:
    run=Path(directory);(run/'out').mkdir()
    ledger=run/'out/test.plummer_health.csv'
    def rows(values):
        with ledger.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    rows([record(0),record(1),record(2)]);assert evaluate(run)['valid']
    rows([record(0),record(1,healthy=0),record(1)])
    h=evaluate(run);assert not h['valid'] and h['last_healthy_time']==0
    rows([record(0),record(1,physical_stop=1,alpha_min=.19)])
    h=evaluate(run);assert h['physical_stop'] and h['last_healthy_time']==1 and not h['valid']
    rows([record(0),record(1,N=3)]);assert evaluate(run)['last_healthy_time']==0
    rows([record(0),record(1,mass_error=1.01e-10)]);assert not evaluate(run)['valid']
    rows([record(0),record(1,alpha_min=float('nan'))]);assert not evaluate(run)['valid']
    rows([record(0),record(1)])
    checkpoint=run/'out/test.rst'
    checkpoint.write_bytes(b'<par_end>\n'+struct.pack('<ii',624,2)+b'\0'*(9*8+2*19*4)+struct.pack('<ddi',1.,.078125,1))
    state=dict(completed=True,returncode=0,checkpoint=str(checkpoint),checkpoint_time=1.,
               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (run/'segment_state.json').write_text(json.dumps(state))
    script=Path(__file__).resolve().parents[1]/'scripts/resume_args.py'
    def resume(*args):return subprocess.run([sys.executable,str(script),str(run),*args],capture_output=True,text=True)
    assert resume().returncode!=0
    assert resume('1').returncode!=0
    assert resume('2').returncode==0
    checkpoint.write_bytes(checkpoint.read_bytes()+b'changed');assert resume('2').returncode!=0
print('PASS: numerical failure is sticky; finite stops bound products; endpoint and checksum gates reject invalid restarts')
