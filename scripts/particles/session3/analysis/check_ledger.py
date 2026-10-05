"""Allocated-node structural checks of the primary angular and health ledgers."""
import json,math,sys
from pathlib import Path
from health import require_window,read_rows
run=Path(sys.argv[1]);window=require_window(run,allow_stopped=False)
health=read_rows(next((run/'out').glob('*.plummer_health.csv')))
modes=read_rows(next((run/'out').glob('*.plummer_modes.csv')))
by_time={}
for row in modes:by_time.setdefault(float(row['time']),{})[int(row['band'])]=row
for t,bands in by_time.items():
    assert len(bands)==5
    assert sum(int(float(bands[b]['N'])) for b in range(4))==int(float(bands[4]['N']))
    for b,row in bands.items():
        n=float(row['N']);s=[float(row[f'sumY{i}']) for i in range(25)]
        assert abs(s[0]/n-1/math.sqrt(4*math.pi))<1.e-10
        for l in range(1,5):
            a=math.sqrt(4*math.pi/(2*l+1)*sum(v*v for v in s[l*l:(l+1)**2]))/n
            assert abs(a-float(row[f'A{l}']))<1.e-13
        for key,index in [('dipole_x',3),('dipole_y',1),('dipole_z',2)]:
            assert abs(float(row[key])-s[index]/(n*math.sqrt(3/(4*math.pi))))<1.e-13
    for i in range(25):assert abs(sum(float(bands[b][f'sumY{i}']) for b in range(4))-float(bands[4][f'sumY{i}']))/float(bands[4]['N'])<1.e-10
speed=max(float(r['coordinate_characteristic_bound']) for r in health)
assert 1.3<speed<1.5, 'initial analytic characteristic bound is unexpected'
result=dict(passed=True,health=window,mode_times=len(by_time),characteristic_bound=speed,
            definition='fixed common untruncated-rest-mass quartiles in COM coordinate radius')
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
