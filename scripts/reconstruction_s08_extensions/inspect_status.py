"""One-shot, read-only AMD metadata inspection; persist the Perseus handoff.

This is an interactive status command, not a controller. It never submits,
cancels, restarts or changes a simulation and does not read bulk particle data.
"""
import datetime
import json
from pathlib import Path
import subprocess

E = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_long_and_inward_20261023')
PROBE = r'''
import collections,csv,json,subprocess
from pathlib import Path
X=Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023')
queue=subprocess.check_output(['squeue','-u','jiaxiwu','-h','-o','%i|%P|%T|%M|%R'],text=True)
jobs=json.loads((X/'evidence/jobs.json').read_text())
accounting=subprocess.check_output(['sacct','-X','-j',','.join(j['id'] for j in jobs),'-n','-P','-o','JobIDRaw,State,ElapsedRaw,AllocNodes'],text=True)
rows=[s.split('|') for s in accounting.splitlines() if s.strip()]
assert {r[0] for r in rows}=={j['id'] for j in jobs}
cases={};telemetry=[]
for mode in ['long','inward0020','inward0050']:
    w=X/mode/'runs/yc0500';state=w/'state.json'
    s=json.loads(state.read_text()) if state.exists() else {}
    d=dict(complete=s.get('complete',False),closed_time=s.get('latest',{}).get('time'),
           closed_AH_onset=s.get('latest',{}).get('ah_onset'),final_fields=(w/'FINAL_FIELDS_OK.json').exists())
    reduction=X/mode/'reduced/yc0500/summary.json'
    if reduction.exists():
        r=json.loads(reduction.read_text())
        d['reduction']=dict(complete=r['complete'],endpoint=r['endpoint'])
    if (w/'yc0500.mass.csv').exists():
        with (w/'yc0500.mass.csv').open() as f:
            reader=csv.DictReader(f);last=collections.deque(reader,maxlen=1)
        if last:d['live_mass']=last[0]
    if (w/'yc0500.ah_quality_0.csv').exists():
        with (w/'yc0500.ah_quality_0.csv').open() as f:a=list(csv.DictReader(f))
        accepted=[r for r in a if float(r['accepted'])==1]
        d.update(AH_searches=len(a),AH_accepted=len(accepted),first_AH=accepted[0] if accepted else None)
    for label,path in [('initial',w/'INITIAL_OK.json'),('admission',X/mode/'evidence/PRODUCTION_ADMISSION.json')]:
        if path.exists():d[label]=json.loads(path.read_text())
    cases[mode]=d
    for path in (X/mode/'evidence/jobs').glob('*/telemetry.jsonl'):
        with path.open() as f:telemetry.extend(json.loads(v) for v in collections.deque(f,maxlen=40))
disk=max((v for v in telemetry if 'extension' in v),key=lambda r:r['utc'])
memory=max((v for v in telemetry if 'vram_max_fraction' in v),key=lambda r:r['utc'])
active=[r.split('|') for r in queue.splitlines() if r.split('|')[0] in {j['id'] for j in jobs}]
print(json.dumps(dict(queue=queue,active_registered_jobs=active,cases=cases,accounting_rows=rows,
    allocated_node_hours=sum(int(r[2] or 0)*int(r[3] or 0)/3600 for r in rows),
    latest_disk=disk,latest_GPU_fraction=memory['vram_max_fraction'],latest_GPU_utc=memory['utc'])))
'''


def main():
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hpcfund.amd.com', 'python3', '-'],
                            input=PROBE, text=True, capture_output=True, timeout=120)
    result.check_returncode()
    observation = json.loads(result.stdout)
    current = datetime.datetime.now(datetime.timezone.utc)
    observation.update(utc=current.isoformat(),
                       classification='verified wait' if observation['active_registered_jobs'] else 'terminal observation',
                       inspection_source=str(Path(__file__).resolve()),
                       scope='Slurm and small native diagnostic/receipt metadata only; no simulation mutation or bulk reduction')
    name = 'evidence/OBSERVE_' + current.strftime('%Y%m%dT%H%M%SZ') + '.json'
    (E / name).write_text(json.dumps(observation, indent=2) + '\n')
    path = E / 'evidence/state.json'
    state = json.loads(path.read_text())
    state.update(utc=observation['utc'], latest_observation=name)
    for mode, case in observation['cases'].items():
        dest = state['observed_progress'].setdefault(mode, {})
        dest.update(validated_closed_time_M=case['closed_time'], complete=case['complete'],
                    final_fields=case['final_fields'], AH_accepted=case.get('AH_accepted', 0) > 0)
        if 'live_mass' in case:
            m = case['live_mass']
            dest.update(time_M=float(m['time']), all_particles=int(m['n_alive']),
                        nblocks=int(m['nblocks']), live_diagnostics=m)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(dict(utc=observation['utc'], queue=observation['queue'],
        cases={k:dict(time_M=v.get('live_mass',{}).get('time'),closed_time=v['closed_time'],
                      complete=v['complete'],AH_accepted=v.get('AH_accepted',0),
                      initial_present='initial' in v,admission_present='admission' in v,
                      final_fields=v['final_fields'],reduction=v.get('reduction')) for k,v in observation['cases'].items()},
        allocated_node_hours=observation['allocated_node_hours'],
        AMD_extension_GiB=observation['latest_disk']['extension']/1024**3,
        latest_GPU_fraction=observation['latest_GPU_fraction'],record=name),indent=2))


if __name__ == '__main__':
    main()
