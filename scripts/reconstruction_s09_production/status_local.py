"""Lightweight scheduler/state inspection from Perseus; no raw-data analysis."""
import datetime
import json
import subprocess
from pathlib import Path
from accounting import summarize, TERMINAL

SESSION = Path(__file__).resolve().parents[4]
AMD = '/work1/eliasmost/jiaxiwu/st_s09_perturbed_production_20261024'


def refresh():
    registry = json.loads((SESSION / 'evidence/job_registry.json').read_text())
    ids = ','.join(row['job_id'] for row in registry['jobs'])
    remote = '''
import json,subprocess,shutil,os,time
from pathlib import Path
os.environ['TZ']='UTC'
time.tzset()
root=Path(AMD_ROOT)
def run(args):
 r=subprocess.run(args,capture_output=True,text=True)
 return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
out=dict(queue=run(['squeue','-u','jiaxiwu','-h','-o','%i|%P|%j|%T|%M|%D|%R']),
 accounting=run(['sacct','-X','-n','-P','-j',JOB_IDS,'-o','JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,Start,End']),
 scheduler_timestamp_timezone='UTC',filesystem_free_bytes=shutil.disk_usage(root).free,cases={})
for c in json.loads((root/'evidence/cases.json').read_text()):
 case=root/c['id'];ident='yc'+format(round(c['yc']*1000),'04d');state=case/'runs'/ident/'state.json'
 value=dict(case=c,root=str(case),initialized=state.exists())
 if state.exists():
  d=json.loads(state.read_text());v=d['latest']
  value.update(complete=d.get('complete',False),segments=len(d['segments']),
   latest={k:v[k] for k in ['path','sha256','time','cycle','dt','nblocks','particle_count','ah_onset']})
 for name in ['INITIAL_CONSTRAINT_ADMISSION','PRODUCTION_ADMISSION','CASE_NUMERICAL_ACCEPTANCE','RESOURCE_STOP']:
  f=case/'evidence'/(name+'.json')
  if f.exists():
   a=json.loads(f.read_text());value[name]={k:a[k] for k in ['constraint_norm_ratios','numerical_checks_passed','reason','ranks','max_nmb_per_rank','interior_grid_max_scaled_error'] if k in a}
 value['review_packet']=(case/'review_packet/MANIFEST.json').exists()
 out['cases'][c['id']]=value
print(json.dumps(out))
'''
    source = 'AMD_ROOT=' + repr(AMD) + '\nJOB_IDS=' + repr(ids) + '\n' + remote
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hpcfund.amd.com', 'python3 -'],
                            input=source, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    state['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state['registry'] = registry
    receipt_root = SESSION / 'evidence/scheduler_receipts/amd'
    receipt_root.mkdir(parents=True, exist_ok=True)
    for line in state['accounting']['stdout'].splitlines():
        fields = line.split('|')
        if len(fields) >= 7 and fields[1].split()[0].rstrip('+') in TERMINAL:
            target = receipt_root / (fields[0] + '.json')
            record = dict(observed_UTC=state['UTC'], machine='amd',
                          scheduler_timestamp_timezone='UTC', scheduler_fields=fields,
                          scope='Actual terminal allocation row from sacct -X -n -P; preserved before scheduler retention expires.')
            if target.exists():
                prior = json.loads(target.read_text())
                if prior['scheduler_fields'] != fields:
                    old = target.with_suffix('.' + state['UTC'].replace(':', '-') + '.prior.json')
                    old.write_text(json.dumps(prior, indent=2) + '\n')
            target.write_text(json.dumps(record, indent=2) + '\n')
    preserved = {p.stem: json.loads(p.read_text())['scheduler_fields']
                 for p in receipt_root.glob('*.json') if '.prior.' not in p.name}
    accounting = summarize(registry, state['accounting']['stdout'], preserved)
    state['accounting_upper'] = accounting['jobs']
    state['committed_node_hour_upper'] = accounting['allocated_node_hour_upper']
    assert state['committed_node_hour_upper'] <= 40
    path = SESSION / 'evidence/state.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)
    print(state['queue']['stdout'], end='')
    print('Committed allocation upper:', state['committed_node_hour_upper'], 'node-hours')
    for case in state['cases'].values():
        print(case['case']['id'], case.get('latest', 'not initialized'),
              'complete=', case.get('complete', False))
    return state


if __name__ == '__main__':
    refresh()
