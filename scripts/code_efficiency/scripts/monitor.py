#!/usr/bin/env python3
"""One lightweight queue/evidence inspection; never builds, simulates, or plots."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(os.environ.get('EFFICIENCY_SESSION', Path(__file__).resolve().parents[1]))


def read(path, default=None):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            time.sleep(.05)
    return default


subprocess.run(['python3', str(ROOT / 'scripts/advance.py')], check=True,
               stdout=subprocess.PIPE, text=True)
summary = read(ROOT / 'results/SUMMARY.json', {})
queue = subprocess.check_output(['squeue', '-u', 'jiaxiwu', '-h', '-o', '%i|%j|%t|%M|%R|%C'], text=True).splitlines()
active = [x for x in queue if '|eff-' in x and '|R|' in x]
waiting = [x for x in queue if '|eff-' in x and '|PD|' in x and '(Dependency)' not in x]
ledger = [json.loads(line) for line in (ROOT / 'provenance/jobs.jsonl').read_text().splitlines()]
by_job = {x['job_id']: x for x in ledger}
latest = {(x['kind'], x['case']): x for x in ledger}
observed = read(ROOT / 'provenance/job_states.json', {})
progress = []
for line in active:
    job = line.split('|')[0]
    row = by_job.get(job)
    if row is None:
        continue
    base = ROOT / 'results' / row['kind'] / row['case']
    for folder in [base] + sorted(base.glob('attempt*')):
        provenance = read(folder / 'provenance.json', {})
        if provenance.get('job_id') != job:
            continue
        log = folder / 'solver.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 12000))
                tail = stream.read().decode(errors='replace')
            matches = re.findall(r'elapsed=([\d.e+-]+) cycle=(\d+) time=([\d.e+-]+)', tail)
            if matches:
                progress.append(dict(job=job, case=row['case'], cycle=int(matches[-1][1]),
                                     physical_M=float(matches[-1][2])))
failures = list(summary.get('failures', []))
failures += [line for line in waiting if 'DependencyNeverSatisfied' in line]
for (kind, case), row in latest.items():
    status = observed.get(row['job_id'], {}).get('JobState')
    receipt = read(ROOT / 'results' / kind / case / 'receipt.json', {})
    if status in ['FAILED', 'OUT_OF_MEMORY'] and not receipt.get('valid'):
        failures.append(f'{kind}/{case}: job{row["job_id"]} {status}')
qa = read(ROOT / 'results/FINAL_QA.json', {})
failures += ['Independent QA: ' + x for x in qa.get('failures', [])]
node = subprocess.check_output(['scontrol', '-o', 'show', 'node', 'perseus'], text=True)
fields = dict(re.findall(r'(\w+)=(\S+)', node))
our_cpus = sum(int(line.split('|')[-1]) for line in active)
free_after = int(fields['CPUEfctv']) - int(fields['CPUAlloc']) + our_cpus
interruptions = read(ROOT / 'provenance/interruptions.json', [])
status = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
              completed=summary.get('completed'), profiles=summary.get('profiled'),
              devices=summary.get('device_diagnostics'), failures=sorted(set(failures)),
              interrupted_attempts=len(interruptions), active=active, waiting=waiting,
              progress=progress, cpu_available_after_active=free_after,
              four_rank_cpu_slot=free_after >= 8, qa_available_measurements_passed=qa.get('available_measurements_passed'),
              complete=qa.get('complete', False) and (ROOT / 'CLOSURE.md').exists()
                       and '\nCOMPLETE\n' in (ROOT / 'CLOSURE.md').read_text())
(ROOT / 'provenance/monitor_status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
