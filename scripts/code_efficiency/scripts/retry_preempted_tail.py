#!/usr/bin/env python3
"""Recover scheduler preemptions at the serialized queue tails; metadata only."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(os.environ.get('EFFICIENCY_SESSION', Path(__file__).resolve().parents[1]))
ALIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def read(path, default=None):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            time.sleep(.05)
    raise RuntimeError('Unreadable metadata: ' + str(path))


def detail(job):
    result = subprocess.run(['scontrol', '-o', 'show', 'job', job], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return dict(re.findall(r'(\w+)=(\S+)', result.stdout)), result.stdout


def main():
    lock = (ROOT / 'provenance/.advance.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    pipeline_path = ROOT / 'provenance/verified_pipeline.json'
    pipeline = read(pipeline_path)
    chains = pipeline['serialized_resource_chains']
    observed = read(ROOT / 'provenance/job_states.json', {})
    recovery = read(ROOT / 'provenance/recovery.json', {})
    ledger = [json.loads(line) for line in (ROOT / 'provenance/jobs.jsonl').read_text().splitlines()]
    latest = {(row['kind'], row['case']): row for row in ledger}
    matrix = {case['name']: case for case in read(ROOT / 'matrix.json')}
    candidates = []
    for (kind, name), row in latest.items():
        if kind not in {'throughput', 'profile', 'device'}:
            continue
        if read(ROOT / 'results' / kind / name / 'receipt.json', {}).get('valid'):
            continue
        current, _ = detail(row['job_id'])
        status = current.get('JobState', observed.get(row['job_id'], {}).get('JobState'))
        if status == 'PREEMPTED':
            candidates.append(row)
    if not candidates:
        print(json.dumps(dict(retried=[])))
        return
    license_snapshot = subprocess.check_output(['scontrol', 'show', 'licenses'], text=True)
    if not re.search(r'LicenseName=debug_gpu_jiaxiwu\s+Total=4\b', license_snapshot):
        raise RuntimeError('The four-license serialization constraint changed; inspect before retrying')
    closure = recovery.get('next_closure_job') or pipeline['closure_job']
    fields, snapshot = detail(closure)
    if fields.get('JobState') != 'PENDING' or not fields.get('UserId', '').startswith('jiaxiwu('):
        raise RuntimeError('Closure is not an owned pending job; inspect before retrying: ' + snapshot)
    record = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  reason='Retry only scheduler PREEMPTED measurements; numerical failures require diagnosis.',
                  closure_job=closure, closure_before=snapshot, license_snapshot=license_snapshot,
                  retries=[], commands=[], status='prepared')
    record_path = ROOT / 'provenance' / f'queue_tail_retry_{candidates[0]["job_id"]}.json'

    def save():
        record_path.write_text(json.dumps(record, indent=2) + '\n')

    def run(command):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record['commands'].append(dict(command=command, returncode=result.returncode,
                                       stdout=result.stdout, stderr=result.stderr))
        save()
        result.check_returncode()
        return result.stdout

    run(['scontrol', 'hold', closure])
    # If any mutation fails, leave closure held and the full transaction recorded.
    # A human/agent can repair it without risking premature final verification.
    try:
        for row in candidates:
            group = 'four_rank' if matrix[row['case']]['ranks'] == 4 else 'one_rank'
            previous = chains[group][-1]
            previous_state, _ = detail(previous)
            command = ['python3', str(ROOT / 'scripts/submit.py'), '--kind', row['kind'],
                       '--names', row['case']]
            if previous_state.get('JobState') in ALIVE:
                command += ['--dependency', previous]
            submission = json.loads(run(command))
            if len(submission['jobs']) != 1:
                raise RuntimeError('Expected one fresh retry: ' + str(submission))
            new = submission['last_job']
            new_fields, new_snapshot = detail(new)
            licenses = 4 if group == 'four_rank' else 1
            if new_fields.get('Licenses') != f'debug_gpu_jiaxiwu:{licenses}':
                raise RuntimeError('Unexpected retry license request: ' + new_snapshot)
            chains[group].append(new)
            order = pipeline['active_execution_order']
            order.insert(order.index(previous) + 1, new)
            record['retries'].append(dict(superseded_job=row['job_id'], job_id=new,
                                          kind=row['kind'], case=row['case'], group=group,
                                          previous_tail=previous, submission=submission))
            save()
        tails = [chains['four_rank'][-1], chains['one_rank'][-1]]
        run(['scontrol', 'update', 'JobId=' + closure, 'Dependency=0'])
        run(['scontrol', 'update', 'JobId=' + closure, 'Dependency=afterany:' + ':'.join(tails)])
        chains['join_tails'] = tails
        pipeline['last_case_job'] = chains['one_rank'][-1]
        pipeline.setdefault('recovery_order_records', []).append(str(record_path.relative_to(ROOT)))
        pipeline['updated_utc'] = record['updated_utc']
        pipeline_path.write_text(json.dumps(pipeline, indent=2) + '\n')
        record['closure_after'] = detail(closure)[1]
        run(['scontrol', 'release', closure])
        record['status'] = 'complete'
        save()
    except Exception as error:
        record['error'] = repr(error)
        record['status'] = 'inspect_held_closure_before_release'
        save()
        raise
    with (ROOT / 'SESSION_LOG.md').open('a') as out:
        out.write('\n' + record['updated_utc'] + ': Queue-tail recovery: '
                  + ', '.join(f'{x["superseded_job"]}→{x["job_id"]} ({x["kind"]}/{x["case"]})'
                              for x in record['retries'])
                  + '. Interrupted evidence preserved; each retry follows its resource-group tail; '
                    'closure joins both new tails and license constraints prevent overlap.\n')
    print(json.dumps(dict(retried=record['retries'], record=str(record_path.relative_to(ROOT)))))


if __name__ == '__main__':
    main()
