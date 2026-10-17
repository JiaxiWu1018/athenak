"""Metadata-only OS gate and same-chat resumption; never launches scientific work directly."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from cfg import PERSEUS as ROOT, THREAD, atomic, load, now
from os_gate import inspect, require_release

def main():
    lock = (ROOT / 'evidence/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / 'evidence/controller.json'
    state = load(path) if path.exists() else dict(events={}, observed={}, heartbeat_at=time.time())
    state.update(pid=os.getpid(), started_utc=now(), thread=THREAD, check_interval_s=300)

    def event(key, description):
        item = state['events'].setdefault(key, dict(last_delivery=0, description=description))
        if (ROOT / 'evidence/events' / (key + '.ack')).exists() or time.time()-item['last_delivery'] < 7200:
            return
        message = (f'ST reconstruction session6 paired-scaling event {key}. {description} '
                   f'Read {ROOT}/HANDOFF.md and evidence/state.json. Continue the already-authorized '
                   'mirror3M/ordinary3M/ordinary6M tests, respecting OS-first priority, through results, '
                   f'movies and amended report. Acknowledge in evidence/events/{key}.ack. '
                   'This event does not replace the task. Do not restart old closed session6 jobs.')
        item['receipt'] = subprocess.check_output(['/home/jiaxiwu/.local/bin/codex', 'queue',
            '--thread', THREAD, '--message', message], text=True, stderr=subprocess.STDOUT, timeout=30)
        item['last_delivery'] = time.time()

    while not (ROOT / 'evidence/COMPLETE').exists():
        try:
            gate = inspect()
            atomic(ROOT / 'evidence/OS_GATE_LATEST.json', dict(utc=now(), **gate))
            state['os_gate'] = gate
            if gate['ready']:
                event('os_completion_candidate', 'Verify the OS owner has completed its final required work; '
                      'only then write OS_PRIORITY_RELEASE.json with verified_complete=true and matching '
                      'os_delivery_sha256. No user reapproval is required. Release the held ST stage only after this check.')
            startup = load(ROOT / 'evidence/state.json')
            stage = next((j for j in startup.get('jobs', []) if j['role'] == 'stage'), None)
            if stage and (ROOT / 'evidence/OS_PRIORITY_RELEASE.json').exists():
                require_release(ROOT)
                if not state.get('stage_released'):
                    info = subprocess.check_output(['scontrol', 'show', 'job', str(stage['id'])], text=True)
                    if f'Command={ROOT}/scripts/stage.sbatch' not in info:
                        raise ValueError('Held stage job identity differs')
                    if 'Reason=JobHeldUser' in info:
                        atomic(ROOT / 'evidence/STAGE_RELEASE_INTENT.json', dict(utc=now(), job=stage['id'], os_gate=gate))
                        subprocess.run(['scontrol', 'release', str(stage['id'])], check=True)
                    state['stage_released'] = dict(job=stage['id'], utc=now())
                    event('stage_released', 'OS completion checked; allocated staging is now eligible. Inspect its receipt and subsequent AMD submissions.')
            jobs = startup.get('jobs', [])
            amdjobs = [str(j['id']) for j in jobs if j['host'] == 'amd']
            if amdjobs:
                reply = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    'hpcfund.amd.com', 'sacct', '-X', '-j', ','.join(amdjobs), '-n', '-P',
                    '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes'], text=True, timeout=45)
                rows = [line.split('|') for line in reply.splitlines() if line.strip()]
                atomic(ROOT / 'evidence/AMD_ACCOUNTING.json', dict(utc=now(), rows=rows,
                    node_hours=sum(int(r[3])*int(r[4]) for r in rows)/3600))
                for row in rows:
                    signature = '|'.join(row[:3])
                    if state['observed'].get(row[0]) != signature:
                        state['observed'][row[0]] = signature
                        if any(x in row[1] for x in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL']):
                            event('job_' + row[0] + '_' + row[1].split()[0], 'Inspect terminal state '+signature)
            event('continuation_test', 'Verify this same-chat continuation and the durable held-stage/OS gate; complete prepared work while OS finishes.')
            if time.time()-state['heartbeat_at'] >= 7200:
                event('heartbeat_'+str(int(time.time()/7200)), 'Review OS gate, held/new jobs, whole-user storage and next authorized actions.')
                state['heartbeat_at'] = time.time()
            state.pop('last_error', None)
        except Exception as error:
            state['last_error'] = repr(error)
            try:
                event('controller_error', 'Inspect the durable controller error and recover metadata transport or receipts without bypassing the OS gate.')
            except Exception as delivery_error:
                state['notification_error'] = repr(delivery_error)
        state['utc'] = now()
        atomic(path, state)
        time.sleep(300)
    state.update(closed=True, utc=now())
    atomic(path, state)

if __name__ == '__main__':
    main()
