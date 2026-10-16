"""Durable metadata-only monitoring and same-chat continuation."""
import fcntl
import json
import os
import subprocess
import sys
import time
from session import ROOT, atomic, now, load

def main():
    thread = sys.argv[1]
    lock = (ROOT / 'evidence/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / 'evidence/controller.json'
    state = load(path) if path.exists() else {'events': {}, 'observed': {}, 'last_heartbeat': time.time()}
    state.update(pid=os.getpid(), thread=thread, heartbeat_interval_seconds=7200)
    def event(key, message):
        state['events'].setdefault(key, {'message': message, 'last_delivery': 0})
    event('continuation_test', 'Verify this same-chat event, acknowledge it, and continue the approved session6 plan.')
    while not (ROOT / 'evidence/CAMPAIGN_COMPLETE').exists():
        try:
            jobs = [j for j in load(ROOT / 'evidence/state.json')['jobs'] if j['host'] == 'amd']
            reply = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    'hpcfund.amd.com', 'sacct', '--allocations', '-j', ','.join(str(j['id']) for j in jobs),
                    '-n', '-P', '--format=JobID,State,ElapsedRaw,AllocNodes'], text=True, timeout=45) if jobs else ''
            rows = [line.split('|') for line in reply.splitlines() if line.strip()]
            used = sum(int(r[2]) * int(r[3]) for r in rows) / 3600
            atomic(ROOT / 'evidence/AMD_ACCOUNTING.json', {'utc': now(), 'allocated_node_hours': used, 'rows': rows})
            for r in rows:
                signature = '|'.join(r[:2])
                if state['observed'].get(r[0]) != signature:
                    state['observed'][r[0]] = signature
                    if any(word in r[1] for word in ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL')):
                        event('job_' + r[0] + '_' + r[1].split()[0], 'Inspect terminal job state: ' + signature)
            if time.time() - state['last_heartbeat'] >= 7200:
                event('heartbeat_' + str(int(time.time() // 7200)), 'Review jobs, storage and progress; continue through actual results, figures and report.')
                state['last_heartbeat'] = time.time()
            for key, ev in state['events'].items():
                if (ROOT / 'evidence/events' / (key + '.ack')).exists() or time.time() - ev['last_delivery'] < 7200:
                    continue
                message = (f'ST reconstruction session6 event {key}. {ev["message"]} '
                           f'Read {ROOT}/HANDOFF.md and evidence/state.json. Continue the already-authorized work; '
                           f'acknowledge in evidence/events/{key}.ack. This event does not replace the task.')
                ev['receipt'] = subprocess.check_output(['/home/jiaxiwu/.local/bin/codex', 'queue', '--thread', thread,
                                                      '--message', message], text=True, stderr=subprocess.STDOUT, timeout=30)
                ev['last_delivery'] = time.time()
            state.pop('last_error', None)
        except Exception as error:
            state['last_error'] = repr(error)
        state['utc'] = now()
        atomic(path, state)
        time.sleep(60)
    state['closed'] = True
    atomic(path, state)

if __name__ == '__main__':
    main()
