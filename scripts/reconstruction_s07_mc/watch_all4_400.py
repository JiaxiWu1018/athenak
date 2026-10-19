"""Perseus metadata-only readiness watch; submit the approved plot once ready."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/07_mc_particle_convergence_20261019')
EVIDENCE = ROOT/'evidence'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def remote(command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                    'hpcfund.amd.com', shlex.join(command)],
                                   text=True, stderr=subprocess.STDOUT, timeout=60)


def check(config):
    code = 'import json;from pathlib import Path;p='+repr(config['ready_paths'])+';print(json.dumps([Path(s).is_file() and Path(s).stat().st_size>0 for s in p]))'
    ready = json.loads(remote(['python3', '-c', code]))
    assert len(ready) == 4
    return dict(utc=now(), ready=ready, all_ready=all(ready),
                condition='All four production402M files exist;400M files are closed')


def notify(config, key, message):
    if (EVIDENCE/'events'/(key+'.ack')).exists():
        return
    subprocess.check_output(['/home/jiaxiwu/.local/bin/codex', 'queue', '--thread', config['thread'],
                             '--message', f'ST session7 event {key}: {message} Read {ROOT}/HANDOFF.md and evidence/state.json. '
                             f'Acknowledge in evidence/events/{key}.ack. All raw and movie caches/frames stay AMD. '
                             'Continue the existing1000M campaign; do not restart closed session6 jobs.'],
                            text=True, stderr=subprocess.STDOUT, timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = load(EVIDENCE/'ALL4_400_WATCH_CONFIG.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == config['watcher_sha256']
    if args.check:
        print(json.dumps(check(config)), flush=True)
        return
    lock = (EVIDENCE/'all4_400_watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(pid=os.getpid(), started_utc=now(), source=config['source'], poll_seconds=60)
    receipt = EVIDENCE/'ALL4_400_SUBMITTED.json'
    intent = EVIDENCE/'ALL4_400_SUBMIT_INTENT.json'
    while not (EVIDENCE/'ALL4_400_WATCH_STOP').exists() and not (EVIDENCE/'COMPLETE').exists():
        try:
            status.update(check(config))
            status.pop('last_error', None)
            atomic(EVIDENCE/'ALL4_400_WATCH_STATUS.json', status)
            if receipt.exists():
                job = load(receipt)
            elif status['all_ready']:
                if intent.exists():
                    notify(config, 'all4_400_submission_review', 'Ambiguous prior plot submission: reconcile ALL4_400_SUBMIT_INTENT before retrying; never submit a duplicate.')
                    status.update(closed=True, submission_reconciliation_required=True)
                    break
                atomic(intent, dict(utc=now(), source=config['source'], command=config['command']))
                reply = remote(config['command'])
                ids = re.findall(r'^(\d+)(?:;[^\n]*)?$', reply, re.M)
                assert len(ids) == 1, reply
                job = dict(id=ids[0], host='amd', role='interim_all4_400', key='ALL4_400',
                           command=config['command'], source=config['source'], utc=now(), reply=reply)
                atomic(receipt, job)
            else:
                time.sleep(60)
                continue
            state = load(EVIDENCE/'state.json')
            if not any(j['id'] == job['id'] and j['host'] == 'amd' for j in state['jobs']):
                state['jobs'].append(job)
            state['interim_request'].update(status='all4_plot_queued', job=job['id'],
                next='Wait for existing all-four production400M figure job; verify compact products and actual PNG/PDF, then deliver the single combined figure. Keep full1000M graph active.')
            state['updated_utc'] = now()
            atomic(EVIDENCE/'state.json', state)
            notify(config, 'all4_400_submitted', f'All four production prefixes reached400M. Prepared plot job{job["id"]} was uniquely submitted; do not duplicate. Actual numeric/PNG-PDF review and delivery are still required.')
            status.update(closed=True, submitted_job=job['id'], utc=now())
            break
        except Exception as error:
            status.update(last_error=repr(error), utc=now())
            atomic(EVIDENCE/'ALL4_400_WATCH_STATUS.json', status)
            time.sleep(60)
    status.update(closed=True, utc=now())
    atomic(EVIDENCE/'ALL4_400_WATCH_STATUS.json', status)


if __name__ == '__main__':
    main()
