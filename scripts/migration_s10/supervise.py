"""Metadata-only same-chat continuation, with durable deduplication and retries."""
import datetime
import fcntl
import json
import subprocess
import time
from submit_local import LOCAL, atomic
from status_local import refresh

THREAD = '01a1700d-9f0a-7801-98d4-0d700ff3c413'

def main():
    lock = (LOCAL/'evidence/supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = LOCAL/'evidence/supervisor_events.json'
    seen = json.loads(path.read_text()) if path.exists() else {}
    def wake(key, message):
        if key in seen:
            return
        q = subprocess.run(['/home/jiaxiwu/.local/bin/codex', 'queue', '--thread', THREAD,
            '--message', message], capture_output=True, text=True, timeout=50)
        atomic(LOCAL/'evidence'/('wake_'+key+'.json'), dict(returncode=q.returncode,
            stdout=q.stdout, stderr=q.stderr, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        if q.returncode == 0:
            seen[key] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            atomic(path, seen)
    while not (LOCAL/'evidence/CLOSED.json').exists():
        try:
            s = refresh()
            for row in s['jobs']:
                if row.get('ALLOCATION_END') or row.get('FAILURE') or ('JobState=' in row['scheduler'] and any('JobState='+v in row['scheduler'] for v in ['FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','COMPLETED'])):
                    wake('job_'+row['id'], 'Continue the approved migration Session10 implementation and campaign. '
                        f"AMD job{row['id']} ({row['role']}, {row['case']}, target{row['target']}M) ended or needs review. "
                        f'Read {LOCAL}/HANDOFF.md, APPROVED_PLAN.md and evidence/remote_status.json. '
                        'Verify actual gates/results, repair safely or submit the next authorized segment. '
                        'Both target1500M; outward+0.005 from0, originalU0 from700. '
                        'Originalmesh +/-512M/initial.25/allowed.0625, 50nodeh/350GiB. '
                        'Preserve other campaigns. Continue through numerical/actual visual delivery, not just submission.')
            for key, case in s['cases'].items():
                if case['resource_stop']:
                    wake('resource_'+key, f'Continue Session10 and inspect the resource stop for{key}; read {LOCAL}/HANDOFF.md and evidence/remote_status.json. Do not clear a real budget/storage stop or enlarge scope without authorization.')
            wake('heartbeat_'+str(int(time.time()//3600)),
                f'Hourly continuation of approved migration Session10. Read {LOCAL}/HANDOFF.md and evidence/remote_status.json. '
                'Inspect registered jobs and progress, keep authorized work moving, and persist next steps. '
                'No original-data cleanup or solver changes. Close this supervisor only after accepted final delivery.')
            atomic(LOCAL/'evidence/supervisor_heartbeat.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                thread=THREAD, job_ids=[r['id'] for r in s['jobs']]))
        except Exception as error:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), repr(error), flush=True)
        time.sleep(60)

if __name__ == '__main__':
    main()
