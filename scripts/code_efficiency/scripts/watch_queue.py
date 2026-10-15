#!/usr/bin/env python3
"""Persist a low-cost scheduler heartbeat until closure; no heavy computation."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(os.environ.get('EFFICIENCY_SESSION', Path(__file__).resolve().parents[1]))
lock = (ROOT / 'provenance/.queue_watch.lock').open('a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('An existing watcher owns the session lock')
record = dict(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256=hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest(),
              interval_seconds=50, scope='Slurm/retained-metadata inspection only; heavy work stays in Slurm')
(ROOT / 'provenance/queue_watch.json').write_text(json.dumps(record, indent=2) + '\n')
previous = None
while not (ROOT / 'provenance/STOP_QUEUE_WATCH').exists():
    result = subprocess.run(['python3', str(ROOT / 'scripts/monitor.py')], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        event = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     monitor_exit=result.returncode, error=result.stderr[-3000:])
    else:
        try:
            event = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            event = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         error='Monitor returned unreadable status', output=result.stdout[-3000:])
    signature = json.dumps({k: v for k, v in event.items() if k != 'updated_utc'}, sort_keys=True)
    # Preserve every completed-case/interruption/resource transition. Solver progress
    # and active elapsed clocks also provide a bounded, compact heartbeat log.
    if signature != previous:
        with (ROOT / 'provenance/queue_watch_events.jsonl').open('a') as out:
            out.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
        previous = signature
    if event.get('complete'):
        record['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record['status'] = 'complete'
        (ROOT / 'provenance/queue_watch.json').write_text(json.dumps(record, indent=2) + '\n')
        break
    time.sleep(50)
