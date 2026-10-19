"""Record read-only scheduler evidence; never submit, cancel, or change jobs."""
import json
import os
import subprocess
import time
from run import RUN, OUT, atomic

if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
    raise RuntimeError('Run inside allocated GG compute')
jobs = [j['job_id'] for j in json.loads((RUN / 'submission.json').read_text())['jobs']]
commands = [
    ['sacct', '-X', '-j', ','.join(jobs), '-n', '-P', '-o', 'JobID,State,ExitCode,ElapsedRaw,AllocNodes,Start,End'],
    ['squeue', '-j', ','.join(jobs), '-o', '%.18i %.10P %.12T %.12M %.12l %.6D %R'],
    ['scontrol', 'show', 'job', '-o', jobs[-1]],
]
results = []
for command in commands:
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    results.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
atomic(OUT / 'SCHEDULER_STATUS.json', dict(checked_epoch=time.time(), read_only=True, results=results))
for row in results:
    print(row['stdout'])
