#!/usr/bin/env python3
"""Save actual pre-submission validation evidence; no scheduler writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import recover


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
        raise RuntimeError('Run the pre-submission verification in allocated GG compute')
    if (recover.RUN / 'submission.json').exists():
        raise RuntimeError('Recovery was already submitted/attempted; do not regenerate preparation evidence')
    recover.verify_frozen(recover.RUN)
    tests = []
    for directory, pattern in [('launch_s10_recovery', 'test_*.py'), ('launch_s10', 'test_launch.py')]:
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', directory, '-p', pattern, '-v']
        p = subprocess.run(command, cwd=recover.BASE, capture_output=True, text=True, timeout=120)
        (recover.RUN / ('TEST_' + directory + '.txt')).write_text(p.stdout + p.stderr)
        tests.append(dict(command=command, returncode=p.returncode))
        if p.returncode:
            raise RuntimeError('Unit suite failed: ' + directory)
    dry = subprocess.run(['bash', str(recover.HERE / 'submit.sh'), '--dry-run'],
                         capture_output=True, text=True, check=True, timeout=30)
    preview = json.loads(dry.stdout)
    if preview['submitted'] or len(preview['jobs']) != 2:
        raise RuntimeError('Unexpected submission preview')
    if any(j['partition'] != 'gh' or j['nodes'] != 16 or j['wall_hours'] != 24 for j in preview['jobs']):
        raise RuntimeError('Resource review mismatch')
    if '--dependency=afterok:FIRST_JOB_ID' not in preview['commands'][1] or '--dependency=' in preview['commands'][0]:
        raise RuntimeError('Incorrect recovery dependencies')
    subprocess.run(['bash', '-n', str(recover.HERE/'submit.sh'), str(recover.HERE/'job.sbatch')], check=True)
    recover.base.atomic(recover.RUN / 'SUBMISSION_PREVIEW.json', preview)
    recover.base.atomic(recover.RUN / 'PRE_SUBMISSION_CHECK.json', dict(passed=True, epoch=time.time(),
        job_id=os.environ['SLURM_JOB_ID'], checkpoint_time=preview['restart_time'],
        tests=tests, native_executable_unchanged=True, original_launcher_unchanged=True,
        new_jobs_submitted=False, physics_changed=False, recovery_files_sha256=recover.base.read(recover.RUN/'RECOVERY_SHA256.json')))
    print('VERIFIED: two dependent 24h GH jobs, 16 nodes each; restart at', preview['restart_time'], '; not submitted')


if __name__ == '__main__': main()
