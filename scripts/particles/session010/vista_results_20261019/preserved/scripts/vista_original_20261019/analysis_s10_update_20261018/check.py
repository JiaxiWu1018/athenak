#!/usr/bin/env python3
"""Run bounded unit suites and retain their actual outputs with the analysis."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import shutil
from run import BASE, OUT, atomic, digest


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
        raise RuntimeError('Run analysis verification inside allocated GG compute')
    logdir = OUT / 'qa/test_logs'; logdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, directory, pattern in [('analysis', 'analysis_s10_update_20261018', 'test_*.py'),
                                      ('launcher', 'launch_s10', 'test_launch.py'),
                                      ('handoff', 'tests', 'test_*.py')]:
        cmd = [sys.executable, '-m', 'unittest', 'discover', '-s', directory, '-p', pattern, '-v']
        result = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        prior = logdir / (name + '.txt')
        if prior.exists():
            archive = logdir / 'history'; archive.mkdir(exist_ok=True)
            shutil.copyfile(prior, archive / f'{time.time_ns()}_{name}.txt')
        (logdir / (name + '.txt')).write_text(output)
        ran = re.search(r'Ran (\d+) tests?', output)
        skipped = re.search(r'OK \(skipped=(\d+)\)', output)
        rows.append(dict(suite=name, command=cmd, returncode=result.returncode,
                         ran=int(ran[1]) if ran else None, skipped=int(skipped[1]) if skipped else 0,
                         log=str((logdir / (name + '.txt')).relative_to(OUT))))
        if result.returncode:
            raise RuntimeError('Unit suite failed: ' + name)
    atomic(OUT / 'TESTS.json', dict(passed=True, epoch=time.time(), suites=rows,
           script_sha256=digest(__file__), total_run=sum(r['ran'] for r in rows),
           total_skipped=sum(r['skipped'] for r in rows)))
    print('TESTS', sum(r['ran'] for r in rows), 'run;', sum(r['skipped'] for r in rows), 'skipped')


if __name__ == '__main__':
    main()
