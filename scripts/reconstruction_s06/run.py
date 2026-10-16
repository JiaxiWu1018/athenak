"""Allocated restart integrity gate and unchanged-binary continuation."""
import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from session import *
import numpy as np
import run_case as native
import reduce_run as reduction

def launch(role, end, walltime):
    require_job()
    work = ROOT / 'runs' / role
    work.mkdir(parents=True, exist_ok=False)
    out = work / 'out'
    out.mkdir()
    exe = ROOT / 'protected/athena'
    if sha(exe) != EXE_SHA:
        raise ValueError('Protected executable changed')
    command = ['prun', str(exe), '--kokkos-map-device-id-by=mpi_rank', '-t', walltime,
               '-r', str(RESTART), '-d', str(out), f'time/tlim={end:g}']
    atomic(work / 'launch.json', {'job': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
           'utc': now(), 'command': command, 'compiled_source': PHYSICS,
           'executable_sha256': EXE_SHA, 'restart_sha256': RST_SHA,
           'scripts_commit': subprocess.check_output(['git', '-C', str(HERE.parents[1]), 'rev-parse', 'HEAD'], text=True).strip(),
           'kokkos': KOKKOS, 'absolute_endpoint_M': end})
    begin = time.monotonic()
    with (work / 'solver.log').open('w') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=work)
        while proc.poll() is None:
            with (work / 'memory_samples.jsonl').open('a') as mem:
                mem.write(json.dumps(native.memory_sample(exe)) + '\n')
            time.sleep(15)
    text = (work / 'solver.log').read_text()
    if proc.returncode or 'Terminating on time limit' not in text or '[conservation OK]' not in text:
        raise RuntimeError(f'{role} did not complete cleanly; inspect {work}/solver.log')
    if re.search(r'search_fail=[1-9]|rejected=[1-9]|nonfinite=[1-9]', text):
        raise ValueError('Invalid motion/deposition')
    states = [native.checkpoint(p, initial=ROOT / 'samples' / (CASE + '.bin'))
              for p in sorted((out / 'rst').glob('*.rst'))]
    if not states or abs(max(s['time'] for s in states) - end) > 1e-7:
        raise ValueError('Final checkpoint missing')
    log = reduction.parse_log(str(work / 'solver.log'))
    atomic(work / 'completion.json', {'role': role, 'job': os.environ['SLURM_JOB_ID'], 'endpoint_M': end,
           'solver_wall_seconds': time.monotonic() - begin, 'checkpoints': states,
           'boris_fallbacks': log['boris_fallbacks'], 'boris_fallback_cycles': log['boris_fallback_cycles'],
           'utc': now()})
    return work

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=['smoke', 'production'])
    args = parser.parse_args()
    require_job()
    if args.role == 'smoke':
        if sha(RESTART) != RST_SHA:
            raise ValueError('Restart checksum mismatch')
        state = native.checkpoint(RESTART, expected=300, initial=ROOT / 'samples' / (CASE + '.bin'))
        atomic(ROOT / 'evidence/RESTART_VERIFIED.json', state)
        work = launch('smoke', 306, '00:10:00')
        data = np.atleast_2d(np.loadtxt(work / 'out' / (CASE + '.user.hst')))
        if data.shape[1] != 19 or not np.isfinite(data).all() or not np.any(data[:, 18] > 0.1):
            raise ValueError('Restart timestep/history did not recover')
        atomic(ROOT / 'evidence/SMOKE_OK.json', {'utc': now(), 'job': os.environ['SLURM_JOB_ID'],
               'endpoint_M': 306, 'history_rows': len(data), 'completed_dt_max': float(data[:, 18].max()),
               'constraints_mask': 'Use genuine t=0 or completed dt > 1e-6M, unchanged from session 5'})
    else:
        load(ROOT / 'evidence/SMOKE_OK.json')
        launch('production', END, '03:35:00')
        atomic(ROOT / 'evidence/EVOLUTION_COMPLETE.json', {'utc': now(), 'job': os.environ['SLURM_JOB_ID'], 'endpoint_M': END})

if __name__ == '__main__':
    main()
