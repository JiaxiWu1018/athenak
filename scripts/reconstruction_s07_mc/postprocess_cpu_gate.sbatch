#!/bin/bash
#SBATCH -J st_s07_cpu_gate
#SBATCH -p mi2101x
#SBATCH -A eliasmost
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 00:05:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s07_mc_20261019/logs/%x.%j.log
set -euo pipefail
export ROOT=/work1/eliasmost/jiaxiwu/st_reconstruction_s07_mc_20261019
export PYTHONPATH="$ROOT/src/athenak/scripts/reconstruction_s07_mc:$ROOT/src/athenak/scripts/reconstruction_s06:$ROOT/src/athenak/scripts/reconstruction_s05:$ROOT/src/athenak/vis/python:$ROOT/dependencies/site"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
record="$ROOT/evidence/jobs/${SLURM_JOB_ID}_single"
mkdir -p "$record"
{ hostname; date -Is; module list 2>&1; python3 --version; hipcc --version; } > "$record/environment.txt"
python3 "$ROOT/src/athenak/scripts/reconstruction_s07_mc/verify_runtime.py"
python3 - "${1:?expected isolated gate source directory}" <<'PY'
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

root = Path(os.environ['ROOT'])
stage = Path(sys.argv[1])
pin = json.loads((stage/'GATE_SOURCE_PIN.json').read_text())
for entry in pin['files']:
    assert hashlib.sha256((stage/entry['path']).read_bytes()).hexdigest() == entry['sha256']
assert os.environ.get('SLURM_JOB_ID') and sys.byteorder == 'little'
assert platform.machine() == 'x86_64'
physical = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
limits = {}
candidates = {Path('/sys/fs/cgroup/memory.max'),
              Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')}
for line in Path('/proc/self/cgroup').read_text().splitlines():
    _, controllers, relative = line.split(':', 2)
    if controllers == '':
        base, filename = Path('/sys/fs/cgroup'), 'memory.max'
    elif 'memory' in controllers.split(','):
        base, filename = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes'
    else:
        continue
    path = base/relative.lstrip('/')
    while path == base or base in path.parents:
        candidates.add(path/filename)
        if path == base:
            break
        path = path.parent
for path in sorted(candidates):
    if path.is_file():
        value = path.read_text().strip()
        if value.isdigit() and int(value) > 0:
            limits[str(path)] = int(value)
effective = min([physical] + list(limits.values()))
modules = {}
for name in ['numpy', 'matplotlib', 'pypdfium2', 'imageio_ffmpeg',
             'analyze_cases', 'compare_cases', 'audit_results', 'movies', 'report']:
    module = importlib.import_module(name)
    modules[name] = dict(file=module.__file__, version=getattr(module, '__version__', None))
import imageio_ffmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
version = subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0]
native = json.loads((root/'evidence/SOURCE_PIN.json').read_text())
used = int(subprocess.check_output(['du', '-s', '-B1', '/work1/eliasmost/jiaxiwu'], text=True).split()[0])
result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(), partition=os.environ.get('SLURM_JOB_PARTITION'),
    gate_source=pin['head'], native_runtime_source=native['head'], python=sys.version,
    physical_memory_bytes=physical, cgroup_memory_limits=limits,
    effective_memory_bytes=effective, minimum_memory_bytes=32*1024**3,
    modules=modules, ffmpeg=dict(path=ffmpeg, version=version),
    whole_user_allocated_bytes=used, cpu_only=True, native_data_read=False,
    passed=effective >= 32*1024**3 and used+600000000000 < 1649267441664)
target = root/'evidence/POSTPROCESS_MI2101X_GATE.json'
temporary = target.with_suffix('.tmp')
temporary.write_text(json.dumps(result, indent=2)+'\n')
temporary.replace(target)
assert result['passed'], 'Environment/resource gate failed; retain existing analysis jobs unchanged'
print('POSTPROCESS_MI2101X_GATE_OK', json.dumps(result), flush=True)
PY
