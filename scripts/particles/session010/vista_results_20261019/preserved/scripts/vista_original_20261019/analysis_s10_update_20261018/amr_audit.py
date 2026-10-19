"""Verify final full-3D mesh spacing and the native relative-level convention."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from run import OUT, SOURCE, atomic, check_time, digest

if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
    raise RuntimeError('Allocated GG compute required')
check_time()
sys.path.insert(0, str(SOURCE / 'vis/python'))
import bin_convert

manifest = json.loads((OUT / 'INPUTS.json').read_text())
files = list(Path(manifest['ordered_outputs'][-1]).rglob('*.mesh.*.cbin'))
path = max(files, key=lambda p: int(p.name.split('.')[-2]))
data = bin_convert.read_coarsened_binary(str(path))
geometry = np.asarray(data['mb_geometry']); levels = np.asarray(data['mb_logical'])[:, 3]
dx = (geometry[:, 1] - geometry[:, 0]) / 32
if not np.allclose(dx, 4 / 2. ** levels, rtol=0, atol=1e-12):
    raise RuntimeError('Native relative levels disagree with block geometry')
lo, hi = geometry[:, ::2], geometry[:, 1::2]
rmin = np.sqrt(np.sum(np.maximum(np.maximum(lo, 0), -hi) ** 2, axis=1))
wave_dx = dx[rmin <= 46 + 1e-10]
atomic(OUT / 'AMR_HEALTH.json', dict(time=data['time'], cycle=data['cycle'],
    leaf_blocks=len(geometry), finest_dx=float(dx.min()), coarsest_dx=float(dx.max()),
    wave_floor_radius=46, wave_floor_dx=.25,
    wave_floor_satisfied=bool(np.all(wave_dx <= .25 + 1e-12)),
    largest_dx_in_blocks_touching_r46=float(wave_dx.max()),
    level_counts={str(int(k)): int((levels == k).sum()) for k in np.unique(levels)},
    level_geometry_consistency_verified=True,
    mesh_file=str(path), mesh_file_sha256=digest(path),
    method='Binary and coarsened-binary outputs store loc.level minus root_level. Relative levels 0..10 correspond to dx 4..1/256. Geometry uses original 32-cell block widths.',
    source_writer_sha256=digest(SOURCE / 'src/outputs/coarsened_binary.cpp'),
    checked_epoch=time.time()))
print((OUT / 'AMR_HEALTH.json').read_text())
